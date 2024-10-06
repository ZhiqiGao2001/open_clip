from PIL import Image
import open_clip
from src.open_clip import CLIPWrapper, load_checkpoint
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import itertools
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from tqdm import tqdm

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.antonym_pairs = set()

    def find(self, word):
        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])  # Path compression
        return self.parent[word]

    def union(self, word1, word2):
        root1 = self.find(word1)
        root2 = self.find(word2)

        if root1 != root2:
            if (root1, root2) in self.antonym_pairs or (root2, root1) in self.antonym_pairs:
                raise ValueError(f"Cannot union antonym groups: {root1} and {root2}")

            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add_word(self, word):
        if word not in self.parent:
            self.parent[word] = word
            self.rank[word] = 0

    def add_antonym_pair(self, word1, word2):
        root1 = self.find(word1)
        root2 = self.find(word2)
        if root1 == root2:
            raise ValueError(f"Antonyms cannot belong to the same synonym group: {word1} and {word2}")
        self.antonym_pairs.add((root1, root2))

    def are_antonyms(self, word1, word2):
        root1 = self.find(word1)
        root2 = self.find(word2)
        return (root1, root2) in self.antonym_pairs or (root2, root1) in self.antonym_pairs


def create_relationship_and_dataset(synonym_list, antonym_pairs, batch_size=10):
    uf = UnionFind()

    # Add synonym words and create union
    for syn_set in synonym_list:
        for word in syn_set:
            uf.add_word(word)
        for i in range(1, len(syn_set)):
            uf.union(syn_set[0], syn_set[i])

    # Add antonym relationships
    for word1, word2 in antonym_pairs:
        uf.add_antonym_pair(word1, word2)

    # Flatten synonym list for dataset creation
    flattened_words = [word for sublist in synonym_list for word in sublist]
    train_dataset_ = DataLoader(flattened_words, batch_size=batch_size, shuffle=True)

    return uf, train_dataset_


class GraphDistanceLoss(nn.Module):
    def __init__(self, synonym_boundary=0.95, synonym_amplification_factor=10, unrelated_boundary=0.7, antonym_boundary=0.3, loss_function="square", base_loss=0.0):
        super(GraphDistanceLoss, self).__init__()
        self.synonym_boundary = synonym_boundary
        self.synonym_amplification_factor = synonym_amplification_factor
        self.unrelated_boundary = unrelated_boundary
        self.antonym_boundary = antonym_boundary
        self.loss_function = loss_function
        self.base_loss = base_loss

    def forward(self, model_touse, disjoint_set, texts, tokenizer_):
        embeddings = {}
        for text in texts:
            encoded_text = model_touse.encode_text(tokenizer_(text).to('cuda'))
            encoded_text = F.normalize(encoded_text, p=2, dim=-1)
            embeddings[text] = encoded_text

        loss = 0.0
        all_combinations = list(itertools.combinations(texts, 2))

        for u, v in all_combinations:
            if disjoint_set.find(u) == disjoint_set.find(v):
                loss += synonym_pair_loss(embeddings[u], embeddings[v], self.synonym_boundary, self.loss_function, self.base_loss) * self.synonym_amplification_factor
            elif self.antonym_boundary is not None and disjoint_set.are_antonyms(u, v):
                loss += unrelated_pair_loss(embeddings[u], embeddings[v], self.antonym_boundary, self.loss_function, self.base_loss)
            else:
                loss += unrelated_pair_loss(embeddings[u], embeddings[v], self.unrelated_boundary, self.loss_function, self.base_loss)

        loss /= len(all_combinations)
        return loss


def unrelated_pair_loss(embedding_1, embedding_2, boundary_value=0.5, loss_function="square", base_loss=0.0):
    cosine_sim = F.cosine_similarity(embedding_1, embedding_2)
    if cosine_sim > boundary_value:
        if loss_function == "square":
            return (cosine_sim - boundary_value) ** 2 + base_loss
        elif loss_function == "linear":
            return (cosine_sim - boundary_value) + base_loss
        elif loss_function == "exp":
            return torch.exp(cosine_sim - boundary_value) + base_loss
        elif loss_function == "sqrt":
            return torch.sqrt(cosine_sim - boundary_value) + base_loss
    else:
        return 0.0


def synonym_pair_loss(embedding_1, embedding_2, boundary_value=0.95, loss_function="square", base_loss=0.0):
    cosine_sim = F.cosine_similarity(embedding_1, embedding_2)
    if cosine_sim < boundary_value:
        if loss_function == "square":
            return (boundary_value - cosine_sim) ** 2 + base_loss
        elif loss_function == "linear":
            return (boundary_value - cosine_sim) + base_loss
        elif loss_function == "exp":
            return torch.exp(boundary_value - cosine_sim) + base_loss
        elif loss_function == "sqrt":
            return torch.sqrt(boundary_value - cosine_sim) + base_loss
    else:
        return 0.0


def train_model(model_to_tune, train_loader, num_epochs, criterion, union_relationship,
                scheduler=None, final_save_name='test'):
    model_to_tune.train()  # Set the model to training mode
    named_parameters = list(model_to_tune.named_parameters())

    text_params = []
    for n, p in named_parameters:
        if 'visual' not in n:
            text_params.append(p)

    # set parameter requires_grad to True
    for param in text_params:
        param.requires_grad = True

    print(f"Number of text parameters: {len(text_params)}")

    optimizer = optim.AdamW(
        [
            {"params": text_params},
        ],
        lr=1e-5,
    )

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        running_loss = 0.0

        # Iterate over training data
        for batch in tqdm(train_loader, desc='Batches', leave=False):
            optimizer.zero_grad()  # Zero the gradients

            # Compute the loss for the current batch
            loss = criterion(model_to_tune, union_relationship, batch, tokenizer) * 100

            # Backward pass and optimization
            if type(loss) == float:
                continue

            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print epoch statistics
        avg_loss = running_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # if avg_loss < 0.005:
        #     break

        # Update the learning rate, if scheduler is provided
        if scheduler:
            scheduler.step()

    # Save the final model
    checkpoint_dict = {
        "epoch": num_epochs,
        "name": '',
        "state_dict": model_to_tune.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint_dict, f"text_model/{final_save_name}.pt")

    return model_to_tune


def check_word_distance(model_to_check, tokenizer_, texts_list_, check_inter_group=False):
    embeddings = []
    # Add a progress bar for processing text groups
    for arr in tqdm(texts_list_, desc='Processing text groups'):
        curr_embeddings = []
        for text in arr:
            encoded_text = model_to_check.encode_text(tokenizer_(text).to('cuda'))
            encoded_text = F.normalize(encoded_text, p=2, dim=-1)
            curr_embeddings.append(encoded_text)
        embeddings.append(curr_embeddings)

    cosine_similarity_same_group, num_of_same_group = 0, 0
    cosine_similarity_diff_group, num_of_diff_group = 0, 0

    # Calculate total number of intra-group pairs for progress bar
    total_intra_pairs = sum(len(group) * (len(group) - 1) // 2 for group in embeddings)

    # Add a progress bar for computing intra-group similarities
    with tqdm(total=total_intra_pairs, desc='Computing intra-group similarities') as pbar_intra:
        for group in embeddings:
            for u, v in itertools.combinations(group, 2):
                cosine_similarity = F.cosine_similarity(u, v)
                cosine_similarity_same_group += cosine_similarity.detach().cpu().numpy()
                num_of_same_group += 1
                pbar_intra.update(1)

    if check_inter_group:
        # Calculate total number of inter-group pairs for progress bar
        inter_group_combinations = list(itertools.combinations(embeddings, 2))
        total_inter_pairs = sum(len(g1) * len(g2) for g1, g2 in inter_group_combinations)

        # Add a progress bar for computing inter-group similarities
        with tqdm(total=total_inter_pairs, desc='Computing inter-group similarities') as pbar_inter:
            for group1, group2 in inter_group_combinations:
                for u in group1:
                    for v in group2:
                        cosine_similarity = F.cosine_similarity(u, v)
                        cosine_similarity_diff_group += cosine_similarity.detach().cpu().numpy()
                        num_of_diff_group += 1
                        pbar_inter.update(1)

        print(f"Average distance within the same group: {cosine_similarity_same_group / num_of_same_group}")
        print(f"Average distance between different groups: {cosine_similarity_diff_group / num_of_diff_group}")
    else:
        print(f"Average distance within the same group: {cosine_similarity_same_group / num_of_same_group}")



def print_distance_2d_array(model_to_check, tokenizer_, texts_list_):
    all_words_flat = [word for sublist in texts_list_ for word in sublist[:2]]

    embeddings = []
    for text in all_words_flat:
        encoded_text = model_to_check.encode_text(tokenizer_(text).to('cuda'))
        encoded_text = F.normalize(encoded_text, p=2, dim=-1)
        embeddings.append(encoded_text)
    cosine_similarity_matrix = np.zeros((len(all_words_flat), len(all_words_flat)))
    for i, j in itertools.combinations(range(len(all_words_flat)), 2):
        cosine_similarity = F.cosine_similarity(embeddings[i], embeddings[j])
        cosine_similarity_matrix[i, j] = cosine_similarity.detach().cpu().numpy()
        cosine_similarity_matrix[j, i] = cosine_similarity.detach().cpu().numpy()
    df = pd.DataFrame(cosine_similarity_matrix, index=all_words_flat, columns=all_words_flat)
    print(df)


def prepare_wordnet_dataset():
    synonym_arrays = []
    antonym_pairs = []
    for synset in wn.all_synsets():
        synonyms = synset.lemma_names()
        # replace underscore with space
        synonyms = [synonym.replace("_", " ") for synonym in synonyms]
        synonym_arrays.append(synonyms)
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonym_pairs.append((lemma.name().replace("_", " "), lemma.antonyms()[0].name().replace("_", " ")))
                break
    # if antonym (a, b) is in antonym_pairs, then (b, a) should not be in antonym_pairs
    unique_pairs = set()

    # Iterate over the original list
    for a, b in antonym_pairs:
        # Sort the pair to standardize the order
        sorted_pair = tuple(sorted((a, b)))
        unique_pairs.add(sorted_pair)

    # Convert the set back to a list of pairs
    antonym_pairs = list(unique_pairs)

    return synonym_arrays, antonym_pairs


wordnet_synonyms, wordnet_antonyms = prepare_wordnet_dataset()
# print(len(wordnet_synonyms), len(wordnet_antonyms))
#
# first_elements = set()
# for synonyms in wordnet_synonyms:
#     if synonyms:  # Ensure the list is not empty
#         first_elements.add(synonyms[0])
# antonym_words = set()
# for a, b in wordnet_antonyms:
#     antonym_words.add(a)
#     antonym_words.add(b)
#
# missing_words = antonym_words - first_elements
# print(len(missing_words))
#
#
# exit()
imagenet_100 = [['restaurant', 'eating_house', 'eating_place', 'eatery'], ['sea_slug', 'nudibranch'], ['cornet', 'horn', 'trumpet', 'trump'], ['indigo_bunting', 'indigo_finch', 'indigo_bird', 'Passerina_cyanea'], ['medicine_chest', 'medicine_cabinet'], ['patas', 'hussar_monkey', 'Erythrocebus_patas'], ['sunscreen', 'sunblock', 'sun_blocker'], ['horned_viper', 'cerastes', 'sand_viper', 'horned_asp', 'Cerastes_cornutus'], ['caldron', 'cauldron'], ['fire_screen', 'fireguard'], ['water_buffalo', 'water_ox', 'Asiatic_buffalo', 'Bubalus_bubalis'], ['axolotl', 'mud_puppy', 'Ambystoma_mexicanum'], ['trailer_truck', 'tractor_trailer', 'trucking_rig', 'rig', 'articulated_lorry', 'semi'], ['Saint_Bernard', 'St_Bernard'], ['crossword_puzzle', 'crossword'], ['Maltese_dog', 'Maltese_terrier', 'Maltese'], ['barn_spider', 'Araneus_cavaticus'], ['oystercatcher', 'oyster_catcher'], ['Shetland_sheepdog', 'Shetland_sheep_dog', 'Shetland'], ['wild_boar', 'boar', 'Sus_scrofa'], ['odometer', 'hodometer', 'mileometer', 'milometer'], ['saltshaker', 'salt_shaker'], ['little_blue_heron', 'Egretta_caerulea'], ['thunder_snake', 'worm_snake', 'Carphophis_amoenus'], ['lawn_mower', 'mower'], ['wallaby', 'brush_kangaroo'], ['gas_pump', 'gasoline_pump', 'petrol_pump', 'island_dispenser'], ['snow_leopard', 'ounce', 'Panthera_uncia'], ['otterhound', 'otter_hound'], ['toilet_tissue', 'toilet_paper', 'bathroom_tissue'], ['Dandie_Dinmont', 'Dandie_Dinmont_terrier'], ['chain_saw', 'chainsaw'], ['radio_telescope', 'radio_reflector'], ['letter_opener', 'paper_knife', 'paperknife'], ['porcupine', 'hedgehog'], ['hot_pot', 'hotpot'], ['dingo', 'warrigal', 'warragal', 'Canis_dingo'], ['bakery', 'bakeshop', 'bakehouse'], ['disk_brake', 'disc_brake'], ['king_snake', 'kingsnake'], ['remote_control', 'remote'], ['grand_piano', 'grand'], ['sulphur_butterfly', 'sulfur_butterfly'], ['spider_monkey', 'Ateles_geoffroyi'], ['horizontal_bar', 'high_bar'], ['golfcart', 'golf_cart'], ['Arctic_fox', 'white_fox', 'Alopex_lagopus'], ['chambered_nautilus', 'pearly_nautilus', 'nautilus'], ['apiary', 'bee_house'], ['megalith', 'megalithic_structure'], ['chocolate_sauce', 'chocolate_syrup'], ['bicycle-built-for-two', 'tandem_bicycle', 'tandem'], ['red_wolf', 'maned_wolf', 'Canis_rufus', 'Canis_niger'], ['motor_scooter', 'scooter'], ['pool_table', 'billiard_table', 'snooker_table'], ['breakwater', 'groin', 'groyne', 'mole', 'bulwark', 'seawall', 'jetty'], ['lycaenid', 'lycaenid_butterfly'], ['French_horn', 'horn'], ['violin', 'fiddle'], ['Pembroke', 'Pembroke_Welsh_corgi'], ['gorilla', 'Gorilla_gorilla'], ['cockroach', 'roach'], ['rubber_eraser', 'rubber', 'pencil_eraser'], ['cougar', 'puma', 'catamount', 'mountain_lion', 'painter', 'panther', 'Felis_concolor'], ['redshank', 'Tringa_totanus'], ['refrigerator', 'icebox'], ['Airedale', 'Airedale_terrier'], ['sports_car', 'sport_car'], ['white_stork', 'Ciconia_ciconia'], ['bookshop', 'bookstore', 'bookstall'], ['cellular_telephone', 'cellular_phone', 'cellphone', 'cell', 'mobile_phone'], ['handkerchief', 'hankie', 'hanky', 'hankey'], ['corkscrew', 'bottle_screw'], ['cello', 'violoncello'], ['black_and_gold_garden_spider', 'Argiope_aurantia'], ['tow_truck', 'tow_car', 'wrecker'], ['frilled_lizard', 'Chlamydosaurus_kingi'], ['American_black_bear', 'black_bear', 'Ursus_americanus', 'Euarctos_americanus'], ['Saluki', 'gazelle_hound'], ['hand-held_computer', 'hand-held_microcomputer'], ['rocking_chair', 'rocker'], ['hippopotamus', 'hippo', 'river_horse', 'Hippopotamus_amphibius'], ['cocker_spaniel', 'English_cocker_spaniel', 'cocker'], ['oboe', 'hautboy', 'hautbois'], ['junco', 'snowbird'], ['shoe_shop', 'shoe-shop', 'shoe_store'], ['timber_wolf', 'grey_wolf', 'gray_wolf', 'Canis_lupus'], ['Indian_cobra', 'Naja_naja'], ['valley', 'vale'], ['promontory', 'headland', 'head', 'foreland'], ['common_newt', 'Triturus_vulgaris'], ['wallet', 'billfold', 'notecase', 'pocketbook'], ['ballpoint', 'ballpoint_pen', 'ballpen', 'Biro'], ['can_opener', 'tin_opener'], ['picket_fence', 'paling'], ['killer_whale', 'killer', 'orca', 'grampus', 'sea_wolf', 'Orcinus_orca'], ['starfish', 'sea_star'], ['tailed_frog', 'bell_toad', 'ribbed_toad', 'tailed_toad', 'Ascaphus_trui'], ['greenhouse', 'nursery', 'glasshouse'], ['Scottish_deerhound', 'deerhound']]

# Union, train_dataset = create_relationship_and_dataset(imagenet_100, [], batch_size=10)

wordnet_union, train_dataset = create_relationship_and_dataset(wordnet_synonyms, [], batch_size=20)

tokenizer = open_clip.get_tokenizer('ViT-B-32')
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model = model.to('cuda')

with torch.no_grad():
    # print_distance_2d_array(model, tokenizer, emotion_synonym)
    check_word_distance(model, tokenizer, wordnet_synonyms)

model_trained_ = train_model(
    model_to_tune=model,
    train_loader=train_dataset,
    num_epochs=25,
    criterion=GraphDistanceLoss(synonym_boundary=0.95, synonym_amplification_factor=20, unrelated_boundary=0.5, antonym_boundary=0.3, loss_function="square", base_loss=0.0),
    union_relationship=wordnet_union,
    final_save_name='wordnet_synonyms'
)

with torch.no_grad():
    check_word_distance(model_trained_, tokenizer, wordnet_synonyms)
    # print_distance_2d_array(model_trained_, tokenizer, emotion_synonym)

