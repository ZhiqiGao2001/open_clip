import networkx as nx
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
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


def train_model(model, train_loader, num_epochs, criterion, graph, optimizer, device='cuda',
                scheduler=None, final_save_name='test'):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over training data
        for batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Compute the loss for the current batch
            loss = criterion(model, graph, batch, tokenizer)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Print epoch statistics
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        # if avg_loss < 0.005:
        #     break

        # Update the learning rate, if scheduler is provided
        if scheduler:
            scheduler.step()

    # Save the final model
    checkpoint_dict = {
        "epoch": num_epochs,
        "name": '',
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint_dict, f"text_model/{final_save_name}.pt")

    return model


class GraphDistanceLoss(nn.Module):
    def __init__(self, scale_factor=2.0, pseudo=0.0):
        super(GraphDistanceLoss, self).__init__()
        self.scale_factor = scale_factor
        self.pseudo = pseudo

    def forward(self, model, graph, texts, tokenizer_):
        embeddings = {}
        for text in texts:
            encoded_text = model.encode_text(tokenizer_(text).to('cuda'))
            encoded_text = F.normalize(encoded_text, p=2, dim=-1)
            embeddings[text] = encoded_text

        loss = 0.0
        all_combinations = list(itertools.combinations(texts, 2))

        for u, v in all_combinations:
            # Scale down the target distance
            weight_edge = graph[u][v]['weight']
            target_distance = weight_edge / self.scale_factor

            if self.pseudo:
                split_part = int(embeddings[u].shape[1] * self.pseudo)
                distance = (torch.mm(embeddings[u][:, :split_part], embeddings[v][:, :split_part].T) -
                            torch.mm(embeddings[u][:, split_part:], embeddings[v][:, split_part:].T))
                curr_loss = (abs(distance) - target_distance) ** 2
            else:
                cosine_similarity = F.cosine_similarity(embeddings[u], embeddings[v])
                curr_loss = (cosine_similarity - (1 - target_distance)) ** 2

            # Compute the loss
            if weight_edge == 0:
                curr_loss = curr_loss * 100
            loss += curr_loss

        loss /= len(all_combinations)
        return loss


def create_graph_and_dataset(synonym_list, weight_default=1, batch_size=10):
    G = nx.Graph()
    for group in synonym_list:
        G.add_nodes_from(group)
        G.add_edges_from(itertools.combinations(group, 2), weight=0)
    for group1, group2 in itertools.combinations(synonym_list, 2):
        G.add_edges_from(itertools.product(group1, group2), weight=weight_default)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    flattened_words = [word for sublist in synonym_list for word in sublist]
    train_dataset = DataLoader(flattened_words, batch_size=batch_size, shuffle=True)
    return G, train_dataset


def check_word_distance(model_to_check, tokenizer_, texts_list_, pseudo=0.0):
    embeddings = []
    for arr in texts_list_:
        curr_embeddings = []
        for text in arr:
            encoded_text = model_to_check.encode_text(tokenizer_(text).to('cuda'))
            encoded_text = F.normalize(encoded_text, p=2, dim=-1)
            curr_embeddings.append(encoded_text)
        embeddings.append(curr_embeddings)
    cosine_similarity_same_group, num_of_same_group = 0, 0
    cosine_similarity_diff_group, num_of_diff_group = 0, 0
    for group in embeddings:
        for u, v in itertools.combinations(group, 2):
            if pseudo == 0.0:
                cosine_similarity = F.cosine_similarity(u, v)
            else:
                split_part = int(u.shape[1] * pseudo)
                distance = (torch.mm(u[:, :split_part], v[:, :split_part].T) -
                            torch.mm(u[:, split_part:], v[:, split_part:].T))
                cosine_similarity = abs(distance)
            cosine_similarity_same_group += cosine_similarity.detach().cpu().numpy()
            num_of_same_group += 1
    for group1, group2 in itertools.combinations(embeddings, 2):
        for u, v in itertools.product(group1, group2):
            if pseudo == 0.0:
                cosine_similarity = F.cosine_similarity(u, v)
            else:
                split_part = int(u.shape[1] * pseudo)
                distance = (torch.mm(u[:, :split_part], v[:, :split_part].T) -
                            torch.mm(u[:, split_part:], v[:, split_part:].T))
                cosine_similarity = abs(distance)
            cosine_similarity_diff_group += cosine_similarity.detach().cpu().numpy()
            num_of_diff_group += 1
    if pseudo == 0.0:
        print(f"Average cosine similarity within the same group: {cosine_similarity_same_group / num_of_same_group}")
        print(f"Average cosine similarity between different groups: {cosine_similarity_diff_group / num_of_diff_group}")
    else:
        print(f"Average distance within the same group: {cosine_similarity_same_group / num_of_same_group}")
        print(f"Average distance between different groups: {cosine_similarity_diff_group / num_of_diff_group}")



emotion_synonym = [
    ['angry', 'furious', 'wild', 'raging', 'tempestuous'],
    ['disgust', 'nauseate', 'churn up', 'sicken', 'gross out', 'revolt', 'repel'],
    ['fear', 'fearfulness', 'reverence', 'fright', 'dread', 'awe'],
    ['happy', 'glad', 'felicitous'],
    ['neutral', 'indifferent'],
    ['sad', 'deplorable', 'distressing', 'lamentable', 'sorry', 'pitiful'],
    ['surprise', "astonished", "amazed", "shocked"]
]

imagenet_20 = [['grey_fox', 'gray_fox', 'Urocyon_cinereoargenteus'], ['loupe', "jeweler's_loupe"], ['prairie_chicken', 'prairie_grouse', 'prairie_fowl'], ['stinkhorn', 'carrion_fungus'], ['backpack', 'back_pack', 'knapsack', 'packsack', 'rucksack', 'haversack'], ['breakwater', 'groin', 'groyne', 'mole', 'bulwark', 'seawall', 'jetty'], ['impala', 'Aepyceros_melampus'], ['brassiere', 'bra', 'bandeau'], ['diamondback', 'diamondback_rattlesnake', 'Crotalus_adamanteus'], ['motor_scooter', 'scooter'], ['tailed_frog', 'bell_toad', 'ribbed_toad', 'tailed_toad', 'Ascaphus_trui'], ['chiffonier', 'commode'], ['tricycle', 'trike', 'velocipede'], ['pay-phone', 'pay-station'], ['ringneck_snake', 'ring-necked_snake', 'ring_snake'], ['hippopotamus', 'hippo', 'river_horse', 'Hippopotamus_amphibius'], ['streetcar', 'tram', 'tramcar', 'trolley', 'trolley_car'], ['pop_bottle', 'soda_bottle'], ['fox_squirrel', 'eastern_fox_squirrel', 'Sciurus_niger'], ['chain_mail', 'ring_mail', 'mail', 'chain_armor', 'chain_armour', 'ring_armor', 'ring_armour']]
imagenet_100 = [['restaurant', 'eating_house', 'eating_place', 'eatery'], ['sea_slug', 'nudibranch'], ['cornet', 'horn', 'trumpet', 'trump'], ['indigo_bunting', 'indigo_finch', 'indigo_bird', 'Passerina_cyanea'], ['medicine_chest', 'medicine_cabinet'], ['patas', 'hussar_monkey', 'Erythrocebus_patas'], ['sunscreen', 'sunblock', 'sun_blocker'], ['horned_viper', 'cerastes', 'sand_viper', 'horned_asp', 'Cerastes_cornutus'], ['caldron', 'cauldron'], ['fire_screen', 'fireguard'], ['water_buffalo', 'water_ox', 'Asiatic_buffalo', 'Bubalus_bubalis'], ['axolotl', 'mud_puppy', 'Ambystoma_mexicanum'], ['trailer_truck', 'tractor_trailer', 'trucking_rig', 'rig', 'articulated_lorry', 'semi'], ['Saint_Bernard', 'St_Bernard'], ['crossword_puzzle', 'crossword'], ['Maltese_dog', 'Maltese_terrier', 'Maltese'], ['barn_spider', 'Araneus_cavaticus'], ['oystercatcher', 'oyster_catcher'], ['Shetland_sheepdog', 'Shetland_sheep_dog', 'Shetland'], ['wild_boar', 'boar', 'Sus_scrofa'], ['odometer', 'hodometer', 'mileometer', 'milometer'], ['saltshaker', 'salt_shaker'], ['little_blue_heron', 'Egretta_caerulea'], ['thunder_snake', 'worm_snake', 'Carphophis_amoenus'], ['lawn_mower', 'mower'], ['wallaby', 'brush_kangaroo'], ['gas_pump', 'gasoline_pump', 'petrol_pump', 'island_dispenser'], ['snow_leopard', 'ounce', 'Panthera_uncia'], ['otterhound', 'otter_hound'], ['toilet_tissue', 'toilet_paper', 'bathroom_tissue'], ['Dandie_Dinmont', 'Dandie_Dinmont_terrier'], ['chain_saw', 'chainsaw'], ['radio_telescope', 'radio_reflector'], ['letter_opener', 'paper_knife', 'paperknife'], ['porcupine', 'hedgehog'], ['hot_pot', 'hotpot'], ['dingo', 'warrigal', 'warragal', 'Canis_dingo'], ['bakery', 'bakeshop', 'bakehouse'], ['disk_brake', 'disc_brake'], ['king_snake', 'kingsnake'], ['remote_control', 'remote'], ['grand_piano', 'grand'], ['sulphur_butterfly', 'sulfur_butterfly'], ['spider_monkey', 'Ateles_geoffroyi'], ['horizontal_bar', 'high_bar'], ['golfcart', 'golf_cart'], ['Arctic_fox', 'white_fox', 'Alopex_lagopus'], ['chambered_nautilus', 'pearly_nautilus', 'nautilus'], ['apiary', 'bee_house'], ['megalith', 'megalithic_structure'], ['chocolate_sauce', 'chocolate_syrup'], ['bicycle-built-for-two', 'tandem_bicycle', 'tandem'], ['red_wolf', 'maned_wolf', 'Canis_rufus', 'Canis_niger'], ['motor_scooter', 'scooter'], ['pool_table', 'billiard_table', 'snooker_table'], ['breakwater', 'groin', 'groyne', 'mole', 'bulwark', 'seawall', 'jetty'], ['lycaenid', 'lycaenid_butterfly'], ['French_horn', 'horn'], ['violin', 'fiddle'], ['Pembroke', 'Pembroke_Welsh_corgi'], ['gorilla', 'Gorilla_gorilla'], ['cockroach', 'roach'], ['rubber_eraser', 'rubber', 'pencil_eraser'], ['cougar', 'puma', 'catamount', 'mountain_lion', 'painter', 'panther', 'Felis_concolor'], ['redshank', 'Tringa_totanus'], ['refrigerator', 'icebox'], ['Airedale', 'Airedale_terrier'], ['sports_car', 'sport_car'], ['white_stork', 'Ciconia_ciconia'], ['bookshop', 'bookstore', 'bookstall'], ['cellular_telephone', 'cellular_phone', 'cellphone', 'cell', 'mobile_phone'], ['handkerchief', 'hankie', 'hanky', 'hankey'], ['corkscrew', 'bottle_screw'], ['cello', 'violoncello'], ['black_and_gold_garden_spider', 'Argiope_aurantia'], ['tow_truck', 'tow_car', 'wrecker'], ['frilled_lizard', 'Chlamydosaurus_kingi'], ['American_black_bear', 'black_bear', 'Ursus_americanus', 'Euarctos_americanus'], ['Saluki', 'gazelle_hound'], ['hand-held_computer', 'hand-held_microcomputer'], ['rocking_chair', 'rocker'], ['hippopotamus', 'hippo', 'river_horse', 'Hippopotamus_amphibius'], ['cocker_spaniel', 'English_cocker_spaniel', 'cocker'], ['oboe', 'hautboy', 'hautbois'], ['junco', 'snowbird'], ['shoe_shop', 'shoe-shop', 'shoe_store'], ['timber_wolf', 'grey_wolf', 'gray_wolf', 'Canis_lupus'], ['Indian_cobra', 'Naja_naja'], ['valley', 'vale'], ['promontory', 'headland', 'head', 'foreland'], ['common_newt', 'Triturus_vulgaris'], ['wallet', 'billfold', 'notecase', 'pocketbook'], ['ballpoint', 'ballpoint_pen', 'ballpen', 'Biro'], ['can_opener', 'tin_opener'], ['picket_fence', 'paling'], ['killer_whale', 'killer', 'orca', 'grampus', 'sea_wolf', 'Orcinus_orca'], ['starfish', 'sea_star'], ['tailed_frog', 'bell_toad', 'ribbed_toad', 'tailed_toad', 'Ascaphus_trui'], ['greenhouse', 'nursery', 'glasshouse'], ['Scottish_deerhound', 'deerhound']]

# G, train_dataset = create_graph_and_dataset(imagenet_20, weight_default=1)
G, train_dataset = create_graph_and_dataset(imagenet_100, weight_default=1, batch_size=10)

tokenizer = open_clip.get_tokenizer('ViT-B-32')
model_, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model_.eval()
model = model_.to('cuda')

# clip, _, preprocess_ = open_clip.create_model_and_transforms('ViT-B-32')
# model_ = CLIPWrapper(clip, projection_dim=64)
# path = f"/hdd5/zhiqi2/open_clip/zero_shot_analyze/models/64_euclidean.pt"
# pseudo_path = f"/hdd5/zhiqi2/open_clip/zero_shot_analyze/models/64_pseudo.pt"
# load_checkpoint(model_, pseudo_path)

named_parameters = list(model_.named_parameters())

text_params = []
for n, p in named_parameters:
    if 'visual' not in n:
        text_params.append(p)

print(f"Number of text parameters: {len(text_params)}")

optimizer = optim.AdamW(
    [
        {"params": text_params},
    ],
    lr=1e-5,
)

with torch.no_grad():
    check_word_distance(model_, tokenizer, imagenet_100 )

model_trained_ = train_model(
    model=model_,
    train_loader=train_dataset,
    num_epochs=20,
    criterion=GraphDistanceLoss(),
    # criterion=GraphDistanceLoss(scale_factor=5, pseudo=0.5),
    graph=G,
    optimizer=optimizer,
    device='cuda',
    final_save_name='imgnet_100_'
)

with torch.no_grad():
    check_word_distance(model_trained_, tokenizer, imagenet_100)

