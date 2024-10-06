import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
from src.open_clip import CLIPWrapper, load_checkpoint
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import itertools
import nltk
from nltk.corpus import wordnet as wn
import random


class custom_classification(Dataset):
    def __init__(self, path_csv, transform=None):
        self.data = pd.read_csv(path_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = Image.open(img_path)
        label = self.data.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label


def run_CLIP(model_, tokenizer_, class_names_, class_templates_, torch_dataset_, directory, pseudo_split=0.0, binary=True):
    loader = torch.utils.data.DataLoader(torch_dataset_, batch_size=500)
    texts = [template.format(cls) for template in class_templates_ for cls in class_names_]
    print('Texts:', texts)
    tokenize_texts = tokenizer_(texts).to('cuda')
    text_features = model_.encode_text(tokenize_texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    top1, n = 0., 0.

    if binary:
        distance_to_class_0 = []
        distance_to_class_1 = []

    model_.eval()
    with torch.no_grad():
        for images, target in loader:
            images = images.to('cuda')
            target = target.to('cuda')
            # predict
            image_features = model_.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # stack text features with batch size
            if pseudo_split == 0.0:
                logits = 100. * image_features @ text_features.t()
            else:
                split_part = int(image_features.shape[1] * pseudo_split)
                image_features_part1 = image_features[:, :split_part]
                image_features_part2 = image_features[:, split_part:]
                text_features_part1 = text_features[:, :split_part]
                text_features_part2 = text_features[:, split_part:]
                logits = 100. * (image_features_part1 @ text_features_part1.T - image_features_part2 @ text_features_part2.T)

            probability = torch.nn.functional.softmax(logits, dim=-1)
            prediction = probability.argmax(dim=-1)
            top1 += (prediction == target).sum().item()
            n += images.size(0)

            if binary:
                for i in range(target.size(0)):
                    euclidean_distance = torch.nn.functional.pairwise_distance(image_features[i], text_features)
                    if target[i] == 0:
                        distance_to_class_0.append(euclidean_distance.detach().cpu().numpy())
                    else:
                        distance_to_class_1.append(euclidean_distance.detach().cpu().numpy())

    top1 = top1 / n * 100
    print(f"Top-1 accuracy: {top1:.2f}%\n")
    del tokenize_texts, text_features, logits, probability, prediction, image_features
    torch.cuda.empty_cache()
    if not binary:
        return top1
    else:
        distance_class_0 = np.array(distance_to_class_0)
        distance_class_1 = np.array(distance_to_class_1)

        fig, ax = plt.subplots()
        plt.scatter([i[0] for i in distance_class_0], [i[1] for i in distance_class_0], color='red', label=texts[0],
                    s=1)
        plt.scatter([i[0] for i in distance_class_1], [i[1] for i in distance_class_1], color='blue', label=texts[1],
                    s=1)

        plt.xlabel(f'Distance to class {texts[0]}')
        plt.ylabel(f'Distance to class {texts[1]}')
        # set accuracy as title
        plt.title(f'Top-1 accuracy: {top1:.2f}%')
        ax.set_aspect('equal', 'box')
        plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
        plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
        ax.plot(plt.xlim(), plt.ylim(), color='black', linestyle='--')

        plt.legend()
        plt.savefig(f'zero_shot_analyze/{directory}/{texts[0]}_{texts[1]}.png')
        plt.close()
        distance_text_0_class_0, distance_text_1_class_0 = distance_class_0[:, 0], distance_class_0[:, 1]
        distance_text_0_class_1, distance_text_1_class_1 = distance_class_1[:, 0], distance_class_1[:, 1]
        mean_0_0, std_0_0 = np.mean(distance_text_0_class_0), np.std(distance_text_0_class_0)
        mean_1_0, std_1_0 = np.mean(distance_text_1_class_0), np.std(distance_text_1_class_0)
        mean_0_1, std_0_1 = np.mean(distance_text_0_class_1), np.std(distance_text_0_class_1)
        mean_1_1, std_1_1 = np.mean(distance_text_1_class_1), np.std(distance_text_1_class_1)
        return [texts[0], texts[1]], [top1, mean_0_0, std_0_0, mean_1_0, std_1_0, mean_0_1, std_0_1, mean_1_1, std_1_1]


def print_pairwise_distance(model_, text_array_, text_template, tokenizer_):
    text_array_ = [template.format(cls) for template in text_template for cls in text_array_]
    copied_text_array_ = list(copy.deepcopy(text_array_))
    text_array_ = tokenizer_(text_array_).to('cuda')
    text_features = model_.encode_text(text_array_)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.detach().cpu().numpy()
    # print euclidean distance between text features by np.linalg.norm()
    distance = np.linalg.norm(text_features[:, None] - text_features, axis=-1)
    # print pairwise distance
    print('Pairwise distance:')
    print(pd.DataFrame(distance, columns=copied_text_array_, index=copied_text_array_))


def compare_templates(model_, tokenizer_, class_1, class_2, template_1, template_2, torch_dataset_):
    all_classes = set(class_1 + class_2)
    print_pairwise_distance(model_, all_classes, template_1, tokenizer_)
    print_pairwise_distance(model_, all_classes, template_2, tokenizer_)
    print('\n')
    run_CLIP(model_, tokenizer_, class_1, template_1, torch_dataset_)
    run_CLIP(model_, tokenizer_, class_2, template_1, torch_dataset_)
    run_CLIP(model_, tokenizer_, class_1, template_2, torch_dataset_)
    run_CLIP(model_, tokenizer_, class_2, template_2, torch_dataset_)
    print('#'*50 + '\n')


def distance_pseudo(split, text_feature_1, text_feature_2):
    split_part = int(text_feature_1.shape[1] * split)
    text_feature_1_part1 = text_feature_1[:, :split_part]
    text_feature_1_part2 = text_feature_1[:, split_part:]
    text_feature_2_part1 = text_feature_2[:, :split_part]
    text_feature_2_part2 = text_feature_2[:, split_part:]
    distance = np.linalg.norm(text_feature_1_part1 - text_feature_2_part1) - np.linalg.norm(text_feature_1_part2 - text_feature_2_part2)
    return distance


def synonym_comparison_binary(model_, tokenizer_, synonym_1, synonym_2, class_templates_, torch_dataset_, pseudo_split=0.0, folder_name="Original", csv_name="result.csv"):
    # make directory under zero_shot_analyze/{synonym_1[0]}_{synonym_2[0]}_result if not exist
    dir_name = f'{folder_name}_{synonym_1[0]}_{synonym_2[0]}_result'
    if pseudo_split != 0.0:
        dir_name += f'_pseudo_{pseudo_split}'
    if not os.path.exists(f'zero_shot_analyze/{dir_name}'):
        os.makedirs(f'zero_shot_analyze/{dir_name}')
    print("Length of dataset:", len(torch_dataset_))
    # initialize a dataframe to store the statistics
    dataframe = pd.DataFrame(columns=['Satisfy distance inequality', 'Text 0', 'Text 1', 'Top-1 Accuracy',
                                        'Mean Distance to text 0 for class 0 image', 'Std Distance to text 0 for class 0 image',
                                        'Mean Distance to text 1 for class 0 image', 'Std Distance to text 1 for class 0 image',
                                        'Mean Distance to text 0 for class 1 image', 'Std Distance to text 0 for class 1 image',
                                        'Mean Distance to text 1 for class 1 image', 'Std Distance to text 1 for class 1 image'])

    texts, stats = run_CLIP(model_, tokenizer_, [synonym_1[0], synonym_2[0]], class_templates_, torch_dataset_, dir_name, pseudo_split)
    dataframe.loc[len(dataframe)] = ["Null"] + [texts[0], texts[1]] + stats

    synonym_1_features = model_.encode_text(tokenizer_([class_templates_[0].format(syn) for syn in synonym_1]).to('cuda'))
    synonym_2_features = model_.encode_text(tokenizer_([class_templates_[0].format(syn) for syn in synonym_2]).to('cuda'))
    synonym_1_features /= synonym_1_features.norm(dim=-1, keepdim=True)
    synonym_2_features /= synonym_2_features.norm(dim=-1, keepdim=True)
    synonym_1_features = synonym_1_features.detach().cpu().numpy()
    synonym_2_features = synonym_2_features.detach().cpu().numpy()
    if pseudo_split == 0.0:
        distance_original = np.linalg.norm(synonym_1_features[0] - synonym_2_features[0])
    else:
        distance_original = distance_pseudo(pseudo_split, synonym_1_features, synonym_2_features)

    for i in range(1, len(synonym_1_features)):
        if pseudo_split == 0.0:
            distance_synonym = np.linalg.norm(synonym_1_features[0] - synonym_1_features[i])
            distance_antonym = np.linalg.norm(synonym_2_features[0] - synonym_1_features[i])
        else:
            distance_synonym = distance_pseudo(pseudo_split, synonym_1_features, synonym_1_features)
            distance_antonym = distance_pseudo(pseudo_split, synonym_2_features, synonym_1_features)
        flag = 0
        if distance_synonym > distance_antonym:
            flag += 1
        if distance_synonym > distance_original:
            flag += 1
        texts, stats = run_CLIP(model_, tokenizer_, [synonym_1[i], synonym_2[0]], class_templates_, torch_dataset_, dir_name, pseudo_split)
        # flag in the first column
        dataframe.loc[len(dataframe)] = [flag] + [texts[0], texts[1]] + stats

    for i in range(1, len(synonym_2_features)):
        if pseudo_split == 0.0:
            distance_synonym = np.linalg.norm(synonym_2_features[0] - synonym_2_features[i])
            distance_antonym = np.linalg.norm(synonym_1_features[0] - synonym_2_features[i])
        else:
            distance_synonym = distance_pseudo(pseudo_split, synonym_2_features, synonym_2_features)
            distance_antonym = distance_pseudo(pseudo_split, synonym_1_features, synonym_2_features)
        flag = 0
        if distance_synonym > distance_antonym:
            flag += 1
        if distance_synonym > distance_original:
            flag += 1
        texts, stats = run_CLIP(model_, tokenizer_, [synonym_1[0], synonym_2[i]], class_templates_, torch_dataset_, dir_name, pseudo_split)
        dataframe.loc[len(dataframe)] = [flag] + [texts[0], texts[1]] + stats

    dataframe.to_csv(f'zero_shot_analyze/{dir_name}/{csv_name}', index=False)


def synonym_comparison(model_, tokenizer_, original, combination_list, class_templates_, torch_dataset_, pseudo_split=0.0, folder_name="Original", csv_name="result.csv"):
    # make directory under zero_shot_analyze/{synonym_1[0]}_{synonym_2[0]}_result if not exist
    dir_name = f'{folder_name}'
    if pseudo_split != 0.0:
        dir_name += f'_pseudo_{pseudo_split}'
    if not os.path.exists(f'zero_shot_analyze/{dir_name}'):
        os.makedirs(f'zero_shot_analyze/{dir_name}')
    print("Length of dataset:", len(torch_dataset_))
    # initialize a dataframe to store the statistics
    dataframe = pd.DataFrame(columns=['Class_name', 'Top-1 Accuracy'])

    accuracy_original = run_CLIP(model_, tokenizer_, original, class_templates_, torch_dataset_, dir_name, pseudo_split, binary=False)
    dataframe.loc[len(dataframe)] = [original, accuracy_original]

    avg_accuracy = 0
    with torch.no_grad():
        for this_comb in combination_list:
            accuracy = run_CLIP(model_, tokenizer_, this_comb, class_templates_, torch_dataset_, dir_name, pseudo_split,
                                binary=False)

            # free cuda memory
            torch.cuda.empty_cache()

            dataframe.loc[len(dataframe)] = [this_comb, accuracy]
            avg_accuracy += accuracy

    avg_accuracy /= len(combination_list)
    print(f"Original accuracy: {accuracy_original:.2f} Average accuracy after replacement: {avg_accuracy:.2f}")
    dataframe.to_csv(f'zero_shot_analyze/{dir_name}/{csv_name}', index=False)
    return accuracy_original, avg_accuracy


def load_model(dimension=None, path=None):
    clip, _, preprocess_ = open_clip.create_model_and_transforms('ViT-B-32')
    if dimension is None:
        load_checkpoint(clip, path)
        clip.to('cuda')
        return clip
    else:
        modified_model = CLIPWrapper(clip, projection_dim=dimension)
        load_checkpoint(modified_model, path)
    return modified_model


def generate_combination(synonym_list_, count=50):
    result = []
    for i in range(count):
        temp = []
        for syn in synonym_list_:
            temp.append(random.choice(syn))
        result.append(temp)
    return result


if __name__ == "__main__":
    imagenet_100_ori = ['restaurant', 'sea_slug', 'cornet', 'indigo_bunting', 'medicine_chest', 'patas', 'sunscreen',
                        'horned_viper', 'caldron', 'fire_screen', 'water_buffalo', 'axolotl', 'trailer_truck',
                        'Saint_Bernard', 'crossword_puzzle', 'Maltese_dog', 'barn_spider', 'oystercatcher',
                        'Shetland_sheepdog', 'wild_boar', 'odometer', 'saltshaker', 'little_blue_heron',
                        'thunder_snake', 'lawn_mower', 'wallaby', 'gas_pump', 'snow_leopard', 'otterhound',
                        'toilet_tissue', 'Dandie_Dinmont', 'chain_saw', 'radio_telescope', 'letter_opener', 'porcupine',
                        'hot_pot', 'dingo', 'bakery', 'disk_brake', 'king_snake', 'remote_control', 'grand_piano',
                        'sulphur_butterfly', 'spider_monkey', 'horizontal_bar', 'golfcart', 'Arctic_fox',
                        'chambered_nautilus', 'apiary', 'megalith', 'chocolate_sauce', 'bicycle-built-for-two',
                        'red_wolf', 'motor_scooter', 'pool_table', 'breakwater', 'lycaenid', 'French_horn', 'violin',
                        'Pembroke', 'gorilla', 'cockroach', 'rubber_eraser', 'cougar', 'redshank', 'refrigerator',
                        'Airedale', 'sports_car', 'white_stork', 'bookshop', 'cellular_telephone', 'handkerchief',
                        'corkscrew', 'cello', 'black_and_gold_garden_spider', 'tow_truck', 'frilled_lizard',
                        'American_black_bear', 'Saluki', 'hand-held_computer', 'rocking_chair', 'hippopotamus',
                        'cocker_spaniel', 'oboe', 'junco', 'shoe_shop', 'timber_wolf', 'Indian_cobra', 'valley',
                        'promontory', 'common_newt', 'wallet', 'ballpoint', 'can_opener', 'picket_fence',
                        'killer_whale', 'starfish', 'tailed_frog', 'greenhouse', 'Scottish_deerhound']

    imagenet_100 = [['restaurant', 'eating_house', 'eating_place', 'eatery'], ['sea_slug', 'nudibranch'],
                    ['cornet', 'horn', 'trumpet', 'trump'],
                    ['indigo_bunting', 'indigo_finch', 'indigo_bird', 'Passerina_cyanea'],
                    ['medicine_chest', 'medicine_cabinet'], ['patas', 'hussar_monkey', 'Erythrocebus_patas'],
                    ['sunscreen', 'sunblock', 'sun_blocker'],
                    ['horned_viper', 'cerastes', 'sand_viper', 'horned_asp', 'Cerastes_cornutus'],
                    ['caldron', 'cauldron'], ['fire_screen', 'fireguard'],
                    ['water_buffalo', 'water_ox', 'Asiatic_buffalo', 'Bubalus_bubalis'],
                    ['axolotl', 'mud_puppy', 'Ambystoma_mexicanum'],
                    ['trailer_truck', 'tractor_trailer', 'trucking_rig', 'rig', 'articulated_lorry', 'semi'],
                    ['Saint_Bernard', 'St_Bernard'], ['crossword_puzzle', 'crossword'],
                    ['Maltese_dog', 'Maltese_terrier', 'Maltese'], ['barn_spider', 'Araneus_cavaticus'],
                    ['oystercatcher', 'oyster_catcher'], ['Shetland_sheepdog', 'Shetland_sheep_dog', 'Shetland'],
                    ['wild_boar', 'boar', 'Sus_scrofa'], ['odometer', 'hodometer', 'mileometer', 'milometer'],
                    ['saltshaker', 'salt_shaker'], ['little_blue_heron', 'Egretta_caerulea'],
                    ['thunder_snake', 'worm_snake', 'Carphophis_amoenus'], ['lawn_mower', 'mower'],
                    ['wallaby', 'brush_kangaroo'], ['gas_pump', 'gasoline_pump', 'petrol_pump', 'island_dispenser'],
                    ['snow_leopard', 'ounce', 'Panthera_uncia'], ['otterhound', 'otter_hound'],
                    ['toilet_tissue', 'toilet_paper', 'bathroom_tissue'], ['Dandie_Dinmont', 'Dandie_Dinmont_terrier'],
                    ['chain_saw', 'chainsaw'], ['radio_telescope', 'radio_reflector'],
                    ['letter_opener', 'paper_knife', 'paperknife'], ['porcupine', 'hedgehog'], ['hot_pot', 'hotpot'],
                    ['dingo', 'warrigal', 'warragal', 'Canis_dingo'], ['bakery', 'bakeshop', 'bakehouse'],
                    ['disk_brake', 'disc_brake'], ['king_snake', 'kingsnake'], ['remote_control', 'remote'],
                    ['grand_piano', 'grand'], ['sulphur_butterfly', 'sulfur_butterfly'],
                    ['spider_monkey', 'Ateles_geoffroyi'], ['horizontal_bar', 'high_bar'], ['golfcart', 'golf_cart'],
                    ['Arctic_fox', 'white_fox', 'Alopex_lagopus'],
                    ['chambered_nautilus', 'pearly_nautilus', 'nautilus'], ['apiary', 'bee_house'],
                    ['megalith', 'megalithic_structure'], ['chocolate_sauce', 'chocolate_syrup'],
                    ['bicycle-built-for-two', 'tandem_bicycle', 'tandem'],
                    ['red_wolf', 'maned_wolf', 'Canis_rufus', 'Canis_niger'], ['motor_scooter', 'scooter'],
                    ['pool_table', 'billiard_table', 'snooker_table'],
                    ['breakwater', 'groin', 'groyne', 'mole', 'bulwark', 'seawall', 'jetty'],
                    ['lycaenid', 'lycaenid_butterfly'], ['French_horn', 'horn'], ['violin', 'fiddle'],
                    ['Pembroke', 'Pembroke_Welsh_corgi'], ['gorilla', 'Gorilla_gorilla'], ['cockroach', 'roach'],
                    ['rubber_eraser', 'rubber', 'pencil_eraser'],
                    ['cougar', 'puma', 'catamount', 'mountain_lion', 'painter', 'panther', 'Felis_concolor'],
                    ['redshank', 'Tringa_totanus'], ['refrigerator', 'icebox'], ['Airedale', 'Airedale_terrier'],
                    ['sports_car', 'sport_car'], ['white_stork', 'Ciconia_ciconia'],
                    ['bookshop', 'bookstore', 'bookstall'],
                    ['cellular_telephone', 'cellular_phone', 'cellphone', 'cell', 'mobile_phone'],
                    ['handkerchief', 'hankie', 'hanky', 'hankey'], ['corkscrew', 'bottle_screw'],
                    ['cello', 'violoncello'], ['black_and_gold_garden_spider', 'Argiope_aurantia'],
                    ['tow_truck', 'tow_car', 'wrecker'], ['frilled_lizard', 'Chlamydosaurus_kingi'],
                    ['American_black_bear', 'black_bear', 'Ursus_americanus', 'Euarctos_americanus'],
                    ['Saluki', 'gazelle_hound'], ['hand-held_computer', 'hand-held_microcomputer'],
                    ['rocking_chair', 'rocker'], ['hippopotamus', 'hippo', 'river_horse', 'Hippopotamus_amphibius'],
                    ['cocker_spaniel', 'English_cocker_spaniel', 'cocker'], ['oboe', 'hautboy', 'hautbois'],
                    ['junco', 'snowbird'], ['shoe_shop', 'shoe-shop', 'shoe_store'],
                    ['timber_wolf', 'grey_wolf', 'gray_wolf', 'Canis_lupus'], ['Indian_cobra', 'Naja_naja'],
                    ['valley', 'vale'], ['promontory', 'headland', 'head', 'foreland'],
                    ['common_newt', 'Triturus_vulgaris'], ['wallet', 'billfold', 'notecase', 'pocketbook'],
                    ['ballpoint', 'ballpoint_pen', 'ballpen', 'Biro'], ['can_opener', 'tin_opener'],
                    ['picket_fence', 'paling'],
                    ['killer_whale', 'killer', 'orca', 'grampus', 'sea_wolf', 'Orcinus_orca'], ['starfish', 'sea_star'],
                    ['tailed_frog', 'bell_toad', 'ribbed_toad', 'tailed_toad', 'Ascaphus_trui'],
                    ['greenhouse', 'nursery', 'glasshouse'], ['Scottish_deerhound', 'deerhound']]

    template_face = ['a face that looks {}']
    template_null = ['{}']
    template_general = ['a photo of {}']

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
    model.eval()
    model = model.to('cuda')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    dataset_imagenet_100 = custom_classification('zero_shot_analyze/imagenet/100.csv', transform=preprocess)

    tuned_model = load_model(path='/hdd5/zhiqi2/open_clip/text_model/imagenet_100_new.pt')
    imagenet_100_random = generate_combination(imagenet_100, count=30)

    accu3, accu4 = synonym_comparison(tuned_model, tokenizer, imagenet_100_ori, imagenet_100_random, template_null,
                                      dataset_imagenet_100, folder_name="imagenet_100")
    accu1, accu2 = synonym_comparison(model, tokenizer, imagenet_100_ori, imagenet_100_random, template_null,
                                      dataset_imagenet_100, folder_name="imagenet_100")

    print(accu1, accu2, accu3, accu4)

