import os
import pandas as pd
import argparse
from tqdm import tqdm
import random
from nltk.corpus import wordnet as wn
import json
import numpy as np


def generate_csv(lower_bound, upper_bound, separator='\t'):
    image_list = []

    for i in tqdm(range(lower_bound, upper_bound + 1)):
        folder_name = f"/hdd5/zhiqi2/datasets/cc12m/{i:05d}"
        if os.path.exists(folder_name):
            for file in os.listdir(folder_name):
                if file.endswith(".jpg"):
                    image_path = folder_name + "/" + file
                    title_path = os.path.splitext(image_path)[0] + ".txt"

                    if os.path.exists(title_path):
                        with open(title_path, 'r') as title_file:
                            title = title_file.read().strip()
                    else:
                        print(f"Title file '{title_path}' not found.")

                    image_list.append({"filepath": image_path, "title": title})

    df = pd.DataFrame(image_list)
    number_of_images = len(image_list)
    csv_filename = f"{lower_bound:05d}-{upper_bound:05d}-{number_of_images}.csv"
    df.to_csv(csv_filename, sep=separator, index=False)

    print(f"CSV file '{csv_filename}' has been generated with {number_of_images} images.")


def fer_csv():
    root_path = "/hdd5/zhiqi2/datasets/fer2013/test/"
    emotion_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    image_list = []
    for i, emotion in enumerate(emotion_list):
        emotion_path = root_path + emotion
        for file in os.listdir(emotion_path)[:100]:
            # select 100 files for each emotion
            if file.endswith(".jpg"):
                image_path = emotion_path + "/" + file
                image_list.append({"label": i, "filepath": image_path})

    df = pd.DataFrame(image_list)
    df.to_csv("fer2013.csv", index=False)
            

def cat_dog():
    cat_path = "/hdd5/zhiqi2/datasets/cat_dog/Cat"
    dog_path = "/hdd5/zhiqi2/datasets/cat_dog/Dog"

    image_list = []
    for file in os.listdir(cat_path):
        # select 5% of the images
        if file.endswith(".jpg") and random.random() < 0.05:
            image_path = cat_path + "/" + file
            image_list.append({"label": 0, "filepath": image_path})

    for file in os.listdir(dog_path):
        if file.endswith(".jpg") and random.random() < 0.05:
            image_path = dog_path + "/" + file
            image_list.append({"label": 1, "filepath": image_path})

    df = pd.DataFrame(image_list)
    df.to_csv("cat_dog.csv", index=False)


def imagenet_csv(number_of_classes=1000):
    root_path = "/hdd5/zhiqi2/datasets/imagenet/train/"

    f = open('/hdd5/zhiqi2/open_clip/zero_shot_analyze/imagenet/dir_label_name_1k.json')
    map_collection = json.load(f)
    f.close()
    class_name_list = []
    directory_name = {}
    for key, values in map_collection.items():
        class_name_list.append(values[1])
        directory_name[values[1]] = values[0]
        if not os.path.exists(f"{root_path}{values[0]}"):
            print(f"{root_path}{values[0]} not exists")
    count = 0
    selected_class = {}
    for i in class_name_list:
        # find the noun synsets
        word_ = wn.synsets(i, pos=wn.NOUN)
        if len(word_) == 1:
            if len(word_[0].lemma_names()) != 1:
                selected_class[i] = word_[0].lemma_names()

    # randomly select number_of_classes classes
    selected_keys = random.sample(list(selected_class.keys()), number_of_classes)

    # Create a new dictionary with the selected entries
    selected_dict = {key: selected_class[key] for key in selected_keys}
    print(selected_keys)
    values_2d_list = list(selected_dict.values())
    print(values_2d_list)
    # store this into a txt file
    with open(f'/hdd5/zhiqi2/open_clip/zero_shot_analyze/imagenet/{number_of_classes}.txt', 'w') as file:
        file.write(str(selected_keys))
        file.write('\n')
        file.write(str(values_2d_list))
        file.write('\n')

    image_list = []
    tokeep = min(50, int(2000/number_of_classes))
    for i, name in enumerate(selected_keys):
        class_path = root_path + directory_name[name]
        for file in os.listdir(class_path)[:tokeep]:
            if file.endswith(".JPEG"):
                image_path = class_path + "/" + file
                image_list.append({"label": i, "filepath": image_path})

    df = pd.DataFrame(image_list)
    df.to_csv(f"/hdd5/zhiqi2/open_clip/zero_shot_analyze/imagenet/{number_of_classes}.csv", index=False)


if __name__ == "__main__":
    imagenet_csv(100)
    # cat_dog()
    # fer_csv()
    # parser = argparse.ArgumentParser(description="Generate a CSV file listing image file paths and titles.")
    #
    # parser.add_argument("--lb", type=int, help="Lower bound for folder names, e.g., 00000.")
    # parser.add_argument("--ub", type=int, help="Upper bound for folder names, e.g., 10000.")
    # parser.add_argument("--separator", type=str, default='\t', help="Separator for the CSV file (default is tab).")
    #
    # args = parser.parse_args()
    #
    # generate_csv(args.lb, args.ub, args.separator)
