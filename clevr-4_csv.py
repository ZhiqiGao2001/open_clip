import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import json
import random

task = "color"
# create dataframes for csv
dataframe_train = pd.DataFrame(columns=["filepath", "title"])
dataframe_val = pd.DataFrame(columns=["filepath", "title"])


def generate_descriptions_count(count_):
    x = 0
    y = 11
    random_int_small = random.randint(x, count_ - 1)
    random_int_large = random.randint(count_ + 1, y)

    descriptions = [
        f"{count_} object",
        f"less than {random_int_large} objects",
        f"more than {random_int_small} objects",
        f"not {random_int_small} objects",
        f"not {random_int_large} objects",
    ]
    return random.choice(descriptions)


def generate_description_color(color_, color_set_):
    descriptions = [
        f"objects that are {color_}",
        f"objects that are not {random.choice(list(color_set_ - {color_}))}",
        f"objects that are not {random.choice(list(color_set_ - {color_}))}",
        f"objects that are not {random.choice(list(color_set_ - {color_}))}"
    ]
    return random.choice(descriptions)


with open('/home/zhiqi/datasets/clevr4_100k/clevr_4_annots.json') as file:
    data = json.load(file)

color_set = set()
for key, value in data.items():
    color = value['color']
    color_set.add(color)


for key, value in data.items():
    split = value['split']
    color = value['color']
    count = int(value['count'])
    image_path = f"/home/zhiqi/datasets/clevr4_100k/images/{key}.png"
    if task == "count":
        description = generate_descriptions_count(count)
    elif task == "color":
        description = generate_description_color(color, color_set)

    new_row = {"filepath": image_path, "title": description}

    if split == "train":
        dataframe_train = pd.concat([dataframe_train, pd.DataFrame([new_row])], ignore_index=True)
    elif split == "val":
        dataframe_val = pd.concat([dataframe_val, pd.DataFrame([new_row])], ignore_index=True)

# print length of train and val dataframes
print(len(dataframe_train), len(dataframe_val))
dataframe_train.to_csv(f"/home/zhiqi/datasets/clevr4_100k/{task}_train.csv", sep='\t',  index=False)
dataframe_val.to_csv(f"/home/zhiqi/datasets/clevr4_100k/{task}_val.csv", sep='\t',  index=False)
