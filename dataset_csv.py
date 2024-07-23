import os
import pandas as pd
import argparse
from tqdm import tqdm
import random


# def generate_csv(lower_bound, upper_bound, separator='\t'):
#     image_list = []
#
#     for i in tqdm(range(lower_bound, upper_bound + 1)):
#         folder_name = f"/hdd5/zhiqi2/datasets/cc12m/{i:05d}"
#         if os.path.exists(folder_name):
#             for file in os.listdir(folder_name):
#                 if file.endswith(".jpg"):
#                     image_path = folder_name + "/" + file
#                     title_path = os.path.splitext(image_path)[0] + ".txt"
#
#                     if os.path.exists(title_path):
#                         with open(title_path, 'r') as title_file:
#                             title = title_file.read().strip()
#                     else:
#                         print(f"Title file '{title_path}' not found.")
#
#                     image_list.append({"filepath": image_path, "title": title})
#
#     df = pd.DataFrame(image_list)
#     number_of_images = len(image_list)
#     csv_filename = f"{lower_bound:05d}-{upper_bound:05d}-{number_of_images}.csv"
#     df.to_csv(csv_filename, sep=separator, index=False)
#
#     print(f"CSV file '{csv_filename}' has been generated with {number_of_images} images.")


def fer_csv():
    happy_path = "/hdd5/zhiqi2/datasets/fer2013/test/happy" # label 0
    sad_path = "/hdd5/zhiqi2/datasets/fer2013/test/sad" # label 1

    image_list = []
    for file in os.listdir(happy_path):
        if file.endswith(".jpg"):
            image_path = happy_path + "/" + file
            image_list.append({"label": 0, "filepath": image_path})

    for file in os.listdir(sad_path):
        if file.endswith(".jpg"):
            image_path = sad_path + "/" + file
            image_list.append({"label": 1, "filepath": image_path})
            
    df = pd.DataFrame(image_list)
    df.to_csv("fer2013_test.csv", index=False)
            

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
    

if __name__ == "__main__":
    cat_dog()
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
