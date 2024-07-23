import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
from src.open_clip import CLIPWrapper, load_checkpoint
import copy
import pandas as pd
import numpy as np
from itertools import combinations


def print_pairwise_distance(model_, tokenizer_, text_array_, text_array_2, satisfy_both=True):
    copied_text_array_ = copy.deepcopy(text_array_)
    text_array_ = tokenizer_(text_array_).to('cuda')
    text_features = model_.encode_text(text_array_)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.detach().cpu().numpy()

    copied_text_array_2 = copy.deepcopy(text_array_2)
    text_array_2 = tokenizer_(text_array_2).to('cuda')
    text_features_2 = model_.encode_text(text_array_2)
    text_features_2 /= text_features_2.norm(dim=-1, keepdim=True)
    text_features_2 = text_features_2.detach().cpu().numpy()

    all_text_features = np.concatenate([text_features, text_features_2], axis=0)
    all_text_array = np.concatenate([copied_text_array_, copied_text_array_2], axis=0)
    result = np.around(np.dot(all_text_features, all_text_features.T), 2)
    print(pd.DataFrame(result, columns=all_text_array, index=all_text_array))

    def calculate_distance(feature1, feature2):
        return np.dot(feature1, feature2)

    if satisfy_both:
        for i in range(len(text_features)):
            dist_syn = calculate_distance(text_features[i], text_features[0])
            dist_ant1 = calculate_distance(text_features[i], text_features_2[0])
            dist_ant2 = calculate_distance(text_features[0], text_features_2[0])
            if dist_syn < dist_ant1 and dist_syn < dist_ant2:
                print(
                    f"Original texts: '{copied_text_array_2[0]}', '{copied_text_array_[0]}', '{copied_text_array_[i]}'")
                print(
                    f"Distances: dist_({copied_text_array_[0]}, {copied_text_array_[i]})={dist_syn:.2f}, dist_({copied_text_array_[0]}, {copied_text_array_2[0]})={dist_ant2:.2f}, dist_({copied_text_array_[i]}, {copied_text_array_2[0]})={dist_ant1:.2f}\n")

        for i in range(len(text_features_2)):
            dist_syn = calculate_distance(text_features_2[i], text_features_2[0])
            dist_ant1 = calculate_distance(text_features_2[i], text_features[0])
            dist_ant2 = calculate_distance(text_features_2[0], text_features[0])
            if dist_syn < dist_ant1 and dist_syn < dist_ant2:
                print(
                    f"Original texts: '{copied_text_array_[0]}', '{copied_text_array_2[0]}', '{copied_text_array_2[i]}'")
                print(
                    f"Distances: dist_({copied_text_array_2[0]}, {copied_text_array_2[i]})={dist_syn:.2f}, dist_({copied_text_array_2[0]}, {copied_text_array_[0]})={dist_ant2:.2f}, dist_({copied_text_array_2[i]}, {copied_text_array_[0]})={dist_ant1:.2f}\n")
    else:
        # Iterate over all combinations of text_features and text_features_2
        pairs = []
        for (i, j) in combinations(range(len(text_features)), 2):
            feature1 = text_features[i]
            feature2 = text_features[j]

            for k in range(len(text_features_2)):
                feature3 = text_features_2[k]

                dist_12 = calculate_distance(feature1, feature2)
                dist_13 = calculate_distance(feature1, feature3)
                dist_23 = calculate_distance(feature2, feature3)

                if dist_12 < dist_13 and dist_12 < dist_23:
                    pairs.append(((i, j), k, dist_12, dist_13, dist_23))

        if print_pair:
            for pair in pairs:
                (idx1, idx2), idx3, dist_12, dist_13, dist_23 = pair
                print(
                    f"Original texts: '{copied_text_array_[idx1]}', '{copied_text_array_[idx2]}', '{copied_text_array_2[idx3]}'")
                print(f"Distances: dist_({copied_text_array_[idx1]}, {copied_text_array_[idx2]})={dist_12:.2f}, dist_({copied_text_array_[idx1]}, {copied_text_array_2[idx3]})={dist_13:.2f}, dist_({copied_text_array_[idx2]}, {copied_text_array_2[idx3]})={dist_23:.2f}\n")

        # also do the same thing for text_features_2
        pairs = []
        for (i, j) in combinations(range(len(text_features_2)), 2):
            feature1 = text_features_2[i]
            feature2 = text_features_2[j]

            for k in range(len(text_features)):
                feature3 = text_features[k]

                dist_12 = calculate_distance(feature1, feature2)
                dist_13 = calculate_distance(feature1, feature3)
                dist_23 = calculate_distance(feature2, feature3)

                if dist_12 < dist_13 and dist_12 < dist_23:
                    pairs.append(((i, j), k, dist_12, dist_13, dist_23))

        if print_pair:
            for pair in pairs:
                (idx1, idx2), idx3, dist_12, dist_13, dist_23 = pair
                print(
                    f"Original texts: '{copied_text_array_2[idx1]}', '{copied_text_array_2[idx2]}', '{copied_text_array_[idx3]}'")
                print(f"Distances: dist_({copied_text_array_2[idx1]}, {copied_text_array_2[idx2]})={dist_12:.2f}, dist_({copied_text_array_2[idx1]}, {copied_text_array_[idx3]})={dist_13:.2f}, dist_({copied_text_array_2[idx2]}, {copied_text_array_[idx3]})={dist_23:.2f}\n")


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()
model = model.to('cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

model_large, _, preprocess_large = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
model_large.eval()
model_large = model_large.to('cuda')
tokenizer_large = open_clip.get_tokenizer('ViT-L-14')


happy_synonyms = [
    "cat", "not dog"
]

sad_synonyms = [
    "dog", "not cat"
]


print_pairwise_distance(model, tokenizer, happy_synonyms, sad_synonyms, satisfy_both=True)
# print_pairwise_distance(model_large, tokenizer_large, happy_synonyms, sad_synonyms, satisfy_both=True)


