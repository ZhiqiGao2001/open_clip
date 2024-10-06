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


def print_distances(text_array_, image1, image2, model_, tokenizer_):
    text = tokenizer_(text_array_).to('cuda')
    with torch.no_grad():
        text_features = model_.encode_text(text)
        image_feature_1 = model_.encode_image(image1)
        image_feature_2 = model_.encode_image(image2)
        # normalize
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_feature_1 /= image_feature_1.norm(dim=-1, keepdim=True)
        image_feature_2 /= image_feature_2.norm(dim=-1, keepdim=True)
        # print pairwise cosine similarities between 2 images and 3 texts
        print("Cosine Similarity between 2 images: ", F.cosine_similarity(image_feature_1, image_feature_2))
        # print dot product between 3 texts, in pair
        print("Dot product between 3 texts: \n", torch.mm(text_features, text_features.T))
        print("Cosine Similarity between image 1 and texts:", F.cosine_similarity(image_feature_1, text_features))
        print("Cosine Similarity between image 2 and texts: ", F.cosine_similarity(image_feature_2, text_features))


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


tokenizer = open_clip.get_tokenizer('ViT-B-32')
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
# model.eval()
# model = model.to('cuda')
#
# model_path = '/hdd5/zhiqi2/open_clip/text_model/happy_sad.pt'
# model_2, _, preprocess_2 = open_clip.create_model_and_transforms('ViT-B-32')
# model_2.eval()
# model_2 = model_2.to('cuda')
# load_checkpoint(model_2, model_path)

vectors = torch.randn(10000, 512).to('cuda')
# normalize
vectors /= vectors.norm(dim=-1, keepdim=True)

split = 400
distance = torch.mm(vectors[:, :split], vectors[:, :split].T) - torch.mm(vectors[:, split:], vectors[:, split:].T)
print("Distance shape: ", distance.shape)
# print max and min distance
print("Max distance: ", distance.max())
print("Min distance: ", distance.min())
exit()

clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
model = CLIPWrapper(clip, projection_dim=64)
path = f"/hdd5/zhiqi2/open_clip/zero_shot_analyze/models/64_euclidean.pt"
load_checkpoint(model, path)

model_2 = CLIPWrapper(clip, projection_dim=64)
path="/hdd5/zhiqi2/open_clip/text_model/imgnet_100_64.pt"
load_checkpoint(model_2, path)
model_2 = load_model(path='/hdd5/zhiqi2/open_clip/text_model/imgnet_20.pt')

# text_array = ['prairie_chicken', 'prairie_grouse', 'prairie_fowl', 'backpack', 'back_pack', 'knapsack', 'packsack', 'rucksack', 'haversack']
text_array = ['horned_viper', 'cerastes', 'sand_viper', 'horned_asp', 'Cerastes_cornutus', 'restaurant', 'eating_house', 'eating_place', 'eatery']
image_path = ["/hdd5/zhiqi2/datasets/fer2013/test/happy/PrivateTest_94181840.jpg", "/hdd5/zhiqi2/datasets/fer2013/test/sad/PrivateTest_28023870.jpg"]
image_1 = Image.open(image_path[0])
image_2 = Image.open(image_path[1])
image_1 = preprocess(image_1).unsqueeze(0).to('cuda')
image_2 = preprocess(image_2).unsqueeze(0).to('cuda')
print("Original model")
print_distances(text_array, image_1, image_2, model, tokenizer)
print("Loaded model")
print_distances(text_array, image_1, image_2, model_2, tokenizer)

# print(model.encode_image(image_1) == model_2.encode_image(image_1))
# print(model.encode_image(image_2) == model_2.encode_image(image_2))

