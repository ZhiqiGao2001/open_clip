import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
from src.open_clip import CLIPWrapper, load_checkpoint
import copy
import numpy


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model_openai, _, preprocess_openai = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model_openai.eval()

model_large, _, preprocess_large = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
model_large.eval()
tokenizer_large = open_clip.get_tokenizer('ViT-L-14')


def print_pairwise_distance(model_, text_array_, tokenizer_, euclidean=False, cosine=True, print_pair=False, pseudo_euclidean=False):
    copied_text_array_ = copy.deepcopy(text_array_)
    text_array_ = tokenizer_(text_array_)
    text_features = model_.encode_text(text_array_)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.detach().cpu().numpy()
    if cosine:
        print("Pairwise cosine distance between text features:")
        result = numpy.around(numpy.dot(text_features, text_features.T), 2)
        print(result)
        if print_pair:
            for i in range(len(text_array_) - 4):
                for j in range(i + 1, len(text_array_) - 4):
                    for k in range(len(text_array_) - 4, len(text_array_)):
                        if result[i][j] < result[i][k]:
                            print(f"d({copied_text_array_[i]}, {copied_text_array_[j]}) > d({copied_text_array_[i]}, {copied_text_array_[k]})")

    if euclidean:
        print("Pairwise Euclidean distance between text features:")
        result = numpy.around(numpy.linalg.norm(text_features[:, None] - text_features, axis=-1), 2)
        # print the pair if the distance between synonyms is more than the distance between antonyms / other words, which is the last four words
        print(result)
        if print_pair:
            for i in range(len(text_array_) - 4):
                for j in range(i + 1, len(text_array_) - 4):
                    for k in range(len(text_array_) - 4, len(text_array_)):
                        if result[i][j] > result[i][k]:
                            print(f"d({copied_text_array_[i]}, {copied_text_array_[j]}) > d({copied_text_array_[i]}, {copied_text_array_[k]})")


# descriptions_with_synonyms_and_antonyms = [
#     ["1 object", "a single object", "one item", "one piece", "multiple objects", "several objects"],
#     ["less than 2 objects", "fewer than 2 objects", "under two objects", "less than a couple of objects", "more than 2 objects", "many objects"],
#     ["2 objects", "a pair of objects", "two items", "a couple of objects", "1 object", "single object"],
#     ["more than 1 object", "multiple objects", "several objects", "more than one item", "1 object", "single object"],
#     ["round object", "circular object", "spherical item", "orb-like object", "square object", "rectangular object"],
#     ["square object", "rectangular object", "quadrilateral item", "four-sided object", "round object", "circular object"],
#     ["on the table", "placed on the table", "situated on the table", "located on the table", "off the table", "under the table"],
#     ["under the table", "beneath the table", "below the table", "underneath the table", "on the table", "above the table"],
#     ["left side", "left-hand side", "leftward side", "leftmost side", "right side", "right-hand side"],
#     ["right side", "right-hand side", "rightward side", "rightmost side", "left side", "left-hand side"],
#     ["near the window", "close to the window", "by the window", "next to the window", "far from the window", "away from the window"],
#     ["far from the window", "away from the window", "distant from the window", "not near the window", "near the window", "close to the window"],
#     ["above the shelf", "over the shelf", "on top of the shelf", "higher than the shelf", "below the shelf", "under the shelf"],
#     ["below the shelf", "under the shelf", "beneath the shelf", "lower than the shelf", "above the shelf", "over the shelf"],
#     ["in the box", "inside the box", "within the box", "contained in the box", "outside the box", "out of the box"],
#     ["outside the box", "out of the box", "beyond the box", "exterior to the box", "in the box", "inside the box"]
# ]
descriptions_with_synonyms_and_antonyms = [["big", "large", "gigantic", "enormous", "small", "tiny"]]


for array in descriptions_with_synonyms_and_antonyms:
    text_array = array + ["NULL", "irrelevant text"]
    print(text_array)
    print_pairwise_distance(model_large, text_array, tokenizer_large, euclidean=True, cosine=True, print_pair=True)
    # print_pairwise_distance(model_large, text_array, tokenizer_large)


# print("Model: ViT-B/32, pretrained: openai")
# print_pairwise_distance(model_openai, text_array, tokenizer)
# print("Model: ViT-L-14, pretrained: laion400m_e31")
# print_pairwise_distance(model_large, text_array, tokenizer_large)

# text_array2 = ["1 object", "2 objects",  "more than 1 object", "less than 2 object"]
# text_array2 = text_array2 + ["NULL", "irrelevant text"]
# print("Text array:", text_array2)
# # Print name of model
# print("Model: ViT-B-32, pretrained: laion400m_e31")
# print_pairwise_distance(model, text_array2, tokenizer)
