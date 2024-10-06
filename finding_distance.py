import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
import copy
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from src.open_clip import CLIPWrapper, load_checkpoint


def print_pairwise_distance(model_, tokenizer_, text_array_, text_array_2, satisfy_both=True, template=None):
    copied_text_array_ = copy.deepcopy(text_array_)
    # use template to replace the text_array_
    if template:
        text_array_ = [template.format(text_) for text_ in text_array_]
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


def check_template(combinations_of_synonyms, sentence_templates, model_, tokenizer_):
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 检查距离关系是否保持不变
    hold, unhold = 0, 0
    for synonyms in tqdm(combinations_of_synonyms):
        # print(f"Checking synonyms: {synonyms}")

        # 计算原始同义词嵌入向量
        original_texts = list(synonyms)
        with torch.no_grad():
            original_text_features = tokenizer_(original_texts).to('cuda')
            original_text_embeddings = model_.encode_text(original_text_features)
            original_text_embeddings /= original_text_embeddings.norm(dim=-1, keepdim=True)
            original_text_embeddings = original_text_embeddings.detach().cpu().numpy()

        # 计算原始同义词之间的余弦距离
        original_distances = {
            (synonyms[0], synonyms[1]): cosine_similarity(original_text_embeddings[0], original_text_embeddings[1]),
            (synonyms[0], synonyms[2]): cosine_similarity(original_text_embeddings[0], original_text_embeddings[2]),
            (synonyms[1], synonyms[2]): cosine_similarity(original_text_embeddings[1], original_text_embeddings[2]),
        }

        # 插入句子模板并计算嵌入向量
        for template in sentence_templates:
            inserted_texts = [template.format(syn) for syn in synonyms]
            with torch.no_grad():
                inserted_text_features = tokenizer_(inserted_texts).to('cuda')
                inserted_text_embeddings = model_.encode_text(inserted_text_features)
                inserted_text_embeddings /= inserted_text_embeddings.norm(dim=-1, keepdim=True)
                inserted_text_embeddings = inserted_text_embeddings.detach().cpu().numpy()

            # 计算插入句子后的余弦距离
            inserted_distances = {
                (synonyms[0], synonyms[1]): cosine_similarity(inserted_text_embeddings[0], inserted_text_embeddings[1]),
                (synonyms[0], synonyms[2]): cosine_similarity(inserted_text_embeddings[0], inserted_text_embeddings[2]),
                (synonyms[1], synonyms[2]): cosine_similarity(inserted_text_embeddings[1], inserted_text_embeddings[2]),
            }

            # 检查距离关系是否保持不变
            original_distance_pairs = sorted(original_distances.items(), key=lambda x: x[1])
            inserted_distance_pairs = sorted(inserted_distances.items(), key=lambda x: x[1])
            original_distance_pairs_ = [pair[0] for pair in original_distance_pairs]
            inserted_distance_pairs_ = [pair[0] for pair in inserted_distance_pairs]

            if original_distance_pairs_ == inserted_distance_pairs_:
                hold += 1
            else:
                unhold += 1
                # print(f"Original: {original_distance_pairs}, Inserted: {inserted_distance_pairs}")

            # 清理 GPU 内存
            del inserted_text_features
            del inserted_text_embeddings
            torch.cuda.empty_cache()

        # 清理 GPU 内存
        del original_text_features
        del original_text_embeddings
        torch.cuda.empty_cache()

    print(f"Hold: {hold}, Unhold: {unhold}")


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()
model = model.to('cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

model_large, _, preprocess_large = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
model_large.eval()
model_large = model_large.to('cuda')
tokenizer_large = open_clip.get_tokenizer('ViT-L-14')


synonyms_1 = [
    "angry", "furious", "irritated", "annoyed", "enraged", "infuriated",
    "wrathful", "mad", "outraged", "fuming", "indignant"
]

synonyms_2 = [
    "surprise", "astonishment", "amazement", "shock", "bewilderment",
    "startle", "wonder", "stupefaction", "disbelief", "awe", "unexpectedness"
]

synonym_1_happy = ['happy', "joyful", "cheerful", "content", "pleased", "delighted", "elated", "ecstatic", "gleeful", "jovial", "merry"]
synonym_2_sad = ['sad', "unhappy", "sorrowful", "dejected", "melancholy", "mournful", "gloomy", "downcast", "disheartened", "dismal", "depressed"]

template_ = 'a face that looks {}'

# print_pairwise_distance(model, tokenizer, synonyms_1, synonyms_2, satisfy_both=True)
# print_pairwise_distance(model_large, tokenizer_large, synonyms_1, synonyms_2, satisfy_both=True)
model_path = '/hdd5/zhiqi2/open_clip/text_model/happy_sad.pt'
model_2, _, preprocess_2 = open_clip.create_model_and_transforms('ViT-B-32')
model_2.eval()
model_2 = model_2.to('cuda')
load_checkpoint(model_2, model_path)
print_pairwise_distance(model, tokenizer, synonym_1_happy, synonym_2_sad, satisfy_both=True, template=template_)
print_pairwise_distance(model_2, tokenizer, synonym_1_happy, synonym_2_sad, satisfy_both=True, template=template_)
exit(0)

adjectives = [
    "beautiful", "ugly", "tall", "short", "intelligent", "stupid",
    "strong", "weak", "brave", "cowardly", "rich", "poor",
    "healthy", "sick", "happy", "sad", "angry", "calm",
    "bright", "dark", "loud", "quiet"
]

sentence_templates_2 = [
    "The {} woman walked into the room.",
    "He gave a {} smile.",
    "She has a {} personality.",
    "The {} man lifted the heavy box.",
    "It was a {} day.",
    "The {} child played in the park.",
    "The {} singer performed on stage.",
    "He is known for his {} behavior.",
    "She looked {} in her new dress.",
    "The {} student answered all the questions correctly.",
    "He felt {} after the meeting.",
    "The {} dog barked loudly.",
    "The {} cat purred softly.",
    "They lived in a {} house.",
    "The {} teacher explained the lesson clearly.",
    "He has a {} outlook on life.",
    "The {} athlete won the race.",
    "The {} flower bloomed in the garden.",
    "The {} sky was filled with stars.",
    "The {} music played in the background."
]

sentence_templates = [
    "I am feeling {} today.",
    "She was {} after hearing the good news.",
    "His {} demeanor was contagious.",
    "They were all {} to see each other.",
    "The {} child ran around the playground.",
    "Everyone felt {} at the party.",
    "It was a {} moment for the team.",
    "He had a {} smile on his face.",
    "We are {} with the results.",
    "Her {} attitude brightened the room.",
    "I felt {} when I received the gift."
]

all_words = happy_synonyms + sad_synonyms

# combinations_of_synonyms = list(combinations(adjectives, 3))
#
# check_template(combinations_of_synonyms, sentence_templates_2, model, tokenizer)

words = [
    "accelerate", "balance", "calibrate", "demonstrate", "elevate", "facilitate",
    "generate", "harmonize", "illustrate", "justify", "knowledge", "leverage",
    "maximize", "navigate", "optimize", "prioritize", "quantify", "recognize",
    "synthesize", "transform", "utilize", "validate", "work", "execute", "assess",
    "boost", "create", "decipher", "enhance", "focus", "grow", "highlight",
    "integrate", "join", "kindle", "lead", "manage", "negotiate", "organize",
    "plan", "question", "refine", "support", "train", "understand", "visualize",
    "win", "yield", "zero", "achieve", "break", "control", "design", "engage",
    "formulate", "guide", "help", "improve", "judge", "keep", "learn", "measure",
    "nurture", "oversee", "present", "query", "reach", "solve", "test", "unify",
    "validate", "watch", "x-ray", "yell", "zoom", "analyze", "build", "connect",
    "define", "empower", "find", "grow", "hold", "implement", "justify", "kick",
    "launch", "maintain", "notice", "open", "predict", "quote", "review",
    "study", "translate", "update", "verify", "write", "maximize", "manage"
]

# 50 sentence templates list
templates = [
    "We need to {} the project timeline to meet our deadline.",
    "It's essential to {} our resources efficiently.",
    "Let's {} the data to ensure accuracy.",
    "We should {} our findings in the next meeting.",
    "We need to {} our team's skills for better results.",
    "It's crucial to {} the new process.",
    "Our goal is to {} more leads through marketing.",
    "We can {} our assets to increase value.",
    "Let's {} our productivity with better tools.",
    "We need to {} the market trends closely.",
    "Our aim is to {} our operations for efficiency.",
    "We must {} our tasks based on importance.",
    "We should {} the impact of our actions.",
    "It's important to {} the contributions of everyone.",
    "We need to {} the data from various sources.",
    "Our goal is to {} our business model.",
    "We should {} our resources effectively.",
    "We must {} our assumptions before proceeding.",
    "Let's {} our goals for the year.",
    "Our plan is to {} customer satisfaction.",
    "We should {} on our main objectives.",
    "It's important to {} our market share.",
    "We need to {} the key points in the report.",
    "Let's {} the new system into our workflow.",
    "We should {} forces with other teams.",
    "We need to {} innovation in our products.",
    "Our team will {} the project to success.",
    "We must {} the workload efficiently.",
    "Let's {} better solutions through discussions.",
    "We need to {} the budget for next year.",
    "It's important to {} our actions with the company goals.",
    "We should {} the project milestones clearly.",
    "Our team must {} the competition closely.",
    "Let's {} the new market opportunities.",
    "We need to {} the process for better efficiency.",
    "Our goal is to {} the project's success.",
    "We must {} our model with accurate data."
]

combinations = list(combinations(words, 3))

check_template(combinations, templates, model, tokenizer)
