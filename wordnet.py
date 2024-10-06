import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


all_words = wn.words()
print("Number of words in WordNet:", len(list(all_words)))
# print all synset that have antonyms
count = 0
for synset in wn.all_synsets():
    if len(synset.lemmas()) > 1:
        for lemma in synset.lemmas():
            if lemma.antonyms():
                count += 1
                print(synset)
                break
print("Number of synsets that have antonyms:", count)
        # synset_lengths = [len(list(synset.lemmas())) for synset in wn.all_synsets()]
# # print distribution of synset lengths
# print("Mean number of words per synset:", sum(synset_lengths) / len(synset_lengths))
#
# unique, counts = np.unique(synset_lengths, return_counts=True)
#
# bars = plt.bar(unique, counts, tick_label=unique)
#
# # Annotate the bars with the count values
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
#
#
# plt.xlabel('Element')
# plt.ylabel('Frequency')
# plt.title('Distribution of number of lemmas per synset')
# plt.savefig('distribution.png')



# synonyms = []
# antonyms = []
#
# for syn in wn.synsets("small"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
#
# print("Lemma:", set(synonyms))
# print("Antonyms:", set(antonyms))
#
# # 获取单词的同义词集
# happy = wn.synsets("happy")[0]
# glad = wn.synsets("glad")[0]
# cheerful = wn.synsets("cheerful")[0]
# joyful = wn.synsets("joyful")[0]
# elated = wn.synsets("elated")[0]
# unhappy = wn.synsets("unhappy")[0]
# sad = wn.synsets("sad")[0]
# depressing = wn.synsets("depressing")[0]
# print(".also_sees() for happy:", happy.also_sees())
# print(".also_sees() for unhappy:", unhappy.also_sees())
#
# # 词列表
# words = [happy, glad, cheerful, joyful, elated, unhappy, sad, depressing]
# word_names = ["happy", "glad", "cheerful", "joyful", "elated", "unhappy", "sad", "depressing"]
#
# # 初始化两个空的 DataFrame 来存储相似度
# path_sim_matrix = pd.DataFrame(index=word_names, columns=word_names)
# wup_sim_matrix = pd.DataFrame(index=word_names, columns=word_names)
#
# # 计算并填充相似度矩阵
# for i in range(len(words)):
#     for j in range(len(words)):
#         path_sim = words[i].path_similarity(words[j])
#         wup_sim = words[i].wup_similarity(words[j])
#         path_sim_matrix.iloc[i, j] = path_sim
#         wup_sim_matrix.iloc[i, j] = wup_sim
#
# # 打印路径相似度矩阵
# print("Path Similarity Matrix:")
# print(path_sim_matrix)
#
# # 打印吴-帕默相似度矩阵
# print("\nWu-Palmer Similarity Matrix:")
# print(wup_sim_matrix)
#
# ############################################################################################################
# small = wn.synsets("small")[0]
# pocket_sized = wn.synsets("pocket-sized")[0]
# modest = wn.synsets("modest")[0]
# lowly = wn.synsets("lowly")[0]
# minor = wn.synsets("minor")[0]
# belittled = wn.synsets("belittled")[0]
# little = wn.synsets("little")[0]
# small_scale = wn.synsets("small-scale")[0]
# big = wn.synsets("big")[0]
# large = wn.synsets("large")[0]
#
# # 打印 also_sees() 信息
# print(".also_sees() for small:", small.also_sees())
#
# # 词列表
# synonyms = [small, pocket_sized, modest, lowly, minor, belittled, little, small_scale]
# antonyms = [big, large]
# words = synonyms + antonyms
# word_names = ["small", "pocket-sized", "modest", "lowly", "minor", "belittled", "little", "small-scale", "big", "large"]
#
# # 初始化两个空的 DataFrame 来存储相似度
# path_sim_matrix = pd.DataFrame(index=word_names, columns=word_names)
# wup_sim_matrix = pd.DataFrame(index=word_names, columns=word_names)
#
# # 计算并填充相似度矩阵
# for i in range(len(words)):
#     for j in range(len(words)):
#         path_sim = words[i].path_similarity(words[j])
#         wup_sim = words[i].wup_similarity(words[j])
#         path_sim_matrix.iloc[i, j] = path_sim
#         wup_sim_matrix.iloc[i, j] = wup_sim
#
# # 打印路径相似度矩阵
# print("Path Similarity Matrix:")
# print(path_sim_matrix)
#
# # 打印吴-帕默相似度矩阵
# print("\nWu-Palmer Similarity Matrix:")
# print(wup_sim_matrix)
#
# # print path similarity of car and automobile
# car = wn.synsets("car")[0]
# automobile = wn.synsets("automobile")[0]
# print("Path similarity between car and automobile:", car.path_similarity(automobile))
