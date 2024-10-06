import numpy as np
import json


def retrieve_dict():
    f = open('../imagenet_meta_data/dir_label_name_21k.json')
    map_collection_21k = json.load(f)
    f.close()

    f = open('../imagenet_meta_data/dir_label_name_1k.json')
    map_collection = json.load(f)
    f.close()

    f = open('../imagenet_meta_data/dir_label_name_21k_Process.json')
    map_collection_21k_Process = json.load(f)
    f.close()

    all_classes_21k = {}
    for key, values in map_collection_21k.items():
        all_classes_21k[int(key)] = values[0]

    all_classes_1k = {}
    for key, values in map_collection.items():
        all_classes_1k[int(key)] = values[0]

    all_classes_21k_Process = {}
    for key, values in map_collection_21k_Process.items():
        all_classes_21k_Process[int(key)] = values[0]

    class_in_21k = []
    class_in_21k_p = []
    for i in range(len(all_classes_1k)):
        if all_classes_1k[i] in all_classes_21k.values():
            class_in_21k.append(i)
        if all_classes_1k[i] in all_classes_21k_Process.values():
            class_in_21k_p.append(i)

    all_class_1k_in_21k = {}
    for i in range(len(class_in_21k)):
        all_class_1k_in_21k[i] = all_classes_1k[class_in_21k[i]]
    all_class_1k_in_21k_p = {}
    for i in range(len(class_in_21k_p)):
        all_class_1k_in_21k_p[i] = all_classes_1k[class_in_21k_p[i]]
    # print('number of 1k classes in 21k:', len(all_class_1k_in_21k))
    # print('number of 1k classes in 21k Process:', len(all_class_1k_in_21k_p))
    # print('number of 1k classes:', len(all_classes_1k))
    # print('number of 21k classes:', len(all_classes_21k))
    # print('number of 21k Process classes:', len(all_classes_21k_Process))

    return all_class_1k_in_21k, all_class_1k_in_21k_p, all_classes_1k, all_classes_21k, all_classes_21k_Process