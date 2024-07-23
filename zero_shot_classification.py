import torch
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


class binary_classification(Dataset):
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


def run_CLIP(model_, tokenizer_, class_names_, class_templates_, torch_dataset_):
    loader = torch.utils.data.DataLoader(torch_dataset_, batch_size=500)
    texts = [template.format(cls) for template in class_templates_ for cls in class_names_]
    print('Texts:', texts)
    tokenize_texts = tokenizer_(texts).to('cuda')
    text_features = model_.encode_text(tokenize_texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    top1, n = 0., 0.

    model_.eval()
    with torch.no_grad():
        for images, target in loader:
            images = images.to('cuda')
            target = target.to('cuda')
            # predict
            image_features = model_.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # print(image_features.shape, text_features.shape)
            # stack text features with batch size
            logits = 100. * image_features @ text_features.t()
            probability = torch.nn.functional.softmax(logits, dim=-1)
            prediction = probability.argmax(dim=-1)
            top1 += (prediction == target).sum().item()
            n += images.size(0)

    top1 = top1 / n * 100
    print(f"Top-1 accuracy: {top1:.2f}%\n")


def print_pairwise_distance(model_, text_array_, text_template, tokenizer_):
    text_array_ = [template.format(cls) for template in text_template for cls in text_array_]
    copied_text_array_ = list(copy.deepcopy(text_array_))
    text_array_ = tokenizer_(text_array_).to('cuda')
    text_features = model_.encode_text(text_array_)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.detach().cpu().numpy()
    result = np.around(np.dot(text_features, text_features.T), 2)
    # print pairwise distance
    print('Pairwise distance:')
    print(pd.DataFrame(result, columns=copied_text_array_, index=copied_text_array_))


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


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()
model = model.to('cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

model_large, _, preprocess_large = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
model_large.eval()
model_large = model_large.to('cuda')
tokenizer_large = open_clip.get_tokenizer('ViT-L-14')

classes = ['happy', 'sad']
classes_another = ['not sad', 'not happy']
template_ = ["a {} face"]
template_another = ['{}']

dataset = binary_classification('fer2013_test.csv', transform=preprocess)
print("Length of dataset:", len(dataset))

compare_templates(model, tokenizer, classes, classes_another, template_, template_another, dataset)
compare_templates(model_large, tokenizer_large, classes, classes_another, template_, template_another, dataset)
