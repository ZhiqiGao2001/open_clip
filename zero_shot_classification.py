import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
from src.open_clip import CLIPWrapper, load_checkpoint
import copy


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

model_large, _, preprocess_large = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model_large.eval()
tokenizer_large = open_clip.get_tokenizer('ViT-L-14')

