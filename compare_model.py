import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests
from src.open_clip import CLIPWrapper, load_checkpoint
import copy



# https://platform.openai.com/docs/guides/vision
def get_alternative_caption(image_path_, content_):
    def encode_image(image_path_):
        with open(image_path_, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path_)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The original caption of the provided image is '{content_}', "
                                f"please generate a new caption of the image using completely different words from the original caption."
                                f" Your answer should include the new caption only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()
    gpt_new_description = response_json['choices'][0]['message']['content']
    print(gpt_new_description)
    return gpt_new_description


def split_result(portion, image, text, model_):
    image_features = model_.encode_image(image)
    text_features = model_.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    split_part = int(image_features.shape[1] * portion)

    image_features_part1 = image_features[:, :split_part]
    image_features_part2 = image_features[:, split_part:]
    text_features_part1 = text_features[:, :split_part]
    text_features_part2 = text_features[:, split_part:]
    text_distance = image_features_part1 @ text_features_part1.T - image_features_part2 @ text_features_part2.T
    print("Pseudo distance in tuned model:", text_distance, "with split portion:", portion,  "with projector dimension:", image_features.shape[1])


def compare_model_result(image_path, content, alternative_description):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer([content, alternative_description, "objects that are not blue", "Irrelavant message"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_distances = image_features @ text_features.T
        print("Euclidean distance in original model:", text_distances)

        image_features = model_euclidean.encode_image(image)
        text_features = model_euclidean.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_distances = image_features @ text_features.T
        print("Euclidean distance in tuned model:", text_distances)

        # split_result(0.5, image, text, model_pseudo64)
        split_result(0.5, image, text, model_pseudo128)
        # split_result(0.5, image, text, model_pseudo256)


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')


euclidean_path = "/home/zhiqi/open_clip/src/logs/2024_07_16-08_03_42-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_25.pt"
pseudo_path = "/home/zhiqi/open_clip/src/logs/2024_07_16-11_58_07-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_25.pt"
# psuedo_256 = "/home/zhiqi/open_clip/src/logs/2024_07_09-21_33_57-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_20.pt"
# psuedo_128 = "/home/zhiqi/open_clip/src/logs/2024_07_09-18_56_27-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_20.pt"


clip_original, _, preprocess_original = open_clip.create_model_and_transforms('ViT-B-32')
wrapper_model = CLIPWrapper(clip_original, projection_dim=128)
load_checkpoint(wrapper_model, pseudo_path)

model_euclidean = wrapper_model

psuedo_model = CLIPWrapper(clip_original, projection_dim=128)
load_checkpoint(psuedo_model, pseudo_path)
model_pseudo128 = psuedo_model


for image_name in ["000102", "000196"]:
    image_path = f"/home/zhiqi/datasets/clevr4_100k/images/CLEVR_new_{image_name}.png"
    compare_model_result(image_path, "objects that are blue", "objects that are not red")
print("-------------------training above-----------------------------------")
for image_name in ["155504", '155625']:
    image_path = f"/home/zhiqi/datasets/clevr4_100k/images/CLEVR_new_{image_name}.png"
    compare_model_result(image_path, "objects that are blue", "objects that are not red")
print("----------------------testing above--------------------------------")




# psuedo_128_model = CLIPWrapper(clip_original, projection_dim=128)
# load_checkpoint(psuedo_128_model, psuedo_128)
# model_pseudo128 = psuedo_128_model
#
# psuedo_256_model = CLIPWrapper(clip_original, projection_dim=256)
# load_checkpoint(psuedo_256_model, psuedo_256)
# model_pseudo256 = psuedo_256_model


# model_pseudo50, _, preprocess_pseudo = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pseudo_path)
# model_pseudo50.eval()
# model_euclidean, _, preprocess_euclidean = open_clip.create_model_and_transforms('ViT-B-32', pretrained=euclidean_path)
# model_euclidean.eval()


# for image_name in ["000009999", "000000095", "000006365", "000000029", "000000000"]:
#     image_path = f"image_test/{image_name}.jpg"
#     with open(f"image_test/{image_name}.txt") as file:
#         content = file.read()
#         print(content)
#     alternative_description = get_alternative_caption(image_path, content)
#     compare_model_result(image_path, content, alternative_description)
# print("----------------------testing above--------------------------------")
#
# for image_name in ["000004370", "000005008", "000006365", "000007686", "000003093"]:
#     image_path = f"image_train/{image_name}.jpg"
#     with open(f"image_train/{image_name}.txt") as file:
#         content = file.read()
#         print(content)
#
#     alternative_description = get_alternative_caption(image_path, content)
#     compare_model_result(image_path, content, alternative_description)
# print("-------------------training above-----------------------------------")





