import torch
from PIL import Image
import open_clip
from openai import OpenAI
import base64
import requests




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
    print("Pseudo cosine distance in tuned model:", text_distance, "with split portion:", portion)


def compare_model_result(image_path, content, alternative_description):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer([content, alternative_description, "random stuff"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_distances = image_features @ text_features.T
        print("Euclidean distance in original model:", text_distances)

        # # pseduo cosine distance in original model
        # image_features_part1 = image_features[:, :split_part]
        # image_features_part2 = image_features[:, split_part:]
        # text_features_part1 = text_features[:, :split_part]
        # text_features_part2 = text_features[:, split_part:]
        # text_distance = image_features_part1 @ text_features_part1.T - image_features_part2 @ text_features_part2.T
        # print("Pseudo cosine distance in original model:", text_distance)

        image_features = model_euclidean.encode_image(image)
        text_features = model_euclidean.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_distances = image_features @ text_features.T
        print("Euclidean distance in tuned model:", text_distances)

        # image_features = model_regression.encode_image(image)
        # text_features = model_regression.encode_text(text)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        #
        # text_distances = image_features @ text_features.T
        # print("Euclidean distance in Regression model:", text_distances)

        # text_probs = (100.0 * text_distances).softmax(dim=-1)
        # print("Euclidean distance softmax:", text_probs)

        split_result(0.75, image, text, model_pseudo)
        split_result(0.25, image, text, model_pseudo25)
        split_result(0.5, image, text, model_pseudo50)


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# # finetune model path
# pseudo_path ='/hdd5/zhiqi2/open_clip/src/logs/2024_06_14-00_43_57-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_latest.pt'
# euclidean_path = '/hdd5/zhiqi2/open_clip/src/logs/2024_06_14-15_32_58-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_latest.pt'

# pretrain model path
pseudo_path = "/hdd5/zhiqi2/open_clip/src/logs/2024_06_26-02_29_45-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_10.pt"
euclidean_path = "/hdd5/zhiqi2/open_clip/src/logs/2024_06_26-00_59_11-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_10.pt"
psuedo_25 = "/hdd5/zhiqi2/open_clip/src/logs/2024_06_26-02_03_12-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_10.pt"

model_pseudo, _, preprocess_pseudo = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pseudo_path)
model_euclidean, _, preprocess_euclidean = open_clip.create_model_and_transforms('ViT-B-32', pretrained=euclidean_path)
model_pseudo.eval()
model_euclidean.eval()
model_pseudo25, _, preprocess_pseudo25 = open_clip.create_model_and_transforms('ViT-B-32', pretrained=psuedo_25)
model_pseudo25.eval()
psuedo_50 = "/hdd5/zhiqi2/open_clip/src/logs/2024_06_26-01_36_28-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_10.pt"
model_pseudo50, _, preprocess_pseudo50 = open_clip.create_model_and_transforms('ViT-B-32', pretrained=psuedo_50)
#
# regression_path = "/hdd5/zhiqi2/open_clip/src/logs/2024_06_25-14_37_42-model_ViT-B-32-lr_5e-06-b_128-j_4-p_amp/checkpoints/epoch_10.pt"
# model_regression, _, preprocess_regression = open_clip.create_model_and_transforms('ViT-B-32', pretrained=regression_path)
# model_regression.eval()


for image_name in ["001110022", "001110003", "001110053", "001110074", "001110092"]:

    image_path = f"image_test/{image_name}.jpg"
    with open(f"image_test/{image_name}.txt") as file:
        content = file.read()
        print(content)
    alternative_description = get_alternative_caption(image_path, content)
    compare_model_result(image_path, content, alternative_description)
print("----------------------testing above--------------------------------")

for image_name in ["000004370", "000005008", "000006364", "000007686", "000003093"]:
    image_path = f"image_train/{image_name}.jpg"
    with open(f"image_train/{image_name}.txt") as file:
        content = file.read()
        print(content)

    alternative_description = get_alternative_caption(image_path, content)
    compare_model_result(image_path, content, alternative_description)
print("-------------------training above-----------------------------------")





