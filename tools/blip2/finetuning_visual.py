# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Blip2Model, Blip2ForConditionalGeneration

from peft import LoraConfig, get_peft_model
from PIL import Image

from transformers import DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pytorch_lightning import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load DETR pre-trained model
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


# output_dim = detr_model.config.d_model

def detect_objects(image):
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)

    # print("detr dim", inputs.shape)

    # Process DETR outputs
    results = detr_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))[0]
    return results

def detect_and_show_objects(image):
    image = image.to(device)
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)

    print("detr dim", inputs.pixel_values.shape)
    print("detr dim", inputs.pixel_mask.shape)

    # Process DETR outputs
    results = detr_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))[0]

    # Set up the figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    #
    # # Draw boxes with labels and scores
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.5:  # Filter detections with confidence above 50%
            x1, y1, x2, y2 = box.tolist()
            print("Box coordinates:", x1, y1, x2, y2)  # These are now plain float values

            # Draw rectangle on the image
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{detr_model.config.id2label[label.item()]}: {score:.2f}',
                    bbox=dict(facecolor='yellow', alpha=0.5))

            # rect = patches.Rectangle((x, y), width - x, height - y, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
    #         ax.text(x, y, f'{detr_model.config.id2label[label.item()]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()
    return results

def detect_and_show_objects_custom(upsampled_feature):
    upsampled_feature.to(device)
    # inputs = detr_processor(images=image, return_tensors="pt")
    single_pixel_mask = torch.ones((1, 750, 1333), dtype=torch.float32, device=device)
    upsampled_feature.to(device)

    inputs =  {'pixel_values': upsampled_feature[0].unsqueeze(0), 'pixel_mask' : single_pixel_mask}

    print(inputs)
    outputs = detr_model(**inputs)


    # print("detr dim", inputs.pixel_values.shape)
    # print("detr dim", inputs.pixel_mask.shape)

    # Process DETR outputs
    results = detr_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))[0]

    # Set up the figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    #
    # # Draw boxes with labels and scores
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.5:  # Filter detections with confidence above 50%
            x1, y1, x2, y2 = box.tolist()
            print("Box coordinates:", x1, y1, x2, y2)  # These are now plain float values

            # Draw rectangle on the image
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{detr_model.config.id2label[label.item()]}: {score:.2f}',
                    bbox=dict(facecolor='yellow', alpha=0.5))

            # rect = patches.Rectangle((x, y), width - x, height - y, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
    #         ax.text(x, y, f'{detr_model.config.id2label[label.item()]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()
    return results


class ProjectionLayer(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        # 텍스트 데이터 차원을 출력 차원에 맞추는 선형 변환
        self.text_projection = nn.Linear(text_dim, output_dim)
        # 이미지 데이터 차원을 출력 차원에 맞추는 선형 변환
        self.image_projection = nn.Linear(image_dim, output_dim)

    def forward(self, text_features, image_features):
        # 각각의 피처를 프로젝션
        # text_projected = self.text_projection(text_features)
        # print("text projected", text_projected.shape)
        # image_projected = self.image_projection(image_features)
        # print("image projected", image_projected.shape)

        text_projected = self.text_projection(text_features.mean(dim=1))
        image_projected = self.image_projection(image_features.mean(dim=1))
        print("text projected", text_projected.shape)
        print("image projected", image_projected.shape)

        # 두 피처를 합치기 (여기서는 단순 합을 사용)
        combined_features = torch.relu(text_projected + image_projected)
        return combined_features

class FeatureUpsample(nn.Module):
    def __init__(self, input_dim, output_channels, target_height, target_width):
        super(FeatureUpsample, self).__init__()
        self.conv = nn.ConvTranspose2d(input_dim, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)  # reshape to [batch, channels, 1, 1]
        x = self.conv(x)  # upsample spatially
        x = self.upsample(x)  # increase to target size
        return x

batch_size = 1
feature_length = 256
target_channels = 3  # for RGB images
target_height = 750
target_width = 1333

# class CustomModel(nn.Module):
#     def __init__(self, text_model, vision_model, projection_layer, detr_model):
#         super(CustomModel, self).__init__()
#         self.text_model = text_model
#         self.vision_model = vision_model
#         self.projection_layer = projection_layer
#         self.detr_model = detr_model
#
#     def forward(self, input_ids, pixel_values, return_detections=False):
#         text_features = self.text_model(input_ids=input_ids).last_hidden_state
#         image_features = self.vision_model(pixel_values=pixel_values).last_hidden_state
#         combined_features = self.projection_layer(text_features, image_features)
#         outputs = self.text_model.generate(input_ids=input_ids, pixel_values=input_ids, attention_mask=None)
#         generated_text = [self.text_model.config.id2label[token] for token in outputs[0] if token in self.text_model.config.id2label]
#         if return_detections:
#             detr_outputs = self.detr_model(inputs_embeds=combined_features)
#             return generated_text, detr_outputs
#         return generated_text

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

torch.cuda.empty_cache()
torch.manual_seed(42)

# We load our model and processor using `transformers`
# model = AutoModelForVision2Seq.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
# )
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")




# 토크나이저에 특수 토큰을 추가합니다.
# processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
processor.tokenizer.add_tokens("[LOC]")


# Get our peft model and print the number of trainable parameters
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Let's load the dataset here!
# dataset = load_dataset("ybelkada/football-dataset", split="train")

# print(dataset[0])

projection_layer = ProjectionLayer(text_dim=2560, image_dim=1408, output_dim=256)
projection_layer.to(device)

# custom_model = CustomModel(text_model=model, vision_model=model, projection_layer=projection_layer, detr_model=detr_model)
# custom_model.to(device)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, max_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

        self.labels = [{
            "labels": torch.tensor([1]),  # 클래스 ID
            "boxes": torch.tensor([[100, 200, 400, 600]])  # [x_min, y_min, x_max, y_max]
        }]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # item = self.dataset[idx]
        question = "What kinds of objects are there?"
        image = Image.open("/home/yeongha/pycharm/blip-vqa-finetune/Data/test_data/0/image.png").convert("RGB")
        # encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length)
        # encoding = self.processor(image, padding="max_length", truncation=True, return_tensors="pt",
        #                           max_length=self.max_length)
        # labels = self.processor.tokenizer.encode(
        #     answer, max_length= 45, pad_to_max_length=True, return_tensors='pt'
        # )
        #
        # remove batch dimension
        # encoding = {k: v.squeeze() for k, v in encoding.items()}

        for k, v in encoding.items():  encoding[k] = v.squeeze()

        # encoding["text"] = item["text"]
        # encoding["text"] = "There are 6 ships and 1 red buoy <inst1> at <LOC>."
        encoding["text"] = "There are 6 ships and 1 red buoy at [LOC]."


        # # # TODO : add labels
        #
        # targets = torch.zeros(6)
        # labels = ['down', 'at table', 'skateboard', 'table']
        # scores = [1.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        # for label, score in zip(labels, scores):
        #     targets[label] = score
        # encoding["labels"] = targets

        return encoding


def collator(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

dataset = load_dataset("json", data_files="Data/total_train.jsonl", split="train")

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=collator)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
detr_optimizer = torch.optim.AdamW(detr_model.parameters(), lr=5e-5)

model.train()
detr_model.train()


for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, output_hidden_states=True)
        # detr_outputs = detr(pixel_values=pixel_values, labels = labels)
        # outputs = custom_model(input_ids=input_ids, pixel_values=pixel_values)

        # print(outputs.language_model_outputs.hidden_states[-1])
        # print("input size=", outputs.language_model_outputs.config.hidden_size)

        text_feature = outputs.language_model_outputs.hidden_states[-1].float()
        # image_feature = pixel_values

        # image_feature = model.extract_image_features(pixel_values)

        image_feature = outputs.vision_outputs[0].float()

        text_dim = model.config.text_config.hidden_size
        image_dim = model.config.vision_config.projection_dim

        # print(text_feature.shape)
        # print(image_feature.shape)
        fused_features = projection_layer(text_feature, image_feature).to(device)

        upsample_model = FeatureUpsample(feature_length, target_channels, target_height, target_width).to(device)
        upsampled_features = upsample_model(fused_features)  # Resulting shape will be [1, 3, 750, 1333]

        print(upsampled_features.shape)

        single_pixel_mask = torch.ones((1, 750, 1333), dtype=torch.float32, device=device)
        upsampled_features.to(device)

        inputs = {'pixel_values': upsampled_features[0].unsqueeze(0), 'pixel_mask': single_pixel_mask}
        labels = [{
            "class_labels": torch.tensor([1], device=device),  # 클래스 ID
            "boxes": torch.tensor([[100, 200, 400, 600]], dtype=torch.float32, device=device),
            # [x_min, y_min, x_max, y_max]
            "area": torch.tensor([(400 - 100) * (600 - 200)], dtype=torch.float32, device=device),
            # Optional, 박스의 면적 계산
            "iscrowd": torch.tensor([0], dtype=torch.int64, device=device)  # Optional, 객체 인스턴스를 설명하는 상태
        }]

        detr_outputs = detr_model(pixel_values=upsampled_features[0].unsqueeze(0), labels=labels)

        loss = outputs.loss
        # detr_loss = detr_

        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ### detr model train
        # labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        if idx % 10 == 0:
            image = Image.open("/home/yeongha/pycharm/blip-vqa-finetune/Data/test_data/0/image.png").convert("RGB")

            # Prepare inputs
            question = "What kinds of objects are there?"
            # encoding = processor(image, question, return_tensors="pt").to(device, torch.float16)
            encoding = processor(image, question, return_tensors="pt").to(device, torch.float16)

            # generated_output = model.generate(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], max_length =30)
            generated_output = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                                              max_length=30)
            decoded_output = processor.batch_decode(generated_output, skip_special_tokens=True)[0]

            print("Generated caption:", decoded_output)
            # out = model.generate(**encoding, max_length = 30)
            # print(processor.batch_decode(out[0], skip_special_tokens=True))

            if "[LOC]" in decoded_output:
                print("Output contains [LOC]. Performing object detection...")
                detect_and_show_objects_custom(upsampled_features)
            else:
                print("Output does not contain [LOC]. No object detection performed.")
                detect_and_show_objects_custom(upsampled_features)