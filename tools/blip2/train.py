import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Blip2Model, Blip2ForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
from peft import LoraConfig, get_peft_model

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.cuda.empty_cache()
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

#--------------------------------------------------------------------------------------------
#                                       Loss Function
#--------------------------------------------------------------------------------------------

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

#--------------------------------------------------------------------------------------------
#                                       Data set
#--------------------------------------------------------------------------------------------

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, max_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

        self.labels = [{
            "labels": torch.tensor([9]),  # 클래스 ID
            "boxes": torch.tensor([[100, 200, 400, 600]])  # [x_min, y_min, x_max, y_max]
        }]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # item = self.dataset[idx]
        question = "What kinds of objects are there?"
        # image_path = item["image_path"]
        # image = Image.open(image_path).convert("RGB")
        image = Image.open("Data/test_data/0/image.png").convert("RGB")
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt",
                                  max_length=self.max_length)

        for k, v in encoding.items():  encoding[k] = v.squeeze()

        encoding["text"] = "There are 6 ships and 1 red buoy at [LOC]."

        return encoding

#--------------------------------------------------------------------------------------------
#                                       Custom Model
#--------------------------------------------------------------------------------------------

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # BLIP -2 Model
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                        quantization_config=BitsAndBytesConfig(
                                                                            load_in_8bit=True))  # Correct initialization

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.processor.tokenizer.add_tokens("[LOC]")

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
        self.blip_model = get_peft_model(self.blip_model, self.lora_config)
        self.blip_model.print_trainable_parameters()

        # DETR Model
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Projection Layer
        self.projection_layer = ProjectionLayer(text_dim=2560, image_dim=1408, output_dim=256)

    def forward(self, image, input_ids, pixel_values=None, **kwargs):
        # BLIP output
        blip_outputs = self.blip_model(input_ids=input_ids, pixel_values=pixel_values, **kwargs )

        text_feature = blip_outputs.language_model_outputs.hidden_states[-1].float()
        image_feature = blip_outputs.vision_outputs[0].float()

        # text_dim = model.config.text_config.hidden_size
        # image_dim = model.config.vision_config.projection_dim

        # Projection Layer
        fused_features = self.projection_layer(text_feature, image_feature).to(device)
        upsample_model = FeatureUpsample(feature_length, target_channels, target_height, target_width).to(device)
        upsampled_features = upsample_model(fused_features)  # Resulting shape will be [1, 3, 750, 1333]

        labels = [{
            "class_labels": torch.tensor([9], device=device),  # 클래스 ID
            "boxes": torch.tensor([[1141, 433, 1529, 608]], dtype=torch.float32, device=device),
            # [x_min, y_min, x_max, y_max]
            "area": torch.tensor([(1529 - 1141) * (608 - 433)], dtype=torch.float32, device=device),
            # Optional, 박스의 면적 계산
            "iscrowd": torch.tensor([0], dtype=torch.int64, device=device)  # Optional, 객체 인스턴스를 설명하는 상태
        }]

        detr_result, detr_outputs = self.detect_and_show_objects_custom(image, upsampled_features, labels)

        return blip_outputs, detr_outputs

    def collator(self, batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = self.processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    def detect_and_show_objects_custom(self, image, upsampled_feature, labels):
        upsampled_feature.to(device)
        # labels.to(device)

        single_pixel_mask = torch.ones((1, 750, 1333), dtype=torch.float32, device=device)

        inputs = {'pixel_values': upsampled_feature[0].unsqueeze(0), 'pixel_mask': single_pixel_mask}
        # inputs = detr_processor(images=image, return_tensors="pt")

        outputs = self.detr_model(**inputs, labels=labels)

        # Process DETR outputs
        results = self.detr_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))[
            0]

        # Set up the figure and axis
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        #
        # # Draw boxes with labels and scores
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.5:  # Filter detections with confidence above 50%
                x1, y1, x2, y2 = box.tolist()
                # print("Box coordinates:", x1, y1, x2, y2)  # These are now plain float values

                # Draw rectangle on the image
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'{self.detr_model.config.id2label[label.item()]}: {score:.2f}',
                        bbox=dict(facecolor='yellow', alpha=0.5))

                # rect = patches.Rectangle((x, y), width - x, height - y, linewidth=1, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
        #         ax.text(x, y, f'{detr_model.config.id2label[label.item()]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
        #         print((x1, y1), x2 - x1, y2 - y1, "score :", score)

        plt.savefig(os.path.join("Detr_result.png"))
        plt.show()

        return results, outputs

class ProjectionLayer(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        # 텍스트 데이터 차원을 출력 차원에 맞추는 선형 변환
        # self.text_projection = nn.Linear(text_dim, output_dim)
        # # 이미지 데이터 차원을 출력 차원에 맞추는 선형 변환
        # self.image_projection = nn.Linear(image_dim, output_dim)

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, output_dim),
            nn.Dropout(0.0)
        )
        # 이미지 데이터 차원을 출력 차원에 맞추는 선형 변환
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim, output_dim),
            nn.Dropout(0.0)
        )

    def forward(self, text_features, image_features):
        # 각각의 피처를 프로젝션
        # text_projected = self.text_projection(text_features)
        # print("text projected", text_projected.shape)
        # image_projected = self.image_projection(image_features)
        # print("image projected", image_projected.shape)

        text_projected = self.text_projection(text_features.mean(dim=1))
        image_projected = self.image_projection(image_features.mean(dim=1))
        # print("text projected", text_projected.shape)
        # print("image projected", image_projected.shape)

        # 두 피처를 합치기 (여기서는 단순 합을 사용)
        combined_features = torch.relu(text_projected + image_projected)
        return combined_features

    def set_train_mode(self):
        self.text_projection.train()
        self.image_projection.train()

    def set_eval_mode(self):
        self.text_projection.eval()
        self.image_projection.eval()


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

feature_length = 256
target_channels = 3  # for RGB images
target_height = 750
target_width = 1333

#--------------------------------------------------------------------------------------------
#                                       Train Part
#--------------------------------------------------------------------------------------------

model = CustomModel()
model.to(device)

dataset = load_dataset("json", data_files="Data/total_train.jsonl", split="train")

train_dataset = ImageCaptioningDataset(dataset, model.processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=model.collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()


for epoch in range(50):
    print("Epoch:", epoch)

    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        image = Image.open("Data/test_data/0/image.png").convert("RGB")

        blip_outputs, detr_outputs = model(image, input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, output_hidden_states=True)

        loss = blip_outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if idx % 10 == 0:
            # Prepare inputs
            question = "What kinds of objects are there?"
            # encoding = processor(image, question, return_tensors="pt").to(device, torch.float16)
            encoding = model.processor(image, question, return_tensors="pt").to(device, torch.float16)

            generated_output = model.blip_model.generate(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], max_length =30)
            print(model.processor.batch_decode(generated_output, skip_special_tokens=True))
            # out = model.generate(**encoding, max_length = 30)
            # print(proces