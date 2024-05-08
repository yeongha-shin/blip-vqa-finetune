
import torchvision
import os
from transformers import DetrImageProcessor
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from coco_eval import CocoEvaluator
from tqdm import tqdm  # tqdm.notebook에서 tqdm으로 변경
import matplotlib.pyplot as plt

# 만약 CUDA가 가능하다면, GPU를 사용하고 그렇지 않다면 CPU를 사용합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "Data/custom_train.json" if train else "Data/custom_val.json")
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # 이미지 및 타깃을 전처리하고 DETR 포맷으로 변환
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze().to(device)  # 디바이스로 전송
        target = encoding["labels"][0]  # 배치 차원 제거

        return pixel_values, target


# DETR 모델 초기화
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder='./', processor=processor)
val_dataset = CocoDetection(img_folder='./', processor=processor, train=False)


# 배치 함수 정의
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [{k: v.to(device) for k, v in item[1].items()} for item in batch]

    batch = {
        'pixel_values': encoding['pixel_values'].to(device),
        'pixel_mask': encoding['pixel_mask'].to(device),
        'labels': labels,
    }

    return batch


train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)


class Detr(pl.LightningModule):
    def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        ).to(device)  # 모델을 디바이스로 전송

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


# 모델을 생성하고 디바이스로 전송
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
model.to(device)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./Model/DETR/',
    filename='detr',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    max_epochs=10,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback],
)

# 이제 모델을 트레이닝
trainer.fit(model)


# 체크포인트 로드 함수
def load_checkpoint(checkpoint_path, model_class, lr, lr_backbone, weight_decay):
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
    )
    model.to(device)  # 디바이스로 전송
    return model


checkpoint_path = './Model/DETR/detr.ckpt'
loaded_model = load_checkpoint(checkpoint_path, Detr, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)


# DETR 평가
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = convert_to_xywh(prediction["boxes"]).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


# CocoEvaluator 초기화
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

    # 모델 예측
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

    predictions = {target["image_id"].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()

# 평가 결과 출력
pixel_values, target = val_dataset[0]
pixel_values = pixel_values.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
print("Outputs:", outputs.keys())

# 결과 시각화
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
          [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), color in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis("off")
    plt.show()
