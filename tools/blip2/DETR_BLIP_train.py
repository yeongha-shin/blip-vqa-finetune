import torchvision
import os

from transformers import DetrImageProcessor

import numpy as np
import os
from PIL import Image, ImageDraw

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import DetrForObjectDetection, AutoProcessor
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration

import torch
from peft import LoraConfig, get_peft_model

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------
#                                           Dataset
#--------------------------------------------------------------------------------------------

class VQADataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, blip_processor, detr_processor, train=True):
        ann_file = os.path.join(img_folder, "Data/custom_train.json" if train else "Data/custom_val.json")
        super(VQADataset, self).__init__(img_folder, ann_file)
        self.blip_processor = blip_processor
        self.detr_processor = detr_processor
        self.max_length = 128

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(VQADataset, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        question = "What kinds of objects are there?"
        blip_encoding = self.blip_processor(img, question, padding="max_length", truncation=True, return_tensors="pt",
                                  max_length=self.max_length)
        detr_encoding = self.detr_processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = blip_encoding["pixel_values"].squeeze() # remove batch dimension
        # target = encoding["labels"][0] # remove batch dimension
        target = detr_encoding["labels"][0]  # remove batch dimension

        blip_encoding["text"] = "There are 6 ships and 1 red buoy at [LOC]."

        return pixel_values, target



#--------------------------------------------------------------------------------------------
#                                           Model
#--------------------------------------------------------------------------------------------


blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = detr_processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  processed_batch = {}
  processed_batch['pixel_values'] = encoding['pixel_values']
  processed_batch['pixel_mask'] = encoding['pixel_mask']
  processed_batch['labels'] = labels


  text_inputs = blip_processor.tokenizer(
      [example["text"] for example in batch], padding=True, return_tensors="pt"
  )
  processed_batch["input_ids"] = text_inputs["input_ids"]
  processed_batch["attention_mask"] = text_inputs["attention_mask"]

  return processed_batch


train_dataset = VQADataset(img_folder='./', blip_processor=blip_processor, detr_processor=detr_processor, train=True)
val_dataset = VQADataset(img_folder='./', blip_processor=blip_processor, detr_processor=detr_processor, train=False)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(train_dataloader))


class CustomModel(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, id2label):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)

        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                                        quantization_config=BitsAndBytesConfig(
                                                                            load_in_8bit=True))  # Correct initialization
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_processor.tokenizer.add_tokens("[LOC]")

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
        self.blip_model = get_peft_model(self.blip_model, self.lora_config)
        self.blip_model.print_trainable_parameters()

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.id2label = id2label  # Adding the id2label mapping to the class

    def forward(self, pixel_values, pixel_mask):
        outputs = self.detr_model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.detr_model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
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
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


#--------------------------------------------------------------------------------------------
#                                      Annotation Check
#--------------------------------------------------------------------------------------------

# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open("Data/image.png")

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

image.show()
image.save("ImageDraw.png")

#--------------------------------------------------------------------------------------------
#                                      Train Part
#--------------------------------------------------------------------------------------------

model = CustomModel(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label={0:"ship"})

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


trainer = Trainer(max_steps=10, gradient_clip_val=0.1)
trainer.fit(model)

#--------------------------------------------------------------------------------------------
#                                      Evaluation Part
#--------------------------------------------------------------------------------------------


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
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

def plot_results(pil_img, scores, labels, boxes, id2label):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("Detr_finetune.png")
    plt.show()

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = detr_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

    # provide to metric
    # metric expects a list of dictionaries, each item
    # containing image_id, category_id, bbox and score keys
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()

pixel_values, target = val_dataset[0]
pixel_values = pixel_values.unsqueeze(0).to(device)
print(pixel_values.shape)

with torch.no_grad():
  # forward pass to get class logits and bounding boxes
  outputs = model(pixel_values=pixel_values, pixel_mask=None)
print("Outputs:", outputs.keys())

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# load image based on ID
image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open("Data/image.png")

# postprocess model outputs
width, height = image.size
postprocessed_outputs = detr_processor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.0)
results = postprocessed_outputs[0]

plot_results(image, results['scores'], results['labels'], results['boxes'], id2label={0:"ship"})

#--------------------------------------------------------------------------------------------
#                                      Evaluation Part
#--------------------------------------------------------------------------------------------

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device, torch.float16)

    question = "What kinds of objects are there?"

    generated_output = model.blip_model.generate(input_ids='input_ids', pixel_values='pixel_values',
                                                 max_length=30)
    print("LLM output", model.blip_processor.batch_decode(generated_output, skip_special_tokens=True))

print("end of algorithm")