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
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Blip2Model, Blip2ForConditionalGeneration

from peft import LoraConfig, get_peft_model
from PIL import Image

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

# Get our peft model and print the number of trainable parameters
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Let's load the dataset here!
# dataset = load_dataset("ybelkada/football-dataset", split="train")

# print(dataset[0])


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, max_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

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
        encoding["text"] = "There are 6 ships and 1 red buoy."


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
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()

for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if idx % 10 == 0:
            image = Image.open("/home/yeongha/pycharm/blip-vqa-finetune/Data/test_data/0/image.png").convert("RGB")

            # Prepare inputs
            question = "What kinds of objects are there?"
            # encoding = processor(image, question, return_tensors="pt").to(device, torch.float16)
            encoding = processor(image, question, return_tensors="pt").to(device, torch.float16)

            generated_output = model.generate(input_ids=encoding['input_ids'], pixel_values=encoding['pixel_values'], max_length =30)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))
            # out = model.generate(**encoding, max_length = 30)
            # print(processor.batch_decode(out[0], skip_special_tokens=True))