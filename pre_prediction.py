from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from PIL import Image
import json, os, csv
import logging
from tqdm import tqdm
import torch
import os

# Set the path to your test data directory
test_data_dir = "Data/test_data/"

# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("test_model/checkpoint-525")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
# model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

# Create a list to store the results
results = []

# # Iterate through each file in the test data directory
# samples = [d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))]

# Iterate through each file in the test data directory, excluding non-directory files
samples = [d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))]
for filename in tqdm(samples, desc="Processing"):
    sample_path = os.path.join(test_data_dir, filename)

    # Read the json file
    json_path = os.path.join(sample_path, "data.json")
    if not os.path.exists(json_path):
        logging.warning(f"JSON file not found for {filename}, skipping.")
        continue

    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        question = data["question"]
        image_id = data["id"]

    # Read the corresponding image
    image_path = os.path.join(sample_path, "image.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image not found for {image_id}, skipping.")
        continue

    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    encoding = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

    out = model.generate(**encoding)
    generated_text = processor.decode(out[0], skip_special_tokens=True)

    results.append((image_id, generated_text))

# Write the results to a CSV file
csv_file_path = "Results/pre_results.csv"
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ID", "Label"])  # Write header
    csv_writer.writerows(results)

print(f"Results saved to {csv_file_path}")
