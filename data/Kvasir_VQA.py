from datasets import load_dataset
import os

HF_TOKEN = "" 
OUTPUT_DIR = "./kvasir_vqa_x1_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading dataset to: {OUTPUT_DIR}...")

dataset = load_dataset(
    "SimulaMet/Kvasir-VQA-x1", 
    cache_dir=OUTPUT_DIR,
    token=HF_TOKEN
)

train_data = dataset['train']
test_data = dataset['test']

print("Dataset downloaded successfully!")
print(f"Number of Train samples: {len(train_data)}")
print("First data sample:")
print(train_data[0])