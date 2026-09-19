import os
import json
from datasets import load_dataset

# Variant of kvasir_create_metadata.py for the official TEST split.
CACHE_DIR = "./kvasir_vqa_x1_data"
OUTPUT_DIR = "./kvasir_vqa_test_ready"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

print("Loading dataset['test']...")
dataset = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CACHE_DIR)

data_subset = dataset['test']

formatted_data = []

print(f"Processing {len(data_subset)} samples...")
for item in data_subset:
    entry = {
        "id": item['img_id'],
        "image": f"images/{item['img_id']}.jpg",
        "width": 0,
        "height": 0,
        "regions": [],
        "text": item['answer'],
        "question": item['question'],
        "complexity": item['complexity']
    }
    formatted_data.append(entry)

with open(os.path.join(OUTPUT_DIR, "dataset.json"), "w", encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"Metadata generation complete at {OUTPUT_DIR}/dataset.json ({len(formatted_data)} QA rows)")
