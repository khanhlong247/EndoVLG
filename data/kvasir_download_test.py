import os
import io
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Variant of kvasir_download.py for the official TEST split (dataset['test'],
# not dataset['train']) -- writes to its own separate directory, without
# touching kvasir_vqa_ready/ (used for train) or any other directory.
CACHE_DIR = "./kvasir_vqa_x1_data"
OUTPUT_IMAGES_DIR = "./kvasir_vqa_test_ready/images"
NUM_THREADS = 4
TIMEOUT_VAL = (10, 120)

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

print("Loading dataset['test'] from cache...")
dataset = load_dataset("SimulaMet/Kvasir-VQA-x1", cache_dir=CACHE_DIR)
data_subset = dataset['test']

unique_images = {}
for item in data_subset:
    if item['img_id'] not in unique_images:
        unique_images[item['img_id']] = item
unique_data_list = list(unique_images.values())
TOTAL_UNIQUE = len(unique_data_list)

def is_valid_image(path):
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def save_and_verify(item):
    img_id = item['img_id']
    img_data = item['image']
    save_path = os.path.join(OUTPUT_IMAGES_DIR, f"{img_id}.jpg")

    if os.path.exists(save_path) and is_valid_image(save_path):
        return None

    try:
        if isinstance(img_data, str):
            if img_data.startswith('http'):
                r = session.get(img_data, timeout=TIMEOUT_VAL)
                if r.status_code == 200:
                    Image.open(io.BytesIO(r.content)).convert('RGB').save(save_path, "JPEG", quality=95)
                else:
                    return f"{img_id}: HTTP {r.status_code}"
            else:
                Image.open(img_data).convert('RGB').save(save_path, "JPEG", quality=95)
        elif hasattr(img_data, 'convert'):
            img_data.convert('RGB').save(save_path, "JPEG", quality=95)
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            Image.open(io.BytesIO(img_data['bytes'])).convert('RGB').save(save_path, "JPEG", quality=95)
    except Exception as e:
        return f"{img_id}: {str(e)}"

    return None

print(f"Downloading {TOTAL_UNIQUE} unique test images...")
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    results = list(tqdm(executor.map(save_and_verify, unique_data_list), total=TOTAL_UNIQUE))

final_count = len([f for f in os.listdir(OUTPUT_IMAGES_DIR) if f.endswith('.jpg')])
print(f"\nStatus: {final_count}/{TOTAL_UNIQUE} images.")

if final_count < TOTAL_UNIQUE:
    failed = [r for r in results if r is not None]
    print(f"Still {len(failed)} images failed to download. You can rerun the script to fill in the rest (it is resumable).")
else:
    print("100% of test images downloaded.")
