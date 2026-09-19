import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# --- PATH CONFIGURATION ---
SOURCE_DIR = Path("kvasir_vqa")
ORIGINAL_JSON = SOURCE_DIR / "dataset.json"
ORIGINAL_IMG_DIR = SOURCE_DIR / "images"
ORIGINAL_MASK_DIR = SOURCE_DIR / "masks"

OUTPUT_ROOT = Path("kvasir_vqa_train")

TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
TEST_RATIO = 0.0

def split_dataset():
    # 1. Load the original data
    if not ORIGINAL_JSON.exists():
        print(f"File not found: {ORIGINAL_JSON}")
        return

    with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 2. Get the list of unique img_ids
    unique_ids = list(set([item['id'] for item in full_data]))
    random.seed(42) # Ensure the split is the same every run
    random.shuffle(unique_ids)

    total_imgs = len(unique_ids)
    train_end = int(total_imgs * TRAIN_RATIO)
    val_end = train_end + int(total_imgs * VAL_RATIO)

    # Split IDs into 3 groups
    train_ids = set(unique_ids[:train_end])
    val_ids = set(unique_ids[train_end:val_end])
    test_ids = set(unique_ids[val_end:])

    print(f"Image allocation: Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # 3. Create the directory structure
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    for split_name in splits:
        (OUTPUT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split_name / "masks").mkdir(parents=True, exist_ok=True)

    # 4. Reverse-map and copy files
    split_contents = {"train": [], "val": [], "test": []}

    print("Distributing data into the splits...")
    for item in tqdm(full_data):
        img_id = item['id']

        # Determine which split this item belongs to
        target_split = ""
        if img_id in train_ids:
            target_split = "train"
        elif img_id in val_ids:
            target_split = "val"
        elif img_id in test_ids:
            target_split = "test"

        if target_split:
            split_contents[target_split].append(item)

            # Copy the image and mask (only if not already present in that split)
            # Assumes the image can be jpg/png, and the mask is png
            img_name = Path(item['image']).name
            mask_name = f"{img_id}.png"

            src_img = ORIGINAL_IMG_DIR / img_name
            src_mask = ORIGINAL_MASK_DIR / mask_name

            dst_img = OUTPUT_ROOT / target_split / "images" / img_name
            dst_mask = OUTPUT_ROOT / target_split / "masks" / mask_name

            if src_img.exists() and not dst_img.exists():
                shutil.copy(str(src_img), str(dst_img))

            if src_mask.exists() and not dst_mask.exists():
                shutil.copy(str(src_mask), str(dst_mask))

    # 5. Save dataset.json for each split
    for split_name, content in split_contents.items():
        output_json = OUTPUT_ROOT / split_name / "dataset.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"Created {split_name}/dataset.json with {len(content)} questions.")

    print(f"\nDONE! Data has been split at: {OUTPUT_ROOT}")

if __name__ == "__main__":
    split_dataset()
