import os
import json
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "data_test_200"
DEST_DIR = "data_train_100"

TOTAL_SAMPLES = 100
MALIGNANT_RATIO = 0.75

def create_debug_dataset():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)

    # 1. Load the original JSON
    json_file = source_path / "dataset.json"
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Classify
    malignant_items = []
    benign_items = []

    for item in data:
        regions = item.get('regions', [])
        is_malignant = False
        for r in regions:
            lbl = r.get('label', '').lower()
            txt = r.get('label_text', '').lower()
            if "malignant" in lbl or "malignant" in txt:
                is_malignant = True
                break

        if is_malignant:
            malignant_items.append(item)
        else:
            benign_items.append(item)

    print(f"Found: {len(malignant_items)} Malignant, {len(benign_items)} Benign")

    # [NEW CHANGE] 3. Compute the number of samples to take based on the ratio
    target_malignant = int(TOTAL_SAMPLES * MALIGNANT_RATIO)
    target_benign = TOTAL_SAMPLES - target_malignant

    print(f"Targeting extraction: {target_malignant} Malignant ({MALIGNANT_RATIO*100}%), {target_benign} Benign")

    # 4. Sample (shuffle for randomness)
    random.shuffle(malignant_items)
    random.shuffle(benign_items)

    # Make sure we don't take more than what's available
    num_mal = min(len(malignant_items), target_malignant)
    num_ben = min(len(benign_items), target_benign)

    # Warn if there isn't enough data
    if num_mal < target_malignant:
        print(f"Warning: Not enough malignant samples. Requested {target_malignant}, found {num_mal}")
    if num_ben < target_benign:
        print(f"Warning: Not enough benign samples. Requested {target_benign}, found {num_ben}")

    selected_items = malignant_items[:num_mal] + benign_items[:num_ben]

    # 5. Copy images and create the new JSON
    # [FIX] Remove the old directory if it exists, to ensure a clean state
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    os.makedirs(dest_path / "images", exist_ok=True)
    os.makedirs(dest_path / "masks", exist_ok=True)

    new_json_data = []

    print(f"Copying {len(selected_items)} items (Images + Masks)...")

    for item in selected_items:
        # --- HANDLE ORIGINAL IMAGES ---
        src_img = source_path / item['image']
        dst_img = dest_path / item['image']

        # --- HANDLE MASK ---
        file_name = src_img.name

        src_mask = source_path / "masks" / file_name
        dst_mask = dest_path / "masks" / file_name

        if src_img.exists():
            # Copy image
            os.makedirs(dst_img.parent, exist_ok=True)
            shutil.copy(src_img, dst_img)

            # Copy mask (if it exists)
            if src_mask.exists():
                os.makedirs(dst_mask.parent, exist_ok=True)
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"Warning: Mask missing for {file_name}")

            new_json_data.append(item)
        else:
            print(f"Warning: Image missing {src_img}")

    # 6. Save the new dataset.json
    with open(dest_path / "dataset.json", 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, indent=2)

    print(f"Created dataset at '{DEST_DIR}' with {len(new_json_data)} samples.")
    print(f"   Structure: {DEST_DIR}/images and {DEST_DIR}/masks")

if __name__ == "__main__":
    create_debug_dataset()
