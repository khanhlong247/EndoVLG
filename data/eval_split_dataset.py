import os
import json
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = "data_train_800"           # Overall source

# [CHANGE] Turned into a list of directories to avoid
EXCLUDE_DIRS = ["data_train_700"]

DEST_DIR = "data_val_100" # New destination directory
NUM_SAMPLES_PER_CLASS = 100    # (This variable is now redundant given the new logic below, but kept for the old format)

# [SAMPLE COUNT & RATIO CONFIGURATION]
TOTAL_SAMPLES = 100           # Total number of samples to take
MALIGNANT_RATIO = None        # None = take at random (exhaustively)

def create_train_dataset_excluding_test():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)

    # ---------------------------------------------------------
    # 1. BUILD THE EXCLUSION LIST (BLACKLIST) FROM MULTIPLE SOURCES
    # ---------------------------------------------------------
    excluded_images = set()

    # [CHANGE] Iterate over the list of directories to exclude
    for ex_dir in EXCLUDE_DIRS:
        exclude_path = Path(ex_dir)
        exclude_json_file = exclude_path / "dataset.json"

        if exclude_json_file.exists():
            print(f"Reading exclude list from: {exclude_json_file}")
            with open(exclude_json_file, 'r', encoding='utf-8') as f:
                exclude_data = json.load(f)
                for item in exclude_data:
                    excluded_images.add(item['image'])
        else:
            print(f"Warning: Exclude directory '{ex_dir}' not found or empty.")

    print(f"Total unique images to exclude: {len(excluded_images)}")

    # ---------------------------------------------------------
    # 2. LOAD THE ORIGINAL JSON AND FILTER
    # ---------------------------------------------------------
    json_file = source_path / "dataset.json"
    if not json_file.exists():
        print(f"Error: Source file not found: {json_file}")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Classify and filter
    malignant_items = []
    benign_items = []

    skipped_count = 0

    for item in data:
        # [IMPORTANT] Check whether the image is on the blacklist
        if item['image'] in excluded_images:
            skipped_count += 1
            continue

        # Old benign/malignant classification logic
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

    print(f"Skipped {skipped_count} images (already in excluded dirs).")
    print(f"Remaining Available: {len(malignant_items)} Malignant, {len(benign_items)} Benign")

    # ---------------------------------------------------------
    # 3. SAMPLE SELECTION (NEW LOGIC)
    # ---------------------------------------------------------
    selected_items = []

    # CASE A: A SPECIFIC RATIO IS SET
    if MALIGNANT_RATIO is not None:
        target_malignant = int(TOTAL_SAMPLES * MALIGNANT_RATIO)
        target_benign = TOTAL_SAMPLES - target_malignant

        print(f"Mode: RATIO ({MALIGNANT_RATIO*100}% Malignant). Target: {target_malignant} Mal, {target_benign} Ben.")

        random.shuffle(malignant_items)
        random.shuffle(benign_items)

        take_mal = min(len(malignant_items), target_malignant)
        take_ben = min(len(benign_items), target_benign)

        if take_mal < target_malignant:
            print(f"Warning: Not enough Malignant samples (Wanted {target_malignant}, got {take_mal})")
        if take_ben < target_benign:
            print(f"Warning: Not enough Benign samples (Wanted {target_benign}, got {take_ben})")

        selected_items = malignant_items[:take_mal] + benign_items[:take_ben]

    # CASE B: NO RATIO SET (RANDOM EXHAUSTIVE PICK)
    else:
        print(f"Mode: RANDOM MIX (No ratio set). Picking {TOTAL_SAMPLES} random samples from remaining pool.")

        all_pool = malignant_items + benign_items
        random.shuffle(all_pool)

        take_total = min(len(all_pool), TOTAL_SAMPLES)
        selected_items = all_pool[:take_total]

        cnt_mal = sum(1 for item in selected_items if item in malignant_items)
        print(f"   -> Randomly selected: {cnt_mal} Malignant, {len(selected_items) - cnt_mal} Benign")

    # ---------------------------------------------------------
    # 4. COPY AND CREATE NEW JSON
    # ---------------------------------------------------------
    if len(selected_items) == 0:
        print("No images selected. Check your paths or numbers.")
        return

    if dest_path.exists():
        shutil.rmtree(dest_path)

    os.makedirs(dest_path / "images", exist_ok=True)
    os.makedirs(dest_path / "masks", exist_ok=True)

    new_json_data = []
    print(f"Copying {len(selected_items)} NEW items to '{DEST_DIR}'...")

    for item in selected_items:
        src_img = source_path / item['image']
        dst_img = dest_path / item['image']

        file_name = src_img.name

        possible_mask_srcs = [
            source_path / "masks" / file_name,
            source_path / "masks" / file_name.replace(".jpeg", ".png").replace(".jpg", ".png")
        ]

        src_mask = next((p for p in possible_mask_srcs if p.exists()), None)
        dst_mask = dest_path / "masks" / file_name

        if src_img.exists():
            os.makedirs(dst_img.parent, exist_ok=True)
            shutil.copy(src_img, dst_img)

            if src_mask:
                os.makedirs(dst_mask.parent, exist_ok=True)
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"  Warning: Mask missing for {file_name}")

            new_json_data.append(item)
        else:
            print(f"  Warning: Image missing {src_img}")

    # 5. Save dataset.json
    with open(dest_path / "dataset.json", 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, indent=2)

    print(f"Done! Created '{DEST_DIR}' with {len(new_json_data)} samples.")
    print(f"   (Excluded contents from {EXCLUDE_DIRS})")

if __name__ == "__main__":
    create_train_dataset_excluding_test()
