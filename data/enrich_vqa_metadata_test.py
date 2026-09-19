import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

# Variant of enrich_vqa_metadata.py for the official TEST split.
BASE_DIR = Path("kvasir_vqa_test")
MASK_DIR = BASE_DIR / "masks"
IMG_DIR = BASE_DIR / "images"
JSON_PATH = BASE_DIR / "dataset.json"
OUTPUT_JSON_PATH = BASE_DIR / "dataset.json"

def get_bbox(pts):
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        int(x1 / width * 1000),
        int(y1 / height * 1000),
        int(x2 / width * 1000),
        int(y2 / height * 1000)
    ]

def extract_boxes_from_binary_mask(mask_path, max_boxes=5, threshold=0.60):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], 0, 0

    h, w = mask.shape
    points = np.column_stack(np.where(mask > 127))

    if len(points) < 20:
        return [], w, h

    full_box = get_bbox(points)
    x1, y1, x2, y2 = full_box
    box_area = (x2 - x1) * (y2 - y1)
    mask_area = len(points)

    if box_area == 0: return [full_box], w, h

    coverage = mask_area / box_area
    if coverage >= threshold:
        return [full_box], w, h

    best_boxes = [full_box]
    best_avg_coverage = coverage

    for k in range(2, max_boxes + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)

            current_boxes = []
            total_box_area = 0

            for i in range(k):
                cluster_pts = points[labels == i]
                if len(cluster_pts) < 15: continue

                c_box = get_bbox(cluster_pts)
                total_box_area += (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])
                current_boxes.append(c_box)

            if total_box_area > 0:
                new_coverage = mask_area / total_box_area
                if new_coverage > best_avg_coverage + 0.05:
                    best_avg_coverage = new_coverage
                    best_boxes = current_boxes
                else:
                    break
        except Exception:
            continue

    return best_boxes, w, h

def enrich_dataset():
    if not JSON_PATH.exists():
        print(f"File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_ids = list(set([item['id'] for item in data]))
    print(f"Found {len(data)} questions for {len(unique_ids)} unique images.")

    id_to_regions = {}
    print("Computing Bounding Boxes with K-Means...")

    for img_id in tqdm(unique_ids):
        mask_file = MASK_DIR / f"{img_id}.png"
        if not mask_file.exists():
            mask_file = MASK_DIR / f"{img_id}.jpg"

        if mask_file.exists():
            boxes, w, h = extract_boxes_from_binary_mask(mask_file)

            regions = []
            for b in boxes:
                regions.append({
                    "box": b,
                    "norm_box": normalize_box(b, w, h)
                })
            id_to_regions[img_id] = regions
        else:
            id_to_regions[img_id] = []

    print("Updating region information in the dataset...")
    for item in data:
        img_id = item['id']
        item['regions'] = id_to_regions.get(img_id, [])

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSUCCESS! Test dataset has been enriched at: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    enrich_dataset()
