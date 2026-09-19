"""
Variant of kvasir_generate_masks.py for the official TEST split -- exactly
the same logic (see the original docstring in kvasir_generate_masks.py for
details on why preprocessing/UNet loading/reverse-resizing must be done this
way), only SOURCE_DIR/DEST_DIR are changed so it doesn't touch kvasir_vqa/
(used for train).

Input:  kvasir_vqa_test_ready/dataset.json, kvasir_vqa_test_ready/images/
Output: kvasir_vqa_test/dataset.json (copied as-is), kvasir_vqa_test/images/,
        kvasir_vqa_test/masks/{id}.png
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.models.unet import UNet

SOURCE_DIR = Path("kvasir_vqa_test_ready")
SOURCE_JSON = SOURCE_DIR / "dataset.json"

DEST_DIR = Path("kvasir_vqa_test")
DEST_IMG_DIR = DEST_DIR / "images"
DEST_MASK_DIR = DEST_DIR / "masks"
DEST_JSON = DEST_DIR / "dataset.json"

CHECKPOINT_PATH = "checkpoints/stage1_seg_unet_full/unet_kvasir_best.pth"

GLOBAL_SIZE = (256, 256)
BINARY_THRESHOLD = 0.5
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.Compose([transforms.ToTensor()])


def load_unet(checkpoint_path: str, device: torch.device) -> UNet:
    model = UNet(channel_in=3, channel_out=1)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, GLOBAL_SIZE, interpolation=cv2.INTER_NEAREST)
    return to_tensor(resized)


@torch.no_grad()
def predict_masks_batch(model: UNet, batch_tensor: torch.Tensor) -> np.ndarray:
    logits = model(batch_tensor.to(DEVICE))
    probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
    return (probs > BINARY_THRESHOLD).astype(np.uint8) * 255


def main():
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

    id_to_rel_path = {}
    for item in data:
        id_to_rel_path.setdefault(item["id"], item.get("image", f"images/{item['id']}.jpg"))

    unique_ids = sorted(id_to_rel_path.keys())
    print(f"{len(data)} QA pairs / {len(unique_ids)} unique images.")

    todo_ids = [
        img_id for img_id in unique_ids
        if not ((DEST_IMG_DIR / Path(id_to_rel_path[img_id]).name).exists()
                and (DEST_MASK_DIR / f"{img_id}.png").exists())
    ]
    print(f"{len(todo_ids)}/{len(unique_ids)} images remaining to process (skipping images that already have both image+mask).")

    if todo_ids:
        print(f"Loading UNet from: {CHECKPOINT_PATH}")
        model = load_unet(CHECKPOINT_PATH, DEVICE)

        failed = []
        for i in tqdm(range(0, len(todo_ids), BATCH_SIZE), desc="Generating masks (test)"):
            chunk_ids = todo_ids[i:i + BATCH_SIZE]

            chunk_bgr, chunk_shapes, valid_ids = [], [], []
            for img_id in chunk_ids:
                src_img_path = SOURCE_DIR / id_to_rel_path[img_id]
                image_bgr = cv2.imread(str(src_img_path))
                if image_bgr is None:
                    failed.append(img_id)
                    continue
                chunk_bgr.append(image_bgr)
                chunk_shapes.append(image_bgr.shape[:2])
                valid_ids.append(img_id)

            if not valid_ids:
                continue

            batch_tensor = torch.stack([preprocess(img) for img in chunk_bgr])
            masks_256 = predict_masks_batch(model, batch_tensor)

            for img_id, image_bgr, (orig_h, orig_w), mask_256 in zip(
                valid_ids, chunk_bgr, chunk_shapes, masks_256
            ):
                mask_orig = cv2.resize(mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(DEST_MASK_DIR / f"{img_id}.png"), mask_orig)

                dst_img_path = DEST_IMG_DIR / Path(id_to_rel_path[img_id]).name
                if not dst_img_path.exists():
                    shutil.copy(str(SOURCE_DIR / id_to_rel_path[img_id]), str(dst_img_path))

        if failed:
            preview = failed[:10]
            print(f"WARNING: {len(failed)} images could not be read and were skipped: {preview}"
                  f"{' ...' if len(failed) > 10 else ''}")
    else:
        print("No images left to process.")

    with open(DEST_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Images -> {DEST_IMG_DIR}, Masks -> {DEST_MASK_DIR}, dataset.json -> {DEST_JSON}")
    print("Next step: run enrich_vqa_metadata_test.py to compute bounding boxes from these masks.")


if __name__ == "__main__":
    main()
