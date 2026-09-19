"""
Step 4 of the Kvasir-VQA-x1 dataset creation pipeline -- the bridge between
kvasir_create_metadata.py (step 3, output: kvasir_vqa_ready/dataset.json +
images/, no masks yet) and enrich_vqa_metadata.py (step 5, input: kvasir_vqa/
with images/ + masks/ + dataset.json).

SimulaMet/Kvasir-VQA-x1 does not ship with lesion masks -- masks are generated
here using the UNet fine-tuned in Stage 1 (checkpoints/stage1_seg_unet_full/
unet_kvasir_best.pth), run on each unique image in the dataset.

Input preprocessing for the UNet MUST match exactly the "global stream"
branch in src/data/dataset.py (EndoDataset.__getitem__, mode != 'train' so
augment_pair is a no-op): BGR->RGB, resize directly to 256x256 with
cv2.INTER_NEAREST (exactly as dataset.py uses for both the image and the
mask, not just the mask), then ToTensor() (divide by 255, no other
mean/std normalization) -- this is exactly the resolution/preprocessing that
unet_kvasir_best.pth was trained with by train_stage1.py; getting any of
these steps wrong will produce a low-quality mask even with the correct
checkpoint.

The checkpoint was saved directly with `torch.save(model.state_dict(), ...)`
using a bare `model = UNet(...)` (train_stage1.py lines 114+172), NOT via
TumorSegmentor/build_unet_encoder in src/models/unet.py (that wrapper wraps
the UNet in `self.unet`, expecting the state_dict to have the "unet." prefix
-- this checkpoint has no such prefix, so loading it via
build_unet_encoder(strict=False) would SILENTLY skip all the weights).
Therefore this script instantiates the UNet directly and loads it with
strict=True to immediately catch any key mismatch.

The predicted mask, at 256x256 resolution (matching the UNet input), is
resized BACK to the original image size before being saved -- because
enrich_vqa_metadata.py computes the bounding box based on the mask file's
own size (mask.shape), and EndoDataset later crops the ROI region directly
from the original (unresized) image using that box's coordinates
(dataset.py line ~138: `image[y1:y2, x1:x2]`) -- if the mask were saved at
256x256 while the original image is larger, the box coordinates would be
completely misaligned with the actual image content.

The script is resumable: it skips images that already have both an image
and a mask in the destination directory, so it can be rerun if interrupted
partway through.

Input:  kvasir_vqa_ready/dataset.json, kvasir_vqa_ready/images/
Output: kvasir_vqa/dataset.json (copied as-is), kvasir_vqa/images/,
        kvasir_vqa/masks/{id}.png
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

SOURCE_DIR = Path("kvasir_vqa_ready")
SOURCE_JSON = SOURCE_DIR / "dataset.json"

DEST_DIR = Path("kvasir_vqa")
DEST_IMG_DIR = DEST_DIR / "images"
DEST_MASK_DIR = DEST_DIR / "masks"
DEST_JSON = DEST_DIR / "dataset.json"

# [New user setup] Stage 1 UNet checkpoint used to auto-generate masks --
# must exist before running this script (scripts/run_stage1.sh).
CHECKPOINT_PATH = "checkpoints/stage1_seg_unet_full/unet_kvasir_best.pth"

GLOBAL_SIZE = (256, 256)   # must match EndoDataset.global_size
BINARY_THRESHOLD = 0.5
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.Compose([transforms.ToTensor()])  # same transform as EndoDataset.normalize


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
        for i in tqdm(range(0, len(todo_ids), BATCH_SIZE), desc="Generating masks"):
            chunk_ids = todo_ids[i:i + BATCH_SIZE]

            chunk_bgr, chunk_shapes, valid_ids = [], [], []
            for img_id in chunk_ids:
                src_img_path = SOURCE_DIR / id_to_rel_path[img_id]
                image_bgr = cv2.imread(str(src_img_path))
                if image_bgr is None:
                    failed.append(img_id)
                    continue
                chunk_bgr.append(image_bgr)
                chunk_shapes.append(image_bgr.shape[:2])  # original (H, W)
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
    print("Next step: run enrich_vqa_metadata.py to compute bounding boxes from these masks.")


if __name__ == "__main__":
    main()
