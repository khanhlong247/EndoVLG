"""
Precompute backbone features (volume_encoder + mask_encoder, frozen
throughout Stage 3) for the entire Stage 3 train set, K=1 variant/sample
(original image only, NO augmentation). Brightness/contrast jitter is no
longer cached statically -- it now runs "live" (randomly on every call)
directly inside EndoRegHD.forward_vision_features (see
src/models/endoreghd.py, the live_augment parameter), to restore unlimited
data diversity instead of being limited to a few fixed variants (confirmed
by experiment to be the cause of the model overfitting faster). This K=1
cache is only used for the batch portion selected to "use the original
image" (ORIGINAL_VARIANT_PROB in train.py), letting that portion skip the
backbone without recomputation.

The geometry branch (mask_encoder) is NOT color-jittered -- brightness/
contrast only makes sense for photographic images (global/texture), not for
the geometric mask, matching the design of the original augment_pair (which
only jitters `img`, never touches `mask`). Since there is no longer any
flip/rotate, geometry_feats does not change across variants of the same
sample -- it only needs to be computed once, not multiplied by K, saving
both compute and RAM.

Does not modify/touch src/data/dataset.py or train_stage1.py/train_stage2.py
-- uses EndoDataset in mode='val' (deterministic, no augmentation) to get the
raw tensor, then applies the jitter itself within this script.

Output: a single .pt file, loaded straight into RAM in train.py.
"""
import os
import random
import torch
from tqdm import tqdm

from src.data.dataset import EndoDataset
from src.models.endoreghd import EndoRegHD

# [New user setup] Run via `bash scripts/precompute_vision_cache.sh`:
#   - DATA_DIR: Kvasir-VQA-x1 train split (from data/split_kvasir_vqa.py).
#   - STAGE2_CKPT_PATH: best checkpoint produced by Stage 2.
DATA_DIR = "kvasir_vqa_train/train"
JSON_PATH = os.path.join(DATA_DIR, "dataset.json")
STAGE2_CKPT_PATH = "checkpoints/stage2_vision_alignment/vision_alignment_best.pth"
CACHE_OUTPUT_PATH = "checkpoints/stage3_vision_cache_K1.pt"

K_VARIANTS = 1                      # only variant 0 = original (fixed jitter removed)
JITTER_ALPHA_RANGE = (0.8, 1.2)     # matches original augment_pair (contrast)
JITTER_BETA_RANGE = (-30, 30)       # matches original augment_pair (brightness), scale [0,255]
JITTER_PROB = 0.8                   # matches the jitter application probability in the original augment_pair
MAX_REGIONS = 6
BATCH_SIZE = 32
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_jitter(t):
    """Approximates cv2.convertScaleAbs(img, alpha, beta) on a ToTensor()'d tensor (values in [0,1]):
    pixel_new = clip(pixel*alpha + beta/255, 0, 1). Only applied with probability JITTER_PROB,
    matching the original augment_pair (`if random.random() < 0.8`)."""
    if random.random() >= JITTER_PROB:
        return t.clone()
    alpha = random.uniform(*JITTER_ALPHA_RANGE)
    beta = random.uniform(*JITTER_BETA_RANGE)
    return torch.clamp(t * alpha + beta / 255.0, 0.0, 1.0)


@torch.no_grad()
def compute_raw_features(model, global_batch, texture_batch):
    """Runs exactly the frozen backbone part (no adapter) -- extracted from
    EndoRegHD.forward_vision_features's cached_raw=None branch, taking only
    the g_feats/t_feats_raw part (does NOT run mask_encoder here, geometry is
    handled separately since it doesn't change across variants)."""
    x_in = model.volume_encoder.unet.initial(global_batch)
    enc0 = model.volume_encoder.unet.down0(torch.nn.functional.max_pool2d(x_in, 2))
    enc1 = model.volume_encoder.unet.down1(torch.nn.functional.max_pool2d(enc0, 2))
    enc2 = model.volume_encoder.unet.down2(torch.nn.functional.max_pool2d(enc1, 2))
    g_map = model.volume_encoder.unet.down3(torch.nn.functional.max_pool2d(enc2, 2))
    g_feats = torch.mean(g_map, dim=[2, 3])

    B, N, C, H, W = texture_batch.shape
    t_flat = texture_batch.reshape(-1, C, H, W)
    t_in = model.volume_encoder.unet.initial(t_flat)
    t_e0 = model.volume_encoder.unet.down0(torch.nn.functional.max_pool2d(t_in, 2))
    t_e1 = model.volume_encoder.unet.down1(torch.nn.functional.max_pool2d(t_e0, 2))
    t_e2 = model.volume_encoder.unet.down2(torch.nn.functional.max_pool2d(t_e1, 2))
    t_map = model.volume_encoder.unet.down3(torch.nn.functional.max_pool2d(t_e2, 2))
    t_feats_raw = torch.mean(t_map, dim=[2, 3]).reshape(B, N, -1)

    return g_feats, t_feats_raw


@torch.no_grad()
def compute_geo_features(model, texture_batch):
    """geo_feats_raw = mask_encoder(t_flat) -- identical to the original
    forward_vision_features, NOT jittered (mask_encoder takes texture_imgs as
    input, see endoreghd.py line 107: geo_feats_raw = self.mask_encoder(t_flat),
    with t_flat derived from texture_imgs)."""
    B, N, C, H, W = texture_batch.shape
    t_flat = texture_batch.reshape(-1, C, H, W)
    geo_feats_raw = model.mask_encoder(t_flat).reshape(B, N, -1)
    return geo_feats_raw


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Initializing EndoRegHD (no LLM needed for this precompute step)...")
    model = EndoRegHD(llm_path="NousResearch/Llama-2-7b-chat-hf", vision_ckpt_path=None, init_llm=False)

    print(f"Loading Stage 2 weights: {STAGE2_CKPT_PATH}")
    state_dict = torch.load(STAGE2_CKPT_PATH, map_location="cpu")
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("llm.")}
    model.load_state_dict(filtered_state_dict, strict=False)

    for name, module in model.named_children():
        if name != "llm":
            module.to(device)
    model.eval()

    dataset = EndoDataset(DATA_DIR, JSON_PATH, tokenizer=None, mode="val", max_regions=MAX_REGIONS)

    # Many QA rows (101,284) point to the same image (4,348 unique images) and
    # share exactly the same set of 'regions' (confirmed: same id -> same
    # regions, since regions are computed per image in the enrich_vqa_metadata.py
    # step) -- so the backbone only needs to run once per image, not once per
    # QA row (avoiding ~23x unnecessary repetition).
    seen_ids = set()
    unique_indices = []
    for i in range(len(dataset)):
        img_id = dataset.data[i]["id"]
        if img_id not in seen_ids:
            seen_ids.add(img_id)
            unique_indices.append(i)

    print(f"{len(dataset)} QA rows / {len(unique_indices)} unique images -- "
          f"precomputing cache per image, K={K_VARIANTS} variant(s)/image...")

    cache = {}
    pbar = tqdm(range(0, len(unique_indices), BATCH_SIZE), desc="Precompute vision cache")
    for start in pbar:
        batch_indices = unique_indices[start:start + BATCH_SIZE]
        items = [dataset[i] for i in batch_indices]
        batch_ids = [dataset.data[i]["id"] for i in batch_indices]

        global_orig = torch.stack([it["global"] for it in items]).to(device)
        texture_orig = torch.stack([it["texture"] for it in items]).to(device)
        num_regions_list = [it["num_regions"] for it in items]

        # region_mask does not change across variants (only depends on num_regions, not the image)
        B, N = global_orig.shape[0], MAX_REGIONS
        region_mask = torch.zeros(B, N, 1, device=device)
        for i, nr in enumerate(num_regions_list):
            region_mask[i, :nr, 0] = 1.0
        region_mask = region_mask.reshape(B * N, 1)

        # geometry: computed only once, shared across all K variants
        geo_feats_raw = compute_geo_features(model, texture_orig)

        g_feats_variants = []
        t_feats_variants = []
        for k in range(K_VARIANTS):
            if k == 0:
                g_batch, t_batch = global_orig, texture_orig
            else:
                g_batch = maybe_jitter(global_orig)
                t_batch = maybe_jitter(texture_orig)
            g_feats, t_feats_raw = compute_raw_features(model, g_batch, t_batch)
            g_feats_variants.append(g_feats.cpu())
            t_feats_variants.append(t_feats_raw.cpu())

        g_feats_stack = torch.stack(g_feats_variants, dim=1)   # (B, K, 1024)
        t_feats_stack = torch.stack(t_feats_variants, dim=1)   # (B, K, N, 1024)
        geo_feats_cpu = geo_feats_raw.cpu()                    # (B, N, 1024)
        region_mask_cpu = region_mask.cpu().reshape(B, N, 1)   # (B, N, 1)

        for i, sample_id in enumerate(batch_ids):
            cache[sample_id] = {
                "g_feats": g_feats_stack[i],          # (K, 1024)
                "t_feats_raw": t_feats_stack[i],       # (K, N, 1024)
                "geo_feats_raw": geo_feats_cpu[i],     # (N, 1024)
                "region_mask": region_mask_cpu[i],     # (N, 1)
                "num_regions": num_regions_list[i],
            }

    print(f"Saving cache ({len(cache)} samples) to {CACHE_OUTPUT_PATH} ...")
    torch.save({"K_VARIANTS": K_VARIANTS, "max_regions": MAX_REGIONS, "cache": cache}, CACHE_OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
