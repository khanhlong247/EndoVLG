import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import gc
import numpy as np
import random
import wandb

from src.data.dataset import EndoDataset
from src.models.endoreghd import EndoRegHD
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- STAGE 3 CONFIG ---
# [New user setup] All paths below are relative to the project root (run this
# script via `bash scripts/train.sh`, or `python -u training/train.py` from
# the project root so relative paths resolve correctly):
#   - DATA_DIR/VAL_DATA_DIR: output of the Kvasir-VQA-x1 build pipeline
#     (see data/, run in order via scripts/kvasir_download.sh -> ... ->
#     scripts/split_kvasir_vqa.sh), i.e. kvasir_vqa_train/{train,val}.
#   - STAGE2_CKPT_PATH: best checkpoint produced by Stage 2
#     (scripts/run_stage2.sh -> checkpoints/stage2_vision_alignment/).
#   - RESUME_CKPT_PATH: optional -- a prior Stage 3 checkpoint to resume
#     from; set to None (or a non-existent path) to train from scratch.
#   - VISION_CACHE_PATH: optional speed-up, produced by
#     scripts/precompute_vision_cache.sh (safe to leave missing).
DATA_DIR = "kvasir_vqa_train/train"
VAL_DATA_DIR = "kvasir_vqa_train/val"
JSON_PATH = os.path.join(DATA_DIR, "dataset.json")
VAL_JSON_PATH = os.path.join(VAL_DATA_DIR, "dataset.json")

LLM_PATH = "NousResearch/Llama-2-7b-chat-hf"

STAGE2_CKPT_PATH = "checkpoints/stage2_vision_alignment/vision_alignment_best.pth"
RESUME_CKPT_PATH = "checkpoints/stage3_kvasir_v4/EndoRegHD_Epoch24.pth"
OUTPUT_DIR = "checkpoints/stage3_kvasir_v5"

BATCH_SIZE = 32
ACCUM_STEPS = 20
EPOCHS = 25
LR_LORA = 2e-5
SAVE_FREQ = 10

STAGE31_EPOCH = 2
STAGE32_EPOCH = 5
STAGE33_EPOCH = 15

STAGE34_LR = 1e-5
STAGE34_WEIGHT_DECAY = 0.1

VISION_CACHE_PATH = "checkpoints/stage3_vision_cache_K1.pt"
ORIGINAL_VARIANT_PROB = 0.4
CLIP_GRAD_NORM = 1.0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- UTILS ---
def collate_fn(batch):
    global_imgs = torch.stack([item['global'] for item in batch])
    texture_imgs = torch.stack([item['texture'] for item in batch])
    geo_imgs = torch.stack([item['geometry'] for item in batch])
    meta = []
    for item in batch:
        meta.append({
            'question': item['question'],
            'answer': item['answer'],
            'num_regions': item['num_regions'],
            'id': item.get('id')
        })
    return global_imgs, texture_imgs, geo_imgs, meta


class _CacheIdDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        item = self.subset[i]
        real_idx = self.subset.indices[i]
        item['id'] = self.subset.dataset.data[real_idx]['id']
        return item


def validate_stage_epochs():
    assert STAGE31_EPOCH < STAGE32_EPOCH, \
        f"STAGE31_EPOCH ({STAGE31_EPOCH}) must be less than STAGE32_EPOCH ({STAGE32_EPOCH})"
    assert STAGE32_EPOCH < STAGE33_EPOCH, \
        f"STAGE32_EPOCH ({STAGE32_EPOCH}) must be less than STAGE33_EPOCH ({STAGE33_EPOCH})"
    assert STAGE33_EPOCH < EPOCHS, \
        f"STAGE33_EPOCH ({STAGE33_EPOCH}) must be less than EPOCHS ({EPOCHS})"


def build_cached_raw(meta, vision_cache, device):
    if vision_cache is None or any(m.get('id') not in vision_cache for m in meta):
        return None
    g_feats_list, t_feats_list, geo_feats_list, region_mask_list = [], [], [], []
    for m in meta:
        entry = vision_cache[m['id']]
        g_feats_list.append(entry['g_feats'][0])
        t_feats_list.append(entry['t_feats_raw'][0])
        geo_feats_list.append(entry['geo_feats_raw'])
        region_mask_list.append(entry['region_mask'])
    B = len(meta)
    N = t_feats_list[0].shape[0]
    return {
        'g_feats': torch.stack(g_feats_list).to(device),
        't_feats_raw': torch.stack(t_feats_list).reshape(B * N, -1).to(device),
        'geo_feats_raw': torch.stack(geo_feats_list).reshape(B * N, -1).to(device),
        'region_mask': torch.stack(region_mask_list).reshape(B * N, 1).to(device),
        'B': B, 'N': N,
    }


def train():
    validate_stage_epochs()

    wandb.init(project="EndoRegHD-Stage3-KvasirVQA", name="Llama2-7b-Progressive-Train")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing Model with Unsloth & Flash Attention-2...")

    model = EndoRegHD(llm_path=LLM_PATH, vision_ckpt_path=None, init_llm=False)

    base_llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_PATH,
        max_seq_length = 2048,
        load_in_4bit = True,
        trust_remote_code = True,
        dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, 'llm'): del model.llm
    gc.collect()
    torch.cuda.empty_cache()

    model.llm = FastLanguageModel.get_peft_model(
        base_llm,
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.llm.print_trainable_parameters()

    for name, _ in model.llm.named_modules():
        if "q_proj" in name:
            print(name)
            break
    
    print(f"Loading Stage 2 weights from {STAGE2_CKPT_PATH}...")
    state_dict = torch.load(STAGE2_CKPT_PATH, map_location='cpu')

    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("llm.")}
    model.load_state_dict(filtered_state_dict, strict=False)

    for name, module in model.named_children():
        if name != "llm":
            module.to(device)

    vision_cache_data = None
    if os.path.exists(VISION_CACHE_PATH):
        print(f"[Cache Stage 3] Loading vision cache from {VISION_CACHE_PATH}...")
        vision_cache_data = torch.load(VISION_CACHE_PATH, map_location='cpu')
        print(f"[Cache Stage 3] Loaded cache for {len(vision_cache_data['cache'])} samples, "
              f"K={vision_cache_data['K_VARIANTS']} variants.")
    else:
        print(f"[Cache Stage 3] Could not find {VISION_CACHE_PATH} -- "
              f"run precompute_vision_cache.py first to enable caching. Training will run live (no cache).")

    full_dataset = EndoDataset(DATA_DIR, JSON_PATH, tokenizer, mode='train', max_regions=6)

    indices = []
    random.seed(42)
    for i, item in enumerate(full_dataset.data):
        complexity = item.get('complexity', 1)
        if complexity == 3:
            indices.append(i)
        else:
            if random.random() < 1.0:
                indices.append(i)

    train_subset = Subset(full_dataset, indices)
    train_subset = _CacheIdDataset(train_subset)  # [Cache Stage 3]

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = EndoDataset(VAL_DATA_DIR, VAL_JSON_PATH, tokenizer, mode='val', max_regions=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    scaler = torch.amp.GradScaler('cuda', init_scale=2**10)
    trainable_params = []

    for m in [model.texture_adapter, model.geometry_projector, model.global_adapter, model.fusion_layer]:
        trainable_params.extend(list(m.parameters()))

    for name, param in model.llm.named_parameters():
        if "lora" in name:
            trainable_params.append(param)

    print(f"[Debug] dtype of {len(trainable_params)} trainable params -- "
          f"LoRA: {next(p.dtype for n, p in model.llm.named_parameters() if 'lora' in n)}, "
          f"Adapter (texture): {model.texture_adapter.parameters().__next__().dtype}")

    optimizer = optim.AdamW(trainable_params, lr=LR_LORA, weight_decay=0.01)

    updates_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    total_training_steps = updates_per_epoch * EPOCHS
    warmup_steps = min(100, max(1, total_training_steps // 10))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    print(f"Start Progressive Training...")
    
    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_CKPT_PATH and os.path.exists(RESUME_CKPT_PATH):
        print(f"\nRESUMING training from {RESUME_CKPT_PATH}...")
        checkpoint = torch.load(RESUME_CKPT_PATH, map_location='cpu')
        saved_state_dict = checkpoint['model_state_dict']
        
        clean_state_dict = {
            k: v for k, v in saved_state_dict.items()
            if not any(ext in k for ext in [".absmax", ".quant_map", ".nested_", ".quant_state"])
        }
        
        model.load_state_dict(clean_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        if checkpoint['epoch'] >= STAGE33_EPOCH:
            resume_lr = 3e-6
            resume_weight_decay = 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = resume_lr
                param_group['weight_decay'] = resume_weight_decay
            remaining_epochs = EPOCHS - start_epoch
            remaining_steps = updates_per_epoch * remaining_epochs
            warmup_steps_resume = max(1, remaining_steps // 20)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps_resume,
                num_training_steps=remaining_steps
            )
            train._lr_reset = True
            train._clip_norm = 2.0
            print(f"[Resume Stage 3.4] checkpoint['epoch']={checkpoint['epoch']} -- "
                  f"LR={resume_lr}, Weight Decay={resume_weight_decay}, "
                  f"Clip Grad Norm={train._clip_norm}, "
                  f"new mini-schedule: {remaining_epochs} epochs ({remaining_steps} steps, "
                  f"warmup={warmup_steps_resume} steps), "
                  f"current optimizer.param_groups[0]['lr']={optimizer.param_groups[0]['lr']}")

        print(f"Resumed successfully! Continuing from Epoch {start_epoch + 1} (Best Val Loss: {best_val_loss:.4f})\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for p in model.parameters(): p.requires_grad = False
        
        if epoch < STAGE31_EPOCH:
            for m in [model.texture_adapter, model.geometry_projector]:
                for p in m.parameters(): p.requires_grad = True
        elif epoch < STAGE32_EPOCH:
            for m in [model.texture_adapter, model.geometry_projector, model.global_adapter, model.fusion_layer]:
                for p in m.parameters(): p.requires_grad = True
        elif epoch < STAGE33_EPOCH:
            for m in [model.texture_adapter, model.geometry_projector, model.global_adapter, model.fusion_layer]:
                for p in m.parameters(): p.requires_grad = True
            for name, param in model.llm.named_parameters():
                if "lora" in name and any(x in name for x in ["q_proj", "v_proj", "k_proj", "o_proj"]):
                    param.requires_grad = True
        else: 
            for m in [model.texture_adapter, model.geometry_projector, model.global_adapter, model.fusion_layer]:
                for p in m.parameters(): p.requires_grad = True
            for name, param in model.llm.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

            if not getattr(train, '_lr_reset', False):
                print(f"[Stage 3.4] Resetting Learning Rate -> {STAGE34_LR}, Weight Decay -> {STAGE34_WEIGHT_DECAY}...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = STAGE34_LR
                    param_group['weight_decay'] = STAGE34_WEIGHT_DECAY
                remaining_epochs = EPOCHS - epoch
                remaining_steps = updates_per_epoch * remaining_epochs
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=max(1, remaining_steps // 10),
                    num_training_steps=remaining_steps
                )
                train._lr_reset = True

        model.volume_encoder.eval()

        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            g_img = batch[0].to(device, non_blocking=True)
            t_img = batch[1].to(device, non_blocking=True)
            geo_img = batch[2].to(device, non_blocking=True)
            meta = batch[3]

            use_cache_this_batch = random.random() < ORIGINAL_VARIANT_PROB
            cached_raw = build_cached_raw(
                meta, vision_cache_data['cache'] if vision_cache_data else None, device
            ) if use_cache_this_batch else None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                g_embeds, r_embeds = model.forward_vision_features(
                    g_img, t_img, geo_img, cached_raw=cached_raw, live_augment=(cached_raw is None)
                )
                prompt_embeds, label_ids, attention_mask = model.prepare_multimodal_prompt(
                    g_embeds, r_embeds, meta, tokenizer, device
                )
                outputs = model.llm(inputs_embeds=prompt_embeds, labels=label_ids, attention_mask=attention_mask)
                loss = outputs.loss / ACCUM_STEPS
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    getattr(train, '_clip_norm', CLIP_GRAD_NORM)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({'loss': f"{loss.item() * ACCUM_STEPS:.4f}"})

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    g_emb, r_emb = model.forward_vision_features(
                        val_batch[0].to(device), val_batch[1].to(device), val_batch[2].to(device)
                    )
                    input_embeds, label_ids, attention_mask = model.prepare_multimodal_prompt(
                        g_emb, r_emb, val_batch[3], tokenizer, device
                    )
                    outputs = model.llm(inputs_embeds=input_embeds, labels=label_ids, attention_mask=attention_mask)
                    if not torch.isnan(outputs.loss):
                        total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": avg_train_loss, "val/loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            local_save_path = os.path.join(OUTPUT_DIR, f"EndoRegHD_Epoch{epoch+1}.pth")
            torch.save(checkpoint, local_save_path)
            print(f"Successfully saved local checkpoint at: {local_save_path}")

    wandb.finish()

if __name__ == "__main__":
    train()