import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import gc
import numpy as np
import torch.nn.functional as F
import wandb 
from src.data.dataset import EndoDataset
from src.models.endoreghd import EndoRegHD

# --- CONFIG ---
# [New user setup] Run via `bash scripts/run_stage2.sh`, or `python
# training/train_stage2.py` from the project root:
#   - DATA_DIR/VAL_DATA_DIR: same curated malignant/benign dataset as Stage 1.
#   - VISION_INIT_CKPT: the UNet checkpoint produced by Stage 1
#     (scripts/run_stage1.sh).
DATA_DIR = "data_train_700"
VAL_DATA_DIR = "data_val_100"
JSON_PATH = os.path.join(DATA_DIR, "dataset.json")
VAL_JSON_PATH = os.path.join(VAL_DATA_DIR, "dataset.json")

LLM_PATH = "NousResearch/Llama-2-7b-chat-hf"
VISION_INIT_CKPT = "checkpoints/stage1_seg_unet_full/unet_kvasir_best.pth"
OUTPUT_DIR = "checkpoints/stage2_vision_alignment"

BATCH_SIZE = 16
ACCUM_STEPS = 2        
EPOCHS = 40            
LR = 3e-5              

def get_label_robust(item):
    """Kept as-is to serve the image-level Sampler"""
    regions = item.get('regions', [])
    if not regions: return 0
    all_text = " ".join([str(r.get('label_text', '')).lower() for r in regions])
    malignant_keywords = ["malignant", "tumor", "cancer", "carcinoma"]
    return 1 if any(word in all_text for word in malignant_keywords) else 0

# --- FOCAL LOSS CLASS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

# --- DATA PROCESSING FUNCTION ---
def collate_fn_vision(batch):
    global_imgs = torch.stack([item['global'] for item in batch])
    texture_imgs = torch.stack([item['texture'] for item in batch])
    geo_imgs = torch.stack([item['geometry'] for item in batch])
    
    batch_region_labels = []
    malignant_keywords = ["malignant", "tumor", "cancer", "carcinoma"]
    
    for item in batch:
        num = item['num_regions']
        regions = item.get('regions', [])
        labels = []
        
        # Fix: iterate over 6 regions (max_regions)
        for i in range(6):
            if i < num and i < len(regions):
                text = str(regions[i].get('label_text', regions[i].get('label', ''))).lower()
                is_mal = 1.0 if any(k in text for k in malignant_keywords) else 0.0
                labels.append(is_mal)
            elif i < num:
                labels.append(0.0)
            else:
                labels.append(-1.0)
                
        batch_region_labels.append(labels)
        
    return global_imgs, texture_imgs, geo_imgs, torch.tensor(batch_region_labels)

# --- UTILS ---
def set_bn_eval(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()
        for param in module.parameters(): param.requires_grad = False

def validate(model, vision_head, val_loader, device, criterion):
    model.eval()
    vision_head.eval()
    reg_stats = {"mal_total": 0, "mal_hit": 0, "ben_total": 0, "ben_hit": 0}

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for batch in val_loader:
                g_img, t_img, geo_img, labels = [b.to(device) for b in batch]
                g_embeds, r_embeds = model.forward_vision_features(g_img, t_img, geo_img)
                
                # Pairwise fusion logic (Global i, Region j)
                B, N, D = r_embeds.shape
                g_exp = g_embeds.expand(-1, N, -1)
                combined = torch.cat([g_exp, r_embeds], dim=-1).view(B * N, -1)
                
                logits = vision_head(combined).squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.35).long()

                flat_labels = labels.view(-1)
                for p, l in zip(preds, flat_labels):
                    if l == -1.0: continue
                    if l == 1.0:
                        reg_stats["mal_total"] += 1
                        if p == 1: reg_stats["mal_hit"] += 1
                    else:
                        reg_stats["ben_total"] += 1
                        if p == 0: reg_stats["ben_hit"] += 1

    acc_mal = reg_stats["mal_hit"] / max(1, reg_stats["mal_total"])
    acc_ben = reg_stats["ben_hit"] / max(1, reg_stats["ben_total"])
    overall_acc = (reg_stats["mal_hit"] + reg_stats["ben_hit"]) / max(1, reg_stats["mal_total"] + reg_stats["ben_total"])
    
    return overall_acc, acc_mal, acc_ben

# --- TRAINING LOOP ---
def train():
    wandb.init(project="EndoReg-HD-Vision-Alignment", config={"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = EndoRegHD(llm_path=LLM_PATH, vision_ckpt_path=VISION_INIT_CKPT)
    if hasattr(model, 'llm'): del model.llm
    torch.cuda.empty_cache()
    gc.collect()
    
    vision_head = nn.Sequential(
        nn.Linear(4096 * 2, 2048), nn.LayerNorm(2048), nn.ReLU(), nn.Dropout(0.4), nn.Linear(2048, 1)
    ).to(device)
    
    nn.init.constant_(vision_head[-1].bias, 0.0)
    model.to(device)
    
    for name, param in model.volume_encoder.named_parameters(): param.requires_grad = False
    for p in [model.global_adapter, model.texture_adapter, model.mask_encoder, model.geometry_projector]:
        for param in p.parameters(): param.requires_grad = True
    
    model.apply(set_bn_eval)
    dataset = EndoDataset(DATA_DIR, JSON_PATH, tokenizer, mode='train', max_regions=6)
    targets = [get_label_robust(item) for item in dataset.data]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor([1.0/(len(targets)-sum(targets)) if t==0 else 1.0/sum(targets) for t in targets]), num_samples=len(targets), replacement=True)
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn_vision, num_workers=4, pin_memory=True)
    val_loader = DataLoader(EndoDataset(VAL_DATA_DIR, VAL_JSON_PATH, tokenizer, mode='val'), batch_size=BATCH_SIZE, collate_fn=collate_fn_vision)

    new_module_params = list(model.mask_encoder.parameters()) + list(model.geometry_projector.parameters()) + \
                        list(model.global_adapter.parameters()) + list(model.texture_adapter.parameters()) + \
                        list(vision_head.parameters())

    optimizer = optim.AdamW([{'params': list(model.volume_encoder.parameters()), 'lr': 1e-6}, {'params': new_module_params, 'lr': LR}], weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, len(train_loader) * EPOCHS)
    criterion = FocalLoss(alpha=0.5, gamma=2.0).to(device)
    scaler = GradScaler()
    best_score = -1.0
    
    for epoch in range(EPOCHS):
        model.train(); vision_head.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (g_img, t_img, geo_img, labels) in enumerate(pbar):
            g_img, t_img, geo_img, labels = [b.to(device) for b in (g_img, t_img, geo_img, labels)]
            
            optimizer.zero_grad()
            with autocast():
                g_embeds, r_embeds = model.forward_vision_features(g_img, t_img, geo_img)
                
                # Region-level Forward
                B, N, D = r_embeds.shape
                g_exp = g_embeds.expand(-1, N, -1)
                combined = torch.cat([g_exp, r_embeds], dim=-1).view(B * N, -1)
                logits = vision_head(combined).squeeze(-1)

                # Masking loss (only compute loss on actual regions)
                flat_labels = labels.view(-1)
                mask = (flat_labels != -1.0)
                loss = criterion(logits[mask], flat_labels[mask])
            
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(new_module_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            
            scheduler.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            wandb.log({"train/step_loss": loss.item(), "train/lr": scheduler.get_last_lr()[1]})
            
        v_acc, v_mal, v_ben = validate(model, vision_head, val_loader, device, criterion)
        current_score = (v_mal + v_ben) * (1.0 - abs(v_mal - v_ben))
        
        print(f"VAL Ep{epoch+1}: RegionAcc={v_acc:.1%} | Mal={v_mal:.1%} | Ben={v_ben:.1%} | Score={current_score:.4f}")
        wandb.log({"epoch": epoch+1, "val/acc": v_acc, "val/acc_mal": v_mal, "val/acc_ben": v_ben, "val/custom_score": current_score})

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vision_alignment_best.pth"))
            print(f"Saved Best Region Model (Score: {best_score:.4f})")

    wandb.finish()

if __name__ == "__main__":
    train()