import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import gc
import cv2
import numpy as np
import wandb 

from src.data.dataset import EndoDataset
from src.models.unet import UNet 

# --- CONFIG ---
# [New user setup] Run via `bash scripts/run_stage1.sh`, or `python
# training/train_stage1.py` from the project root:
#   - DATA_DIR/VAL_DATA_DIR: a curated malignant/benign polyp dataset with
#     image+mask pairs (see data/split_dataset.py, data/eval_split_dataset.py
#     -- their source directory is external and must be supplied separately).
#   - KVASIR_PTH: an externally pretrained Kvasir-SEG UNet checkpoint used to
#     initialize this stage; not produced by any script in this repo.
DATA_DIR = "data_train_700"
VAL_DATA_DIR = "data_val_100"
JSON_PATH = os.path.join(DATA_DIR, "dataset.json")
VAL_JSON_PATH = os.path.join(VAL_DATA_DIR, "dataset.json")
OUTPUT_DIR = "checkpoints/stage1_seg_unet_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KVASIR_PTH = "checkpoints/kvasir_seg/UNet_IoUBCELoss_augmented.pth"
START_EPOCH = 0
EPOCHS = 100 
BATCH_SIZE = 16
ACCUM_STEPS = 2
BASE_LR = 1e-4
PROJECT_NAME = "EndoReg-HD-Segmentation" 

# --- MINIMAL CHANGE: custom batch collate function for Stage 1 ---
def collate_fn_stage1(batch):
    global_imgs = torch.stack([item['global'] for item in batch])
    global_masks = torch.stack([item['global_mask'] for item in batch])
    return {
        'global': global_imgs,
        'global_mask': global_masks
    }

# --- LOSS FUNCTION FROM THE KVASIR-SEG REPO ---
class IoUBCELoss(nn.Module):
    def __init__(self):
        super(IoUBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs_after_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_after_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        total = (inputs_flat + targets_flat).sum()
        union = total - intersection
        
        iou_loss = 1 - (intersection + smooth) / (union + smooth)
        bce_loss = self.bce(inputs, targets)
        return iou_loss + bce_loss

def tversky_loss(inputs, targets, alpha=0.3, beta=0.7, smooth=1e-4):
    inputs = torch.sigmoid(inputs).view(-1)
    targets = targets.view(-1)
    tp = (inputs * targets).sum()
    fp = (inputs * (1 - targets)).sum()
    fn = ((1 - inputs) * targets).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

# --- VALIDATE ---
def validate(model, val_loader, device, criterion, epoch, best_dice):
    model.eval()
    total_dice, steps = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Read from the dict according to EndoDataset's new structure
            images = batch['global'].to(device)
            masks = batch['global_mask'].to(device).float()
            
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            intersect = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum()
            dice = ((2. * intersect + 1e-6) / (union + 1e-6)).item()
            total_dice += dice
            steps += 1
            
            if i == 0 and dice > best_dice:
                pred_mask = (pred_binary[0][0].cpu().numpy() * 255).astype(np.uint8)
                wandb.log({"val/prediction": wandb.Image(pred_mask, caption=f"Ep {epoch+1} Dice {dice:.4f}")}, commit=False)
                
    return total_dice / steps if steps > 0 else 0

def train():
    wandb.init(project=PROJECT_NAME, name="UNet_Kvasir_FineTune", reinit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIXED: Added collate_fn=collate_fn_stage1 for both DataLoaders to stop the metadata size-mismatch error
    train_loader = DataLoader(
        EndoDataset(DATA_DIR, JSON_PATH, mode='train'), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn_stage1
    )
    val_loader = DataLoader(
        EndoDataset(VAL_DATA_DIR, VAL_JSON_PATH, mode='val'), 
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn_stage1
    )

    # Initialize the standard UNet
    model = UNet(channel_in=3, channel_out=1)

    # Load all pretrained weights
    if os.path.exists(KVASIR_PTH):
        print(f"Loading pretrained UNet weights from: {KVASIR_PTH}")
        state = torch.load(KVASIR_PTH, map_location='cpu')
        model.load_state_dict(state.get('net', state), strict=True)
        print("Successfully loaded pretrained weights!")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = IoUBCELoss()
    scaler = GradScaler('cuda')
    best_val_dice = 0.0

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for i, batch in enumerate(pbar):
            images = batch['global'].to(device)
            masks = batch['global_mask'].to(device).float()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks) + 0.5 * tversky_loss(outputs, masks)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({'Loss': f"{loss.item()*ACCUM_STEPS:.4f}"})

        scheduler.step()

        # Validation
        avg_dice = validate(model, val_loader, device, criterion, epoch, best_val_dice)

        wandb.log({
            "val/dice": avg_dice, 
            "train/loss": epoch_loss / len(train_loader),
            "train/lr": optimizer.param_groups[0]['lr'], 
            "epoch": epoch + 1
        })
        
        tqdm.write(f"Ep {epoch+1} | Val Dice: {avg_dice:.4f}")

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            # Save the best model
            save_path = os.path.join(OUTPUT_DIR, "unet_kvasir_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Dice: {best_val_dice:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()