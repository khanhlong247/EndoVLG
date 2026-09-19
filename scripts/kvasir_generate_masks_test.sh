#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 3/4 (official test split): generate
# lesion masks with the Stage 1 UNet checkpoint. Requires Stage 1 to have
# produced checkpoints/stage1_seg_unet_full/unet_kvasir_best.pth.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/kvasir_generate_masks_test.py
