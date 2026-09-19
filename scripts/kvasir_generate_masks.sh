#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 3/5 (train split): generate lesion masks
# with the Stage 1 UNet checkpoint (checkpoints/stage1_seg_unet_full/...).
# Requires Stage 1 (scripts/run_stage1.sh) to have produced that checkpoint.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/kvasir_generate_masks.py
