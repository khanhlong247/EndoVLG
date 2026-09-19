#!/bin/bash
# Precomputes frozen backbone vision features for kvasir_vqa_train/train
# (K=1, no augmentation) to speed up Stage 3. Run after Stage 2
# (checkpoints/stage2_vision_alignment/vision_alignment_best.pth must exist)
# and after the Kvasir-VQA-x1 train split is built.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/precompute_vision_cache.py
