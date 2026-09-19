#!/bin/bash
# Launcher for Stage 1 (UNet segmentation pretraining).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

echo "Running Stage 1: Detection Pre-training..."
python training/train_stage1.py
echo "TRAINING FINISHED SUCCESSFULLY"
