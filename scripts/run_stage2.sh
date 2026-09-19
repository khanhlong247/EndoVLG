#!/bin/bash
# Launcher for Stage 2 (vision-diagnostic alignment).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

echo "Running Stage 2: Vision Encoder training"
python training/train_stage2.py
echo "TRAINING FINISHED SUCCESSFULLY"
