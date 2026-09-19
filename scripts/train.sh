#!/bin/bash
# Launcher for Stage 3 (SFT / VQA fine-tuning). Always resolves the project
# root from this script's own location, so it can be run from anywhere,
# e.g. `bash scripts/train.sh`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/train_run_$TIMESTAMP.log"

if [ ! -f "training/train.py" ]; then
    echo "Error: training/train.py not found under project root ($PROJECT_ROOT)!"
    exit 1
fi

python -u training/train.py 2>&1 | tee "$LOG_FILE"

if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "=================================================================="
    echo "TRAINING FINISHED SUCCESSFULLY"
    echo "=================================================================="
else
    echo "=================================================================="
    echo "TRAINING FAILED"
    echo "Please check the log file: $LOG_FILE"
    echo "=================================================================="
fi
