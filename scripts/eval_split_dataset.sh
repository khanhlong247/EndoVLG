#!/bin/bash
# Builds the Stage 1/2 validation split (data_val_100) from the debug dataset.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/eval_split_dataset.py
