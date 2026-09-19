#!/bin/bash
# Builds the small curated malignant/benign debug dataset used by Stage 1/2
# (source data directory is external -- see comments in data/split_dataset.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/split_dataset.py
