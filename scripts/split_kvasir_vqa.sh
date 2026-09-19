#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 5/5 (train split only): split
# kvasir_vqa/ into kvasir_vqa_train/{train,val} (90/10 by image, seed 42).
# Run after scripts/enrich_vqa_metadata.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/split_kvasir_vqa.py
