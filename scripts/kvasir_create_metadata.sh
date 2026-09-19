#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 2/5 (train split): build dataset.json
# (id/image/question/answer/complexity) from the HuggingFace dataset.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/kvasir_create_metadata.py
