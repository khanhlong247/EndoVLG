#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 2/4 (official test split): build
# dataset.json from the HuggingFace dataset.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/kvasir_create_metadata_test.py
