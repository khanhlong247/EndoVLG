#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 4/4 (official test split): K-Means
# region enrichment. Run after kvasir_generate_masks_test.py. There is no
# separate split step for the test set -- kvasir_vqa_test/ is used directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/enrich_vqa_metadata_test.py
