#!/bin/bash
# Launcher for the K-Means region-enrichment step of the Kvasir-VQA-x1 build
# pipeline (train split). Run after kvasir_generate_masks.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

echo "Running enrich_vqa_metadata"
python data/enrich_vqa_metadata.py
echo "FINISHED SUCCESSFULLY"
