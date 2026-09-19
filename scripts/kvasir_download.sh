#!/bin/bash
# Kvasir-VQA-x1 build pipeline, step 1/5 (train split): download raw images
# from the SimulaMet/Kvasir-VQA-x1 dataset on HuggingFace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

python data/kvasir_download.py
