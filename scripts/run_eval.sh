#!/bin/bash
# Launcher for Stage-3 checkpoint evaluation (NLG metrics on the test set).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

echo "Running Evaluate..."
python -u training/run_eval.py
echo "EVALUATE FINISHED SUCCESSFULLY"
