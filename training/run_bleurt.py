"""
Computes the BLEURT metric separately on existing eval results (results/*/predictions_debug.csv) --
split out from run_eval.py because the `bleurt` library needs an extra install and isn't in the
base requirements:

    uv pip install git+https://github.com/google-research/bleurt.git

Uses the default checkpoint `bleurt-base-128` (evaluate.load("bleurt") without a config_name)
-- matches how the paper 2506.09958v1 (Kvasir-VQA-x1) most likely computed its baseline: the
paper's results table has both NEGATIVE scores (e.g. -0.723, -0.557) and scores around 0.3-0.4,
which matches the score distribution typical of bleurt-base-128 -- quite different from
BLEURT-20 (a newer RemBERT checkpoint that rarely produces negative scores; testing it gave an
average score of ~0.76, too high for a fair comparison).

No need to rerun the model/inference -- this script only reads `predictions_debug.csv` (already
containing Reference/Prediction from run_eval.py), computes the score, and updates the BLEURT
column in `overall_metrics.csv` in the same directory.
"""
import os
import pandas as pd
import evaluate

PREDICTIONS_CSV_PATH = "results/full_kvasir_eval_e24_v4/predictions_debug.csv"
OVERALL_METRICS_CSV_PATH = "results/full_kvasir_eval_e24_v4/overall_metrics.csv"

BATCH_SIZE = 64
LOWERCASE = True


def main():
    if not os.path.exists(PREDICTIONS_CSV_PATH):
        print(f"Could not find {PREDICTIONS_CSV_PATH} -- run run_eval.py first.")
        return

    df = pd.read_csv(PREDICTIONS_CSV_PATH, encoding="utf-8-sig")
    references = df["Reference"].astype(str).tolist()
    predictions = df["Prediction"].astype(str).tolist()

    if LOWERCASE:
        references = [r.lower() for r in references]
        predictions = [p.lower() for p in predictions]

    print(f"Loading BLEURT (bleurt-base-128, ~405MB, first download may take a few minutes)...")
    bleurt = evaluate.load("bleurt", "bleurt-base-128")

    print(f"Computing BLEURT on {len(predictions)} samples (batch_size={BATCH_SIZE})...")
    scores = []
    for start in range(0, len(predictions), BATCH_SIZE):
        end = start + BATCH_SIZE
        result = bleurt.compute(predictions=predictions[start:end], references=references[start:end])
        scores.extend(result["scores"])
        print(f"  {min(end, len(predictions))}/{len(predictions)}", end="\r")

    # DO NOT multiply by 100 -- unlike BLEU/ROUGE/METEOR/CHRF++/BERT-F1 (natural [0,1]
    # fractions), BLEURT is a regression score and doesn't follow a percentage convention --
    # the baseline being compared against is also in raw form (e.g. 0.404), not a percentage.
    bleurt_score = sum(scores) / len(scores)
    print(f"\n\nAverage BLEURT: {bleurt_score:.4f}")

    if os.path.exists(OVERALL_METRICS_CSV_PATH):
        metrics_df = pd.read_csv(OVERALL_METRICS_CSV_PATH)
        metrics_df["BLEURT"] = bleurt_score
        metrics_df.to_csv(OVERALL_METRICS_CSV_PATH, index=False)
        print(f"Updated the BLEURT column in {OVERALL_METRICS_CSV_PATH}")
    else:
        print(f"Could not find {OVERALL_METRICS_CSV_PATH} to update -- just printing the result above.")


if __name__ == "__main__":
    main()
