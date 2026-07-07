"""
validate_metric.py

Answers: "does the automatic composite score actually track human judgment of
quality, or is it just a number that looks principled?"

We run the exact same scoring pipeline used in evaluate.py against
data/human_eval_calibration.jsonl -- 13 (incoming email, candidate reply)
pairs that were independently given a 1-5 human quality score with a written
rationale at dataset-build time (see data/generate_dataset.py:
build_calibration_set). These pairs deliberately include both strong and
weak replies to the *same* email, so the calibration set actually tests
whether the metric can tell good from bad, not just recover surface
similarity to one reference.

We report:
  - Pearson correlation (linear agreement) between our composite score and
    the human 1-5 score
  - Spearman correlation (rank agreement -- arguably more meaningful here,
    since we mainly need the metric to *rank* replies correctly, e.g. for
    routing low-scoring drafts to human review)
  - The full table of (human score, auto score, rationale) for manual
    inspection, so a reviewer can see any specific disagreements

A metric with weak/no correlation here should not be trusted, regardless of
how well-motivated it seems, and we say so explicitly if that happens.
"""
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from llm_client import LLMClient
from evaluate import evaluate_one

DATA_PATH = Path(__file__).parent.parent / "data" / "human_eval_calibration.jsonl"
OUT_PATH = Path(__file__).parent.parent / "outputs" / "metric_validation.json"


def main():
    client = LLMClient()
    calib = [json.loads(l) for l in open(DATA_PATH)]

    rows = []
    for c in calib:
        record = {
            "id": c["id"],
            "incoming_email": c["email"],
            "suggested_reply": c["reply"],
            "reference_reply": "",  # no single reference in calibration mode: judged on its own merits
            "key_points": c.get("key_points", []),
            "retrieved_example_texts": [c["email"]],
        }
        result = evaluate_one(record, client)
        rows.append({
            "id": c["id"],
            "human_score": c["human_score"],
            "human_rationale": c["human_rationale"],
            "auto_composite_1_5": result["composite_score_1_5"],
            "auto_llm_judge_1_5": round(result["scores"]["llm_judge"] * 5, 2),
            "auto_scores": result["scores"],
        })

    human = [r["human_score"] for r in rows]
    auto_composite = [r["auto_composite_1_5"] for r in rows]
    auto_judge = [r["auto_llm_judge_1_5"] for r in rows]

    pearson_composite, _ = pearsonr(human, auto_composite)
    spearman_composite, _ = spearmanr(human, auto_composite)
    pearson_judge, _ = pearsonr(human, auto_judge)
    spearman_judge, _ = spearmanr(human, auto_judge)

    summary = {
        "n_calibration_pairs": len(rows),
        "composite_score_vs_human": {
            "pearson_r": round(pearson_composite, 3),
            "spearman_rho": round(spearman_composite, 3),
        },
        "llm_judge_alone_vs_human": {
            "pearson_r": round(pearson_judge, 3),
            "spearman_rho": round(spearman_judge, 3),
        },
        "used_real_llm_judge": client.using_real_llm,
        "interpretation": (
            "Spearman rho > 0.7 with the composite score => the metric reliably ranks "
            "better vs worse replies the way a human does, which is the property we actually "
            "need (e.g. to decide which drafts need human review before sending). "
            "If run with the offline mock judge (used_real_llm_judge=false), correlations will "
            "be noisier because the mock judge is a crude heuristic; re-run with "
            "ANTHROPIC_API_KEY set for the real validation number to trust for decision-making."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "pairs": rows}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nPer-pair detail:")
    for r in rows:
        print(f"  {r['id']}: human={r['human_score']}  auto_composite={r['auto_composite_1_5']}  "
              f"({r['human_rationale'][:60]}...)")
    print(f"\nFull detail written to {OUT_PATH}")


if __name__ == "__main__":
    main()
