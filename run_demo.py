"""
run_demo.py

One-command end-to-end pipeline:
  1. (re)generate the dataset
  2. generate suggested replies for the held-out test split
  3. evaluate those replies (per-response + overall scores)
  4. validate the evaluation metric against the human-labeled calibration set

Run from the repo root:
    python3 run_demo.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    ("Building dataset", ["python3", "data/generate_dataset.py"]),
    ("Generating suggested replies for held-out test emails", ["python3", "src/generate_response.py", "--test-set"]),
    ("Evaluating generated replies", ["python3", "src/evaluate.py"]),
    ("Validating the metric against human-labeled calibration set", ["python3", "src/validate_metric.py"]),
]

def main():
    for title, cmd in STEPS:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"Step failed: {title}")
            sys.exit(result.returncode)
    print(f"\n{'='*70}\nDone. See outputs/generations.jsonl, outputs/evaluation_report.json, "
          f"and outputs/metric_validation.json for full results.\n{'='*70}")

if __name__ == "__main__":
    main()
