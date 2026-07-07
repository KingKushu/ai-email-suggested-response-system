"""
evaluate.py

The core accuracy/evaluation system.

======================================================================
WHAT DOES "ACCURATE" MEAN FOR A SUGGESTED REPLY?
======================================================================
Exact string match to the reply that was actually sent is the wrong target:
two replies can be equally good while using completely different words ("I've
issued your refund" vs "Your refund has been processed"), and the single
historical reply is not the only correct answer -- it's *one* sample from a
space of acceptable replies. So we decompose "accurate" into the properties
we actually care about, each independently measurable:

  1. COVERAGE      - does the reply address every question/request in the
                      incoming email? (An email with 3 questions answered by
                      a reply that only addresses 1 is not "accurate" no
                      matter how well-written that 1 answer is.)
  2. RELEVANCE      - is the reply's content/topic aligned with what a human
                      actually sent for a similar email? (semantic
                      similarity to the reference reply, not exact match)
  3. FAITHFULNESS   - does the reply avoid inventing facts (dates, numbers,
                      policies) not present in the incoming email or
                      supported by the retrieved examples? A confident,
                      well-written, hallucinated reply is worse than an
                      honest, vaguer one.
  4. TONE/STYLE FIT - does the reply match the company's established voice
                      (greeting, sign-off, register) seen in the dataset?
  5. HOLISTIC JUDGE - an LLM-as-judge score (1-5) with a rationale, acting as
                      a human reviewer would: catches things the above
                      structural checks miss (e.g. genuinely confusing
                      phrasing, wrong tone for the situation, an answer that
                      is technically on-topic but not actually useful).

We do NOT report a single opaque number as ground truth. We report all five,
plus a weighted composite, precisely so a reviewer can see *why* a response
scored the way it did, per the task's emphasis on explaining accuracy, not
just producing a score.

======================================================================
HOW EACH METRIC IS COMPUTED
======================================================================
- Coverage: for each key_point in the dataset record (hand-labeled at
  dataset-build time, see data/generate_dataset.py), ask the LLM judge
  (or, offline, use a lexical-overlap heuristic) whether the candidate reply
  addresses it. Score = fraction addressed.
- Relevance: TF-IDF cosine similarity between candidate reply and the
  reference (sent) reply. Cheap, deterministic, reproducible -- a good
  complement to the LLM judge, which is powerful but non-deterministic.
- Faithfulness: flags candidate replies that contain specific facts (numbers,
  dates, order IDs) not present anywhere in the incoming email or the
  retrieved few-shot examples used to ground generation -- a simple, auditable
  proxy for hallucination.
- Tone/style fit: checks for a greeting and sign-off consistent with the
  dataset's conventions (structural, not stylistic-nitpicking).
- LLM judge: holistic 1-5 score with a short rationale, from a rubric prompt.

======================================================================
HOW WE VALIDATE THE METRIC REFLECTS REAL QUALITY (not just a number)
======================================================================
See validate_metric.py: we run this same composite scorer on the
hand-labeled data/human_eval_calibration.jsonl set (13 (email, reply)
pairs independently rated 1-5 by a human, with a written rationale for each
score) and report the Spearman/Pearson correlation between our composite
score and the human score. A metric that doesn't correlate with human
judgment on that calibration set is not trustworthy, no matter how
principled it looks on paper -- so we treat that correlation, not face
validity, as the actual bar the metric must clear.
"""
import json
import re
import argparse
from pathlib import Path
from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import LLMClient

WEIGHTS = {
    "coverage": 0.35,
    "relevance": 0.20,
    "faithfulness": 0.20,
    "tone_fit": 0.10,
    "llm_judge": 0.15,
}
# Coverage and faithfulness weighted heaviest: an unfaithful or incomplete
# reply is a bigger real-world problem than a merely awkwardly-worded one.


def relevance_score(candidate: str, reference: str) -> float:
    """TF-IDF cosine similarity between candidate and reference reply, 0-1."""
    try:
        vec = TfidfVectorizer(stop_words="english").fit([candidate, reference])
        m = vec.transform([candidate, reference])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


NUM_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def faithfulness_score(candidate: str, incoming_email: str, grounding_texts: list) -> tuple:
    """
    Extracts numeric/date-like tokens from the candidate reply and checks
    whether each also appears somewhere in the incoming email or in the
    retrieved grounding examples. Returns (score in [0,1], list of
    unsupported tokens found) so a reviewer can see exactly what's flagged.
    This is a precision-oriented proxy for hallucination, not a proof of
    factual correctness -- it catches invented numbers, not invented prose.
    """
    source_text = " ".join([incoming_email] + grounding_texts)
    source_tokens = set(NUM_RE.findall(source_text))
    candidate_tokens = set(NUM_RE.findall(candidate))
    if not candidate_tokens:
        return 1.0, []
    unsupported = [t for t in candidate_tokens if t not in source_tokens]
    score = 1.0 - (len(unsupported) / len(candidate_tokens))
    return score, unsupported


def tone_fit_score(candidate: str) -> float:
    """Structural check: greeting present + sign-off present -> 1.0, else partial credit."""
    has_greeting = bool(re.match(r"^\s*(hi|hello|hey|dear)\b", candidate.strip(), re.I))
    has_signoff = bool(re.search(r"\n\s*(best|thanks|regards|warm regards|kind regards|sincerely)\b.*\n?", candidate, re.I))
    return 0.5 * has_greeting + 0.5 * has_signoff


JUDGE_SYSTEM = (
    "You are an impartial reviewer scoring a suggested email reply for quality, the way a "
    "careful support-team lead would when auditing an AI-drafted response before it's sent."
)

JUDGE_PROMPT_TMPL = """INCOMING EMAIL:
{incoming_email}

CANDIDATE REPLY:
{candidate_reply}

KEY POINTS THE REPLY SHOULD ADDRESS:
{key_points}

Score the candidate reply from 1 (bad: ignores the email, wrong/harmful, or unusable) to
5 (excellent: fully addresses every key point, correct tone, ready to send as-is).
Consider completeness (all key points addressed), correctness/faithfulness (no invented
facts), tone, and clarity.

Respond ONLY with JSON, no other text, in this exact shape:
{{"score": <1-5 integer>, "addressed_all_points": <true/false>, "rationale": "<one sentence>"}}
"""


def llm_judge_score(client: LLMClient, incoming_email: str, candidate_reply: str, key_points: list) -> dict:
    prompt = JUDGE_PROMPT_TMPL.format(
        incoming_email=incoming_email,
        candidate_reply=candidate_reply,
        key_points="\n".join(f"- {kp}" for kp in key_points) if key_points else "(none specified)",
    )
    raw = client.complete(JUDGE_SYSTEM, prompt)
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        data = json.loads(cleaned)
        data["score"] = max(1, min(5, int(data.get("score", 3))))
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 3, "addressed_all_points": None, "rationale": "[judge output unparsable, defaulted to neutral score]"}


def coverage_score(client: LLMClient, incoming_email: str, candidate_reply: str, key_points: list, judge_result: dict) -> float:
    """
    Reuses the LLM judge's addressed_all_points signal when available and
    unambiguous; otherwise falls back to a lexical-overlap heuristic per key
    point so the pipeline still works offline / when the judge is uncertain.
    """
    if not key_points:
        return 1.0
    if judge_result.get("addressed_all_points") is True:
        return 1.0
    if judge_result.get("addressed_all_points") is False:
        # partial credit via lexical overlap rather than an all-or-nothing 0
        pass
    hits = 0
    reply_lower = candidate_reply.lower()
    for kp in key_points:
        kp_words = [w for w in re.findall(r"[a-z]{4,}", kp.lower())]
        if any(w in reply_lower for w in kp_words):
            hits += 1
    return hits / len(key_points)


def evaluate_one(record: dict, client: LLMClient) -> dict:
    """
    record needs: incoming_email, suggested_reply, reference_reply (sent_reply),
    key_points, and optionally retrieved_example_texts for faithfulness grounding.
    """
    candidate = record["suggested_reply"]
    reference = record.get("reference_reply", "")
    incoming = record["incoming_email"]
    key_points = record.get("key_points", [])
    grounding_texts = record.get("retrieved_example_texts", [reference] if reference else [])

    judge = llm_judge_score(client, incoming, candidate, key_points)
    cov = coverage_score(client, incoming, candidate, key_points, judge)
    rel = relevance_score(candidate, reference) if reference else None
    faith, unsupported = faithfulness_score(candidate, incoming, grounding_texts)
    tone = tone_fit_score(candidate)

    scored = {
        "coverage": round(cov, 3),
        "relevance": round(rel, 3) if rel is not None else None,
        "faithfulness": round(faith, 3),
        "tone_fit": round(tone, 3),
        "llm_judge": round(judge["score"] / 5.0, 3),
    }
    weights_used = {k: v for k, v in WEIGHTS.items() if scored[k] is not None}
    norm = sum(weights_used.values())
    composite = sum(scored[k] * w for k, w in weights_used.items()) / norm

    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "scores": scored,
        "composite_score_0_1": round(composite, 3),
        "composite_score_1_5": round(1 + composite * 4, 2),
        "llm_judge_rationale": judge.get("rationale"),
        "unsupported_facts_flagged": unsupported,
        "used_real_llm_judge": client.using_real_llm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=str,
                     default=str(Path(__file__).parent.parent / "outputs" / "generations.jsonl"))
    ap.add_argument("--out", type=str,
                     default=str(Path(__file__).parent.parent / "outputs" / "evaluation_report.json"))
    args = ap.parse_args()

    client = LLMClient()
    records = [json.loads(l) for l in open(args.generations)]
    per_response = []
    for r in records:
        r["reference_reply"] = r.get("reference_reply", "")
        per_response.append(evaluate_one(r, client))

    def agg(key):
        vals = [r["scores"][key] for r in per_response if r["scores"][key] is not None]
        return round(mean(vals), 3) if vals else None

    overall = {
        "n_responses": len(per_response),
        "mean_composite_score_0_1": round(mean(r["composite_score_0_1"] for r in per_response), 3),
        "mean_composite_score_1_5": round(mean(r["composite_score_1_5"] for r in per_response), 3),
        "mean_by_metric": {k: agg(k) for k in WEIGHTS},
        "mean_composite_by_category": {},
        "used_real_llm_judge": client.using_real_llm,
    }
    cats = sorted(set(r["category"] for r in per_response if r["category"]))
    for c in cats:
        vals = [r["composite_score_0_1"] for r in per_response if r["category"] == c]
        overall["mean_composite_by_category"][c] = round(mean(vals), 3)

    report = {"overall": overall, "per_response": per_response}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"\nFull per-response report written to {args.out}")


if __name__ == "__main__":
    main()
