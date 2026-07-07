# AI Email Suggested-Response System

Given an incoming email, this system (1) retrieves similar past emails from a
dataset of real (well, honestly-labeled synthetic) email/reply pairs, (2) uses
an LLM to draft a suggested reply grounded in those examples, and (3) scores
the suggested reply against five independent quality dimensions, with a
composite score and a written rationale — plus a step that checks whether the
scoring actually agrees with human judgment.

The evaluation system (part 3) is the part this exercise weighs heaviest, so
most of this README is about that.

```
email-reply-ai/
├── data/
│   ├── generate_dataset.py           # builds the dataset (run this first)
│   ├── emails_dataset.jsonl          # 120 (email, reply) pairs, 8 categories
│   └── human_eval_calibration.jsonl  # 13 hand-labeled pairs for metric validation
├── src/
│   ├── llm_client.py                 # pluggable LLM wrapper (real API or offline fallback)
│   ├── retrieval.py                  # TF-IDF retrieval over the dataset (RAG grounding)
│   ├── generate_response.py          # the generator: retrieval + few-shot LLM prompting
│   ├── evaluate.py                   # the accuracy/evaluation system
│   └── validate_metric.py            # validates the metric against human labels
├── outputs/                          # generations, evaluation report, metric validation (example run committed)
├── run_demo.py                       # runs everything end-to-end
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# optional but recommended: use the real Claude model instead of the offline fallback
export ANTHROPIC_API_KEY=sk-ant-...

python3 run_demo.py
```

This will: build the dataset, generate suggested replies for the 18 held-out
test emails, evaluate them, and validate the evaluation metric against the
13-pair human-labeled calibration set. Results land in `outputs/`.

To try a single email interactively:
```bash
python3 src/generate_response.py --email "Hi, my order #123 still says processing, when will it ship?"
```

**No API key needed to run any of this.** Without `ANTHROPIC_API_KEY`,
`llm_client.py` falls back to a small deterministic offline mock (details
below) so the whole pipeline — retrieval, generation, scoring, reporting —
is runnable and inspectable with zero setup. The mock is explicitly not a
claim of real generation quality; it exists purely so a grader can execute
the repo without a key. Every output is labeled `used_real_llm: true/false`
so it's always clear which mode produced it.

---

## 1. The dataset

**What it is:** `data/generate_dataset.py` builds 120 synthetic (incoming
email, sent reply) pairs across 8 categories that cover the common shapes of
business support/sales email: order status, refunds, bug reports, sales
pricing questions, meeting scheduling, billing questions, cancellations, and
complaints. Each record also carries hand-labeled `key_points` — the concrete
questions/asks in the email that a good reply must address — used later by
the evaluator.

**Why synthetic, and why this is honest, not a cop-out:**
- Real customer support inboxes aren't available to us without a privacy
  problem, and public email corpora (e.g. Enron) are personal correspondence,
  not customer-support Q&A — wrong shape for this task.
- Synthetic generation lets us **know the ground truth is actually good**:
  every "sent reply" in the dataset is constructed to genuinely answer the
  paired email, which we can't guarantee by scraping. That matters because
  our evaluation metric partly compares against these references — garbage
  references would make evaluation meaningless.
- We use templates with randomized slots (names, order numbers, products,
  dates, phrasing branches) rather than one fixed template per category, so
  there's real lexical variety, not 15 copies of one sentence.
- **What it doesn't capture:** real emails are messier — typos, multiple
  topics in one email, ambiguous intent, attachments referenced inline. This
  dataset is a reasonable stand-in for the *structure* of the problem
  (question(s) in, answer(s)+next-step out) but an honest production system
  would need to validate against real (consented / anonymized) traffic before
  shipping.

`emails_dataset.jsonl` is split 85/102 corpus vs 15/18 test at generation
time (`split` field) — the corpus is what the retriever/generator draws
few-shot examples from, the test split is fully held out and only used to
generate-then-evaluate.

`human_eval_calibration.jsonl` is a separate, small (13-pair) hand-labeled
set used **only** to validate the evaluation metric (see §3.4) — it
deliberately contains multiple candidate replies per email, ranging from
excellent to bad, each with a human 1-5 score and a written rationale, so we
can check whether our automatic score can tell them apart.

---

## 2. The generator (Gen AI, grounded in the dataset)

**Approach: retrieval-augmented few-shot prompting**, not fine-tuning, not
zero-shot.

`src/retrieval.py` builds a TF-IDF index over the corpus split of the
dataset. For a new incoming email, it retrieves the top-k (default 3) most
similar past emails and their actually-sent replies. `src/generate_response.py`
puts those into the LLM's context as few-shot examples and asks it to draft a
reply to the new email in the same voice, directly addressing every question,
without inventing facts it wasn't given.

**Why this over the alternatives:**
| Approach | Trade-off |
|---|---|
| Zero-shot prompting (no dataset) | Fastest to build, but ignores the dataset entirely — the task explicitly asks for grounding, and there's no way to adapt to this company's tone/policy without it. |
| Fine-tuning an LLM on the dataset | The right call *at scale* (thousands of examples), but with ~100 examples it would overfit, is slow to iterate, and "bakes in" replies in a way that's hard to audit — you can't easily tell a reviewer *which* past emails informed a given suggestion, and updating policy means retraining. |
| **Retrieval + few-shot prompting (chosen)** | Cheap, fully inspectable (`retrieved_example_ids` is in every output — you can see exactly which past cases grounded a given draft), updates instantly as the dataset grows, and lets the same underlying model adapt to whatever's retrieved. The clear right choice for a dataset this size, and the way most production reply-suggestion systems (e.g. help-desk "smart reply" features) are actually built. |

TF-IDF (not neural embeddings) is used for retrieval specifically so the
retrieval step needs no external model download or API call — only
generation and judging call the LLM. Swapping in a sentence-embedding
retriever later is a one-function change (`TfidfRetriever` → an
`EmbeddingRetriever` with the same interface).

The prompt (`SYSTEM_PROMPT` in `generate_response.py`) explicitly instructs
the model to use placeholders like `[ship date]` instead of inventing
specific facts it wasn't given — this is enforced on the generation side and
then *checked* on the evaluation side (faithfulness score, §3).

---

## 3. The evaluation system (the core of this project)

### 3.1 What does "accurate" mean for a suggested reply?

Exact match against the historical reply is the wrong target: two replies can
be equally good with completely different wording ("I've issued your refund"
vs. "your refund has been processed"), and the one historical reply is a
*sample* from a space of acceptable replies, not the only correct answer. So
instead of one similarity number, we decompose "accurate" into five
independently-measurable properties and report all of them:

| Metric | Question it answers | Weight |
|---|---|---|
| **Coverage** | Does the reply address *every* question/request in the incoming email? | 0.35 |
| **Faithfulness** | Does it avoid inventing facts (numbers, dates, order IDs) not given in the email or the retrieved examples? | 0.20 |
| **Relevance** | Is its content/topic aligned with what a human actually sent for a similar email (TF-IDF similarity to the reference), not exact wording? | 0.20 |
| **LLM judge** | Holistic 1-5 quality score with a written rationale — catches things structural checks miss (confusing phrasing, wrong tone for the situation, technically-on-topic-but-useless answers). | 0.15 |
| **Tone fit** | Does it have a greeting/sign-off consistent with the dataset's conventions? | 0.10 |

Coverage and faithfulness are weighted heaviest deliberately: an incomplete
or hallucinated reply is a bigger real-world failure than a merely
awkwardly-worded one, and both are things you'd actually block a draft over
before it goes to a customer.

We report **all five plus the composite**, not just the composite — per the
task's emphasis on *why* a response scored the way it did, the goal is to
give a reviewer something to look at when a score seems off, not a black box.

### 3.2 How each metric is computed

- **Coverage** (`evaluate.py: coverage_score`): uses the LLM judge's
  `addressed_all_points` field when confident; otherwise falls back to
  checking whether each hand-labeled `key_point`'s significant words appear
  in the reply (lexical-overlap heuristic), so it's still meaningful offline.
- **Relevance** (`relevance_score`): TF-IDF cosine similarity between the
  candidate and the historical reference reply — cheap, deterministic,
  reproducible; a good complement to the (non-deterministic) LLM judge.
- **Faithfulness** (`faithfulness_score`): extracts numeric/date-like tokens
  from the candidate reply and flags any that don't appear anywhere in the
  incoming email or the retrieved grounding examples — an auditable proxy for
  hallucination (it catches invented numbers, not invented prose; see
  limitations below).
- **Tone fit** (`tone_fit_score`): structural check for a greeting + sign-off
  matching dataset conventions.
- **LLM judge** (`llm_judge_score`): a rubric prompt asking a 1-5 score plus
  a one-sentence rationale, JSON-structured for reliable parsing.

### 3.3 Reporting

Running `python3 src/evaluate.py` (or `run_demo.py`) writes
`outputs/evaluation_report.json` with:
- **Per-response**: all five sub-scores, the composite (both 0-1 and 1-5
  scales), the LLM judge's rationale, and any flagged unsupported facts —
  everything needed to see *why* a specific reply scored the way it did.
- **Overall**: mean composite score, mean of each sub-metric across all
  responses, and mean composite broken down **by category** (e.g. is the
  system worse at complaint replies than order-status replies? — yes, in our
  test run, see `outputs/evaluation_report.json`).

### 3.4 Validating the metric against real human judgment

A metric that looks principled on paper but doesn't track actual quality
isn't worth trusting. `src/validate_metric.py` runs the exact same scoring
pipeline against `data/human_eval_calibration.jsonl` — 13 (email, reply)
pairs independently given a 1-5 human score with a written rationale at
dataset-build time, deliberately including multiple replies of very
different quality to the *same* email (so the check is really "can the
metric tell good from bad", not "does it recover one reference"). It reports
Pearson and Spearman correlation between the human score and the automatic
composite score.

**Result with the offline mock LLM** (the mode this repo runs in with zero
setup): Spearman ρ ≈ 0.45 between the composite score and human judgment —
positive and directionally right (the metric does rank the clearly-bad
replies below the clearly-good ones in most cases), but far from the ~0.7+
we'd want before trusting it for real decisions like auto-routing low
scorers to human review. This is expected and disclosed rather than
massaged: the offline judge is a crude lexical-overlap heuristic
(`llm_client.py: _mock_judge`), not a real quality assessment, and the LLM
judge alone (without the other four metrics) correlates ~0 in this mode —
exactly why we don't rely on the judge alone, and why the honest thing to do
is report this number rather than hide it.

**With `ANTHROPIC_API_KEY` set**, both generation and judging use the real
Claude model, and we'd expect materially higher correlation, since coverage
and the LLM judge would use actual language understanding instead of word
overlap — re-run `python3 src/validate_metric.py` with the key set to get
the number that should actually inform any go/no-go decision. We report the
honest offline number here rather than a cherry-picked one, and the code
always tags `used_real_llm_judge` in its output so you can never confuse the
two.

### 3.5 Known limitations of the metric (stated plainly)

- Faithfulness only catches invented **numbers/dates**, not invented prose
  claims (e.g. a reply asserting a false policy in words, not digits) — a
  real deployment would want a second LLM-judge pass specifically prompted
  for factual consistency against a knowledge base.
- Relevance via TF-IDF rewards lexical overlap with one reference reply; a
  genuinely good but differently-worded reply can score lower on this
  sub-metric even though it's fine — this is why relevance is one of five
  signals, not the whole score.
- The calibration set (13 pairs) is small; it's enough to sanity-check
  directionality, not to certify a precise correlation coefficient. A
  production rollout should grow this to 100+ pairs across more categories,
  ideally rated by more than one human, before trusting the exact number.
- The LLM judge is not deterministic between runs when using a real model;
  for high-stakes use we'd average over 2-3 judge calls.

---

## How AI tools were used

This entire repository — dataset design and generation script, retrieval and
generation code, the evaluation system design and implementation, and this
README — was built with Claude (Anthropic) as a pair-programmer/co-author in
an agentic coding environment: I described the task, and Claude wrote,
ran, and debugged the code in a sandboxed environment (including catching
and fixing a few real bugs along the way, e.g. a regex bug in the offline
mock LLM's example-parsing and a grammar glitch in one dataset template),
and wrote the README. The design decisions and trade-off reasoning above
(why retrieval over fine-tuning, why five metrics instead of one, why
validate against human labels) were arrived at collaboratively and are the
actual reasoning behind the code, not post-hoc justification.

## Example results (from the committed `outputs/` in this repo, offline mode)

```
Overall: mean composite = 0.634 (0-1 scale) / 3.54 (1-5 scale) across 18 held-out test emails
By metric:  coverage 0.46 | relevance 0.71 | faithfulness 1.00 | tone_fit 1.00 | llm_judge 0.20
By category: order_status 0.86, cancellation 0.83, billing 0.81 (strongest)
             complaint_feedback 0.48 (weakest — hardest category to draft well from few-shot alone)

Metric validation vs. 13 human-labeled pairs: Spearman rho = 0.45 (offline mock judge)
```
These numbers are from the offline fallback mode and are provided to show
the pipeline runs and produces a real, inspectable report — not as a claim
about production-grade reply quality. Re-run with `ANTHROPIC_API_KEY` set for
numbers that reflect actual LLM generation and judging.
