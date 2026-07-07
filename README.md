AI Email Suggested-Response System
An end-to-end AI system that generates context-aware email replies using Retrieval-Augmented Generation (RAG) and evaluates their quality using a multi-metric evaluation framework. The project includes dataset generation, response generation, automated evaluation, and metric validation against human judgments.
## Repository Structure

```text
email-reply-ai/
├── .gitignore
├── README.md
├── requirements.txt
├── run_demo.py
├── generate_dataset.py              # Builds the synthetic dataset
├── emails_dataset.jsonl             # 120 email–reply pairs
├── human_eval_calibration.jsonl     # Human-labeled calibration set
├── llm_client.py                    # LLM wrapper (API or offline fallback)
├── retrieval.py                     # TF-IDF retrieval for RAG
├── generate_response.py             # Response generator
├── evaluate.py                      # Evaluation framework
├── validate_metric.py               # Metric validation
├── generations.jsonl                # Example generated replies
├── evaluation_report.json           # Example evaluation report
└── metric_validation.json           # Example metric validation results
```

## Quickstart

Install the dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Set an Anthropic API key to use the real Claude model instead of the offline fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run the complete pipeline:

```bash
python run_demo.py
```

This will:

1. Build the synthetic email dataset.
2. Generate suggested replies for the held-out test emails.
3. Evaluate every generated reply using the multi-metric evaluation framework.
4. Validate the evaluation metric against the human-labeled calibration set.

The following files will be generated (or updated):

* `generations.jsonl`
* `evaluation_report.json`
* `metric_validation.json`

To generate a reply for a single email:

```bash
python generate_response.py --email "Hi, my order #123 still says processing, when will it ship?"
```

No API key is required to run the project. If `ANTHROPIC_API_KEY` is not provided, `llm_client.py` automatically uses a deterministic offline mock so the entire pipeline remains fully runnable without external services. All outputs indicate whether they were produced using the real LLM or the offline fallback.
