"""
generate_response.py

Given a new incoming email, retrieves the most similar past (email, reply)
pairs from the dataset and uses them as few-shot grounding for an LLM call
that drafts a suggested reply.

Usage:
    python3 generate_response.py --email "Hi, my order #123 hasn't shipped..."
    python3 generate_response.py --test-set   # generate for every held-out test email
"""
import argparse
import json
from pathlib import Path

from retrieval import TfidfRetriever
from llm_client import LLMClient

DATA_PATH = Path(__file__).parent.parent / "data" / "emails_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are an assistant that drafts suggested email replies for a company's support/sales "
    "inbox. You will be shown 2-3 examples of past incoming emails and the replies that were "
    "actually sent for them (the company's real tone, policies, and level of detail). Study "
    "those examples, then draft a reply to the NEW incoming email in the same voice and level "
    "of specificity. Directly address every question or request in the new email. Do not "
    "invent specific facts (order numbers, dates, prices) that are not given to you or "
    "reasonably implied by the examples' style -- if a concrete fact is unknown, use a "
    "placeholder like [ship date] rather than fabricating one. Keep it concise, address the "
    "sender by first name if given, and match the sign-off style used in the examples. "
    "Output only the reply text, nothing else."
)


def build_prompt(new_email: str, examples: list) -> str:
    parts = []
    for i, (rec, score) in enumerate(examples, 1):
        parts.append(
            f"EXAMPLE INCOMING EMAIL {i}:\n{rec['incoming_email']}\n\n"
            f"EXAMPLE REPLY {i}:\n{rec['sent_reply']}\n"
        )
    examples_block = "\n".join(parts)
    return f"{examples_block}\nNEW INCOMING EMAIL:\n\n{new_email}\n\nDraft the suggested reply now."


def generate_reply(new_email: str, retriever: TfidfRetriever, client: LLMClient, k: int = 3):
    examples = retriever.retrieve(new_email, k=k)
    prompt = build_prompt(new_email, examples)
    reply = client.complete(SYSTEM_PROMPT, prompt)
    return {
        "incoming_email": new_email,
        "suggested_reply": reply,
        "retrieved_example_ids": [rec["id"] for rec, _ in examples],
        "retrieved_scores": [round(s, 3) for _, s in examples],
        "retrieved_example_texts": [rec["incoming_email"] + " " + rec["sent_reply"] for rec, _ in examples],
        "used_real_llm": client.using_real_llm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", type=str, help="Raw text of a new incoming email")
    ap.add_argument("--test-set", action="store_true", help="Generate replies for the held-out test split")
    ap.add_argument("--k", type=int, default=3, help="Number of retrieved examples to use")
    ap.add_argument("--out", type=str, default=None, help="Where to write JSONL output (test-set mode)")
    args = ap.parse_args()

    retriever = TfidfRetriever(str(DATA_PATH), split="corpus")
    client = LLMClient()
    if not client.using_real_llm:
        print("[generate_response] No ANTHROPIC_API_KEY found -> using offline mock LLM fallback. "
              "Set the env var and `pip install anthropic` for real generations.\n")

    if args.email:
        result = generate_reply(args.email, retriever, client, k=args.k)
        print(json.dumps(result, indent=2))
        return

    if args.test_set:
        test_records = [json.loads(l) for l in open(DATA_PATH) if json.loads(l)["split"] == "test"]
        out_path = args.out or str(Path(__file__).parent.parent / "outputs" / "generations.jsonl")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for rec in test_records:
                result = generate_reply(rec["incoming_email"], retriever, client, k=args.k)
                result["id"] = rec["id"]
                result["category"] = rec["category"]
                result["reference_reply"] = rec["sent_reply"]
                result["key_points"] = rec["key_points"]
                f.write(json.dumps(result) + "\n")
        print(f"Wrote {len(test_records)} generations to {out_path}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
