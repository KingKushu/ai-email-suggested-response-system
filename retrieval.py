"""
retrieval.py

Retrieves the most similar past (email, reply) pairs from the corpus split of
our dataset, to ground the generator via few-shot / RAG-style prompting.

Why TF-IDF retrieval instead of e.g. fine-tuning:
- Fine-tuning needs a much larger dataset than we can honestly hand-author,
  is slow to iterate on, and "bakes in" the past replies in a way that's hard
  to audit or update (new policy? re-train). It's the right call at scale
  with thousands of labeled examples; not here.
- Pure zero-shot prompting (no retrieval) ignores the dataset entirely, which
  the task explicitly asks us to ground generation in.
- Retrieval + few-shot prompting is cheap, fully inspectable (you can see
  exactly which past emails informed a given suggestion), updates instantly
  as the dataset grows, and lets the *same* underlying LLM adapt its
  reply style/policy to whatever examples are retrieved -- a good trade-off
  for a small/medium support-inbox dataset like this one.
- We use TF-IDF rather than neural embeddings so the whole system runs with
  no external model download / API call for the retrieval step itself
  (only the generation and judging steps need the LLM). Swapping in a
  sentence-embedding model later is a one-function change (see
  `TfidfRetriever` vs a hypothetical `EmbeddingRetriever`).
"""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, dataset_path: str, split: str = "corpus"):
        self.records = []
        with open(dataset_path) as f:
            for line in f:
                r = json.loads(line)
                if split is None or r["split"] == split:
                    self.records.append(r)
        corpus_texts = [r["incoming_email"] for r in self.records]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query_email: str, k: int = 3):
        q_vec = self.vectorizer.transform([query_email])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [(self.records[i], float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    r = TfidfRetriever(str(Path(__file__).parent.parent / "data" / "emails_dataset.jsonl"))
    test_email = "Hi, my order #12345 still shows processing, when will it ship?"
    for rec, score in r.retrieve(test_email, k=3):
        print(f"{score:.3f}  [{rec['category']}]  {rec['subject']}")
