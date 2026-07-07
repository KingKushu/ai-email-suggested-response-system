"""
llm_client.py

Thin wrapper around the generative model used for (a) drafting the suggested
reply and (b) acting as an LLM judge in the evaluation system.

Design:
- If ANTHROPIC_API_KEY is set in the environment and the `anthropic` package
  is installed, we call the real Claude API (model configurable, defaults to
  a fast Claude model). This is the "real" path described in the README.
- Otherwise we fall back to a small deterministic local "mock LLM" so that
  the whole pipeline (dataset -> generation -> evaluation -> report) can be
  run end-to-end with zero setup and zero cost, e.g. by a grader who hasn't
  configured an API key. The mock is clearly labeled as a fallback, not a
  claim that it is competitive with a real LLM: it does light template
  completion grounded in the retrieved examples, only good enough to exercise
  the rest of the system (retrieval, prompting, evaluation, reporting).

This separation means swapping in a real key is a one-line env var change,
no code changes needed anywhere else in the pipeline.
"""
import os
import re
import json


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 600):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self.api_key:
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("[llm_client] `anthropic` package not installed; falling back to mock LLM. "
                      "Run `pip install anthropic` to use the real API.")
                self._client = None

    @property
    def using_real_llm(self) -> bool:
        return self._client is not None

    def complete(self, system: str, prompt: str) -> str:
        if self._client is not None:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return self._mock_complete(system, prompt)

    # ---------------- Offline fallback ----------------
    def _mock_complete(self, system: str, prompt: str) -> str:
        """
        Deterministic stand-in used only when no API key is configured.
        For the *generation* task, it builds a reply out of the single most
        similar retrieved example, lightly substituting the new email's
        name/subject so the pipeline is fully runnable offline.
        For the *judge* task (prompt contains 'Respond ONLY with JSON'), it
        returns a heuristic JSON score so evaluate.py's plumbing can be
        exercised without a live model.
        """
        if "Respond ONLY with JSON" in prompt or "respond only with json" in prompt.lower():
            return self._mock_judge(prompt)
        return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        m = re.search(r"EXAMPLE REPLY 1:\n(.*?)(?:\nEXAMPLE INCOMING EMAIL \d+:|\nNEW INCOMING EMAIL:)", prompt, re.S)
        base_reply = m.group(1).strip() if m else "Thanks for your email, we'll take a look and get back to you shortly."
        name_m = re.search(r"NEW INCOMING EMAIL:\n\n(.*?)\n\nDraft the suggested reply now\.", prompt, re.S)
        incoming = name_m.group(1) if name_m else ""
        sender_m = re.search(r"\n([A-Z][a-z]+ [A-Z][a-z]+)\s*$", incoming.strip())
        sender = sender_m.group(1).split()[0] if sender_m else "there"
        base_reply = re.sub(r"^Hi \w+,", f"Hi {sender},", base_reply)
        return base_reply

    def _mock_judge(self, prompt: str) -> str:
        # Extremely lightweight heuristic so the offline demo still produces
        # *some* signal (not random): reward overlap of key content words
        # between the "candidate reply" and "incoming email" sections found
        # in the prompt. Real scoring, when a key is configured, is done by
        # the actual LLM per the rubric in evaluate.py.
        email_m = re.search(r"INCOMING EMAIL:\n(.*?)\n\n", prompt, re.S)
        reply_m = re.search(r"CANDIDATE REPLY:\n(.*?)\n\n", prompt, re.S)
        email = (email_m.group(1) if email_m else "").lower()
        reply = (reply_m.group(1) if reply_m else "").lower()
        email_words = set(re.findall(r"[a-z]{4,}", email))
        reply_words = set(re.findall(r"[a-z]{4,}", reply))
        overlap = len(email_words & reply_words) / max(1, len(email_words))
        score = max(1, min(5, round(1 + overlap * 5)))
        return json.dumps({
            "score": score,
            "addressed_all_points": overlap > 0.25,
            "rationale": "[offline mock judge - heuristic word overlap only; configure ANTHROPIC_API_KEY for real judgments]"
        })
