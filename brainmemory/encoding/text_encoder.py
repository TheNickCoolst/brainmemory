"""Text → concept tokens.

An external LLM may be used ONLY to extract concepts. The default path is a
deterministic tokenizer so the system works without API keys. The encoder is
not the memory store.
"""

from __future__ import annotations

import os
import re
from typing import Callable

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "that",
    "this",
    "those",
    "these",
    "it",
    "its",
    "as",
    "into",
    "over",
    "after",
    "before",
    "about",
    "against",
    "between",
    "during",
    "without",
    "within",
    "often",
    "likes",
    "like",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "does",
    "do",
    "did",
    "has",
    "have",
    "had",
    "not",
    "no",
    "yes",
    "very",
    "too",
    "than",
    "who",
    "whom",
    "which",
    "what",
    "when",
    "where",
    "how",
    "there",
    "here",
    "their",
    "his",
    "her",
    "our",
    "your",
    "my",
    "me",
    "we",
    "they",
    "them",
    "he",
    "she",
    "you",
    "i",
}


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class TextEncoder:
    def __init__(self, extractor: Callable[[str], list[str]] | None = None):
        self.extractor = extractor

    def extract(self, text: str) -> list[str]:
        if self.extractor is not None:
            concepts = [c.strip().lower() for c in self.extractor(text) if str(c).strip()]
            if concepts:
                return _dedupe(concepts)
        if os.environ.get("BRAINMEMORY_LLM_EXTRACTOR") == "1":
            llm = _try_llm_extract(text)
            if llm:
                return _dedupe(llm)
        return extract_concepts_deterministic(text)


# Lightweight sensory/semantic features (not stored memories). These give the
# encoder extra overlapping bits the way a perceptual system might already
# know that a retriever is a dog. Associations among episodes still require
# synaptic change.
FEATURE_EXPANSION = {
    "retriever": ["dog"],
    "golden retriever": ["dog"],
    "apple": ["fruit"],
    "strawberry": ["fruit"],
    "banana": ["fruit"],
}


def extract_concepts_deterministic(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text.lower())
    tokens = [t for t in raw if t not in STOPWORDS and len(t) > 1]
    concepts: list[str] = []
    for i, tok in enumerate(tokens):
        concepts.append(tok)
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # adjective-ish / compound: keep bigrams of consecutive content words
            if len(tok) >= 3 and len(nxt) >= 3:
                concepts.append(f"{tok} {nxt}")
    expanded = list(concepts)
    for c in concepts:
        expanded.extend(FEATURE_EXPANSION.get(c, []))
    return _dedupe(expanded)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = " ".join(x.split())
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _try_llm_extract(text: str) -> list[str]:
    """Optional concept extractor. Never stores memories; returns tokens only."""
    # Intentionally conservative: only runs if an API key exists AND the env flag is set.
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("XAI_API_KEY")
    if not api_key:
        return []
    try:
        import json
        import urllib.request

        body = json.dumps(
            {
                "model": os.environ.get("BRAINMEMORY_LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract a JSON list of short memory concepts from the text. No commentary.",
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            os.environ.get("BRAINMEMORY_LLM_URL", "https://api.openai.com/v1/chat/completions"),
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        return []
    return []
