"""Curiosity: turn weak or sparse recall into the next forage queries.

This is an approximation of information-seeking, not a dopamine circuit.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from brainmemory.brain import Brain
from brainmemory.encoding.text_encoder import STOPWORDS


def knowledge_gaps(brain: Brain, sample: int = 10, min_associates: int = 2, min_confidence: float = 0.35) -> list[str]:
    concepts = [c for c in brain.encoder.known_concepts() if _query_like(c)]
    if not concepts:
        return []
    # prefer rarer / shorter content words
    concepts = sorted(concepts, key=lambda c: (c.count(" "), len(c)))[: max(sample * 3, sample)]
    gaps: list[str] = []
    for c in concepts:
        rec = brain.recall([c], top_k=8)
        n_assoc = len(rec.get("associated_concepts") or [])
        if rec["confidence"] < min_confidence or n_assoc < min_associates:
            gaps.append(c)
        if len(gaps) >= sample:
            break
    return gaps


def seed_queue(seeds: Iterable[str], extra: Iterable[str] | None = None) -> deque[str]:
    q: deque[str] = deque()
    seen: set[str] = set()
    for item in list(seeds) + list(extra or []):
        t = " ".join(str(item).strip().split())
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            q.append(t)
    return q


def _query_like(concept: str) -> bool:
    c = concept.strip().lower()
    if not c or c in STOPWORDS:
        return False
    if len(c) < 4:
        return False
    if c.count(" ") > 2:
        return False
    return True
