"""Autonomous forage → novelty gate → synaptic learning → sleep.

External pages are sensory input. They are not stored as the memory.
If synapses are deleted, what was foraged is gone.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from brainmemory.autonomy.curiosity import knowledge_gaps, seed_queue
from brainmemory.autonomy.senses import CompositeSense, Document, LocalFolderSense, WikipediaSense, chunk_text
from brainmemory.brain import Brain
from brainmemory.config import AutonomyConfig
from brainmemory.encoding.text_encoder import STOPWORDS


class AutonomousLearner:
    def __init__(
        self,
        brain: Brain,
        config: AutonomyConfig | None = None,
        senses: list | None = None,
    ):
        self.brain = brain
        self.config = config or brain.config.autonomy
        if not hasattr(brain, "seen_docs"):
            brain.seen_docs = set()
        if not hasattr(brain, "seen_queries"):
            brain.seen_queries = set()
        if senses is not None:
            self.sense = CompositeSense(senses) if len(senses) > 1 else senses[0]
        else:
            built = []
            if self.config.local_dir:
                built.append(LocalFolderSense(self.config.local_dir))
            if self.config.wikipedia:
                built.append(WikipediaSense(language=self.config.language))
            self.sense = CompositeSense(built) if len(built) > 1 else (built[0] if built else LocalFolderSense("."))

    def forage(self, topic: str) -> dict[str, Any]:
        """Fetch one topic and learn novel chunks."""
        docs = self.sense.search(topic, limit=max(1, self.config.follow_links))
        self.brain.seen_queries.add(topic.lower().strip())
        learned = []
        skipped = 0
        follow: list[str] = []
        for doc in docs:
            key = doc.title.lower().strip()
            if key in self.brain.seen_docs:
                skipped += 1
                continue
            self.brain.seen_docs.add(key)
            stats = self._ingest_document(doc)
            learned.extend(stats["episodes"])
            skipped += stats["skipped"]
            follow.extend(doc.related[: self.config.follow_links])
            if self.config.rate_limit_s:
                time.sleep(float(self.config.rate_limit_s))
        return {
            "topic": topic,
            "documents": [d.title for d in docs],
            "episodes_learned": len(learned),
            "skipped": skipped,
            "follow": follow,
            "episodes": learned,
        }

    def wander(self, seeds: Iterable[str], steps: int | None = None) -> dict[str, Any]:
        """Curiosity loop: fetch, learn, sleep, chase knowledge gaps."""
        n_steps = int(steps if steps is not None else self.config.max_pages)
        queue = seed_queue(seeds)
        log: list[dict[str, Any]] = []
        pages = 0
        learned_eps = 0
        sleeps: list[dict[str, Any]] = []
        while queue and pages < n_steps:
            topic = queue.popleft()
            if topic.lower().strip() in self.brain.seen_queries:
                continue
            result = self.forage(topic)
            pages += 1
            learned_eps += int(result["episodes_learned"])
            log.append(result)
            for nxt in result.get("follow") or []:
                if nxt.lower().strip() not in self.brain.seen_queries and nxt.lower().strip() not in self.brain.seen_docs:
                    queue.append(nxt)
            if pages % max(1, int(self.config.sleep_every)) == 0:
                sleeps.append(self.brain.sleep())
            for gap in knowledge_gaps(self.brain, sample=4):
                if gap.lower().strip() not in self.brain.seen_queries:
                    queue.append(gap)
        if learned_eps and (not sleeps or pages % max(1, int(self.config.sleep_every)) != 0):
            sleeps.append(self.brain.sleep())
        return {
            "steps": pages,
            "episodes_learned": learned_eps,
            "queue_remaining": list(queue)[:12],
            "known_concepts": len(self.brain.encoder.sdrs),
            "traces": len(self.brain.traces),
            "sleeps": sleeps,
            "log": log,
        }

    def ingest_path(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        if p.is_dir():
            sense = LocalFolderSense(p)
            docs = []
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in {".txt", ".md"}:
                    doc = sense.fetch(str(f))
                    if doc:
                        docs.append(doc)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs = [Document(title=p.stem, text=text, source=str(p))]
        episodes = []
        skipped = 0
        for doc in docs:
            stats = self._ingest_document(doc)
            episodes.extend(stats["episodes"])
            skipped += stats["skipped"]
            self.brain.seen_docs.add(doc.title.lower().strip())
        return {"documents": [d.title for d in docs], "episodes_learned": len(episodes), "skipped": skipped}

    def _ingest_document(self, doc: Document) -> dict[str, Any]:
        episodes = []
        skipped = 0
        chunks = chunk_text(doc.text)
        if not chunks:
            return {"episodes": [], "skipped": 1}
        for chunk in chunks:
            concepts = _cap_concepts(self.brain.text_encoder.extract(chunk), self.config.max_concepts)
            if len(concepts) < 2:
                skipped += 1
                continue
            content = self.brain.cortex.content(concepts)
            novelty = float(self.brain._novelty(content))
            if novelty < float(self.config.min_novelty):
                skipped += 1
                continue
            preview = self.brain.learn(concepts, importance=0.35 + 0.55 * novelty)
            episodes.append(
                {
                    "title": doc.title,
                    "concepts": concepts,
                    "novelty": novelty,
                    "engram_size": preview.get("engram_size", 0),
                    "synapses_grown": preview.get("synapses_grown", 0),
                }
            )
        return {"episodes": episodes, "skipped": skipped}


def _cap_concepts(concepts: list[str], max_n: int) -> list[str]:
    scored: list[tuple[int, str]] = []
    for c in concepts:
        if c in STOPWORDS:
            continue
        score = 0
        if " " not in c and len(c) >= 4:
            score += 2
        if len(c) >= 6:
            score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    out: list[str] = []
    seen: set[str] = set()
    for _, c in scored:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= int(max_n):
            break
    return out
