"""Distributed memory traces (engrams).

A memory is a sparse ensemble, not a row in a table. Traces store neuron IDs
and neuromodulatory tags used for replay priority. They do NOT store the
concept strings that should be recalled — recall must come from network
activity plus the sensory codebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HippocampalTrace:
    neuron_ids: np.ndarray
    salience: float
    novelty: float
    reward: float
    created_at: float
    last_replay: float = -1.0
    last_retrieval: float = -1.0
    replay_count: int = 0
    retrieval_count: int = 0
    consolidation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "neuron_ids": self.neuron_ids.astype(np.int64),
            "salience": float(self.salience),
            "novelty": float(self.novelty),
            "reward": float(self.reward),
            "created_at": float(self.created_at),
            "last_replay": float(self.last_replay),
            "last_retrieval": float(self.last_retrieval),
            "replay_count": int(self.replay_count),
            "retrieval_count": int(self.retrieval_count),
            "consolidation": float(self.consolidation),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HippocampalTrace":
        return cls(
            neuron_ids=np.asarray(d["neuron_ids"], dtype=np.int64),
            salience=float(d["salience"]),
            novelty=float(d.get("novelty", 0.0)),
            reward=float(d.get("reward", 0.0)),
            created_at=float(d["created_at"]),
            last_replay=float(d.get("last_replay", -1.0)),
            last_retrieval=float(d.get("last_retrieval", -1.0)),
            replay_count=int(d.get("replay_count", 0)),
            retrieval_count=int(d.get("retrieval_count", 0)),
            consolidation=float(d.get("consolidation", 0.0)),
        )


class TraceBook:
    def __init__(self) -> None:
        self.traces: list[HippocampalTrace] = []

    def add(self, trace: HippocampalTrace) -> None:
        self.traces.append(trace)

    def __len__(self) -> int:
        return len(self.traces)

    def sample_for_replay(self, rng: np.random.RandomState, count: int) -> list[HippocampalTrace]:
        if not self.traces:
            return []
        weights = np.array([max(1e-6, t.salience) * (1.0 + 0.15 * t.retrieval_count) for t in self.traces])
        # less-consolidated traces replay more (systems consolidation)
        weights *= 1.25 - 0.6 * np.array([t.consolidation for t in self.traces])
        weights = weights / weights.sum()
        k = min(int(count), len(self.traces))
        idx = rng.choice(len(self.traces), size=k, replace=True, p=weights)
        return [self.traces[i] for i in idx]

    def overlapping(self, min_jaccard: float = 0.08) -> list[tuple[HippocampalTrace, HippocampalTrace, float]]:
        pairs = []
        sets = [set(int(x) for x in t.neuron_ids.tolist()) for t in self.traces]
        for i in range(len(self.traces)):
            for j in range(i + 1, len(self.traces)):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j]) or 1
                jac = inter / union
                if jac >= min_jaccard:
                    pairs.append((self.traces[i], self.traces[j], jac))
        return pairs

    def bump_retrieval(self, active: np.ndarray, t: float, min_overlap: float = 0.25) -> None:
        active_set = set(int(x) for x in np.flatnonzero(active > 0.5).tolist()) if active.dtype != np.int64 else set(
            int(x) for x in active.tolist()
        )
        if not active_set:
            return
        for tr in self.traces:
            ids = tr.neuron_ids
            if ids.size == 0:
                continue
            hit = sum(1 for i in ids.tolist() if int(i) in active_set) / float(ids.size)
            if hit >= min_overlap:
                tr.retrieval_count += 1
                tr.last_retrieval = float(t)
                tr.salience = min(1.0, tr.salience * 1.02 + 0.01)

    def state_dict(self) -> dict:
        return {"traces": [t.to_dict() for t in self.traces]}

    def load_state(self, data: dict) -> None:
        self.traces = [HippocampalTrace.from_dict(t) for t in data.get("traces", [])]


def engram_overlap(ids_a: np.ndarray, ids_b: np.ndarray) -> dict[str, float]:
    a = set(int(x) for x in np.asarray(ids_a).tolist())
    b = set(int(x) for x in np.asarray(ids_b).tolist())
    inter = a & b
    union = a | b
    return {
        "intersection": float(len(inter)),
        "union": float(len(union)),
        "jaccard": float(len(inter) / len(union)) if union else 0.0,
        "overlap_a": float(len(inter) / len(a)) if a else 0.0,
        "overlap_b": float(len(inter) / len(b)) if b else 0.0,
    }
