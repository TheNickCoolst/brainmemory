"""Deterministic sparse concept encoder.

Each concept is assigned a fixed sparse set of excitatory cortical neurons
(an SDR). This mapping is a sensory codebook, not a memory: deleting synapses
destroys recall even though the codebook remains.

Related memories share neurons because they share concept tokens, not because
we store a table of facts.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _stable_rng(seed: int, *parts: str) -> np.random.RandomState:
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for p in parts:
        h.update(b"\0")
        h.update(p.encode("utf-8"))
    material = int.from_bytes(h.digest()[:8], "little") % (2**32)
    return np.random.RandomState(material)


class ConceptEncoder:
    def __init__(
        self,
        n: int,
        cortex_pool: np.ndarray,
        hippo_pool: np.ndarray,
        assoc_pool: np.ndarray,
        bits_per_concept: int,
        index_size: int,
        seed: int,
    ):
        self.n = int(n)
        self.cortex_pool = np.asarray(cortex_pool, dtype=np.int64)
        self.hippo_pool = np.asarray(hippo_pool, dtype=np.int64)
        self.assoc_pool = np.asarray(assoc_pool, dtype=np.int64)
        self.bits = int(bits_per_concept)
        self.index_size = int(index_size)
        self.seed = int(seed)
        self.sdrs: dict[str, np.ndarray] = {}
        self._episode_counter = 0

    def canonicalize(self, concept: str) -> str:
        return " ".join(str(concept).strip().lower().split())

    def encode(self, concept: str) -> np.ndarray:
        key = self.canonicalize(concept)
        if not key:
            return np.zeros(0, dtype=np.int64)
        cached = self.sdrs.get(key)
        if cached is not None:
            return cached
        k = min(self.bits, max(4, self.cortex_pool.size // 8))
        rng = _stable_rng(self.seed, "concept", key)
        sdr = np.sort(rng.choice(self.cortex_pool, size=min(k, self.cortex_pool.size), replace=False))
        self.sdrs[key] = sdr
        return sdr

    def encode_many(self, concepts: Iterable[str]) -> dict[str, np.ndarray]:
        return {self.canonicalize(c): self.encode(c) for c in concepts if self.canonicalize(c)}

    def content_union(self, concepts: Iterable[str]) -> np.ndarray:
        parts = [self.encode(c) for c in concepts]
        parts = [p for p in parts if p.size]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))

    def episode_index(self, concepts: Iterable[str], unique: bool = True) -> np.ndarray:
        """Dentate-gyrus-like sparse index: mostly unique, weakly content-derived.

        Pattern separation: similar concept sets get largely orthogonal hippo codes
        because the full-set hash changes when any distinguishing token changes.
        """
        tokens = sorted({self.canonicalize(c) for c in concepts if self.canonicalize(c)})
        content_key = "|".join(tokens)
        k = min(self.index_size, max(4, self.hippo_pool.size // 4))
        k_unique = max(2, int(0.8 * k))
        k_shared = max(1, k - k_unique)
        self._episode_counter += 1
        uniq_salt = f"{content_key}#{self._episode_counter}" if unique else content_key
        rng_u = _stable_rng(self.seed, "hippo-unique", uniq_salt)
        rng_s = _stable_rng(self.seed, "hippo-content", content_key)
        u = rng_u.choice(self.hippo_pool, size=min(k_unique, self.hippo_pool.size), replace=False)
        s = rng_s.choice(self.hippo_pool, size=min(k_shared, self.hippo_pool.size), replace=False)
        return np.unique(np.concatenate([u, s]))

    def association_code(self, concepts: Iterable[str]) -> np.ndarray:
        tokens = sorted({self.canonicalize(c) for c in concepts if self.canonicalize(c)})
        if not tokens or self.assoc_pool.size == 0:
            return np.zeros(0, dtype=np.int64)
        rng = _stable_rng(self.seed, "assoc", "|".join(tokens))
        k = min(max(6, self.bits // 2), self.assoc_pool.size)
        return np.sort(rng.choice(self.assoc_pool, size=k, replace=False))

    def known_concepts(self) -> list[str]:
        return list(self.sdrs.keys())

    def overlap_score(self, sdr: np.ndarray, active: np.ndarray | set[int]) -> float:
        if sdr.size == 0:
            return 0.0
        if isinstance(active, np.ndarray):
            active_set = set(int(x) for x in active.tolist())
        else:
            active_set = active
        hit = sum(1 for i in sdr.tolist() if int(i) in active_set)
        return hit / float(sdr.size)

    def state_dict(self) -> dict:
        return {
            "sdrs": {k: v.astype(np.int64) for k, v in self.sdrs.items()},
            "episode_counter": self._episode_counter,
            "bits": self.bits,
            "index_size": self.index_size,
            "seed": self.seed,
        }

    def load_state(self, data: dict) -> None:
        self.sdrs = {k: np.asarray(v, dtype=np.int64) for k, v in data.get("sdrs", {}).items()}
        self._episode_counter = int(data.get("episode_counter", 0))
