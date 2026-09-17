"""Hippocampal subsystem (complementary learning systems, fast learner).

Approximations:
- Dentate-gyrus-like expansion via unique sparse indices (pattern separation)
- CA3-like recurrent binding via high learning rate + dense engram wiring
- Indexing theory: hippo stores a pointer ensemble, not a JSON episode

This is not a anatomical reconstruction of CA1/CA3/DG.
"""

from __future__ import annotations

import numpy as np

from brainmemory.config import BrainConfig
from brainmemory.encoding.concept_encoder import ConceptEncoder
from brainmemory.network import SparseNetwork


class Hippocampus:
    def __init__(self, net: SparseNetwork, encoder: ConceptEncoder, cfg: BrainConfig):
        self.net = net
        self.encoder = encoder
        self.cfg = cfg
        self.enabled = True

    @property
    def learning_rate(self) -> float:
        return float(self.cfg.hippocampus.learning_rate)

    def form_episode(self, concepts: list[str]) -> np.ndarray:
        return self.encoder.episode_index(concepts, unique=True)

    def neurons(self) -> np.ndarray:
        return self.net.hippo_exc()
