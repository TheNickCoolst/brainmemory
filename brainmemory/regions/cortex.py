"""Neocortical subsystem (slow semantic learner).

Overlapping concept SDRs live here. Repeated / replayed co-activation carves
shared structure (e.g. many apple episodes → an APPLE-like assembly).
Learning rate is lower than hippocampus. Not a layered neocortical column model.
"""

from __future__ import annotations

import numpy as np

from brainmemory.config import BrainConfig
from brainmemory.encoding.concept_encoder import ConceptEncoder
from brainmemory.network import SparseNetwork


class Cortex:
    def __init__(self, net: SparseNetwork, encoder: ConceptEncoder, cfg: BrainConfig):
        self.net = net
        self.encoder = encoder
        self.cfg = cfg
        self.enabled = True

    @property
    def learning_rate(self) -> float:
        return float(self.cfg.cortex.learning_rate)

    def content(self, concepts: list[str]) -> np.ndarray:
        return self.encoder.content_union(concepts)

    def neurons(self) -> np.ndarray:
        return self.net.cortex_exc()
