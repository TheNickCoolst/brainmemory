"""Pattern completion / associative recall via network dynamics."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from brainmemory.config import BrainConfig
from brainmemory.encoding.concept_encoder import ConceptEncoder
from brainmemory.network import SparseNetwork


def recall_pattern(
    net: SparseNetwork,
    cue_neurons: np.ndarray,
    cfg: BrainConfig,
    use_hippocampus: bool = True,
) -> tuple[Tensor, dict]:
    spikes = net.run_clamped(
        clamp_idx=np.asarray(cue_neurons, dtype=np.int64),
        steps=cfg.recall.steps,
        clamp_steps=cfg.recall.clamp_steps,
        k_wta=cfg.recall.k_wta,
        clamp_current=cfg.recall.clamp_current,
        use_hippocampus=use_hippocampus,
    )
    active_idx = torch.nonzero(spikes > 0.5, as_tuple=False).view(-1)
    stats = {
        "engram_size": int(active_idx.numel()),
        "activation_strength": float(net.state.v[active_idx].mean().item()) if active_idx.numel() else 0.0,
        "sparsity": float(active_idx.numel()) / float(net.n),
    }
    return spikes, stats


def decode_concepts(
    encoder: ConceptEncoder,
    spikes: Tensor,
    threshold: float,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    active = torch.nonzero(spikes > 0.5, as_tuple=False).view(-1).detach().cpu().numpy()
    active_set = set(int(x) for x in active.tolist())
    scored: list[tuple[str, float]] = []
    exclude = exclude or set()
    for concept, sdr in encoder.sdrs.items():
        if concept in exclude:
            continue
        score = encoder.overlap_score(sdr, active_set)
        if score >= float(threshold):
            scored.append((concept, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
