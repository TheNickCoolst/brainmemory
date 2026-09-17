"""Synaptic decay, unused-trace fade, and pruning.

Forgetting is a network process: unused weights shrink, then drop out of the
sparse graph. We do not delete named memory records on a timer.
"""

from __future__ import annotations

from brainmemory.config import BrainConfig
from brainmemory.network import SparseNetwork


def forget_step(net: SparseNetwork, cfg: BrainConfig, steps: int = 1) -> dict[str, float]:
    before = net.syn.nnz
    mean_before = net.mean_weight()
    for _ in range(max(1, int(steps))):
        net.decay(
            cfg.forgetting.decay_rate,
            cfg.forgetting.unused_decay,
            cfg.forgetting.unused_window,
        )
        net.t += 1.0
    pruned = net.prune()
    return {
        "mean_weight_before": mean_before,
        "mean_weight_after": net.mean_weight(),
        "synapses_pruned": float(pruned),
        "synapse_count": float(net.syn.nnz),
        "synapses_before": float(before),
    }
