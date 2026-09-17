"""Artificial sleep: replay, consolidation, homeostasis, pruning, linking."""

from __future__ import annotations

from brainmemory.config import BrainConfig
from brainmemory.memory.consolidation import consolidate_step
from brainmemory.memory.engram import TraceBook
from brainmemory.memory.forgetting import forget_step
from brainmemory.network import SparseNetwork
from brainmemory.simulation.replay import link_overlapping_traces


def sleep_cycle(net: SparseNetwork, traces: TraceBook, cfg: BrainConfig) -> dict[str, float]:
    cons = consolidate_step(net, traces, cfg)
    new_assoc = link_overlapping_traces(net, traces)
    net.downscale(cfg.sleep.downscale)
    forget = forget_step(net, cfg, steps=1)
    return {
        "memories_replayed": int(cons["memories_replayed"]),
        "synapses_strengthened": int(cons["synapses_strengthened"]),
        "synapses_weakened": int(cons["synapses_weakened"]),
        "synapses_pruned": int(forget["synapses_pruned"]),
        "new_associations": int(new_assoc),
        "mean_weight": net.mean_weight(),
        "synapse_count": net.syn.nnz,
    }
