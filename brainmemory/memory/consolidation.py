"""Hippocampus → cortex systems consolidation.

Replay hippocampal ensembles, drive cortical cells, and apply slow Hebbian
updates so repeated structure becomes cortical. Dependence on the original
hippocampal index is reduced as cortical recurrent weights grow.

Inspired by complementary learning systems (McClelland et al.), not a
literal NREM/REM simulator.
"""

from __future__ import annotations

import numpy as np
import torch

from brainmemory.config import BrainConfig
from brainmemory.memory.engram import TraceBook
from brainmemory.network import SparseNetwork


def consolidate_step(
    net: SparseNetwork,
    traces: TraceBook,
    cfg: BrainConfig,
    count: int | None = None,
) -> dict[str, float]:
    rng = net.rng
    n_replay = int(count if count is not None else cfg.sleep.replay_count)
    selected = traces.sample_for_replay(rng, n_replay)
    strengthened = 0
    weakened = 0
    for tr in selected:
        ids = tr.neuron_ids
        if ids.size == 0:
            continue
        active = torch.zeros(net.n, device=net.device, dtype=torch.float32)
        tidx = torch.from_numpy(ids.astype(np.int64)).to(net.device)
        tidx = tidx[(tidx >= 0) & (tidx < net.n)]
        if tidx.numel() == 0:
            continue
        active[tidx] = 1.0
        # extra cortical growth among the replayed ensemble
        net.grow_random_pairs(ids, probability=0.05, weight=0.12, t=net.t)
        lr_c = cfg.cortex.learning_rate * cfg.sleep.cortex_lr_boost
        stats = net.hebbian(
            active,
            hippo_lr=cfg.hippocampus.learning_rate * 0.15,
            cortex_lr=lr_c,
            assoc_lr=cfg.association.learning_rate * 2.0,
            salience=tr.salience,
            depress=0.01,
        )
        if cfg.stdp.enabled:
            # give replayed cells a near-synchronous spike time with tiny jitter
            times = torch.full((net.n,), -1.0e9, device=net.device)
            jitter = torch.rand(tidx.numel(), device=net.device) * 4.0
            times[tidx] = net.t + jitter
            net.state.last_spike_time = times
            stdp_stats = net.stdp(lr_scale=0.5 * tr.salience)
            strengthened += stdp_stats["synapses_strengthened"]
            weakened += stdp_stats["synapses_weakened"]
        strengthened += stats["synapses_strengthened"]
        weakened += stats["synapses_weakened"]
        tr.replay_count += 1
        tr.last_replay = net.t
        tr.consolidation = min(1.0, tr.consolidation + 0.08 * tr.salience)
        net.t += 1.0
    return {
        "memories_replayed": float(len(selected)),
        "synapses_strengthened": float(strengthened),
        "synapses_weakened": float(weakened),
    }
