"""Offline replay of hippocampal ensembles."""

from __future__ import annotations

import numpy as np
import torch

from brainmemory.config import BrainConfig
from brainmemory.memory.engram import TraceBook
from brainmemory.network import SparseNetwork


def replay_traces(
    net: SparseNetwork,
    traces: TraceBook,
    cfg: BrainConfig,
    count: int | None = None,
) -> dict[str, float]:
    selected = traces.sample_for_replay(net.rng, int(count or cfg.sleep.replay_count))
    strengthened = 0
    for tr in selected:
        ids = tr.neuron_ids
        if ids.size == 0:
            continue
        active = torch.zeros(net.n, device=net.device, dtype=torch.float32)
        tidx = torch.from_numpy(ids.astype(np.int64)).to(net.device)
        tidx = tidx[(tidx >= 0) & (tidx < net.n)]
        active[tidx] = 1.0
        stats = net.hebbian(
            active,
            hippo_lr=cfg.hippocampus.learning_rate * 0.25,
            cortex_lr=cfg.cortex.learning_rate * cfg.sleep.cortex_lr_boost,
            assoc_lr=cfg.association.learning_rate,
            salience=tr.salience,
            depress=0.008,
        )
        strengthened += stats["synapses_strengthened"]
        tr.replay_count += 1
        tr.last_replay = net.t
        net.t += 1.0
    return {"memories_replayed": float(len(selected)), "synapses_strengthened": float(strengthened)}


def link_overlapping_traces(net: SparseNetwork, traces: TraceBook, max_pairs: int = 12) -> int:
    """Memories that share neurons become more associated (sleep linking)."""
    pairs = traces.overlapping(min_jaccard=0.02)
    if not pairs:
        return 0
    pairs = sorted(pairs, key=lambda p: p[2], reverse=True)[:max_pairs]
    new_assoc = 0
    for a, b, jac in pairs:
        union = np.unique(np.concatenate([a.neuron_ids, b.neuron_ids]))
        grown = net.grow_random_pairs(union, probability=0.08 + 0.25 * jac, weight=0.16, t=net.t)
        sa = set(int(x) for x in a.neuron_ids.tolist())
        sb = set(int(x) for x in b.neuron_ids.tolist())
        only_a = np.array(list(sa - sb), dtype=np.int64)
        only_b = np.array(list(sb - sa), dtype=np.int64)
        if only_a.size and only_b.size:
            deg = int(min(14, only_a.size, only_b.size))
            pre_ab, post_ab, pre_ba, post_ba = [], [], [], []
            for tgt in only_b:
                srcs = net.rng.choice(only_a, size=deg, replace=False)
                pre_ab.append(srcs.astype(np.int64))
                post_ab.append(np.full(deg, int(tgt), dtype=np.int64))
            for tgt in only_a:
                srcs = net.rng.choice(only_b, size=deg, replace=False)
                pre_ba.append(srcs.astype(np.int64))
                post_ba.append(np.full(deg, int(tgt), dtype=np.int64))
            pre = torch.from_numpy(np.concatenate(pre_ab + pre_ba))
            post = torch.from_numpy(np.concatenate(post_ab + post_ba))
            w = torch.full((pre.numel(),), 0.28, dtype=torch.float32)
            grown += net.syn.add(pre, post, weight=w, t=net.t)
            net.potentiate_engram(union, delta=0.08)
        active = torch.zeros(net.n, device=net.device, dtype=torch.float32)
        tidx = torch.from_numpy(union.astype(np.int64)).to(net.device)
        tidx = tidx[(tidx >= 0) & (tidx < net.n)]
        active[tidx] = 1.0
        net.hebbian(
            active,
            hippo_lr=0.02,
            cortex_lr=0.03,
            assoc_lr=0.08,
            salience=0.5 * (a.salience + b.salience),
            depress=0.0,
        )
        new_assoc += 1 if grown or union.size else 0
    return new_assoc
