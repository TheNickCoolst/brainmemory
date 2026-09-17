"""Spike-timing-dependent plasticity.

pre before post (dt = t_post - t_pre > 0) → LTP
post before pre (dt < 0) → LTD

Exponential windows, configurable amplitudes. This is the classical
Bi & Poo / Song-Miller-Abbott kernel, not a detailed STDP cascade.
"""

from __future__ import annotations

import torch
from torch import Tensor

from brainmemory.config import STDPConfig
from brainmemory.synapses.synapse import SparseSynapses


def stdp_update(
    syn: SparseSynapses,
    last_spike: Tensor,
    cfg: STDPConfig,
    w_max: float = 1.0,
    w_min: float = 0.0,
    lr_scale: float = 1.0,
) -> dict[str, float]:
    if not cfg.enabled or syn.nnz == 0:
        return {"synapses_strengthened": 0, "synapses_weakened": 0, "mean_dw": 0.0}

    t_pre = last_spike[syn.pre]
    t_post = last_spike[syn.post]
    valid = (t_pre > -1.0e8) & (t_post > -1.0e8)
    dt = t_post - t_pre
    window = float(cfg.window)
    in_win = valid & (dt.abs() > 1e-6) & (dt.abs() <= window)

    dw = torch.zeros_like(syn.weight)
    ltp = in_win & (dt > 0)
    ltd = in_win & (dt < 0)
    if ltp.any():
        dw[ltp] = float(cfg.a_plus) * torch.exp(-dt[ltp] / max(float(cfg.tau_plus), 1e-6))
    if ltd.any():
        dw[ltd] = -float(cfg.a_minus) * torch.exp(dt[ltd] / max(float(cfg.tau_minus), 1e-6))

    dw.mul_(syn.plasticity * float(lr_scale))
    # multiplicative bound: LTP saturates near w_max, LTD near 0
    dw = torch.where(dw >= 0, dw * (w_max - syn.weight), dw * (syn.weight - w_min))
    syn.weight.add_(dw)
    syn.clip(w_min, w_max)
    return {
        "synapses_strengthened": int((dw > 1e-8).sum().item()),
        "synapses_weakened": int((dw < -1e-8).sum().item()),
        "mean_dw": float(dw.mean().item()) if dw.numel() else 0.0,
    }


def apply_pair_times(
    syn: SparseSynapses,
    pre_idx: int,
    post_idx: int,
    t_pre: float,
    t_post: float,
    n: int,
    cfg: STDPConfig,
    w_max: float = 1.0,
    w_min: float = 0.0,
) -> float:
    """Convenience for unit tests: set two spike times and apply STDP."""
    last = torch.full((n,), -1.0e9, device=syn.device, dtype=torch.float32)
    last[pre_idx] = float(t_pre)
    last[post_idx] = float(t_post)
    before = syn.weight.clone()
    stdp_update(syn, last, cfg, w_max=w_max, w_min=w_min)
    return float((syn.weight - before).sum().item())
