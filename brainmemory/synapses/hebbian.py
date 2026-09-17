"""Rate-based Hebbian plasticity with soft bounds and synaptic scaling.

Δw = lr * pre * post * plasticity * (w_max - w)   (potentiation toward bound)
plus a small heterosynaptic depression term so unused partners weaken.
This is an approximation of Hebb's rule, not a biophysical calcium model.
"""

from __future__ import annotations

import torch
from torch import Tensor

from brainmemory.synapses.synapse import SparseSynapses


def hebbian_update(
    syn: SparseSynapses,
    pre_act: Tensor,
    post_act: Tensor,
    lr: Tensor | float,
    w_max: float = 1.0,
    w_min: float = 0.0,
    t: float = 0.0,
    depress: float = 0.02,
) -> dict[str, float]:
    pre_a = pre_act[syn.pre]
    post_a = post_act[syn.post]
    co = pre_a * post_a
    if not torch.is_tensor(lr):
        lr_vec = torch.full_like(syn.weight, float(lr))
    else:
        lr_vec = lr
    # LTP with soft upper bound (keeps weights from exploding)
    dw_plus = lr_vec * co * syn.plasticity * (w_max - syn.weight)
    # heterosynaptic LTD: pre fires, post silent
    dw_minus = depress * lr_vec * pre_a * (1.0 - post_a) * syn.weight * syn.plasticity
    dw = dw_plus - dw_minus
    syn.weight.add_(dw)
    syn.clip(w_min, w_max)
    syn.mark_used(pre_act, post_act, t)
    strengthened = int((dw > 1e-8).sum().item())
    weakened = int((dw < -1e-8).sum().item())
    return {
        "mean_dw": float(dw.mean().item()) if dw.numel() else 0.0,
        "synapses_strengthened": strengthened,
        "synapses_weakened": weakened,
    }


def incoming_scale(syn: SparseSynapses, target: float = 1.0, excitatory_only: bool = True) -> None:
    """Homeostatic scaling of incoming excitatory weights per postsynaptic cell."""
    if syn.nnz == 0:
        return
    mask = syn.sign > 0 if excitatory_only else torch.ones_like(syn.sign, dtype=torch.bool)
    if not mask.any():
        return
    sums = torch.zeros(syn.n, device=syn.device, dtype=torch.float32)
    sums.index_add_(0, syn.post[mask], syn.weight[mask])
    post_sum = sums[syn.post]
    scale = torch.ones_like(syn.weight)
    need = mask & (post_sum > target) & (post_sum > 1e-8)
    scale[need] = target / post_sum[need]
    syn.weight.mul_(scale)
