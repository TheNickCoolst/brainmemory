"""Structure-of-arrays neuron state.

Neurons are stored as tensors, not Python objects, so 10k–1M units remain
practical. `neuron_view` exposes the per-cell fields required by the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


REGION_HIPPOCAMPUS = 0
REGION_CORTEX = 1
REGION_ASSOCIATION = 2

REGION_NAMES = {
    REGION_HIPPOCAMPUS: "hippocampus",
    REGION_CORTEX: "cortex",
    REGION_ASSOCIATION: "association",
}


@dataclass
class NeuronState:
    v: Tensor
    threshold: Tensor
    activation: Tensor
    refractory: Tensor
    excitability: Tensor
    last_spike_time: Tensor
    activity_ema: Tensor
    region: Tensor
    is_inhibitory: Tensor

    @property
    def n(self) -> int:
        return int(self.v.numel())

    def reset_activity(self) -> None:
        self.v.zero_()
        self.activation.zero_()
        self.refractory.zero_()


class NeuronModel(Protocol):
    name: str

    def step(self, state: NeuronState, current: Tensor, t: float) -> Tensor:
        """Advance one timestep. Returns binary spike vector."""


def neuron_view(state: NeuronState, idx: int) -> dict[str, float | int | str | bool]:
    """Diagnostic snapshot of a single neuron (not used as the runtime model)."""
    i = int(idx)
    region = int(state.region[i].item())
    return {
        "id": i,
        "membrane_potential": float(state.v[i].item()),
        "threshold": float(state.threshold[i].item()),
        "activation": float(state.activation[i].item()),
        "refractory_state": int(state.refractory[i].item()),
        "excitability": float(state.excitability[i].item()),
        "last_spike_time": float(state.last_spike_time[i].item()),
        "activity_history": float(state.activity_ema[i].item()),
        "region": REGION_NAMES.get(region, str(region)),
        "inhibitory": bool(state.is_inhibitory[i].item()),
    }


def allocate_state(
    n: int,
    region: Tensor,
    is_inhibitory: Tensor,
    threshold: float,
    device: torch.device,
) -> NeuronState:
    z = torch.zeros(n, device=device, dtype=torch.float32)
    return NeuronState(
        v=z.clone(),
        threshold=torch.full((n,), float(threshold), device=device, dtype=torch.float32),
        activation=z.clone(),
        refractory=torch.zeros(n, device=device, dtype=torch.int16),
        excitability=torch.ones(n, device=device, dtype=torch.float32),
        last_spike_time=torch.full((n,), -1.0e9, device=device, dtype=torch.float32),
        activity_ema=z.clone(),
        region=region.to(device=device, dtype=torch.uint8),
        is_inhibitory=is_inhibitory.to(device=device, dtype=torch.bool),
    )
