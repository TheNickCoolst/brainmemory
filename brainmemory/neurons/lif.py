"""Current-based leaky integrate-and-fire neuron.

This is a discrete-time approximation, not a biophysically detailed cell:
    v <- v + dt * (-(v - v_rest)/tau + I * excitability)
    spike if v >= threshold and refractory == 0
"""

from __future__ import annotations

import torch
from torch import Tensor

from brainmemory.config import NeuronConfig
from brainmemory.neurons.neuron import NeuronState


class LIFNeuronModel:
    name = "lif"

    def __init__(self, cfg: NeuronConfig):
        self.tau = float(cfg.tau_mem)
        self.v_rest = float(cfg.v_rest)
        self.v_reset = float(cfg.v_reset)
        self.dt = float(cfg.dt)
        self.refrac = int(cfg.refractory_steps)

    def step(self, state: NeuronState, current: Tensor, t: float) -> Tensor:
        leak = (state.v - self.v_rest) / max(self.tau, 1e-6)
        state.v.add_(self.dt * (-leak + current * state.excitability))

        can_fire = state.refractory <= 0
        spikes = (state.v >= state.threshold) & can_fire

        if spikes.any():
            state.v.masked_fill_(spikes, self.v_reset)
            state.last_spike_time = torch.where(
                spikes, torch.full_like(state.last_spike_time, float(t)), state.last_spike_time
            )
            if self.refrac > 0:
                state.refractory = torch.where(
                    spikes,
                    torch.full_like(state.refractory, self.refrac),
                    torch.clamp(state.refractory - 1, min=0),
                )
            else:
                state.refractory.zero_()
        else:
            if self.refrac > 0:
                state.refractory = torch.clamp(state.refractory - 1, min=0)

        # cells still in refractory cannot accumulate unbounded voltage
        in_refrac = state.refractory > 0
        if in_refrac.any():
            state.v = torch.where(in_refrac & ~spikes, torch.minimum(state.v, state.threshold * 0.5), state.v)

        state.activation = spikes.to(state.activation.dtype)
        state.activity_ema.mul_(0.95).add_(state.activation * 0.05)
        return spikes
