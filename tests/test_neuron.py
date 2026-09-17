from __future__ import annotations

import torch

from brainmemory.config import NeuronConfig
from brainmemory.neurons.lif import LIFNeuronModel
from brainmemory.neurons.neuron import allocate_state, neuron_view


def test_neuron_firing():
    cfg = NeuronConfig(tau_mem=10.0, v_threshold=1.0, v_reset=0.0, refractory_steps=2, dt=1.0)
    model = LIFNeuronModel(cfg)
    device = torch.device("cpu")
    state = allocate_state(
        1,
        torch.zeros(1, dtype=torch.uint8),
        torch.zeros(1, dtype=torch.bool),
        threshold=1.0,
        device=device,
    )
    spiked = False
    for t in range(40):
        spikes = model.step(state, torch.tensor([0.4]), float(t))
        if bool(spikes[0].item()):
            spiked = True
            break
    assert spiked, "LIF neuron should fire under sustained suprathreshold current"
    view = neuron_view(state, 0)
    assert view["id"] == 0
    assert view["last_spike_time"] >= 0
    assert view["refractory_state"] >= 0


def test_subthreshold_does_not_fire():
    cfg = NeuronConfig(tau_mem=20.0, v_threshold=1.0, dt=1.0, refractory_steps=0)
    model = LIFNeuronModel(cfg)
    state = allocate_state(
        1,
        torch.zeros(1, dtype=torch.uint8),
        torch.zeros(1, dtype=torch.bool),
        threshold=1.0,
        device=torch.device("cpu"),
    )
    for t in range(30):
        spikes = model.step(state, torch.tensor([0.01]), float(t))
        assert not bool(spikes[0].item())


def test_refractory_blocks_immediate_refire():
    cfg = NeuronConfig(tau_mem=5.0, v_threshold=0.5, refractory_steps=5, dt=1.0)
    model = LIFNeuronModel(cfg)
    state = allocate_state(
        1,
        torch.zeros(1, dtype=torch.uint8),
        torch.zeros(1, dtype=torch.bool),
        threshold=0.5,
        device=torch.device("cpu"),
    )
    spikes = model.step(state, torch.tensor([5.0]), 0.0)
    assert bool(spikes[0].item())
    spikes2 = model.step(state, torch.tensor([5.0]), 1.0)
    assert not bool(spikes2[0].item())
