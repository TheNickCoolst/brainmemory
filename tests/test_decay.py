from __future__ import annotations

import torch

from brainmemory.synapses.synapse import SparseSynapses


def test_synaptic_decay_reduces_weight():
    syn = SparseSynapses.empty(2, torch.device("cpu"))
    syn.add(torch.tensor([0]), torch.tensor([1]), weight=torch.tensor([0.5]))
    before = float(syn.weight[0])
    syn.weight.mul_(1.0 - 0.1)
    assert float(syn.weight[0]) < before
    assert abs(float(syn.weight[0]) - 0.45) < 1e-5


def test_brain_tick_decays(tiny_brain):
    tiny_brain.learn(["alpha", "beta"], importance=0.3)
    before = tiny_brain.net.mean_weight()
    tiny_brain.tick(steps=25)
    after = tiny_brain.net.mean_weight()
    assert after < before
