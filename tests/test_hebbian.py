from __future__ import annotations

import torch

from brainmemory.synapses.hebbian import hebbian_update
from brainmemory.synapses.synapse import SparseSynapses


def test_hebbian_strengthens_coactive_synapse():
    device = torch.device("cpu")
    syn = SparseSynapses.empty(4, device)
    syn.add(torch.tensor([0]), torch.tensor([1]), weight=torch.tensor([0.2]))
    act = torch.tensor([1.0, 1.0, 0.0, 0.0])
    before = float(syn.weight[0])
    hebbian_update(syn, act, act, lr=0.2, w_max=1.0, depress=0.0)
    assert float(syn.weight[0]) > before


def test_hebbian_weight_bounded():
    device = torch.device("cpu")
    syn = SparseSynapses.empty(2, device)
    syn.add(torch.tensor([0]), torch.tensor([1]), weight=torch.tensor([0.9]))
    act = torch.tensor([1.0, 1.0])
    for _ in range(50):
        hebbian_update(syn, act, act, lr=0.5, w_max=1.0, depress=0.0)
    assert float(syn.weight[0]) <= 1.0 + 1e-5
