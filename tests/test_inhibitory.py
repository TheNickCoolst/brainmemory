from __future__ import annotations

import torch

from brainmemory.synapses.synapse import SparseSynapses


def test_inhibitory_current_is_negative():
    device = torch.device("cpu")
    syn = SparseSynapses.empty(3, device)
    syn.add(
        torch.tensor([0]),
        torch.tensor([1]),
        weight=torch.tensor([0.4]),
        sign=torch.tensor([-1.0]),
    )
    act = torch.tensor([1.0, 0.0, 0.0])
    I = syn.current(act)
    assert I[1].item() < 0
    assert I[0].item() == 0


def test_excitatory_current_is_positive():
    device = torch.device("cpu")
    syn = SparseSynapses.empty(3, device)
    syn.add(
        torch.tensor([0]),
        torch.tensor([1]),
        weight=torch.tensor([0.4]),
        sign=torch.tensor([1.0]),
    )
    I = syn.current(torch.tensor([1.0, 0.0, 0.0]))
    assert I[1].item() > 0
