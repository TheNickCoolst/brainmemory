from __future__ import annotations

import torch

from brainmemory.config import STDPConfig
from brainmemory.synapses.stdp import stdp_update
from brainmemory.synapses.synapse import SparseSynapses


def _syn(w: float = 0.4):
    syn = SparseSynapses.empty(2, torch.device("cpu"))
    syn.add(torch.tensor([0]), torch.tensor([1]), weight=torch.tensor([w]))
    return syn


def test_stdp_pre_before_post_potentiates():
    syn = _syn()
    cfg = STDPConfig(a_plus=0.08, a_minus=0.08, tau_plus=20.0, tau_minus=20.0, window=40.0)
    last = torch.tensor([10.0, 15.0])  # pre=10, post=15
    before = float(syn.weight[0])
    stdp_update(syn, last, cfg)
    assert float(syn.weight[0]) > before


def test_stdp_post_before_pre_depresses():
    syn = _syn()
    cfg = STDPConfig(a_plus=0.08, a_minus=0.08, tau_plus=20.0, tau_minus=20.0, window=40.0)
    last = torch.tensor([15.0, 10.0])  # pre after post
    before = float(syn.weight[0])
    stdp_update(syn, last, cfg)
    assert float(syn.weight[0]) < before


def test_stdp_disabled_is_noop():
    syn = _syn()
    cfg = STDPConfig(enabled=False)
    last = torch.tensor([10.0, 15.0])
    before = float(syn.weight[0])
    stdp_update(syn, last, cfg)
    assert float(syn.weight[0]) == before
