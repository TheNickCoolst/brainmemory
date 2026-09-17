from __future__ import annotations

import torch

from brainmemory.synapses.synapse import SparseSynapses


def test_pruning_removes_weak_synapses():
    syn = SparseSynapses.empty(4, torch.device("cpu"))
    syn.add(torch.tensor([0, 1]), torch.tensor([1, 2]), weight=torch.tensor([0.0001, 0.4]))
    removed = syn.prune(0.01)
    assert removed == 1
    assert syn.nnz == 1
    assert float(syn.weight[0]) > 0.3


def test_brain_prunes_after_decay(tiny_brain):
    tiny_brain.config.forgetting.pruning_threshold = 0.05
    tiny_brain.config.forgetting.decay_rate = 0.2
    tiny_brain.config.forgetting.unused_decay = 0.2
    before = tiny_brain.net.syn.nnz
    tiny_brain.tick(steps=40)
    assert tiny_brain.net.syn.nnz <= before
