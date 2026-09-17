from __future__ import annotations

import numpy as np


def test_engram_creation_changes_network(tiny_brain):
    w_before = tiny_brain.net.syn.weight.clone()
    nnz_before = tiny_brain.net.syn.nnz
    result = tiny_brain.learn(["apple", "red", "fruit", "sweet"], importance=0.8)
    assert result["engram_size"] > 0
    assert result["synapses_grown"] > 0 or tiny_brain.net.syn.nnz >= nnz_before
    assert not torch_allclose(tiny_brain.net.syn.weight, w_before) or tiny_brain.net.syn.nnz > nnz_before


def test_engram_is_distributed(tiny_brain):
    result = tiny_brain.learn(["dog", "park", "ball"], importance=0.7)
    assert result["engram_size"] >= 10
    # not a single neuron memory
    assert result["engram_size"] > 1


def test_engram_overlap_related_concepts(small_brain):
    small_brain.learn(["apple", "red", "fruit"], importance=0.8)
    small_brain.learn(["strawberry", "red", "fruit"], importance=0.8)
    ov = small_brain.engram_overlap(["apple", "red", "fruit"], ["strawberry", "red", "fruit"])
    assert ov["jaccard"] > 0.0
    assert ov["intersection"] > 0


def torch_allclose(a, b) -> bool:
    import torch

    if a.numel() != b.numel():
        return False
    n = min(a.numel(), b.numel())
    if n == 0:
        return True
    x = a.flatten()[:n]
    y = b.flatten()[:n]
    return bool(torch.allclose(x, y))
