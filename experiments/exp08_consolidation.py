"""Experiment 8 — cortical recall after hippocampal contribution is disabled."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import recall_accuracy


def run(seed: int = 28) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    mem = ["lantern", "harbor", "fog", "tide"]
    brain.learn(mem, importance=0.92)
    for _ in range(5):
        brain.consolidate(count=16)
    with_h = brain.recall(["lantern"], use_hippocampus=True)
    without = brain.recall(["lantern"], use_hippocampus=False)
    return {
        "name": "consolidation",
        "with_hippocampus": with_h["recalled_concepts"],
        "without_hippocampus": without["recalled_concepts"],
        "accuracy_cortical": recall_accuracy(without["recalled_concepts"], mem),
        "pass": recall_accuracy(without["recalled_concepts"], mem) >= 0.4,
        "mean_consolidation": sum(t.consolidation for t in brain.traces.traces) / max(1, len(brain.traces)),
    }


if __name__ == "__main__":
    print(run())
