"""Experiment 7 — recall before vs after sleep."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import recall_accuracy


def run(seed: int = 27) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    mem = ["violin", "stage", "velvet"]
    brain.learn(mem, importance=0.7)
    before = brain.recall(["violin"])
    sleep_stats = brain.sleep()
    after = brain.recall(["violin"])
    return {
        "name": "sleep",
        "before": before["recalled_concepts"],
        "after": after["recalled_concepts"],
        "accuracy_before": recall_accuracy(before["recalled_concepts"], mem),
        "accuracy_after": recall_accuracy(after["recalled_concepts"], mem),
        "sleep": sleep_stats,
        "pass": recall_accuracy(after["recalled_concepts"], mem)
        >= recall_accuracy(before["recalled_concepts"], mem) - 0.05,
    }


if __name__ == "__main__":
    print(run())
