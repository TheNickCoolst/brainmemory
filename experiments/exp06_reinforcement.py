"""Experiment 6 — repetition strengthens recall."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import recall_accuracy


def run(seed: int = 26) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    mem = ["compass", "ridge", "wind"]
    brain.learn(mem, importance=0.35)
    weak = brain.recall(["compass"])
    for _ in range(6):
        brain.learn(mem, importance=0.9)
    strong = brain.recall(["compass"])
    return {
        "name": "reinforcement",
        "accuracy_weak": recall_accuracy(weak["recalled_concepts"], mem),
        "accuracy_strong": recall_accuracy(strong["recalled_concepts"], mem),
        "confidence_weak": weak["confidence"],
        "confidence_strong": strong["confidence"],
        "pass": recall_accuracy(strong["recalled_concepts"], mem)
        >= recall_accuracy(weak["recalled_concepts"], mem),
    }


if __name__ == "__main__":
    print(run())
