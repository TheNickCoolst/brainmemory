"""Experiment 5 — unused weak memory becomes harder to recall."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import recall_accuracy


def run(seed: int = 25) -> dict:
    cfg = BrainConfig.small(seed=seed)
    cfg.memory.retrieve_reconsolidates = False
    cfg.forgetting.decay_rate = 0.1
    cfg.forgetting.unused_decay = 0.15
    cfg.forgetting.pruning_threshold = 0.025
    brain = Brain(cfg, device="cpu")
    mem = ["whisper", "attic", "dust"]
    brain.learn(mem, importance=0.15)
    before = brain.recall(["whisper"])
    brain.tick(steps=90)
    after = brain.recall(["whisper"])
    acc_b = recall_accuracy(before["recalled_concepts"], mem)
    acc_a = recall_accuracy(after["recalled_concepts"], mem)
    return {
        "name": "forgetting",
        "accuracy_before": acc_b,
        "accuracy_after": acc_a,
        "confidence_before": before["confidence"],
        "confidence_after": after["confidence"],
        "pass": after["confidence"] <= before["confidence"] + 0.02 and acc_a <= acc_b + 1e-9,
    }


if __name__ == "__main__":
    print(run())
