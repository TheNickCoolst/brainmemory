"""Experiment 2 — partial cue reconstructs missing features."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import completion_from_result


def run(seed: int = 22) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    memory = ["dog", "leash", "sunrise", "meadow", "whistle"]
    brain.learn(memory, importance=0.9)
    cue = ["leash", "whistle"]
    result = brain.recall(cue)
    metrics = completion_from_result(result, memory, cue)
    recovered = [m for m in memory if m not in cue and m in result["recalled_concepts"]]
    return {
        "name": "partial_recall",
        "cue": cue,
        "recovered": recovered,
        "metrics": metrics,
        "pass": len(recovered) >= 2,
        "recalled": result["recalled_concepts"],
    }


if __name__ == "__main__":
    print(run())
