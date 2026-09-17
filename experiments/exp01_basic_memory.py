"""Experiment 1 — basic memory: teach apple/red/fruit/sweet, cue red."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.metrics import completion_from_result


def run(seed: int = 21) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    memory = ["apple", "red", "fruit", "sweet"]
    learned = brain.learn(memory, importance=0.85)
    result = brain.recall(["red"])
    metrics = completion_from_result(result, memory, ["red"])
    return {
        "name": "basic_memory",
        "learned": learned,
        "result": {k: result[k] for k in result if k != "activated_neurons"},
        "metrics": metrics,
        "pass": metrics["recall_accuracy"] >= 0.5 and "apple" in set(result["recalled_concepts"]) | set(result["associated_concepts"]),
    }


if __name__ == "__main__":
    print(run())
