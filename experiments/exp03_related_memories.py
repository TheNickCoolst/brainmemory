"""Experiment 3 — shared structure across related fruit memories."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig


def run(seed: int = 23) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    apple = ["apple", "fruit", "red"]
    straw = ["strawberry", "fruit", "red"]
    banana = ["banana", "fruit", "yellow"]
    brain.learn(apple, importance=0.8)
    brain.learn(straw, importance=0.8)
    brain.learn(banana, importance=0.8)
    ov_ar = brain.engram_overlap(apple, straw)
    ov_ab = brain.engram_overlap(apple, banana)
    fruit = brain.recall(["fruit"])
    rec = set(fruit["recalled_concepts"]) | set(fruit["associated_concepts"])
    return {
        "name": "related_memories",
        "overlap_apple_strawberry": ov_ar,
        "overlap_apple_banana": ov_ab,
        "fruit_recall": fruit["recalled_concepts"],
        "pass": ov_ar["jaccard"] > 0 and ov_ar["jaccard"] >= ov_ab["jaccard"] * 0.6,
        "shared_from_fruit": sorted(rec & {"apple", "strawberry", "banana", "red", "yellow"}),
    }


if __name__ == "__main__":
    print(run())
