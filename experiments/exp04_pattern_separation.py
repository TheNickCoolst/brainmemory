"""Experiment 4 — similar episodes remain distinguishable."""

from __future__ import annotations

from brainmemory import Brain, BrainConfig


def run(seed: int = 24) -> dict:
    brain = Brain(BrainConfig.small(seed=seed), device="cpu")
    a = ["nick", "monday", "math", "room204"]
    b = ["nick", "tuesday", "physics", "room204"]
    brain.learn(a, importance=0.88)
    brain.learn(b, importance=0.88)
    r = brain.recall(["nick", "monday"])
    scores = r.get("scores", {})
    rec = {c.lower() for c in r["recalled_concepts"]}
    ov = brain.engram_overlap(a, b)
    math_ok = scores.get("math", 0) >= scores.get("physics", 0) and (
        "math" in rec or scores.get("math", 0) > 0.2
    )
    return {
        "name": "pattern_separation",
        "recalled": r["recalled_concepts"],
        "scores": scores,
        "overlap": ov,
        "pass": math_ok and ov["jaccard"] < 0.9,
    }


if __name__ == "__main__":
    print(run())
