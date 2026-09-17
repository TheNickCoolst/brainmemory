from __future__ import annotations


def test_reconsolidation_updates_association(small_brain):
    small_brain.learn(["alex", "blue car"], importance=0.85)
    first = small_brain.recall(["alex"])
    rec1 = set(first["recalled_concepts"]) | set(first["associated_concepts"])
    assert "blue car" in rec1 or any("blue" in c for c in rec1)
    small_brain.learn(["alex", "sold", "red car"], importance=0.9, reconsolidate=True)
    second = small_brain.recall(["alex"])
    rec2 = set(second["recalled_concepts"]) | set(second["associated_concepts"])
    assert "red car" in rec2 or "sold" in rec2 or any("red" in c for c in rec2)
