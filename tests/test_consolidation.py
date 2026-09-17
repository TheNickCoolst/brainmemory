from __future__ import annotations


def test_consolidation_builds_cortical_recall(small_brain):
    memory = ["lantern", "harbor", "fog"]
    small_brain.learn(memory, importance=0.9)
    for _ in range(4):
        small_brain.consolidate(count=12)
    result = small_brain.recall(["lantern"], use_hippocampus=False)
    rec = {c.lower() for c in result["recalled_concepts"]} | {
        c.lower() for c in result.get("associated_concepts", [])
    }
    assert "harbor" in rec or "fog" in rec, f"cortical recall failed: {result}"
    assert any(t.consolidation > 0 for t in small_brain.traces.traces)
