from __future__ import annotations


def test_pattern_completion_recovers_missing_features(small_brain):
    memory = ["dog", "golden retriever", "park", "red ball", "max"]
    small_brain.learn(memory, importance=0.9)
    result = small_brain.recall(["golden retriever", "red ball"])
    recalled = set(result["recalled_concepts"]) | set(result.get("associated_concepts", []))
    missing = {"dog", "park", "max"}
    recovered = missing & {c.lower() for c in recalled}
    assert len(recovered) >= 2, f"expected pattern completion, got {recalled} scores={result.get('scores')}"


def test_memory_is_in_synapses_not_a_table(small_brain):
    small_brain.learn(["apple", "red", "fruit", "sweet"], importance=0.9)
    ok = small_brain.recall(["red"])
    associated = set(ok.get("associated_concepts", [])) | set(ok["recalled_concepts"])
    assert "apple" in associated or "fruit" in associated or "sweet" in associated
    small_brain.net.zero_weights()
    gone = small_brain.recall(["red"])
    associated2 = set(gone.get("associated_concepts", []))
    assert "apple" not in associated2
    assert "fruit" not in associated2
    assert gone["confidence"] <= ok["confidence"]
