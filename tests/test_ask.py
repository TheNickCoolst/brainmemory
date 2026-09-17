from __future__ import annotations


def test_ask_uses_network_completion(tiny_brain):
    tiny_brain.learn(["hippocampus", "cortex", "schlaf"], importance=0.95)
    rec = tiny_brain.ask("Was macht der Hippocampus?")
    assert "answer" in rec
    bag = set(rec.get("recalled_concepts") or []) | set(rec.get("associated_concepts") or [])
    assert "cortex" in bag or "schlaf" in bag
    assert "hippocampus" in " ".join(rec.get("cue_concepts") or [])
    assert rec["engram_size"] > 0
