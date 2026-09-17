from __future__ import annotations


def test_repetition_strengthens_recall(small_brain):
    cue = ["ember"]
    mem = ["ember", "hearth", "night"]
    small_brain.config.memory.retrieve_reconsolidates = False
    small_brain.learn(mem, importance=0.4)
    weak = small_brain.recall(cue)
    for _ in range(5):
        small_brain.learn(mem, importance=0.9)
    strong = small_brain.recall(cue)
    rec_strong = set(strong.get("associated_concepts", [])) | set(strong["recalled_concepts"])
    assert "hearth" in rec_strong or "night" in rec_strong
    assert strong["confidence"] >= weak["confidence"] - 1e-6


def test_unused_memory_fades(small_brain):
    small_brain.config.memory.retrieve_reconsolidates = False
    small_brain.config.forgetting.decay_rate = 0.12
    small_brain.config.forgetting.unused_decay = 0.18
    small_brain.config.forgetting.pruning_threshold = 0.03
    mem = ["whisper", "attic", "dust"]
    small_brain.learn(mem, importance=0.2)
    before = small_brain.recall(["whisper"])
    small_brain.tick(steps=80)
    after = small_brain.recall(["whisper"])
    assert after["confidence"] <= before["confidence"] + 0.05
    rec_after = set(after.get("associated_concepts", []))
    rec_before = set(before.get("associated_concepts", [])) | set(before["recalled_concepts"])
    # accessibility reduced: fewer associates or lower scores
    if rec_before:
        assert len(rec_after) <= len(rec_before) or after["confidence"] < before["confidence"]
