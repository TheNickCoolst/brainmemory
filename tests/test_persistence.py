from __future__ import annotations


def test_save_load_preserves_recall(small_brain, tmp_path):
    memory = ["max", "golden retriever", "park"]
    small_brain.learn(memory, importance=0.9)
    small_brain.config.memory.retrieve_reconsolidates = False
    before = small_brain.recall(["golden retriever"])
    path = tmp_path / "brain.pt"
    small_brain.save(path)
    loaded = type(small_brain).load(path, device="cpu")
    import torch

    assert loaded.net.syn.nnz == small_brain.net.syn.nnz
    assert torch.allclose(
        loaded.net.syn.weight.cpu(),
        small_brain.net.syn.weight.cpu(),
        atol=1e-5,
    )
    loaded.config.memory.retrieve_reconsolidates = False
    after = loaded.recall(["golden retriever"])
    rec_before = set(before.get("associated_concepts", [])) | set(before["recalled_concepts"])
    rec_after = set(after.get("associated_concepts", [])) | set(after["recalled_concepts"])
    assert "max" in rec_after or "park" in rec_after
    assert rec_after & rec_before


def test_recall_after_restart_uses_weights_not_answers(small_brain, tmp_path):
    small_brain.learn(["cello", "concert", "hall"], importance=0.9)
    path = tmp_path / "brain.pt"
    small_brain.save(path)
    loaded = type(small_brain).load(path, device="cpu")
    loaded.net.zero_weights()
    result = loaded.recall(["cello"])
    assoc = set(result.get("associated_concepts", []))
    assert "concert" not in assoc
    assert "hall" not in assoc
