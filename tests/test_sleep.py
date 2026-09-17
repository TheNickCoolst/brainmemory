from __future__ import annotations


def test_sleep_replay_returns_statistics(tiny_brain):
    tiny_brain.learn(["river", "stone"], importance=0.7)
    tiny_brain.learn(["river", "fish"], importance=0.7)
    stats = tiny_brain.sleep()
    for key in (
        "memories_replayed",
        "synapses_strengthened",
        "synapses_weakened",
        "synapses_pruned",
        "new_associations",
    ):
        assert key in stats
    assert stats["memories_replayed"] >= 1
