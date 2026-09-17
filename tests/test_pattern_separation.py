from __future__ import annotations


def test_pattern_separation_keeps_similar_episodes_distinct(small_brain):
    a = ["nick", "monday", "math", "room204"]
    b = ["nick", "tuesday", "physics", "room204"]
    small_brain.learn(a, importance=0.85)
    small_brain.learn(b, importance=0.85)
    r = small_brain.recall(["nick", "monday"])
    scores = {k.lower(): v for k, v in r.get("scores", {}).items()}
    rec = {c.lower() for c in r["recalled_concepts"]}
    math_score = scores.get("math", 1.0 if "math" in rec else 0.0)
    physics_score = scores.get("physics", 1.0 if "physics" in rec else 0.0)
    assert math_score > physics_score or ("math" in rec and "physics" not in rec), (
        f"episodes collapsed: {r['recalled_concepts']} scores={r.get('scores')}"
    )
    ov = small_brain.engram_overlap(a, b)
    assert ov["jaccard"] < 0.95, "engrams should not be identical"
