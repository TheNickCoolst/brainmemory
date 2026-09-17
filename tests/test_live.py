from __future__ import annotations

from brainmemory import Brain, BrainConfig
from brainmemory.config import AutonomyConfig
from brainmemory.live import Continuum
from tests.test_autonomy import FakeSense


def test_grow_keeps_old_memory(tiny_brain):
    tiny_brain.learn(["apfel", "rot", "suess"], importance=0.9)
    before = tiny_brain.net.n
    rec1 = tiny_brain.recall(["apfel"])
    bag1 = set(rec1["recalled_concepts"]) | set(rec1["associated_concepts"])
    grown = tiny_brain.grow(40)
    assert grown["added"] == 40
    assert tiny_brain.net.n == before + 40
    rec2 = tiny_brain.recall(["apfel"])
    bag2 = set(rec2["recalled_concepts"]) | set(rec2["associated_concepts"])
    assert "rot" in bag2 or "suess" in bag2 or bag2 & bag1


def test_grow_save_load(tiny_brain, tmp_path):
    tiny_brain.learn(["cello", "konzert"], importance=0.9)
    tiny_brain.grow(24)
    path = tmp_path / "grown.pt"
    tiny_brain.save(path)
    loaded = Brain.load(path, device="cpu")
    assert loaded.net.n == tiny_brain.net.n
    rec = loaded.recall(["cello"])
    found = set(rec["recalled_concepts"]) | set(rec["associated_concepts"])
    assert "konzert" in found or "cello" in found


def test_live_continuum_same_file_grows(tmp_path):
    path = tmp_path / "live.pt"
    cfg = BrainConfig.tiny(seed=4)
    cfg.autonomy = AutonomyConfig(
        rate_limit_s=0.0,
        wikipedia=False,
        min_novelty=0.0,
        pages_per_cycle=2,
        grow_neurons=20,
        grow_every_cycles=1,
        max_neurons=2000,
        sleep_every=99,
    )
    c1 = Continuum.open(path=path, config=cfg, senses=[FakeSense()], web=False)
    n0 = c1.brain.net.n
    r1 = c1.cycle(["hippocampus"], pages=2, grow=True)
    assert r1["neurons"] >= n0
    assert path.exists()
    c2 = Continuum.open(path=path, senses=[FakeSense()], web=False)
    assert c2.brain.net.n == r1["neurons"]
    assert c2.brain.life["cycles"] >= 1
    r2 = c2.cycle(["cortex"], pages=2, grow=True)
    assert r2["cycle"] > r1["cycle"]
    assert r2["neurons"] >= r1["neurons"]
    assert c2.brain.life["episodes_total"] >= c1.brain.life["episodes_total"]
