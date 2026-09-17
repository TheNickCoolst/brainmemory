from __future__ import annotations

from pathlib import Path

from brainmemory import Brain, BrainConfig
from brainmemory.autonomy.learner import AutonomousLearner
from brainmemory.autonomy.senses import Document, chunk_text
from brainmemory.config import AutonomyConfig


class FakeSense:
    def __init__(self):
        self.docs = {
            "hippocampus": Document(
                title="Hippocampus",
                text=(
                    "Der Hippocampus speichert episodische Erinnerungen. "
                    "Schlaf konsolidiert Spuren in den Cortex. "
                    "Mustertrennung hält ähnliche Episoden getrennt."
                ),
                source="fake:hippocampus",
                related=["Cortex", "Schlaf"],
            ),
            "cortex": Document(
                title="Cortex",
                text="Der Cortex lernt langsam semantische Assoziationen durch Wiederholung und Replay.",
                source="fake:cortex",
                related=["Hippocampus"],
            ),
            "schlaf": Document(
                title="Schlaf",
                text="Schlaf replayt wichtige Muster und beschneidet schwache Synapsen.",
                source="fake:schlaf",
                related=[],
            ),
        }

    def search(self, query: str, limit: int = 5):
        q = query.lower()
        hits = [d for k, d in self.docs.items() if q in k or q in d.title.lower() or q in d.text.lower()]
        return hits[:limit]


def test_chunk_text_splits_sentences():
    chunks = chunk_text("Eins. Zwei. Drei ist ein etwas längerer Satz.", max_chars=20)
    assert len(chunks) >= 2


def test_ingest_local_corpus_then_recall(tiny_brain):
    corpus = Path(__file__).resolve().parents[1] / "examples" / "corpus"
    cfg = AutonomyConfig(rate_limit_s=0.0, wikipedia=False, min_novelty=0.0, sleep_every=99)
    result = tiny_brain.ingest(corpus, config=cfg)
    assert result["episodes_learned"] >= 1
    rec = tiny_brain.recall(["hippocampus"])
    found = set(rec["recalled_concepts"]) | set(rec["associated_concepts"])
    assert any(x in found for x in ("cortex", "episodisches", "gedaechtnis", "gedächtnis", "mustertrennung", "schlaf"))


def test_wander_with_fake_sense_learns(tiny_brain):
    cfg = AutonomyConfig(rate_limit_s=0.0, wikipedia=False, min_novelty=0.0, max_pages=3, sleep_every=2, follow_links=2)
    learner = AutonomousLearner(tiny_brain, config=cfg, senses=[FakeSense()])
    out = learner.wander(["hippocampus"], steps=3)
    assert out["episodes_learned"] >= 1
    assert out["known_concepts"] >= 3
    rec = tiny_brain.recall(["hippocampus"])
    bag = " ".join(rec["recalled_concepts"])
    assert "cortex" in bag or "schlaf" in bag or rec["confidence"] > 0


def test_novelty_gate_skips_repeat(tiny_brain):
    cfg = AutonomyConfig(rate_limit_s=0.0, wikipedia=False, min_novelty=0.5, max_pages=2, sleep_every=99)
    learner = AutonomousLearner(tiny_brain, config=cfg, senses=[FakeSense()])
    first = learner.forage("hippocampus")
    # same title is marked seen, second forage of same topic skips
    second = learner.forage("hippocampus")
    assert first["episodes_learned"] >= 1
    assert second["episodes_learned"] == 0
