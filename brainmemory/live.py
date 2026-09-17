"""Lifelong brain: one network that keeps learning and growing.

Open the same file every time. Do not create a new brain for each session.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from brainmemory.autonomy.curiosity import knowledge_gaps
from brainmemory.autonomy.learner import AutonomousLearner
from brainmemory.autonomy.senses import LocalFolderSense, WikipediaSense
from brainmemory.brain import Brain
from brainmemory.config import BrainConfig

DEFAULT_LIVE = Path.home() / ".brainmemory" / "live.pt"
DEFAULT_SEEDS = [
    "hippocampus",
    "gedächtnis",
    "hebb",
    "schlaf",
    "cortex",
    "neuron",
    "synapsen",
    "plastizität",
]


class Continuum:
    def __init__(self, brain: Brain, learner: AutonomousLearner):
        self.brain = brain
        self.learner = learner

    @classmethod
    def open(
        cls,
        path: str | Path | None = None,
        config: BrainConfig | None = None,
        senses: list | None = None,
        device: str | None = None,
        local_dir: str | Path | None = None,
        web: bool | None = None,
    ) -> "Continuum":
        dest = Path(path) if path else None
        brain = Brain.live(path=dest, config=config, device=device)
        auto = brain.config.autonomy
        if local_dir is not None:
            auto.local_dir = str(local_dir)
        if web is not None:
            auto.wikipedia = bool(web)
        if senses is None:
            senses = []
            if auto.local_dir:
                senses.append(LocalFolderSense(auto.local_dir))
            if auto.wikipedia:
                senses.append(WikipediaSense(language=auto.language))
            if not senses:
                senses = [LocalFolderSense(Path.cwd())]
        learner = AutonomousLearner(brain, config=auto, senses=senses)
        return cls(brain, learner)

    def cycle(self, seeds: Sequence[str] | None = None, pages: int | None = None, grow: bool = True) -> dict[str, Any]:
        brain = self.brain
        auto = brain.config.autonomy
        n_pages = int(pages if pages is not None else auto.pages_per_cycle)
        topics = list(seeds or DEFAULT_SEEDS)
        topics.extend(knowledge_gaps(brain, sample=6))
        before = {
            "neurons": brain.net.n,
            "synapses": brain.net.syn.nnz,
            "concepts": len(brain.encoder.sdrs),
        }
        learned = self.learner.wander(topics, steps=n_pages)
        rehearsal = brain.rehearse(sample=12)
        brain.sleep()
        grown = {"added": 0}
        every = max(1, int(auto.grow_every_cycles))
        if grow and (int(brain.life.get("cycles", 0)) + 1) % every == 0:
            grown = brain.grow(auto.grow_neurons)
        brain.life["cycles"] = int(brain.life.get("cycles", 0)) + 1
        brain.life["episodes_total"] = int(brain.life.get("episodes_total", 0)) + int(learned.get("episodes_learned", 0))
        saved = brain.checkpoint()
        report = {
            "cycle": brain.life["cycles"],
            "episodes_this_cycle": learned.get("episodes_learned", 0),
            "episodes_total": brain.life["episodes_total"],
            "neurons_before": before["neurons"],
            "neurons": brain.net.n,
            "synapses_before": before["synapses"],
            "synapses": brain.net.syn.nnz,
            "concepts_before": before["concepts"],
            "concepts": len(brain.encoder.sdrs),
            "grown": grown.get("added", 0),
            "rehearse": rehearsal,
            "saved": str(saved),
            "log": learned.get("log", []),
        }
        _append_journal(saved, report)
        return report

    def run(
        self,
        cycles: int = 1,
        forever: bool = False,
        seeds: Iterable[str] | None = None,
        pages: int | None = None,
    ) -> list[dict[str, Any]]:
        seed_list = list(seeds or DEFAULT_SEEDS)
        out: list[dict[str, Any]] = []
        i = 0
        try:
            while forever or i < int(cycles):
                out.append(self.cycle(seed_list, pages=pages))
                i += 1
        except KeyboardInterrupt:
            self.brain.checkpoint()
        return out


def _append_journal(save_path: Path, report: dict[str, Any]) -> None:
    journal = Path(save_path).with_name("journal.jsonl")
    row = {"ts": time.time(), **{k: v for k, v in report.items() if k != "log"}}
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
