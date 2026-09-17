"""Lifelong brain: one network that keeps learning and growing.

Open the same file every time. Do not create a new brain for each session.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import psutil

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
        rehearsal = brain.rehearse(sample=24)
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
        until_limit: bool = False,
        ram_cap_gb: float | None = None,
        ram_floor_gb: float = 2.5,
    ) -> list[dict[str, Any]]:
        seed_list = list(seeds or DEFAULT_SEEDS)
        out: list[dict[str, Any]] = []
        i = 0
        try:
            if until_limit or forever:
                return self.run_until_limit(
                    seeds=seed_list,
                    pages=pages,
                    ram_cap_gb=ram_cap_gb,
                    ram_floor_gb=ram_floor_gb,
                    stop_at_limit=bool(until_limit) and not forever,
                )
            while forever or i < int(cycles):
                out.append(self.cycle(seed_list, pages=pages))
                i += 1
        except KeyboardInterrupt:
            self.brain.checkpoint()
        return out

    def run_until_limit(
        self,
        seeds: Sequence[str] | None = None,
        pages: int | None = None,
        ram_cap_gb: float | None = None,
        ram_floor_gb: float = 2.5,
        max_cycles: int = 10_000,
        stop_at_limit: bool = True,
    ) -> list[dict[str, Any]]:
        """Learn and grow. If stop_at_limit is False, keep looping without growing when RAM is tight."""
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        cap = float(ram_cap_gb) if ram_cap_gb is not None else min(total_gb * 0.62, total_gb - float(ram_floor_gb) - 1.5)
        cap = max(1.0, cap)
        floor = float(ram_floor_gb)
        self.brain.config.autonomy.max_neurons = max(
            int(self.brain.config.autonomy.max_neurons), 5_000_000
        )
        seed_list = list(seeds or DEFAULT_SEEDS)
        out: list[dict[str, Any]] = []
        mode = "bis Stopp/Shutdown" if not stop_at_limit else "bis RAM-Limit"
        n_loops = 10**12 if not stop_at_limit else int(max_cycles)
        print(
            f"Lauf ({mode}): RAM {total_gb:.1f} GB total, cap {cap:.1f} GB RSS, "
            f"floor {floor:.1f} GB frei. Speichert nach jedem Zyklus.",
            flush=True,
        )
        try:
            for i in range(n_loops):
                ram = _ram()
                disk_free = shutil.disk_usage(str(self.brain.live_path or Path.home())).free / (1024**3)
                reason = _limit_reason(ram, cap, floor, disk_free, self.brain.net.n, self.brain.config.autonomy.max_neurons)
                extra = 0 if reason else _grow_chunk(self.brain, ram, cap)
                if reason and stop_at_limit:
                    self.brain.checkpoint()
                    print(f"LIMIT_REACHED: {reason}", flush=True)
                    print(
                        f"final neurons={self.brain.net.n:,} synapses={self.brain.net.syn.nnz:,} "
                        f"rss_gb={ram['rss_gb']:.2f} avail_gb={ram['available_gb']:.2f}",
                        flush=True,
                    )
                    break
                if reason and not stop_at_limit:
                    print(f"HOLD_GROWTH: {reason} — weiter lernen ohne neue Neuronen", flush=True)
                self.brain.config.autonomy.grow_neurons = extra
                try:
                    report = self.cycle(seed_list, pages=pages or 1, grow=bool(extra))
                except (MemoryError, RuntimeError) as exc:
                    try:
                        self.brain.checkpoint()
                    except Exception:
                        pass
                    if stop_at_limit:
                        print(f"LIMIT_REACHED: allocation_failed {type(exc).__name__}: {exc}", flush=True)
                        break
                    print(f"HOLD_GROWTH: allocation_failed {type(exc).__name__} — retry", flush=True)
                    time.sleep(5)
                    continue
                ram2 = _ram()
                report["rss_gb"] = ram2["rss_gb"]
                report["available_gb"] = ram2["available_gb"]
                report["ram_cap_gb"] = cap
                out.append(report)
                if not stop_at_limit and len(out) > 32:
                    out = out[-16:]
                print(
                    f"Zyklus {report['cycle']}: N {report['neurons']:,}  syn {report['synapses']:,}  "
                    f"konzepte {report['concepts']}  +{report['grown']}N  "
                    f"rss {ram2['rss_gb']:.2f}/{cap:.1f} GB  frei {ram2['available_gb']:.2f} GB",
                    flush=True,
                )
            else:
                self.brain.checkpoint()
                print("LIMIT_REACHED: max_cycles", flush=True)
        except KeyboardInterrupt:
            self.brain.checkpoint()
            print("STOPPED: interrupted_saved", flush=True)
        return out


def _ram() -> dict[str, float]:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    return {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
        "percent": float(vm.percent),
        "rss_gb": rss / (1024**3),
        "rss_bytes": float(rss),
    }


def _limit_reason(
    ram: dict[str, float],
    cap: float,
    floor: float,
    disk_free_gb: float,
    n: int,
    max_n: int,
) -> str | None:
    if ram["rss_gb"] >= cap:
        return f"rss {ram['rss_gb']:.2f} GB >= cap {cap:.1f} GB"
    if ram["available_gb"] <= floor:
        return f"only {ram['available_gb']:.2f} GB free (floor {floor:.1f})"
    if disk_free_gb < 4.0:
        return f"disk free {disk_free_gb:.1f} GB < 4 GB"
    if n >= int(max_n):
        return f"max_neurons {max_n:,}"
    return None


def _grow_chunk(brain: Brain, ram: dict[str, float], cap: float) -> int:
    n = max(1, brain.net.n)
    per = ram["rss_bytes"] / n
    per = max(per, 4000.0)
    remain = max(0.0, (cap - ram["rss_gb"]) * (1024**3))
    # take a fraction of remaining budget so one expand cannot OOM
    extra = int(0.28 * remain / per)
    extra = max(0, extra)
    chunk_cap = 12_000 if getattr(brain, "device", None) is not None and str(brain.device).startswith("mps") else 25_000
    extra = min(extra, chunk_cap, int(brain.config.autonomy.max_neurons) - n)
    if extra < 512 and remain > 0.4 * (1024**3):
        extra = min(4096, int(brain.config.autonomy.max_neurons) - n)
    return max(0, extra)


def _append_journal(save_path: Path, report: dict[str, Any]) -> None:
    journal = Path(save_path).with_name("journal.jsonl")
    row = {"ts": time.time(), **{k: v for k, v in report.items() if k != "log"}}
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
