#!/usr/bin/env python3
"""Scale benchmarks: learn/recall/sleep time and RAM at several network sizes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

from brainmemory import Brain, BrainConfig


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def bench_size(n: int, seed: int = 0) -> dict:
    cfg = BrainConfig(neurons=n, seed=seed)
    cfg.synapses.avg_connections = 48 if n <= 25000 else 32
    cfg.memory.bits_per_concept = 24 if n < 5000 else 40
    cfg.sleep.replay_count = 12 if n >= 25000 else 24
    t0 = time.perf_counter()
    mem0 = rss_mb()
    brain = Brain(cfg)
    init_s = time.perf_counter() - t0
    mem1 = rss_mb()
    t1 = time.perf_counter()
    brain.learn(["max", "golden retriever", "park", "red ball"], importance=0.8)
    learn_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    rec = brain.recall(["golden retriever"])
    recall_s = time.perf_counter() - t2
    t3 = time.perf_counter()
    sleep = brain.sleep()
    sleep_s = time.perf_counter() - t3
    return {
        "neurons": n,
        "synapses": brain.net.syn.nnz,
        "init_s": round(init_s, 4),
        "learn_s": round(learn_s, 4),
        "recall_s": round(recall_s, 4),
        "sleep_s": round(sleep_s, 4),
        "ram_delta_mb": round(mem1 - mem0, 2),
        "rss_mb": round(rss_mb(), 2),
        "device": brain.stats()["device"],
        "recalled": rec["recalled_concepts"][:6],
        "sleep_replayed": sleep["memories_replayed"],
    }


def main() -> int:
    sizes = [1000, 2500, 5000, 10000]
    # larger sizes if the first ones are fast
    rows = []
    for n in sizes:
        print(f"benchmarking n={n} ...", flush=True)
        row = bench_size(n)
        rows.append(row)
        print(row, flush=True)
    if rows[-1]["learn_s"] < 2.0:
        for n in (25000, 50000):
            print(f"benchmarking n={n} ...", flush=True)
            row = bench_size(n)
            rows.append(row)
            print(row, flush=True)
            if row["learn_s"] > 8.0:
                break
    print("\n=== summary ===")
    hdr = f"{'N':>8} {'synapses':>12} {'init_s':>8} {'learn_s':>8} {'recall_s':>9} {'sleep_s':>8} {'RAM_MB':>8}"
    print(hdr)
    for r in rows:
        print(
            f"{r['neurons']:8d} {r['synapses']:12d} {r['init_s']:8.3f} {r['learn_s']:8.3f} "
            f"{r['recall_s']:9.3f} {r['sleep_s']:8.3f} {r['rss_mb']:8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
