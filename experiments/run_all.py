#!/usr/bin/env python3
"""Run the eight v0.1 experiments and print a compact report."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp01_basic_memory import run as e1
from experiments.exp02_partial_recall import run as e2
from experiments.exp03_related_memories import run as e3
from experiments.exp04_pattern_separation import run as e4
from experiments.exp05_forgetting import run as e5
from experiments.exp06_reinforcement import run as e6
from experiments.exp07_sleep import run as e7
from experiments.exp08_consolidation import run as e8


EXPERIMENTS = [e1, e2, e3, e4, e5, e6, e7, e8]


def main() -> int:
    rows = []
    for fn in EXPERIMENTS:
        try:
            out = fn()
        except Exception as exc:
            out = {"name": getattr(fn, "__name__", "unknown"), "pass": False, "error": str(exc)}
            traceback.print_exc()
        rows.append(out)
        status = "PASS" if out.get("pass") else "FAIL"
        print(f"[{status}] {out.get('name')}")
        compact = {k: v for k, v in out.items() if k not in {"learned", "result"}}
        print(json.dumps(compact, default=str, indent=2)[:1200])
        print()
    n_ok = sum(1 for r in rows if r.get("pass"))
    print(f"{n_ok}/{len(rows)} experiments passed")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
