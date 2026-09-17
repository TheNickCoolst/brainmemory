#!/usr/bin/env python3
"""Talk to the lifelong brain. Answers come from network completion, not a lookup table.

    python examples/talk.py
    python examples/talk.py --device mps
    python examples/talk.py "Was ist der Hippocampus?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmemory import Brain
from brainmemory.device import select_device
from brainmemory.live import DEFAULT_LIVE


def main() -> int:
    p = argparse.ArgumentParser(description="Ask the lifelong BrainMemory")
    p.add_argument("question", nargs="*", help="one-shot question; omit for a prompt loop")
    p.add_argument("--path", default=None)
    p.add_argument("--device", default=None, help="cpu / mps / cuda")
    p.add_argument("--lang", default="de")
    args = p.parse_args()

    dest = Path(args.path) if args.path else DEFAULT_LIVE
    if not dest.exists():
        print(f"Kein Gehirn unter {dest}")
        print("Zuerst: python examples/live_brain.py")
        return 1

    device = args.device or str(select_device())
    print(f"Lade Gehirn ({device}) von {dest} ...", flush=True)
    brain = Brain.load(dest, device=device)
    st = brain.stats()
    print(
        f"Neuronen {st['neurons']:,}  Synapsen {st['synapses']:,}  "
        f"Konzepte {st['known_concepts']}  device {st['device']}\n"
    )

    def once(q: str) -> None:
        rec = brain.ask(q, language=args.lang)
        print(f"Cue:     {', '.join(rec.get('cue_concepts') or []) or '(keine)'}")
        print(f"Recall:  {', '.join(rec.get('recalled_concepts') or []) or '(leer)'}")
        print(f"Aktiv:   {rec.get('engram_size', 0)} Neuronen  Konfidenz {rec.get('confidence', 0):.2f}")
        print(f"Antwort: {rec.get('answer')}\n")

    if args.question:
        once(" ".join(args.question))
        return 0

    print("Frag das Netz. Leerzeile oder 'quit' beendet.\n")
    while True:
        try:
            q = input("du> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"quit", "exit", "q"}:
            break
        once(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
