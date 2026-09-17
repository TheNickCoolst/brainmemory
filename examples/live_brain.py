#!/usr/bin/env python3
"""One lifelong brain. Each run continues the same network — it is never reset.

    python examples/live_brain.py              # one cycle, then save
    python examples/live_brain.py --cycles 5
    python examples/live_brain.py --forever    # Ctrl+C saves
    python examples/live_brain.py --status
    python examples/live_brain.py --web
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmemory import Brain, BrainConfig
from brainmemory.live import DEFAULT_LIVE, DEFAULT_SEEDS, Continuum


def main() -> int:
    p = argparse.ArgumentParser(description="Lifelong BrainMemory")
    p.add_argument("seeds", nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--forever", action="store_true")
    p.add_argument("--web", action="store_true", help="Wikipedia as extra sense")
    p.add_argument("--status", action="store_true")
    p.add_argument("--path", default=None, help="checkpoint path (default ~/.brainmemory/live.pt)")
    p.add_argument("--pages", type=int, default=None)
    args = p.parse_args()

    dest = Path(args.path) if args.path else DEFAULT_LIVE
    corpus = ROOT / "examples" / "corpus"

    if args.status:
        if not dest.exists():
            print(f"Noch kein Gehirn unter {dest}")
            print("Starte mit: python examples/live_brain.py")
            return 0
        brain = Brain.load(dest)
        st = brain.stats()
        print(f"Pfad:      {dest}")
        print(f"Neuronen:  {st['neurons']:,}")
        print(f"Synapsen:  {st['synapses']:,}")
        print(f"Konzepte:  {st['known_concepts']:,}")
        print(f"Zyklen:    {st['cycles']}")
        print(f"Episoden:  {st['episodes_total']}")
        print(f"Dokumente: {st['seen_docs']}")
        return 0

    cfg_path = ROOT / "config" / "default.yaml"
    cfg = BrainConfig.from_yaml(cfg_path) if cfg_path.exists() else BrainConfig.demo()
    cfg.autonomy.local_dir = str(corpus)
    cfg.autonomy.wikipedia = bool(args.web)
    cfg.autonomy.rate_limit_s = 0.7 if args.web else 0.0
    cfg.autonomy.pages_per_cycle = args.pages or cfg.autonomy.pages_per_cycle

    existed = dest.exists()
    continuum = Continuum.open(
        path=dest,
        config=None if existed else cfg,
        local_dir=corpus,
        web=args.web,
    )
    brain = continuum.brain
    st = brain.stats()
    print("Ein Gehirn, das bleibt.\n")
    print(f"{'Fortsetzung' if existed else 'Geburt'}: {dest}")
    print(f"Neuronen: {st['neurons']:,}  Synapsen: {st['synapses']:,}  Konzepte: {st['known_concepts']}")
    print(f"Zyklen bisher: {st['cycles']}  Episoden bisher: {st['episodes_total']}")
    print()

    if args.forever:
        print("Läuft weiter bis Ctrl+C — speichert nach jedem Zyklus.\n")
    reports = continuum.run(cycles=args.cycles, forever=args.forever, seeds=args.seeds, pages=args.pages)
    for r in reports:
        print(
            f"Zyklus {r['cycle']}: "
            f"+{r['episodes_this_cycle']} Episoden  "
            f"Neuronen {r['neurons_before']:,}→{r['neurons']:,}  "
            f"Synapsen {r['synapses_before']:,}→{r['synapses']:,}  "
            f"Konzepte {r['concepts_before']}→{r['concepts']}"
        )
        if r.get("grown"):
            print(f"  Neurogenese: +{r['grown']} Neuronen")
    print(f"\nGespeichert: {brain.live_path}")
    print("Nächster Start lädt genau dieses Gehirn — kein neues Training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
