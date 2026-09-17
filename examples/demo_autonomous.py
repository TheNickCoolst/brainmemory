#!/usr/bin/env python3
"""The brain fetches data itself, then learns by changing synapses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmemory import Brain, BrainConfig
from brainmemory.autonomy.learner import AutonomousLearner
from brainmemory.autonomy.senses import LocalFolderSense, WikipediaSense
from brainmemory.config import AutonomyConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous BrainMemory foraging")
    parser.add_argument("seeds", nargs="*", default=["hippocampus", "hebb", "schlaf"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--web", action="store_true", help="also fetch Wikipedia (needs network)")
    parser.add_argument("--lang", default="de")
    args = parser.parse_args()

    cfg_path = ROOT / "config" / "default.yaml"
    brain_cfg = BrainConfig.from_yaml(cfg_path) if cfg_path.exists() else BrainConfig.small()
    brain_cfg.neurons = min(brain_cfg.neurons, 4000)
    auto = AutonomyConfig(
        language=args.lang,
        max_pages=args.steps,
        sleep_every=2,
        min_novelty=0.08,
        rate_limit_s=0.6 if args.web else 0.0,
        follow_links=2,
        wikipedia=args.web,
        local_dir=str(ROOT / "examples" / "corpus"),
    )
    brain_cfg.autonomy = auto

    print("Creating BrainMemory for autonomous learning...\n")
    brain = Brain(brain_cfg)
    print(f"Neurons: {brain.stats()['neurons']:,}  synapses: {brain.stats()['synapses']:,}  device: {brain.stats()['device']}")

    senses = [LocalFolderSense(ROOT / "examples" / "corpus")]
    if args.web:
        senses.append(WikipediaSense(language=args.lang))
        print("Senses: local corpus + Wikipedia")
    else:
        print("Senses: local corpus (pass --web to add Wikipedia)")

    learner = AutonomousLearner(brain, config=auto, senses=senses)
    print(f"\nWandering from seeds: {', '.join(args.seeds)}")
    report = learner.wander(args.seeds, steps=args.steps)

    print(f"\nSteps: {report['steps']}")
    print(f"Episodes learned: {report['episodes_learned']}")
    print(f"Known concepts: {report['known_concepts']}")
    print(f"Hippocampal traces: {report['traces']}")
    for item in report["log"]:
        print(f"  - {item['topic']}: {item['episodes_learned']} episodes from {item['documents']}")

    print("\nProbing the network (not a lookup table)...")
    for cue in args.seeds[:3]:
        rec = brain.recall_text(cue)
        print(f"  cue '{cue}' → {', '.join(rec['recalled_concepts'][:8]) or '(nothing)'}  conf={rec['confidence']:.2f}")

    print("\nDone. Memory lives in synapses; deleting weights forgets the foraged knowledge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
