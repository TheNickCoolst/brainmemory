#!/usr/bin/env python3
"""End-to-end BrainMemory demonstration. Values come from the live network."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmemory import Brain, BrainConfig


def _fmt_list(items, n=8) -> str:
    items = list(items)[:n]
    return ", ".join(items) if items else "(none)"


def main() -> int:
    cfg_path = ROOT / "config" / "default.yaml"
    if cfg_path.exists():
        cfg = BrainConfig.from_yaml(cfg_path)
    else:
        cfg = BrainConfig.demo()
    # keep the demo responsive on CPU while remaining in the 10k-neuron regime
    cfg.neurons = min(cfg.neurons, 10000)

    print("Creating BrainMemory...\n")
    brain = Brain(cfg)
    st = brain.stats()
    print(f"Neurons: {st['neurons']:,}")
    print(f"Synapses: {st['synapses']:,}")
    print(f"Device: {st['device']}")
    print()

    sentences = [
        "Max is a golden retriever.",
        "Max likes playing with a red ball.",
        "Max often plays in the park.",
    ]
    print("Teaching:")
    total_grown = 0
    total_strengthened = 0
    for s in sentences:
        print(f'  "{s}"')
        r = brain.learn_text(s, importance=0.85)
        total_grown += r["synapses_grown"]
        total_strengthened += r["synapses_strengthened"]
        print(f"    concepts: {_fmt_list(r.get('extracted_concepts', r.get('concepts', [])))}")
        print(f"    engram size: {r['engram_size']}")

    print("\nNeural changes:")
    print("+ engram created")
    print(f"+ synapses grown: {total_grown:,}")
    print(f"+ synapses strengthened: {total_strengthened:,}")

    print('\nCue:\n  "golden retriever"\n')
    print("Neural propagation...")
    rec = brain.recall_text("golden retriever")
    print("\nRecalled:")
    shown = []
    for c in rec["recalled_concepts"][:20]:
        print(f"  {c}")
        shown.append(c)
    print(f"confidence: {rec['confidence']:.3f}  engram_size: {rec['engram_size']}")
    print(f"scores: { {k: round(v, 2) for k, v in list(rec.get('scores', {}).items())[:8]} }")

    print("\nRunning sleep...")
    sleep = brain.sleep()
    print(f"Replayed memories: {sleep['memories_replayed']}")
    print(f"Strengthened synapses: {sleep['synapses_strengthened']}")
    print(f"Weakened synapses: {sleep['synapses_weakened']}")
    print(f"Pruned synapses: {sleep['synapses_pruned']}")
    print(f"New associations: {sleep['new_associations']}")

    print('\nPartial cue:\n  "red ball"\n')
    rec2 = brain.recall_text("red ball")
    print("Recovered:")
    for c in rec2["recalled_concepts"][:20]:
        print(f"  {c}")
    print(f"confidence: {rec2['confidence']:.3f}  engram_size: {rec2['engram_size']}")

    vis = ROOT / "artifacts" / "demo_memory.png"
    try:
        path = brain.visualize_memory(["golden retriever"], path=vis)
        print(f"\nVisualization written to {path}")
    except Exception as exc:
        print(f"\nVisualization skipped: {exc}")

    save_path = ROOT / "artifacts" / "demo_brain.pt"
    brain.save(save_path)
    loaded = Brain.load(save_path)
    rec3 = loaded.recall_text("golden retriever")
    print(f"\nAfter save/load, recall still returns: {_fmt_list(rec3['recalled_concepts'])}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
