"""Evaluation metrics for BrainMemory experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from brainmemory.brain import Brain


def recall_accuracy(recalled: Sequence[str], expected: Sequence[str]) -> float:
    exp = {e.lower() for e in expected}
    if not exp:
        return 0.0
    rec = {r.lower() for r in recalled}
    return len(exp & rec) / float(len(exp))


def pattern_completion_accuracy(
    recalled: Sequence[str], missing: Sequence[str], cue: Sequence[str]
) -> float:
    return recall_accuracy(recalled, missing)


def false_recall_rate(recalled: Sequence[str], allowed: Sequence[str]) -> float:
    rec = [r.lower() for r in recalled]
    allow = {a.lower() for a in allowed}
    if not rec:
        return 0.0
    false = sum(1 for r in rec if r not in allow)
    return false / float(len(rec))


def engram_sparsity(engram_size: int, n_neurons: int) -> float:
    if n_neurons <= 0:
        return 0.0
    return float(engram_size) / float(n_neurons)


def completion_from_result(result: dict, expected: Sequence[str], cue: Sequence[str]) -> dict[str, float]:
    rec = result.get("recalled_concepts", [])
    missing = [e for e in expected if e.lower() not in {c.lower() for c in cue}]
    allowed = list(expected) + list(cue)
    return {
        "recall_accuracy": recall_accuracy(rec, expected),
        "pattern_completion_accuracy": recall_accuracy(rec, missing) if missing else 1.0,
        "false_recall_rate": false_recall_rate(rec, allowed),
        "confidence": float(result.get("confidence", 0.0)),
        "engram_sparsity": float(result.get("sparsity", 0.0)),
        "activation_strength": float(result.get("activation_strength", 0.0)),
    }


def network_metrics(brain: Brain) -> dict[str, float]:
    st = brain.stats()
    return {
        "average_synaptic_weight": float(st["mean_weight"]),
        "synapse_count": float(st["synapses"]),
        "neuronal_activity": float(st["mean_activity_ema"]),
        "known_concepts": float(st["known_concepts"]),
        "hippocampal_traces": float(st["hippocampal_traces"]),
        "consolidation_strength": float(
            np.mean([t.consolidation for t in brain.traces.traces]) if brain.traces.traces else 0.0
        ),
    }


def memory_retention(before: dict, after: dict, expected: Sequence[str]) -> float:
    b = recall_accuracy(before.get("recalled_concepts", []), expected)
    a = recall_accuracy(after.get("recalled_concepts", []), expected)
    return float(a / b) if b > 1e-9 else float(a > 0)
