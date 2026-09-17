"""Public BrainMemory API.

Learning writes into synaptic weights and grown connections.
Recall is recurrent pattern completion. There is no hidden answer table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from brainmemory.config import BrainConfig
from brainmemory.device import device_name, select_device
from brainmemory.encoding.concept_encoder import ConceptEncoder
from brainmemory.encoding.text_encoder import TextEncoder
from brainmemory.memory.consolidation import consolidate_step
from brainmemory.memory.engram import HippocampalTrace, TraceBook, engram_overlap
from brainmemory.memory.forgetting import forget_step
from brainmemory.memory.recall import decode_concepts, recall_pattern
from brainmemory.network import SparseNetwork
from brainmemory.neurons.neuron import REGION_NAMES
from brainmemory.regions.cortex import Cortex
from brainmemory.regions.hippocampus import Hippocampus
from brainmemory.simulation.sleep import sleep_cycle


class Brain:
    def __init__(self, config: BrainConfig | None = None, device: str | None = None):
        self.config = config or BrainConfig()
        self.device = select_device(device or self.config.device)
        self.net = SparseNetwork(self.config, self.device)
        self.encoder = ConceptEncoder(
            n=self.net.n,
            cortex_pool=self.net.cortex_exc(),
            hippo_pool=self.net.hippo_exc(),
            assoc_pool=self.net.assoc_exc(),
            bits_per_concept=self.config.memory.bits_per_concept,
            index_size=self.config.hippocampus.index_size,
            seed=self.config.seed,
        )
        self.text_encoder = TextEncoder()
        self.hippocampus = Hippocampus(self.net, self.encoder, self.config)
        self.cortex = Cortex(self.net, self.encoder, self.config)
        self.traces = TraceBook()
        self._last_engrams: dict[str, np.ndarray] = {}
        # diagnostic only: maps a label to the last engram neuron ids used for overlap plots
        # NOT consulted by recall()
        self.seen_docs: set[str] = set()
        self.seen_queries: set[str] = set()
        self.live_path: Path | None = None
        self.life: dict[str, Any] = {
            "cycles": 0,
            "episodes_total": 0,
            "growth_events": 0,
            "born_neurons": self.net.n,
        }

    # ------------------------------------------------------------------ learn
    def learn(
        self,
        concepts: Sequence[str],
        importance: float = 0.5,
        reward: float = 0.0,
        reconsolidate: bool = True,
    ) -> dict[str, Any]:
        tokens = _clean_concepts(concepts)
        if not tokens:
            return {"engram_size": 0, "synapses_grown": 0, "synapses_strengthened": 0}

        content = self.cortex.content(tokens)
        hippo_idx = self.hippocampus.form_episode(tokens) if self.hippocampus.enabled else np.zeros(0, np.int64)
        assoc = self.encoder.association_code(tokens)
        engram = np.unique(np.concatenate([p for p in (content, hippo_idx, assoc) if p.size]))

        if reconsolidate and self.traces:
            recalled, _ = recall_pattern(self.net, content, self.config, use_hippocampus=self.hippocampus.enabled)
            rec_idx = torch.nonzero(recalled > 0.5, as_tuple=False).view(-1).cpu().numpy()
            if rec_idx.size:
                # mix old active ensemble with new content (memory updating)
                engram = np.unique(np.concatenate([engram, rec_idx[: max(16, content.size)]]))

        novelty = self._novelty(engram)
        salience = float(
            np.clip(
                0.28 * novelty
                + 0.42 * float(importance)
                + 0.20 * float(reward)
                + 0.10,
                0.05,
                1.0,
            )
        )

        w0 = self.net.mean_weight()
        grown = self.net.bind_engram(
            engram,
            degree=self.config.memory.bind_degree,
            weight=0.28 + 0.35 * salience,
            t=self.net.t,
        )
        grown += self.net.grow_random_pairs(
            engram,
            probability=self.config.memory.grow_probability * salience,
            weight=0.20 + 0.25 * salience,
            t=self.net.t,
        )
        self.net.potentiate_engram(engram, delta=0.12 + 0.25 * salience)

        active = torch.zeros(self.net.n, device=self.device, dtype=torch.float32)
        tidx = torch.from_numpy(engram.astype(np.int64)).to(self.device)
        active[tidx] = 1.0
        self.net.state.activation = active
        self.net.state.v[tidx] = self.config.neuron.v_threshold + 0.5
        self.net.state.last_spike_time[tidx] = self.net.t

        hebb_stats = {"synapses_strengthened": 0, "synapses_weakened": 0, "mean_dw": 0.0}
        for k in range(int(self.config.memory.hebb_steps)):
            # slight sequential jitter so STDP has a causal direction within the ensemble
            if self.config.stdp.enabled:
                jitter = torch.rand(tidx.numel(), device=self.device) * 6.0
                times = torch.full((self.net.n,), -1.0e9, device=self.device)
                times[tidx] = self.net.t + jitter
                self.net.state.last_spike_time = times
                self.net.stdp(lr_scale=0.4 + 0.6 * salience)
            stats = self.net.hebbian(
                active,
                hippo_lr=self.hippocampus.learning_rate,
                cortex_lr=self.cortex.learning_rate,
                assoc_lr=self.config.association.learning_rate,
                salience=salience,
                depress=0.015,
            )
            hebb_stats["synapses_strengthened"] += stats["synapses_strengthened"]
            hebb_stats["synapses_weakened"] += stats["synapses_weakened"]
            hebb_stats["mean_dw"] += stats["mean_dw"]
            self.net.t += 1.0

        self.traces.add(
            HippocampalTrace(
                neuron_ids=engram.copy(),
                salience=salience,
                novelty=novelty,
                reward=float(reward),
                created_at=self.net.t,
            )
        )
        label = " ".join(tokens)
        self._last_engrams[label] = engram.copy()

        return {
            "engram_size": int(engram.size),
            "synapses_grown": int(grown),
            "synapses_strengthened": int(hebb_stats["synapses_strengthened"]),
            "synapses_weakened": int(hebb_stats["synapses_weakened"]),
            "mean_weight_delta": float(self.net.mean_weight() - w0),
            "novelty": float(novelty),
            "salience": float(salience),
            "activated_neurons": engram[:48].tolist(),
            "concepts": tokens,
        }

    def learn_text(self, text: str, importance: float = 0.5, reward: float = 0.0) -> dict[str, Any]:
        concepts = self.text_encoder.extract(text)
        result = self.learn(concepts, importance=importance, reward=reward)
        result["extracted_concepts"] = concepts
        result["text"] = text
        return result

    def explore(self, topic: str, steps: int = 6, **kwargs: Any) -> dict[str, Any]:
        """Autonomously fetch data about a topic and learn it synaptically."""
        from brainmemory.autonomy.learner import AutonomousLearner

        return AutonomousLearner(self, **kwargs).wander([topic], steps=steps)

    def wander(self, seeds: Sequence[str], steps: int = 8, **kwargs: Any) -> dict[str, Any]:
        from brainmemory.autonomy.learner import AutonomousLearner

        return AutonomousLearner(self, **kwargs).wander(seeds, steps=steps)

    def ingest(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Read a local file or folder and learn its contents."""
        from brainmemory.autonomy.learner import AutonomousLearner

        return AutonomousLearner(self, **kwargs).ingest_path(path)

    def grow(self, extra: int | None = None) -> dict[str, Any]:
        """Add neurons to THIS brain. Old memories keep their indices."""
        extra = int(extra if extra is not None else self.config.autonomy.grow_neurons)
        cap = int(self.config.autonomy.max_neurons)
        extra = min(extra, max(0, cap - self.net.n))
        if extra <= 0:
            return {"added": 0, "neurons": self.net.n, "synapses": self.net.syn.nnz, "capped": True}
        stats = self.net.expand(extra)
        self._sync_anatomy()
        self.life["growth_events"] = int(self.life.get("growth_events", 0)) + 1
        return stats

    def rehearse(self, sample: int = 12) -> dict[str, Any]:
        """Retrieval practice: recall known concepts so synapses deepen."""
        concepts = [c for c in self.encoder.known_concepts() if " " not in c and len(c) >= 4]
        if not concepts:
            return {"rehearsed": 0}
        rng = self.net.rng
        k = min(int(sample), len(concepts))
        idx = np.atleast_1d(rng.choice(len(concepts), size=k, replace=False))
        pick = [concepts[int(i)] for i in idx]
        conf = []
        for c in pick:
            rec = self.recall([c], top_k=8)
            conf.append(float(rec.get("confidence", 0.0)))
        return {"rehearsed": len(pick), "mean_confidence": float(np.mean(conf) if conf else 0.0)}

    def _sync_anatomy(self) -> None:
        self.encoder.n = self.net.n
        self.encoder.cortex_pool = self.net.cortex_exc()
        self.encoder.hippo_pool = self.net.hippo_exc()
        self.encoder.assoc_pool = self.net.assoc_exc()
        self.config.neurons = self.net.n

    # ----------------------------------------------------------------- recall
    def recall(
        self,
        concepts: Sequence[str],
        use_hippocampus: bool | None = None,
        top_k: int = 24,
    ) -> dict[str, Any]:
        tokens = _clean_concepts(concepts)
        cue_neurons = self.encoder.content_union(tokens)
        use_h = self.hippocampus.enabled if use_hippocampus is None else bool(use_hippocampus)
        spikes, dyn = recall_pattern(self.net, cue_neurons, self.config, use_hippocampus=use_h)
        scored = decode_concepts(self.encoder, spikes, self.config.recall.decode_threshold)
        cue_set = set(tokens)
        associated = [(c, s) for c, s in scored if c not in cue_set]
        recalled = [c for c, _ in scored[:top_k]]
        conf = float(np.mean([s for _, s in associated[:5]])) if associated else 0.0
        active_idx = torch.nonzero(spikes > 0.5, as_tuple=False).view(-1).cpu().numpy()

        if self.config.memory.retrieve_reconsolidates and active_idx.size:
            act = torch.zeros(self.net.n, device=self.device, dtype=torch.float32)
            act[torch.from_numpy(active_idx.astype(np.int64)).to(self.device)] = 1.0
            self.net.hebbian(
                act,
                hippo_lr=self.hippocampus.learning_rate * 0.05,
                cortex_lr=self.cortex.learning_rate * 0.25,
                assoc_lr=self.config.association.learning_rate * 0.2,
                salience=0.3,
                depress=0.0,
            )
            self.traces.bump_retrieval(active_idx.astype(np.int64), self.net.t)

        return {
            "recalled_concepts": recalled,
            "cue_concepts": tokens,
            "associated_concepts": [c for c, _ in associated[:top_k]],
            "scores": {c: s for c, s in scored[:top_k]},
            "activated_neurons": active_idx.tolist(),
            "confidence": conf,
            "engram_size": dyn["engram_size"],
            "activation_strength": dyn["activation_strength"],
            "sparsity": dyn["sparsity"],
            "used_hippocampus": use_h,
        }

    def recall_text(self, text: str, **kwargs: Any) -> dict[str, Any]:
        concepts = self.text_encoder.extract(text)
        result = self.recall(concepts, **kwargs)
        result["extracted_concepts"] = concepts
        return result

    # --------------------------------------------------------- sleep / time
    def sleep(self) -> dict[str, Any]:
        return sleep_cycle(self.net, self.traces, self.config)

    def consolidate(self, count: int | None = None) -> dict[str, Any]:
        return consolidate_step(self.net, self.traces, self.config, count=count)

    def tick(self, steps: int = 1) -> dict[str, Any]:
        return forget_step(self.net, self.config, steps=steps)

    # -------------------------------------------------------------- metrics
    def stats(self) -> dict[str, Any]:
        region_counts = {
            name: int(np.sum(self.net.region == rid)) for rid, name in REGION_NAMES.items()
        }
        inh = int(self.net.is_inhibitory.sum())
        active = int((self.net.state.activation > 0.5).sum().item())
        return {
            "neurons": self.net.n,
            "synapses": self.net.syn.nnz,
            "mean_weight": self.net.mean_weight(),
            "inhibitory_neurons": inh,
            "excitatory_neurons": self.net.n - inh,
            "regions": region_counts,
            "device": device_name(self.device),
            "time": float(self.net.t),
            "known_concepts": len(self.encoder.sdrs),
            "hippocampal_traces": len(self.traces),
            "active_neurons": active,
            "mean_activity_ema": float(self.net.state.activity_ema.mean().item()),
            "cycles": int(self.life.get("cycles", 0)),
            "episodes_total": int(self.life.get("episodes_total", 0)),
            "seen_docs": len(self.seen_docs),
        }

    def engram_overlap(self, memory_a: Sequence[str], memory_b: Sequence[str]) -> dict[str, float]:
        a = self._ensemble_for(memory_a)
        b = self._ensemble_for(memory_b)
        return engram_overlap(a, b)

    def visualize_memory(
        self,
        concepts: Sequence[str] | None = None,
        path: str | Path | None = None,
        max_edges: int = 400,
    ) -> str:
        from brainmemory.visualization.network import render_memory

        cue = _clean_concepts(concepts or [])
        result = self.recall(cue) if cue else None
        return render_memory(self, concepts=cue, recall=result, path=path, max_edges=max_edges)

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "0.1.0",
            "config": self.config.to_dict(),
            "device_type": self.device.type,
            "network": self.net.snapshot_state(),
            "encoder": self.encoder.state_dict(),
            "traces": self.traces.state_dict(),
            "last_engrams": {k: v.astype(np.int64) for k, v in self._last_engrams.items()},
            "life": dict(self.life),
            "seen_docs": sorted(self.seen_docs),
            "seen_queries": sorted(self.seen_queries),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Brain":
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
        cfg = BrainConfig.from_dict(payload["config"])
        snap = payload["network"]
        cfg.neurons = int(np.asarray(snap["region"]).shape[0])
        brain = cls(cfg, device=device)
        brain.net.t = float(snap["t"])
        brain.net.region = np.asarray(snap["region"], dtype=np.uint8)
        brain.net.is_inhibitory = np.asarray(snap["is_inhibitory"], dtype=np.bool_)
        st = brain.net.state
        st.v.copy_(torch.as_tensor(snap["v"], device=brain.device, dtype=torch.float32))
        st.threshold.copy_(torch.as_tensor(snap["threshold"], device=brain.device, dtype=torch.float32))
        st.activation.copy_(torch.as_tensor(snap["activation"], device=brain.device, dtype=torch.float32))
        st.refractory.copy_(torch.as_tensor(snap["refractory"], device=brain.device, dtype=torch.int16))
        st.excitability.copy_(torch.as_tensor(snap["excitability"], device=brain.device, dtype=torch.float32))
        st.last_spike_time.copy_(torch.as_tensor(snap["last_spike_time"], device=brain.device, dtype=torch.float32))
        st.activity_ema.copy_(torch.as_tensor(snap["activity_ema"], device=brain.device, dtype=torch.float32))
        st.region.copy_(torch.as_tensor(snap["region"], device=brain.device, dtype=torch.uint8))
        st.is_inhibitory.copy_(torch.as_tensor(snap["is_inhibitory"], device=brain.device, dtype=torch.bool))
        from brainmemory.synapses.synapse import SparseSynapses

        brain.net.syn = SparseSynapses.from_state(brain.net.n, snap["syn"], brain.device)
        brain.encoder.load_state(payload["encoder"])
        brain.traces.load_state(payload["traces"])
        brain._last_engrams = {
            k: np.asarray(v, dtype=np.int64) for k, v in payload.get("last_engrams", {}).items()
        }
        brain.life = dict(payload.get("life") or brain.life)
        brain.seen_docs = set(payload.get("seen_docs") or [])
        brain.seen_queries = set(payload.get("seen_queries") or [])
        brain._sync_anatomy()
        return brain

    @classmethod
    def live(cls, path: str | Path | None = None, config: BrainConfig | None = None, device: str | None = None) -> "Brain":
        """Open the one lifelong brain, or create it. Never starts over if the file exists."""
        dest = Path(path or os.environ.get("BRAINMEMORY_LIVE_PATH") or (Path.home() / ".brainmemory" / "live.pt"))
        if dest.exists():
            brain = cls.load(dest, device=device)
            brain.live_path = dest
            return brain
        brain = cls(config or BrainConfig.demo(), device=device)
        brain.live_path = dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        brain.save(dest)
        return brain

    def checkpoint(self, path: str | Path | None = None) -> Path:
        dest = Path(path or self.live_path or (Path.home() / ".brainmemory" / "live.pt"))
        self.live_path = dest
        return self.save(dest)

    # -------------------------------------------------------------- internals
    def _ensemble_for(self, concepts: Sequence[str]) -> np.ndarray:
        tokens = _clean_concepts(concepts)
        key = " ".join(tokens)
        if key in self._last_engrams:
            return self._last_engrams[key]
        return self.encoder.content_union(tokens)

    def _novelty(self, engram: np.ndarray) -> float:
        if engram.size == 0 or self.net.syn.nnz == 0:
            return 1.0
        ids = np.asarray(engram, dtype=np.int64)
        pre = self.net.syn.pre.detach().cpu().numpy()
        post = self.net.syn.post.detach().cpu().numpy()
        w = self.net.syn.weight.detach().cpu().numpy()
        intra = np.isin(pre, ids) & np.isin(post, ids)
        if not intra.any():
            return 1.0
        strong = float((w[intra] > 0.25).mean())
        return float(np.clip(1.0 - strong, 0.0, 1.0))


def _clean_concepts(concepts: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in concepts:
        t = " ".join(str(c).strip().lower().split())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
