"""YAML/dataclass configuration for BrainMemory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HippocampusConfig:
    fraction: float = 0.20
    learning_rate: float = 0.18
    index_size: int = 72


@dataclass
class CortexConfig:
    fraction: float = 0.70
    learning_rate: float = 0.02


@dataclass
class AssociationConfig:
    fraction: float = 0.10
    learning_rate: float = 0.05


@dataclass
class SynapseConfig:
    avg_connections: int = 64
    w_init_mean: float = 0.04
    w_init_std: float = 0.02
    w_max: float = 1.0
    w_min: float = 0.0
    inhibitory_strength: float = 0.25
    max_degree: int = 480


@dataclass
class NeuronConfig:
    model: str = "lif"
    tau_mem: float = 20.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_steps: int = 2
    dt: float = 1.0


@dataclass
class MemoryConfig:
    bits_per_concept: int = 40
    sparsity: float = 0.02
    bind_degree: int = 22
    grow_probability: float = 0.28
    hebb_steps: int = 6
    retrieve_reconsolidates: bool = True


@dataclass
class RecallConfig:
    steps: int = 10
    clamp_steps: int = 3
    k_wta: int = 700
    decode_threshold: float = 0.20
    clamp_current: float = 3.0


@dataclass
class SleepConfig:
    replay_count: int = 48
    replay_steps: int = 6
    downscale: float = 0.985
    cortex_lr_boost: float = 6.0


@dataclass
class ForgettingConfig:
    decay_rate: float = 0.008
    unused_decay: float = 0.02
    pruning_threshold: float = 0.004
    unused_window: int = 30


@dataclass
class STDPConfig:
    enabled: bool = True
    a_plus: float = 0.04
    a_minus: float = 0.042
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    window: float = 40.0


@dataclass
class AutonomyConfig:
    language: str = "de"
    max_pages: int = 10
    sleep_every: int = 3
    min_novelty: float = 0.12
    rate_limit_s: float = 0.8
    follow_links: int = 3
    max_concepts: int = 14
    wikipedia: bool = True
    local_dir: str | None = None
    grow_every_cycles: int = 1
    grow_neurons: int = 64
    max_neurons: int = 200000
    pages_per_cycle: int = 4


@dataclass
class BrainConfig:
    neurons: int = 10000
    seed: int = 42
    inhibitory_fraction: float = 0.18
    device: str | None = None
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    cortex: CortexConfig = field(default_factory=CortexConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    synapses: SynapseConfig = field(default_factory=SynapseConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)

    @classmethod
    def tiny(cls, seed: int = 1) -> "BrainConfig":
        """Small network for unit tests (~400 neurons)."""
        cfg = cls(neurons=400, seed=seed, inhibitory_fraction=0.15)
        cfg.synapses.avg_connections = 24
        cfg.synapses.max_degree = 80
        cfg.memory.bits_per_concept = 14
        cfg.memory.bind_degree = 10
        cfg.memory.hebb_steps = 6
        cfg.hippocampus.index_size = 20
        cfg.recall.k_wta = 90
        cfg.recall.steps = 10
        cfg.recall.decode_threshold = 0.22
        cfg.sleep.replay_count = 12
        return cfg

    @classmethod
    def small(cls, seed: int = 7) -> "BrainConfig":
        """Development-scale network for integration tests (~1500 neurons)."""
        cfg = cls(neurons=1500, seed=seed, inhibitory_fraction=0.16)
        cfg.synapses.avg_connections = 36
        cfg.synapses.max_degree = 160
        cfg.memory.bits_per_concept = 22
        cfg.memory.bind_degree = 16
        cfg.memory.hebb_steps = 6
        cfg.hippocampus.index_size = 40
        cfg.recall.k_wta = 80
        cfg.recall.decode_threshold = 0.22
        cfg.recall.steps = 12
        cfg.recall.clamp_steps = 4
        cfg.sleep.replay_count = 20
        return cfg

    @classmethod
    def demo(cls, seed: int = 42) -> "BrainConfig":
        return cls(neurons=10000, seed=seed)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BrainConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BrainConfig":
        brain = dict(raw.get("brain", {}))
        cfg = cls(
            neurons=int(brain.get("neurons", 10000)),
            seed=int(brain.get("seed", 42)),
            inhibitory_fraction=float(brain.get("inhibitory_fraction", 0.18)),
            device=brain.get("device"),
        )
        cfg.hippocampus = _fill(HippocampusConfig, raw.get("hippocampus"))
        cfg.cortex = _fill(CortexConfig, raw.get("cortex"))
        cfg.association = _fill(AssociationConfig, raw.get("association"))
        cfg.synapses = _fill(SynapseConfig, raw.get("synapses"))
        cfg.neuron = _fill(NeuronConfig, raw.get("neuron"))
        cfg.memory = _fill(MemoryConfig, raw.get("memory"))
        cfg.recall = _fill(RecallConfig, raw.get("recall"))
        cfg.sleep = _fill(SleepConfig, raw.get("sleep"))
        cfg.forgetting = _fill(ForgettingConfig, raw.get("forgetting"))
        cfg.stdp = _fill(STDPConfig, raw.get("stdp"))
        cfg.autonomy = _fill(AutonomyConfig, raw.get("autonomy"))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        neurons = d.pop("neurons")
        seed = d.pop("seed")
        inh = d.pop("inhibitory_fraction")
        device = d.pop("device")
        return {
            "brain": {
                "neurons": neurons,
                "seed": seed,
                "inhibitory_fraction": inh,
                "device": device,
            },
            **d,
        }

    def scaled(self, neurons: int) -> "BrainConfig":
        cfg = replace(self, neurons=int(neurons))
        return cfg


def _fill(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**fields)
