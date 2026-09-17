"""Sparse recurrent network: LIF neurons + COO synapses.

This is the substrate of memory. Learning changes `syn.weight` (and grows
edges). Recall is recurrent propagation, not table lookup.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from brainmemory.config import BrainConfig
from brainmemory.neurons.lif import LIFNeuronModel
from brainmemory.neurons.neuron import (
    REGION_ASSOCIATION,
    REGION_CORTEX,
    REGION_HIPPOCAMPUS,
    NeuronState,
    allocate_state,
    neuron_view,
)
from brainmemory.synapses.hebbian import hebbian_update, incoming_scale
from brainmemory.synapses.stdp import stdp_update
from brainmemory.synapses.synapse import SparseSynapses, pack_keys


class SparseNetwork:
    def __init__(self, cfg: BrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n = int(cfg.neurons)
        self.t = 0.0
        self.rng = np.random.RandomState(cfg.seed)
        self.torch_gen = torch.Generator(device="cpu")
        self.torch_gen.manual_seed(cfg.seed)

        self.region, self.is_inhibitory, self.slices = _partition_regions(cfg, self.n, self.rng)
        self.state: NeuronState = allocate_state(
            self.n,
            torch.from_numpy(self.region),
            torch.from_numpy(self.is_inhibitory),
            cfg.neuron.v_threshold,
            device,
        )
        self.model = LIFNeuronModel(cfg.neuron)
        self.syn = _init_synapses(cfg, self.n, self.is_inhibitory, self.rng, device)
        self._degree_cache: Tensor | None = None

    @property
    def excitatory_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.is_inhibitory)

    def region_mask(self, region: int, excitatory: bool = True) -> np.ndarray:
        m = self.region == region
        if excitatory:
            m = m & (~self.is_inhibitory)
        return np.flatnonzero(m)

    def hippo_exc(self) -> np.ndarray:
        return self.region_mask(REGION_HIPPOCAMPUS)

    def cortex_exc(self) -> np.ndarray:
        return self.region_mask(REGION_CORTEX)

    def assoc_exc(self) -> np.ndarray:
        return self.region_mask(REGION_ASSOCIATION)

    def reset(self) -> None:
        self.state.reset_activity()

    def inject(self, idx: np.ndarray | Tensor, current: float) -> Tensor:
        I = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        if isinstance(idx, np.ndarray):
            if idx.size == 0:
                return I
            tidx = torch.from_numpy(idx.astype(np.int64)).to(self.device)
        else:
            tidx = idx.to(self.device, dtype=torch.int64)
            if tidx.numel() == 0:
                return I
        I[tidx] = float(current)
        return I

    def step(self, external: Tensor | None = None) -> Tensor:
        syn_I = self.syn.current(self.state.activation)
        I = syn_I if external is None else syn_I + external
        spikes = self.model.step(self.state, I, self.t)
        self.t += self.cfg.neuron.dt
        self.syn.age.add_(self.cfg.neuron.dt)
        return spikes

    def k_wta(self, current: Tensor, k: int) -> Tensor:
        """Approximate inhibition: keep top-k excitatory currents, plus any inhibitory that exceed threshold."""
        k = max(1, min(int(k), self.n))
        inh = self.state.is_inhibitory
        scores = current.clone()
        scores = torch.where(inh, torch.full_like(scores, -1.0e9), scores)
        if k >= self.n:
            winners = torch.ones(self.n, device=self.device, dtype=torch.bool)
        else:
            topv, topi = torch.topk(scores, k)
            winners = torch.zeros(self.n, device=self.device, dtype=torch.bool)
            # ignore padded -inf
            valid = topv > -1.0e8
            winners[topi[valid]] = True
        return winners

    def excitatory_current(self, activation: Tensor) -> Tensor:
        if self.syn.nnz == 0:
            return torch.zeros_like(activation)
        pre_a = activation[self.syn.pre]
        contrib = self.syn.weight * pre_a
        contrib = torch.where(self.syn.sign > 0, contrib, torch.zeros_like(contrib))
        out = torch.zeros_like(activation)
        out.index_add_(0, self.syn.post, contrib)
        return out

    def run_clamped(
        self,
        clamp_idx: np.ndarray,
        steps: int,
        clamp_steps: int,
        k_wta: int,
        clamp_current: float,
        use_hippocampus: bool = True,
    ) -> Tensor:
        """Rate-based attractor completion on excitatory weights.

        LIF refractory would blank the ensemble every other step, which destroys
        pattern completion, so recall uses thresholded recurrent current (still
        purely synaptic — not a lookup table).
        """
        self.reset()
        clamp = torch.from_numpy(np.unique(clamp_idx.astype(np.int64))).to(self.device)
        hippo = torch.from_numpy(self.region).to(self.device) == REGION_HIPPOCAMPUS
        inh = self.state.is_inhibitory
        active = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        if clamp.numel():
            active[clamp] = 1.0
        # Attractors stay sparse even if the brain has grown. Scaling k with N
        # drowns old engrams in random cells.
        k = min(max(24, int(k_wta)), 280, self.n)
        for s in range(int(steps)):
            I_syn = self.excitatory_current(active)
            if not use_hippocampus:
                I_syn = torch.where(hippo, torch.zeros_like(I_syn), I_syn)
            I_syn = torch.where(inh, torch.zeros_like(I_syn), I_syn)
            I = I_syn.clone()
            if s < int(clamp_steps) and clamp.numel():
                I[clamp] = I[clamp] + float(clamp_current)
            pos = I_syn > 0
            if pos.any():
                peak = float(I_syn[pos].max().item())
                rel = I_syn >= max(0.05 * peak, 0.04)
            else:
                rel = torch.zeros_like(I_syn, dtype=torch.bool)
            winners = self.k_wta(I, k) & (rel | ((s < int(clamp_steps)) & (I > 0)))
            if clamp.numel():
                winners[clamp] = True
            active = winners.to(dtype=torch.float32)
            self.state.activation = active
            self.state.v = torch.clamp(I, min=0.0)
            self.t += self.cfg.neuron.dt
        # final settled state only — OR-ing every step piles noise in large nets
        return active

    def bind_engram(self, neurons: np.ndarray, degree: int, weight: float, t: float) -> int:
        """Grow recurrent synapses among an ensemble (structural plasticity)."""
        ids = np.unique(neurons.astype(np.int64))
        ids = ids[(ids >= 0) & (ids < self.n)]
        if ids.size < 2:
            return 0
        # do not bind onto inhibitory cells as sources of memory
        exc = ids[~self.is_inhibitory[ids]]
        if exc.size < 2:
            return 0
        deg = max(1, min(int(degree), exc.size - 1))
        # respect max_degree approximately by sampling
        rng = self.rng
        pre_list = []
        post_list = []
        for i, src in enumerate(exc):
            # sample `deg` targets other than self
            # cheap: random permutation slice
            choice = rng.randint(0, exc.size - 1, size=deg)
            choice = choice + (choice >= i).astype(np.int64)
            tgt = exc[choice % exc.size]
            pre_list.append(np.full(deg, src, dtype=np.int64))
            post_list.append(tgt.astype(np.int64))
        pre = torch.from_numpy(np.concatenate(pre_list))
        post = torch.from_numpy(np.concatenate(post_list))
        w = torch.full((pre.numel(),), float(weight), dtype=torch.float32)
        return self.syn.add(pre, post, weight=w, t=t)

    def potentiate_engram(self, neurons: np.ndarray, delta: float) -> int:
        """Raise existing excitatory weights inside an ensemble."""
        if self.syn.nnz == 0 or neurons.size < 2:
            return 0
        ids = torch.from_numpy(np.unique(neurons.astype(np.int64))).to(self.device)
        member = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        valid = (ids >= 0) & (ids < self.n)
        member[ids[valid]] = True
        pre_in = member[self.syn.pre]
        post_in = member[self.syn.post]
        intra = pre_in & post_in & (self.syn.sign > 0)
        if not intra.any():
            return 0
        self.syn.weight[intra] = self.syn.weight[intra] + float(delta)
        self.syn.clip(self.cfg.synapses.w_min, self.cfg.synapses.w_max)
        return int(intra.sum().item())

    def grow_random_pairs(self, neurons: np.ndarray, probability: float, weight: float, t: float) -> int:
        ids = np.unique(neurons.astype(np.int64))
        ids = ids[(ids >= 0) & (ids < self.n) & (~self.is_inhibitory[ids])]
        k = ids.size
        if k < 2:
            return 0
        n_cand = int(probability * k * (k - 1))
        n_cand = max(0, min(n_cand, k * min(32, k - 1)))
        if n_cand == 0:
            return 0
        pre = torch.from_numpy(self.rng.choice(ids, size=n_cand).astype(np.int64))
        post = torch.from_numpy(self.rng.choice(ids, size=n_cand).astype(np.int64))
        w = torch.full((n_cand,), float(weight), dtype=torch.float32)
        return self.syn.add(pre, post, weight=w, t=t)

    def region_learning_rates(self, hippo_lr: float, cortex_lr: float, assoc_lr: float) -> Tensor:
        region = torch.from_numpy(self.region.astype(np.int64)).to(self.device)
        pre_r = region[self.syn.pre]
        post_r = region[self.syn.post]
        lr = torch.full((self.syn.nnz,), float(cortex_lr), device=self.device)
        hippo_involved = (pre_r == REGION_HIPPOCAMPUS) | (post_r == REGION_HIPPOCAMPUS)
        assoc_involved = (pre_r == REGION_ASSOCIATION) | (post_r == REGION_ASSOCIATION)
        lr = torch.where(assoc_involved, torch.full_like(lr, float(assoc_lr)), lr)
        lr = torch.where(hippo_involved, torch.full_like(lr, float(hippo_lr)), lr)
        return lr

    def hebbian(
        self,
        active: Tensor,
        hippo_lr: float,
        cortex_lr: float,
        assoc_lr: float,
        salience: float = 1.0,
        depress: float = 0.02,
    ) -> dict[str, float]:
        scale = 0.35 + 0.65 * float(np.clip(salience, 0.0, 1.0))
        lr = self.region_learning_rates(hippo_lr, cortex_lr, assoc_lr) * scale
        stats = hebbian_update(
            self.syn,
            active,
            active,
            lr,
            w_max=self.cfg.synapses.w_max,
            w_min=self.cfg.synapses.w_min,
            t=self.t,
            depress=depress,
        )
        incoming_scale(self.syn, target=max(6.0, self.cfg.memory.bind_degree * 0.45))
        return stats

    def stdp(self, lr_scale: float = 1.0) -> dict[str, float]:
        return stdp_update(
            self.syn,
            self.state.last_spike_time,
            self.cfg.stdp,
            w_max=self.cfg.synapses.w_max,
            w_min=self.cfg.synapses.w_min,
            lr_scale=lr_scale,
        )

    def decay(self, rate: float, unused_rate: float, unused_window: float) -> None:
        if self.syn.nnz == 0:
            return
        self.syn.weight.mul_(1.0 - float(rate))
        stale = (self.t - self.syn.last_used) > float(unused_window)
        if stale.any():
            extra = torch.ones_like(self.syn.weight)
            extra[stale] = 1.0 - float(unused_rate)
            self.syn.weight.mul_(extra)
        self.syn.clip(self.cfg.synapses.w_min, self.cfg.synapses.w_max)

    def prune(self) -> int:
        return self.syn.prune(self.cfg.forgetting.pruning_threshold)

    def downscale(self, factor: float) -> None:
        self.syn.weight.mul_(float(factor))

    def zero_weights(self) -> None:
        self.syn.weight.zero_()

    def mean_weight(self) -> float:
        if self.syn.nnz == 0:
            return 0.0
        exc = self.syn.sign > 0
        if not exc.any():
            return float(self.syn.weight.mean().item())
        return float(self.syn.weight[exc].mean().item())

    def get_neuron(self, idx: int) -> dict:
        return neuron_view(self.state, idx)

    def expand(self, extra: int) -> dict[str, int]:
        """Append neurons (neurogenesis). Old indices stay valid, so old memories remain."""
        extra = int(max(0, extra))
        if extra == 0:
            return {"added": 0, "neurons": self.n, "synapses": self.syn.nnz}
        old_n = self.n
        new_n = old_n + extra
        rng = self.rng
        frac = float(self.cfg.inhibitory_fraction)
        f_h = self.cfg.hippocampus.fraction
        f_c = self.cfg.cortex.fraction
        f_a = self.cfg.association.fraction
        s = f_h + f_c + f_a
        new_region = rng.choice(
            np.array([REGION_HIPPOCAMPUS, REGION_CORTEX, REGION_ASSOCIATION], dtype=np.uint8),
            size=extra,
            p=[f_h / s, f_c / s, f_a / s],
        ).astype(np.uint8)
        new_inh = rng.rand(extra) < frac

        def _pad(t: Tensor, fill, dtype=None) -> Tensor:
            dt = dtype or t.dtype
            ext = torch.full((extra,), fill, device=self.device, dtype=dt)
            return torch.cat([t, ext], dim=0)

        st = self.state
        thr = float(self.cfg.neuron.v_threshold)
        st.v = _pad(st.v, 0.0)
        st.threshold = _pad(st.threshold, thr)
        st.activation = _pad(st.activation, 0.0)
        st.refractory = _pad(st.refractory, 0, dtype=torch.int16)
        st.excitability = _pad(st.excitability, 1.0)
        st.last_spike_time = _pad(st.last_spike_time, -1.0e9)
        st.activity_ema = _pad(st.activity_ema, 0.0)
        st.region = torch.cat(
            [st.region, torch.from_numpy(new_region).to(device=self.device, dtype=torch.uint8)]
        )
        st.is_inhibitory = torch.cat(
            [st.is_inhibitory, torch.from_numpy(new_inh).to(device=self.device, dtype=torch.bool)]
        )

        self.region = np.concatenate([self.region, new_region])
        self.is_inhibitory = np.concatenate([self.is_inhibitory, new_inh])
        self.n = new_n
        self.cfg.neurons = new_n
        self.syn.n = new_n
        self.syn.keys = set(int(k) for k in pack_keys(self.syn.pre, self.syn.post, new_n))

        avg = max(8, int(self.cfg.synapses.avg_connections))
        new_ids = np.arange(old_n, new_n, dtype=np.int64)
        all_ids = np.arange(new_n, dtype=np.int64)
        pre = rng.choice(new_ids, size=extra * avg).astype(np.int64)
        post = rng.choice(all_ids, size=extra * avg).astype(np.int64)
        back_pre = rng.choice(np.arange(old_n, dtype=np.int64), size=extra * max(4, avg // 4)).astype(np.int64)
        back_post = rng.choice(new_ids, size=back_pre.size).astype(np.int64)
        pre = np.concatenate([pre, back_pre])
        post = np.concatenate([post, back_post])
        w = np.abs(rng.normal(self.cfg.synapses.w_init_mean, self.cfg.synapses.w_init_std, size=pre.size)).astype(
            np.float32
        )
        sign = np.where(self.is_inhibitory[pre], -1.0, 1.0).astype(np.float32)
        added = self.syn.add(
            torch.from_numpy(pre),
            torch.from_numpy(post),
            weight=torch.from_numpy(w),
            sign=torch.from_numpy(sign),
            t=self.t,
        )
        return {"added": extra, "synapses_grown": int(added), "neurons": self.n, "synapses": self.syn.nnz}

    def snapshot_state(self) -> dict:
        s = self.state
        return {
            "v": s.v.detach().cpu().numpy(),
            "threshold": s.threshold.detach().cpu().numpy(),
            "activation": s.activation.detach().cpu().numpy(),
            "refractory": s.refractory.detach().cpu().numpy(),
            "excitability": s.excitability.detach().cpu().numpy(),
            "last_spike_time": s.last_spike_time.detach().cpu().numpy(),
            "activity_ema": s.activity_ema.detach().cpu().numpy(),
            "region": self.region,
            "is_inhibitory": self.is_inhibitory,
            "t": self.t,
            "syn": self.syn.cpu_state(),
        }


def _partition_regions(cfg: BrainConfig, n: int, rng: np.random.RandomState):
    f_h = cfg.hippocampus.fraction
    f_c = cfg.cortex.fraction
    f_a = cfg.association.fraction
    s = f_h + f_c + f_a
    f_h, f_c, f_a = f_h / s, f_c / s, f_a / s
    n_h = max(8, int(n * f_h))
    n_a = max(8, int(n * f_a))
    n_c = max(8, n - n_h - n_a)
    region = np.empty(n, dtype=np.uint8)
    region[:n_h] = REGION_HIPPOCAMPUS
    region[n_h : n_h + n_c] = REGION_CORTEX
    region[n_h + n_c :] = REGION_ASSOCIATION
    slices = {
        "hippocampus": slice(0, n_h),
        "cortex": slice(n_h, n_h + n_c),
        "association": slice(n_h + n_c, n),
    }
    inh = np.zeros(n, dtype=np.bool_)
    frac = float(cfg.inhibitory_fraction)
    for sl in slices.values():
        idx = np.arange(sl.start, sl.stop)
        k = max(1, int(len(idx) * frac))
        chosen = rng.choice(idx, size=k, replace=False)
        inh[chosen] = True
    return region, inh, slices


def _init_synapses(
    cfg: BrainConfig,
    n: int,
    is_inh: np.ndarray,
    rng: np.random.RandomState,
    device: torch.device,
) -> SparseSynapses:
    avg = int(cfg.synapses.avg_connections)
    nnz = int(n * avg)
    pre = rng.randint(0, n, size=nnz, dtype=np.int64)
    post = rng.randint(0, n, size=nnz, dtype=np.int64)
    mask = pre != post
    pre, post = pre[mask], post[mask]
    keys = pre * n + post
    _, uniq = np.unique(keys, return_index=True)
    pre, post = pre[uniq], post[uniq]
    w = rng.normal(cfg.synapses.w_init_mean, cfg.synapses.w_init_std, size=pre.size).astype(np.float32)
    w = np.clip(np.abs(w), 0.005, cfg.synapses.w_max * 0.3)
    sign = np.where(is_inh[pre], -1.0, 1.0).astype(np.float32)
    w = np.where(sign < 0, np.clip(w * 0 + cfg.synapses.inhibitory_strength, 0.05, 0.8), w)
    syn = SparseSynapses.empty(n, device)
    syn.add(
        torch.from_numpy(pre),
        torch.from_numpy(post),
        weight=torch.from_numpy(w),
        sign=torch.from_numpy(sign),
        t=0.0,
    )
    return syn
