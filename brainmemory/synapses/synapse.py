"""Sparse COO synapse store.

Memories live here: if these weights are deleted, learned associations vanish.
No dense N×N matrix is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def pack_keys(pre: Tensor, post: Tensor, n: int) -> np.ndarray:
    pre_np = pre.detach().to(dtype=torch.int64, device="cpu").numpy()
    post_np = post.detach().to(dtype=torch.int64, device="cpu").numpy()
    return pre_np.astype(np.int64) * np.int64(n) + post_np.astype(np.int64)


@dataclass
class SparseSynapses:
    n: int
    pre: Tensor
    post: Tensor
    weight: Tensor
    sign: Tensor
    plasticity: Tensor
    last_used: Tensor
    age: Tensor
    eligibility: Tensor
    keys: set[int]

    @property
    def nnz(self) -> int:
        return int(self.weight.numel())

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def current(self, activation: Tensor) -> Tensor:
        """Sparse synaptic current: I_j = sum_i S_ij * a_i * w_ij."""
        if self.nnz == 0:
            return torch.zeros_like(activation)
        pre_a = activation[self.pre]
        contrib = self.weight * self.sign * pre_a
        out = torch.zeros_like(activation)
        out.index_add_(0, self.post, contrib)
        return out

    def add(
        self,
        pre: Tensor,
        post: Tensor,
        weight: Tensor | None = None,
        sign: Tensor | None = None,
        plasticity: Tensor | None = None,
        t: float = 0.0,
    ) -> int:
        """Append synapses that do not already exist. Returns number added."""
        if pre.numel() == 0:
            return 0
        pre = pre.to(dtype=torch.int64)
        post = post.to(dtype=torch.int64)
        mask = pre != post
        pre, post = pre[mask], post[mask]
        if weight is not None:
            weight = weight[mask.to(weight.device)]
        if sign is not None:
            sign = sign[mask.to(sign.device)]
        if plasticity is not None:
            plasticity = plasticity[mask.to(plasticity.device)]
        if pre.numel() == 0:
            return 0
        keys = pack_keys(pre, post, self.n)
        keep_np = np.fromiter((int(k) not in self.keys for k in keys), dtype=np.bool_, count=keys.size)
        if not keep_np.any():
            return 0
        keep = torch.from_numpy(np.ascontiguousarray(keep_np)).to(pre.device)
        pre, post = pre[keep], post[keep]
        new_keys = keys[keep_np]
        added = int(pre.numel())
        keep_w = keep.to(weight.device) if weight is not None else keep
        if weight is None:
            w = torch.full((added,), 0.08, device=self.device, dtype=torch.float32)
        else:
            w = weight[keep_w].to(self.device, torch.float32)
        if sign is None:
            s = torch.ones(added, device=self.device, dtype=torch.float32)
        else:
            s = sign[keep.to(sign.device)].to(self.device, torch.float32)
        if plasticity is None:
            p = torch.ones(added, device=self.device, dtype=torch.float32)
        else:
            p = plasticity[keep.to(plasticity.device)].to(self.device, torch.float32)
        t_vec = torch.full((added,), float(t), device=self.device, dtype=torch.float32)
        z = torch.zeros(added, device=self.device, dtype=torch.float32)
        self.pre = torch.cat([self.pre, pre.to(self.device)])
        self.post = torch.cat([self.post, post.to(self.device)])
        self.weight = torch.cat([self.weight, w])
        self.sign = torch.cat([self.sign, s])
        self.plasticity = torch.cat([self.plasticity, p])
        self.last_used = torch.cat([self.last_used, t_vec])
        self.age = torch.cat([self.age, z])
        self.eligibility = torch.cat([self.eligibility, z])
        self.keys.update(int(k) for k in new_keys)
        return added

    def prune(self, threshold: float, protect: Tensor | None = None) -> int:
        mag = self.weight.abs()
        keep = mag >= float(threshold)
        if protect is not None:
            keep = keep | protect
        if bool(keep.all()):
            return 0
        removed = int((~keep).sum().item())
        if removed == 0:
            return 0
        dropped = pack_keys(self.pre[~keep], self.post[~keep], self.n)
        for k in dropped:
            self.keys.discard(int(k))
        self.pre = self.pre[keep]
        self.post = self.post[keep]
        self.weight = self.weight[keep]
        self.sign = self.sign[keep]
        self.plasticity = self.plasticity[keep]
        self.last_used = self.last_used[keep]
        self.age = self.age[keep]
        self.eligibility = self.eligibility[keep]
        return removed

    def clip(self, w_min: float, w_max: float) -> None:
        self.weight.clamp_(min=float(w_min), max=float(w_max))

    def mark_used(self, pre_active: Tensor, post_active: Tensor, t: float) -> None:
        used = (pre_active[self.pre] > 0) & (post_active[self.post] > 0)
        if used.any():
            self.last_used = torch.where(
                used, torch.full_like(self.last_used, float(t)), self.last_used
            )
            self.eligibility = torch.where(
                used, torch.clamp(self.eligibility + 0.2, max=1.0), self.eligibility * 0.95
            )

    def cpu_state(self) -> dict[str, np.ndarray]:
        return {
            "pre": self.pre.detach().to("cpu", torch.int64).numpy(),
            "post": self.post.detach().to("cpu", torch.int64).numpy(),
            "weight": self.weight.detach().to("cpu", torch.float32).numpy(),
            "sign": self.sign.detach().to("cpu", torch.float32).numpy(),
            "plasticity": self.plasticity.detach().to("cpu", torch.float32).numpy(),
            "last_used": self.last_used.detach().to("cpu", torch.float32).numpy(),
            "age": self.age.detach().to("cpu", torch.float32).numpy(),
            "eligibility": self.eligibility.detach().to("cpu", torch.float32).numpy(),
        }

    @classmethod
    def from_state(cls, n: int, data: dict[str, np.ndarray], device: torch.device) -> "SparseSynapses":
        pre = torch.as_tensor(data["pre"], device=device, dtype=torch.int64)
        post = torch.as_tensor(data["post"], device=device, dtype=torch.int64)
        syn = cls(
            n=n,
            pre=pre,
            post=post,
            weight=torch.as_tensor(data["weight"], device=device, dtype=torch.float32),
            sign=torch.as_tensor(data["sign"], device=device, dtype=torch.float32),
            plasticity=torch.as_tensor(data["plasticity"], device=device, dtype=torch.float32),
            last_used=torch.as_tensor(data["last_used"], device=device, dtype=torch.float32),
            age=torch.as_tensor(data["age"], device=device, dtype=torch.float32),
            eligibility=torch.as_tensor(data["eligibility"], device=device, dtype=torch.float32),
            keys=set(),
        )
        syn.keys = set(int(k) for k in pack_keys(syn.pre, syn.post, n))
        return syn

    @classmethod
    def empty(cls, n: int, device: torch.device) -> "SparseSynapses":
        z64 = torch.zeros(0, device=device, dtype=torch.int64)
        z = torch.zeros(0, device=device, dtype=torch.float32)
        return cls(
            n=n,
            pre=z64.clone(),
            post=z64.clone(),
            weight=z.clone(),
            sign=z.clone(),
            plasticity=z.clone(),
            last_used=z.clone(),
            age=z.clone(),
            eligibility=z.clone(),
            keys=set(),
        )
