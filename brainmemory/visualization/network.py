"""Sampled local-graph visualization. Never attempts to draw millions of edges."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from brainmemory.neurons.neuron import REGION_ASSOCIATION, REGION_CORTEX, REGION_HIPPOCAMPUS


def render_memory(brain, concepts: Sequence[str], recall: dict | None, path: str | Path | None, max_edges: int = 400) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    out = Path(path) if path else Path("artifacts") / "memory.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    region = brain.net.region
    n = brain.net.n
    rng = np.random.RandomState(0)
    xs = rng.rand(n)
    ys = rng.rand(n)
    xs[region == REGION_HIPPOCAMPUS] = xs[region == REGION_HIPPOCAMPUS] * 0.30
    xs[region == REGION_CORTEX] = 0.35 + xs[region == REGION_CORTEX] * 0.40
    xs[region == REGION_ASSOCIATION] = 0.80 + xs[region == REGION_ASSOCIATION] * 0.20

    active = set()
    if recall:
        active = set(int(i) for i in recall.get("activated_neurons", []))
    cue_ids = set(int(i) for i in brain.encoder.content_union(list(concepts)).tolist()) if concepts else set()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    # subsample background
    bg = rng.choice(n, size=min(n, 1200), replace=False)
    ax.scatter(xs[bg], ys[bg], s=4, c="#d0d4dc", alpha=0.4, linewidths=0, label="neurons")
    if cue_ids:
        cidx = np.fromiter(cue_ids, dtype=np.int64)
        ax.scatter(xs[cidx], ys[cidx], s=18, c="#f4c430", label="cue", zorder=3)
    if active:
        aidx = np.fromiter(active, dtype=np.int64)
        ax.scatter(xs[aidx], ys[aidx], s=10, c="#e4572e", alpha=0.7, label="active", zorder=2)
    ax.set_title("Regions (left hippo · mid cortex · right assoc)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    # sampled strong local graph around active/cue neurons
    ax2 = axes[1]
    focus = np.array(sorted(active | cue_ids), dtype=np.int64)
    if focus.size == 0:
        focus = rng.choice(n, size=min(80, n), replace=False)
    G = nx.DiGraph()
    pre = brain.net.syn.pre.detach().cpu().numpy()
    post = brain.net.syn.post.detach().cpu().numpy()
    w = brain.net.syn.weight.detach().cpu().numpy()
    intra = np.isin(pre, focus) & np.isin(post, focus)
    cand = np.flatnonzero(intra)
    if cand.size:
        order = np.argsort(-w[cand])[: max_edges]
        pick = cand[order]
        for i in pick:
            if w[i] < 0.05:
                continue
            G.add_edge(int(pre[i]), int(post[i]), weight=float(w[i]))
    for node in list(G.nodes()):
        G.nodes[node]["region"] = int(region[node])
    pos = {i: (float(xs[i]), float(ys[i])) for i in G.nodes()}
    if G.number_of_edges():
        colors = []
        for node in G.nodes():
            if node in cue_ids:
                colors.append("#f4c430")
            elif node in active:
                colors.append("#e4572e")
            else:
                colors.append("#4c78a8")
        widths = [0.4 + 2.2 * G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx(
            G,
            pos=pos,
            ax=ax2,
            node_size=40,
            with_labels=False,
            node_color=colors,
            arrows=False,
            width=widths,
            edge_color="#8898aa",
        )
    ax2.set_title(f"Sampled engram graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.suptitle("BrainMemory visualization (sampled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)
