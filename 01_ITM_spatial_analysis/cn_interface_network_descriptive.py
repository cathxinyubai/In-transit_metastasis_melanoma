#!/usr/bin/env python3
"""
Descriptive CN interface NETWORK  --  exploratory Fig. 3F panel.
===============================================================

Three views, all DESCRIPTIVE (no p-values, no significance markers; group
differences were not significant at the patient level, so nothing here should be
read as a statistical comparison):

  A. observed_adjacency  -- edge width = how often the two CNs are spatially
                            adjacent (raw co-occurrence, row/total normalised).
                            Intuitive "who touches whom"; influenced by CN
                            abundance (note this in the legend).
  B. log2_obs_exp        -- edge colour/width = log2(observed/expected) vs a
                            within-sample random null (size-invariant). Red =
                            interface more than expected, blue = less. Most
                            between-CN pairs read as <1 (depleted) because CNs
                            are spatially contiguous; this is expected.
  C. responders_only     -- a single panel (PRE responders) illustrating the
                            triad adjacency without inviting a (null) 3-group
                            comparison.

Triad edges (T/B--Immune interface, Immune interface--Tumour core, T/B--Tumour
core) are highlighted in green in every view. Node size = CN abundance.

Requires: scanpy/anndata, numpy, pandas, scikit-learn, matplotlib; scipy.sparse.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

# ----------------------------------------------------------------------------
H5AD_PATH   = "itm18_annotated_cn_Oct25.h5ad"
CN_COL      = "CN_k20_n7_annot"
SAMPLE_COL  = "sample"
GROUP_COL   = "response_group"
X_COL, Y_COL = "x", "y"
OUTDIR      = "cn_interface_output"

TRIAD   = ["T/B cell enriched", "Immune interface", "Tumour core"]
K_ADJ   = 6
P_PERM  = 200          # for the expected (mean) counts in log2(O/E)
RNG_SEED = 0
EDGE_MIN  = 0.02       # conditional-adjacency floor for drawing an arrow (2%)
LABEL_MIN = 0.0        # label triad arrows regardless; raise to also gate by value

GROUP_ORDER  = ["PRE_Responsive", "PRE_Resistant", "PROG_Resistant"]
GROUP_TITLES = {"PRE_Responsive": "PRE Responder",
                "PRE_Resistant":  "PRE Resistant",
                "PROG_Resistant": "PROG Resistant"}
CN_PALETTE = {
    "Immune interface": "#006FA6", "Tumour core": "#A30059",
    "Tumour vessel": "#FFDBE5", "Immune vessel": "#7A4900",
    "TAM niche": "#0000A6", "T/B cell enriched": "#63FFAC",
    "Myeloid enriched": "#B79762"}
DISPLAY_ORDER = TRIAD + [c for c in CN_PALETTE if c not in TRIAD]

os.makedirs(OUTDIR, exist_ok=True)


def sample_adjacency(coords):
    n = coords.shape[0]; k = min(K_ADJ, max(1, n - 1))
    A = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    return ((A + A.T) > 0).astype(np.float64)


def onehot(codes, n_cn):
    n = len(codes)
    return sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_cn))


def group_matrices(samples, n_cn, rng):
    """Return observed counts O and expected counts E (perm mean), pooled."""
    O = np.zeros((n_cn, n_cn))
    cache = []
    for A, codes in samples:
        C = onehot(codes, n_cn)
        O += np.asarray((C.T @ A @ C).todense())
        cache.append((A, codes))
    E = np.zeros((n_cn, n_cn))
    for _ in range(P_PERM):
        M = np.zeros((n_cn, n_cn))
        for A, codes in cache:
            C = onehot(rng.permutation(codes), n_cn)
            M += np.asarray((C.T @ A @ C).todense())
        E += M
    E /= P_PERM
    return O, E


# ----------------------------------------------------------------------------
def _layout(order):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(order), endpoint=False)
    return {c: (np.cos(a), np.sin(a)) for c, a in zip(order, ang)}


def _node_sizes(order, abundance, code):
    return {c: 110 + 2200 * abundance[code[c]] for c in order}


def _node_radius_pts(size):
    return np.sqrt(size / np.pi) + 2


def _draw_nodes(ax, order, pos, abundance, code):
    sizes = _node_sizes(order, abundance, code)
    smax = max(sizes.values()) or 1.0
    for c in order:
        x, y = pos[c]
        ax.scatter([x], [y], s=sizes[c], color=CN_PALETTE[c], edgecolor="black", lw=0.8, zorder=3)
        # push label radially outward beyond the node (extra clearance for big nodes)
        rl = 1.42 + 0.30 * (sizes[c] / smax)
        ha = "left" if x > 0.15 else "right" if x < -0.15 else "center"
        va = "bottom" if y > 0.15 else "top" if y < -0.15 else "center"
        ax.text(x * rl, y * rl, c, ha=ha, va=va, fontsize=6.5)
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-2.25, 2.25); ax.set_aspect("equal"); ax.axis("off")


def conditional_matrix(O):
    """Cond[i,j] = fraction of CN i's contacts going to CN j (row-normalised)."""
    M = O.copy(); np.fill_diagonal(M, 0)
    rs = M.sum(1, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)


def draw_conditional(ax, O, abundance, order, code, tri_set, cmax):
    """Directional conditional-adjacency network: arrow i->j width = fraction of
    CN i's spatial contacts going to CN j. Triad arrows green + %-labelled; others
    faint grey (drawn, not hidden)."""
    pos = _layout(order)
    Cond = conditional_matrix(O)
    sizes = _node_sizes(order, abundance, code)
    for a in range(len(order)):
        for b in range(len(order)):
            if a == b:
                continue
            ca, cb = order[a], order[b]
            v = Cond[code[ca], code[cb]]
            if v < EDGE_MIN:
                continue
            is_tri = frozenset((ca, cb)) in tri_set
            lw = 0.4 + 9.0 * (v / cmax)
            col = "#2ca02c" if is_tri else "#bdbdbd"
            ax.annotate("", xy=pos[cb], xytext=pos[ca],
                        arrowprops=dict(arrowstyle="-|>", lw=lw, color=col,
                                        alpha=0.97 if is_tri else 0.3,
                                        shrinkA=_node_radius_pts(sizes[ca]),
                                        shrinkB=_node_radius_pts(sizes[cb]) + 3,
                                        connectionstyle="arc3,rad=0.13"),
                        zorder=2 if is_tri else 1)
            if is_tri and v >= LABEL_MIN:
                # place the label on THIS arrow's own arc so it matches its width.
                # matplotlib arc3(rad) control point = midpoint + (rad*dy, -rad*dx);
                # offset along the same vector keeps the label on the correct side.
                mx, my = (pos[ca][0] + pos[cb][0]) / 2, (pos[ca][1] + pos[cb][1]) / 2
                dx, dy = pos[cb][0] - pos[ca][0], pos[cb][1] - pos[ca][1]
                rad, f = 0.13, 0.95
                lx, ly = mx + f * rad * dy, my - f * rad * dx
                ax.text(lx, ly, f"{v*100:.0f}%", fontsize=6.5, color="#176d17",
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.7))
    _draw_nodes(ax, order, pos, abundance, code)


def draw_log2oe(ax, O, E, abundance, order, code, tri_set, vmax):
    pos = _layout(order)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where((O > 0) & (E > 0), np.log2(O / E), 0.0)
    np.fill_diagonal(L, 0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            ca, cb = order[a], order[b]
            v = L[code[ca], code[cb]]
            if abs(v) < vmax * 0.05:
                continue
            is_tri = frozenset((ca, cb)) in tri_set
            lw = 0.6 + 5.0 * (abs(v) / vmax)
            ax.plot(*zip(pos[ca], pos[cb]), color=cmap(norm(v)), lw=lw,
                    alpha=0.95, zorder=1, solid_capstyle="round")
            if is_tri:  # green halo behind triad edges
                ax.plot(*zip(pos[ca], pos[cb]), color="#2ca02c", lw=lw + 3.0,
                        alpha=0.5, zorder=0, solid_capstyle="round")
    _draw_nodes(ax, order, pos, abundance, code)
    return cmap, norm


def main():
    import scanpy as sc
    print(f"Loading {H5AD_PATH} ...")
    adata = sc.read(H5AD_PATH)
    obs = adata.obs[[CN_COL, SAMPLE_COL, GROUP_COL]].copy()
    if X_COL in adata.obs and Y_COL in adata.obs:
        obs[X_COL] = adata.obs[X_COL].to_numpy(); obs[Y_COL] = adata.obs[Y_COL].to_numpy()
    else:
        obs[X_COL] = adata.obsm["spatial"][:, 0]; obs[Y_COL] = adata.obsm["spatial"][:, 1]
    obs = obs.dropna(subset=[CN_COL, SAMPLE_COL, GROUP_COL, X_COL, Y_COL])
    cn_categories = sorted(obs[CN_COL].astype(str).unique().tolist())
    code = {c: i for i, c in enumerate(cn_categories)}
    obs["_cn_code"] = obs[CN_COL].astype(str).map(code).to_numpy()
    n_cn = len(cn_categories); rng = np.random.default_rng(RNG_SEED)
    order = [c for c in DISPLAY_ORDER if c in cn_categories]
    tri_set = {frozenset((a, b)) for a in TRIAD for b in TRIAD if a != b}

    O_g, E_g, ab_g = {}, {}, {}
    for g in GROUP_ORDER:
        sub_g = obs[obs[GROUP_COL] == g]
        samples = []
        for _, sub in sub_g.groupby(SAMPLE_COL, observed=True):
            if sub.shape[0] < 5:
                continue
            samples.append((sample_adjacency(sub[[X_COL, Y_COL]].to_numpy(float)),
                            sub["_cn_code"].to_numpy()))
        if not samples:
            continue
        print(f"  {g}: {len(samples)} samples")
        O, E = group_matrices(samples, n_cn, rng)
        O_g[g] = O; E_g[g] = E
        ab = sub_g["_cn_code"].value_counts(normalize=True)
        ab_g[g] = np.array([ab.get(code[c], 0.0) for c in cn_categories])

    groups = [g for g in GROUP_ORDER if g in O_g]

    # ---- A. conditional adjacency (3 panels) ----
    cmax = max(conditional_matrix(O_g[g]).max() for g in groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.8))
    if len(groups) == 1: axes = [axes]
    for ax, g in zip(axes, groups):
        draw_conditional(ax, O_g[g], ab_g[g], order, code, tri_set, cmax)
        ax.set_title(GROUP_TITLES[g], fontsize=10)
    fig.suptitle("CN interface network — conditional adjacency (arrow i→j width = % of CN i's "
                 "contacts going to j; abundance-corrected for the focal node). "
                 "Green = triad; node size = abundance. Exploratory; no group test.",
                 fontsize=9, y=1.03)
    fig.savefig(os.path.join(OUTDIR, "network_conditional_adjacency.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "network_conditional_adjacency.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- B. log2(obs/exp) (3 panels) ----
    def _l(g):
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where((O_g[g] > 0) & (E_g[g] > 0), np.log2(O_g[g] / E_g[g]), 0.0)
        np.fill_diagonal(L, 0); return L
    vmax = max(np.abs(_l(g)).max() for g in groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.4))
    if len(groups) == 1: axes = [axes]
    cmap = norm = None
    for ax, g in zip(axes, groups):
        cmap, norm = draw_log2oe(ax, O_g[g], E_g[g], ab_g[g], order, code, tri_set, vmax)
        ax.set_title(GROUP_TITLES[g], fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="log2(observed / expected)")
    fig.suptitle("CN interface network — log2(obs/exp) vs within-sample null (descriptive; "
                 "red=more, blue=less; green halo = triad). Exploratory; no group test.",
                 fontsize=9.5, y=1.03)
    fig.savefig(os.path.join(OUTDIR, "network_log2_obs_exp.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "network_log2_obs_exp.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- C. responders only (single panel, conditional adjacency) ----
    if "PRE_Responsive" in O_g:
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        draw_conditional(ax, O_g["PRE_Responsive"], ab_g["PRE_Responsive"], order, code, tri_set, cmax)
        ax.set_title("PRE Responder — conditional adjacency (descriptive)", fontsize=10)
        fig.suptitle("Arrow i→j width = fraction of CN i's contacts going to j (abundance-corrected). "
                     "Green = triad; labels show %. Exploratory.", fontsize=8, y=0.04)
        fig.savefig(os.path.join(OUTDIR, "network_conditional_responders_only.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, "network_conditional_responders_only.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    # export the matrices used (observed counts, conditional %, log2 O/E)
    for g in groups:
        pd.DataFrame(O_g[g], index=cn_categories, columns=cn_categories)\
            .to_csv(os.path.join(OUTDIR, f"adjacency_observed_{g}.csv"))
        pd.DataFrame(conditional_matrix(O_g[g]), index=cn_categories, columns=cn_categories)\
            .to_csv(os.path.join(OUTDIR, f"adjacency_conditional_{g}.csv"))
        pd.DataFrame(_l(g), index=cn_categories, columns=cn_categories)\
            .to_csv(os.path.join(OUTDIR, f"adjacency_log2oe_{g}.csv"))
    print(f"Saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
