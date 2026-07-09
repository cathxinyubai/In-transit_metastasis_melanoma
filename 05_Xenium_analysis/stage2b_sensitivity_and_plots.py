"""
Stage 2b - bandwidth sensitivity analysis + plotting
=====================================================
Two parts, run on the WSL `liana` env after stage 2:

  PART A (heavy): rerun per-core spatial rank_aggregate at bandwidth 50/150/250
                  (1000 perms, matching the main 100 um run), combine with the
                  existing 100 um per-core table, save a combined long table.
                  Set RERUN=False to skip and reuse a previously saved combined
                  table (so you can iterate on plots without recomputing).

  PART B (fast):  targeted interactions across bandwidths (does the R-vs-NR
                  signal hold?), plus three figures:
                    1. bandwidth sensitivity (median strength R vs NR vs bw)
                    2. targeted dotplot at primary bandwidth (R vs NR)
                    3. detection-frequency heatmap for targeted interactions

Uses magnitude_strength = -log10(magnitude_rank), which is expression-derived and
independent of permutation count; specificity (perm-based) is reported separately.
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import liana as li
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config (keep in sync with stage 2)
# --------------------------------------------------------------------------- #
IN_H5AD = os.path.expanduser("~/projects/itm/data_itm_xenium_stage1_for_liana.h5ad")
OUT_DIR = os.path.expanduser("~/projects/itm/stage2_liana_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BW100_CSV = os.path.join(OUT_DIR, "liana_per_core_spatial.csv")   # from stage 2
COMBINED_CSV = os.path.join(OUT_DIR, "liana_per_core_bandwidth_sweep.csv")

RERUN = True                 # False -> reuse COMBINED_CSV, plot only
BANDWIDTHS_NEW = [50, 150, 250]
PRIMARY_BW = 100

GROUPBY, RESOURCE, SPATIAL_KEY = "cell_type", "consensus", "spatial"
EXPR_PROP, MIN_CELLS, N_PERMS, N_JOBS, SEED = 0.1, 10, 1000, 8, 1337
MIN_CELLS_PER_CORE, MIN_CELLTYPES_PER_CORE = 200, 2
RESPONSE_ORDER, REGION_ORDER = ["R", "NR"], ["high_TILs", "peritumour", "high_tumour"]

TARGET_LR = {
    "i_immune_activating": [
        ("CXCL9", "CXCR3"), ("CXCL10", "CXCR3"), ("CXCL11", "CXCR3"),
        ("CCL5", "CCR5"), ("IFNG", "IFNGR1"),
        ("CD80", "CD28"), ("CD86", "CD28"), ("CD40", "CD40LG"),
    ],
    "iii_cdc1_trafficking_crosstalk": [
        ("CCL19", "CCR7"), ("CCL21", "CCR7"), ("CXCL16", "CXCR6"),
        ("CXCL12", "CXCR4"), ("ICAM1", "ITGAL"), ("VCAM1", "ITGA4"),
        ("SELE", "GLG1"),
    ],
    "ii_tumour_vasculature": [
        ("VEGFA", "KDR"), ("VEGFA", "FLT1"), ("VEGFA", "CD44"), ("VEGFA", "EGFR"),
        ("ANGPT1", "TIE1"),
    ],
}
TARGET_CT_PAIRS = {
    "i_immune_activating": [("DC", "CD8+ T"), ("DC", "CD4+ T"),
                            ("CD8+ T", "CD8+ T"), ("M1 macrophage", "CD8+ T")],
    "iii_cdc1_trafficking_crosstalk": [
        ("DC", "CD8+ T"), ("Endothelial", "DC"), ("Endothelial", "CD8+ T"),
        ("DC", "DC"), ("Endothelial", "CD4+ T")],
    "ii_tumour_vasculature": [("Melanoma", "Endothelial"),
                              ("Endothelial", "Melanoma")],
}

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _members(s):
    return set(str(s).split("_"))

def add_ids(df):
    df = df.copy()
    df["ct_pair"] = df["source"] + " -> " + df["target"]
    df["lr_pair"] = df["ligand_complex"] + " -> " + df["receptor_complex"]
    df["interaction"] = df["ct_pair"] + " | " + df["lr_pair"]
    df["magnitude_strength"] = -np.log10(df["magnitude_rank"].clip(lower=1e-10))
    return df

def targeted_mask(df, claim):
    pairs, cts = TARGET_LR[claim], set(TARGET_CT_PAIRS[claim])
    def row_ok(r):
        if (r["source"], r["target"]) not in cts:
            return False
        lig, rec = _members(r["ligand_complex"]), _members(r["receptor_complex"])
        return any(lg in lig and rc in rec for lg, rc in pairs)
    return df[df.apply(row_ok, axis=1)].copy()

def run_all_cores(adata, bandwidth, core_meta):
    out = []
    for core, idx in adata.obs.groupby("sample", observed=True).groups.items():
        ad_c = adata[idx].copy()
        if ad_c.n_obs < MIN_CELLS_PER_CORE or ad_c.obs["cell_type"].nunique() < MIN_CELLTYPES_PER_CORE:
            continue
        try:
            li.mt.rank_aggregate(
                ad_c, groupby=GROUPBY, resource_name=RESOURCE, spatial_key=SPATIAL_KEY,
                spatial_kwargs={"kernel": "gaussian", "bandwidth": bandwidth},
                expr_prop=EXPR_PROP, min_cells=MIN_CELLS, use_raw=False,
                n_perms=N_PERMS, n_jobs=N_JOBS, seed=SEED, verbose=False,
            )
        except Exception as e:
            print(f"    FAILED {core} @ bw{bandwidth}: {e}")
            continue
        d = ad_c.uns["liana_res"].copy()
        d["sample"] = core
        d["patient_id"] = core_meta.loc[core, "patient_id"]
        d["response"] = core_meta.loc[core, "response"]
        d["region"] = core_meta.loc[core, "region"]
        d["bandwidth"] = bandwidth
        out.append(d)
    return pd.concat(out, ignore_index=True)

# --------------------------------------------------------------------------- #
# PART A - bandwidth sweep
# --------------------------------------------------------------------------- #
if RERUN:
    print("Loading", IN_H5AD)
    adata = sc.read_h5ad(IN_H5AD)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    core_meta = (adata.obs[["sample", "patient_id", "response", "region"]]
                 .drop_duplicates().set_index("sample"))

    frames = []
    # reuse the existing 100 um run
    if os.path.exists(BW100_CSV):
        d100 = pd.read_csv(BW100_CSV)
        d100["bandwidth"] = PRIMARY_BW
        frames.append(d100)
        print(f"Loaded existing bw{PRIMARY_BW} table: {d100.shape}")
    else:
        print(f"WARNING: {BW100_CSV} not found; bw{PRIMARY_BW} will be missing.")

    for bw in BANDWIDTHS_NEW:
        print(f"\n=== bandwidth {bw} um (1000 perms, 64 cores - slow) ===")
        frames.append(run_all_cores(adata, bw, core_meta))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)
    print("\nSaved combined sweep:", combined.shape, "->", COMBINED_CSV)
else:
    combined = pd.read_csv(COMBINED_CSV)
    print("Loaded combined sweep:", combined.shape)

combined = add_ids(combined)

# --------------------------------------------------------------------------- #
# PART B1 - sensitivity table: targeted R vs NR across bandwidths (pooled)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70, "\nSENSITIVITY: targeted R vs NR by bandwidth\n", "=" * 70)
targeted = []
for claim in TARGET_LR:
    t = targeted_mask(combined, claim)
    if len(t):
        t["claim"] = claim
        targeted.append(t)
targeted = pd.concat(targeted, ignore_index=True)
print("Targeted per-core rows:", targeted.shape,
      "| interactions:", targeted["interaction"].nunique())

recs = []
for (claim, inter, bw), sub in targeted.groupby(["claim", "interaction", "bandwidth"], observed=True):
    pat = sub.groupby(["patient_id", "response"], observed=True)["magnitude_strength"].mean().reset_index()
    g = {r: v["magnitude_strength"].values for r, v in pat.groupby("response", observed=True)}
    if not {"NR", "R"}.issubset(g) or len(g["NR"]) < 3 or len(g["R"]) < 3:
        continue
    _, p = mannwhitneyu(g["NR"], g["R"], alternative="two-sided")
    recs.append({
        "claim": claim, "interaction": inter, "bandwidth": bw,
        "n_NR": len(g["NR"]), "n_R": len(g["R"]),
        "median_NR": float(np.median(g["NR"])), "median_R": float(np.median(g["R"])),
        "delta_R_minus_NR": float(np.median(g["R"]) - np.median(g["NR"])),
        "pvalue": p,
    })
sens = pd.DataFrame(recs).sort_values(["claim", "interaction", "bandwidth"])
sens.to_csv(os.path.join(OUT_DIR, "sensitivity_targeted_by_bandwidth.csv"), index=False)
print(sens.head(20).to_string(index=False))

# --------------------------------------------------------------------------- #
# FIG 1 - bandwidth sensitivity (delta R-NR vs bandwidth, per interaction)
# --------------------------------------------------------------------------- #
if len(sens):
    claims = sens["claim"].unique()
    fig, axes = plt.subplots(1, len(claims), figsize=(6 * len(claims), 5), squeeze=False)
    for ax, claim in zip(axes[0], claims):
        sub = sens[sens["claim"] == claim]
        for inter, g in sub.groupby("interaction"):
            g = g.sort_values("bandwidth")
            ax.plot(g["bandwidth"], g["delta_R_minus_NR"], marker="o",
                    label=inter.split(" | ")[1] + " (" + inter.split(" -> ")[0] + ")")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_title(claim); ax.set_xlabel("bandwidth (um)")
        ax.set_ylabel("delta magnitude_strength (R - NR)")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Bandwidth sensitivity of targeted R vs NR contrasts", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_bandwidth_sensitivity.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_bandwidth_sensitivity.png")

# --------------------------------------------------------------------------- #
# FIG 2 - targeted dotplot at primary bandwidth (R vs NR)
#   x = response, y = interaction; dot size = detection freq, colour = median strength
# --------------------------------------------------------------------------- #
prim = targeted[targeted["bandwidth"] == PRIMARY_BW].copy()
prim["is_specific"] = prim["specificity_rank"] <= 0.05
dot = (prim.groupby(["claim", "interaction", "response"], observed=True)
       .agg(med_strength=("magnitude_strength", "median"),
            frac_spec=("is_specific", "mean")).reset_index())
dot.to_csv(os.path.join(OUT_DIR, "targeted_dotplot_data_primary_bw.csv"), index=False)

if len(dot):
    order = (dot.groupby("interaction")["med_strength"].max()
             .sort_values().index.tolist())
    ymap = {it: i for i, it in enumerate(order)}
    xmap = {"R": 0, "NR": 1}
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(order))))
    sc_ = ax.scatter(
        [xmap[r] for r in dot["response"]],
        [ymap[i] for i in dot["interaction"]],
        s=dot["frac_spec"] * 320 + 10,
        c=dot["med_strength"], cmap="viridis", edgecolor="k", linewidth=0.4,
    )
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R", "NR"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([it.replace(" | ", "\n") for it in order], fontsize=6)
    ax.set_title(f"Targeted interactions @ bandwidth {PRIMARY_BW} um")
    cbar = fig.colorbar(sc_, ax=ax); cbar.set_label("median magnitude_strength")
    # size legend
    for f in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=f * 320 + 10, c="grey", edgecolor="k",
                   label=f"{int(f*100)}% cores specific")
    ax.legend(title="dot size", bbox_to_anchor=(1.25, 1), loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_targeted_dotplot.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_targeted_dotplot.png")

# --------------------------------------------------------------------------- #
# FIG 3 - detection-frequency heatmap (R vs NR) for targeted interactions
# --------------------------------------------------------------------------- #
if len(dot):
    hm = dot.pivot_table(index="interaction", columns="response",
                         values="frac_spec").reindex(columns=RESPONSE_ORDER)
    hm = hm.reindex(order)
    fig, ax = plt.subplots(figsize=(5, max(4, 0.32 * len(hm))))
    sns.heatmap(hm, cmap="magma", vmin=0, vmax=1, annot=True, fmt=".2f",
                cbar_kws={"label": "fraction of cores specific"}, ax=ax)
    ax.set_yticklabels([t.get_text().replace(" | ", "\n") for t in ax.get_yticklabels()],
                       fontsize=6, rotation=0)
    ax.set_title(f"Detection frequency (specificity_rank<=0.05) @ bw {PRIMARY_BW}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_detection_freq_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_detection_freq_heatmap.png")

print("\nDONE. Figures in:", FIG_DIR)
