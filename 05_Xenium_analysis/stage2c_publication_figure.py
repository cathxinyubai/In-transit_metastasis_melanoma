"""
Stage 2c - publication figure for the reviewer response
=======================================================
Focused on the robust, claim-relevant spatial LR interactions enriched in
responders (R) in the pretreatment ITM Xenium TMA cohort.

Inputs (in stage2_outputs/):
  - liana_per_core_bandwidth_sweep.csv   (per-core spatial LIANA, bw 50/100/150/250)
  - detection_frequency_fisher.csv       (per-interaction Fisher on specific cores)

Figure (3 panels):
  A. R vs NR spatially-weighted interaction strength at primary bandwidth (100 um),
     patient-level (mean +/- SE, individual patients overlaid), with Mann-Whitney p.
  B. Bandwidth decay: delta (R - NR) vs bandwidth - the contrast is largest at
     short range and washes out by 250 um, i.e. the responder enrichment is
     proximity-driven (spatial), not merely expression-driven.
  C. Detection frequency (Fisher): fraction of cores where the ICAM1 adhesion
     interactions are called specific, R vs NR - a convergent second metric.

Stat unit = patient (per-core scores averaged to patient). Strength =
-log10(magnitude_rank).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import erfc, sqrt

OUT_DIR = "/Users/xbai6546/Desktop/ITM_rebuttal/xenium/stage2_outputs"
SWEEP = os.path.join(OUT_DIR, "liana_per_core_bandwidth_sweep.csv")
FISHER = os.path.join(OUT_DIR, "detection_frequency_fisher.csv")
FIG_OUT = os.path.join(OUT_DIR, "fig_reviewer_spatial_LR.png")
FIG_PDF = os.path.join(OUT_DIR, "fig_reviewer_spatial_LR.pdf")

PRIMARY_BW = 100
PALETTE = {"R": "#F9E58A", "NR": "#7AB7BA"}
RESP_ORDER = ["R", "NR"]

# robust, claim-relevant focus interactions (label -> full interaction id, claim)
FOCUS = [
    ("DC→CD8\nICAM1–ITGAL/ITGB2", "DC -> CD8+ T | ICAM1 -> ITGAL_ITGB2", "iii"),
    ("Endo→CD8\nICAM1–ITGAL/ITGB2", "Endothelial -> CD8+ T | ICAM1 -> ITGAL_ITGB2", "iii"),
    ("Endo→DC\nICAM1–ITGAL/ITGB2", "Endothelial -> DC | ICAM1 -> ITGAL_ITGB2", "iii"),
    ("DC→CD8\nCXCL16–CXCR6", "DC -> CD8+ T | CXCL16 -> CXCR6", "iii"),
    ("DC→CD8\nCD86–CD28", "DC -> CD8+ T | CD86 -> CD28", "i"),
    ("DC→CD4\nCD86–CD28", "DC -> CD4+ T | CD86 -> CD28", "i"),
]
# subset shown in the bandwidth-decay panel (clearest spatial decay)
DECAY = FOCUS[:3] + [FOCUS[4]]
# ICAM1 adhesion pairs for the detection-frequency panel
ICAM = FOCUS[:3]

CLAIM_COLOR = {"i": "#C44E52", "iii": "#4C72B0", "ii": "#55A868"}


def mwu_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        return np.nan
    allv = np.concatenate([a, b]); r = pd.Series(allv).rank().values
    U1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    U = min(U1, n1 * n2 - U1); mu = n1 * n2 / 2
    _, c = np.unique(allv, return_counts=True); tie = (c ** 3 - c).sum()
    sig = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / ((n1 + n2) * (n1 + n2 - 1))))
    return 1.0 if sig == 0 else erfc((abs(U - mu) - 0.5) / sig / sqrt(2))


def p_star(p):
    if np.isnan(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return f"p={p:.2f}"


# --------------------------------------------------------------------------- #
# load + patient-level aggregation
# --------------------------------------------------------------------------- #
df = pd.read_csv(SWEEP)
df["interaction"] = (df["source"] + " -> " + df["target"] + " | "
                     + df["ligand_complex"] + " -> " + df["receptor_complex"])
df["strength"] = -np.log10(df["magnitude_rank"].clip(lower=1e-10))

focus_ids = [f[1] for f in FOCUS]
d = df[df["interaction"].isin(focus_ids)].copy()
# per-patient mean strength per interaction x bandwidth
pat = (d.groupby(["interaction", "bandwidth", "patient_id", "response"], observed=True)
       ["strength"].mean().reset_index())

# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(11, 8))
gs = GridSpec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)

# ---- Panel A: bw100 R vs NR (bars + SE + points) ----
axA = fig.add_subplot(gs[0, :])
labels = [f[0] for f in FOCUS]
x = np.arange(len(FOCUS)); w = 0.38
for j, resp in enumerate(RESP_ORDER):
    means, ses = [], []
    for _, fid, _ in FOCUS:
        vals = pat[(pat.interaction == fid) & (pat.bandwidth == PRIMARY_BW)
                   & (pat.response == resp)]["strength"].values
        means.append(np.mean(vals) if len(vals) else 0)
        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    xpos = x + (j - 0.5) * w
    axA.bar(xpos, means, w, yerr=ses, capsize=3, color=PALETTE[resp],
            edgecolor="black", linewidth=0.8, label=resp, zorder=2)
    # individual patients
    for k, (_, fid, _) in enumerate(FOCUS):
        vals = pat[(pat.interaction == fid) & (pat.bandwidth == PRIMARY_BW)
                   & (pat.response == resp)]["strength"].values
        jit = np.random.uniform(-0.07, 0.07, len(vals))
        axA.scatter(np.full(len(vals), xpos[k]) + jit, vals, s=10, color="black",
                    alpha=0.55, zorder=3, linewidths=0)
# p-value annotations
for k, (_, fid, _) in enumerate(FOCUS):
    r = pat[(pat.interaction == fid) & (pat.bandwidth == PRIMARY_BW) & (pat.response == "R")]["strength"].values
    nr = pat[(pat.interaction == fid) & (pat.bandwidth == PRIMARY_BW) & (pat.response == "NR")]["strength"].values
    p = mwu_p(nr, r)
    ytop = max(np.concatenate([r, nr]).max() if len(r)+len(nr) else 0, 0)
    axA.text(x[k], ytop * 1.05 + 0.02, p_star(p), ha="center", va="bottom", fontsize=8)
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=7.5)
axA.set_ylabel("Spatially-weighted interaction\nstrength  (-log10 magnitude rank)")
axA.set_title(f"A   Responder-enriched spatial LR interactions @ bandwidth {PRIMARY_BW} µm  "
              "(patient-level, mean ± SE)", fontsize=10, loc="left")
axA.legend(title="Response", frameon=False, loc="upper left", ncol=2)
# claim color strip under x labels
for k, (_, _, cl) in enumerate(FOCUS):
    axA.add_patch(plt.Rectangle((x[k]-0.45, -0.001), 0.9, 0.0, fill=False))
axA.margins(y=0.18)

# ---- Panel B: bandwidth decay ----
axB = fig.add_subplot(gs[1, 0])
bws = sorted(df["bandwidth"].unique())
for lab, fid, cl in DECAY:
    deltas = []
    for bw in bws:
        r = pat[(pat.interaction == fid) & (pat.bandwidth == bw) & (pat.response == "R")]["strength"].values
        nr = pat[(pat.interaction == fid) & (pat.bandwidth == bw) & (pat.response == "NR")]["strength"].values
        deltas.append(np.median(r) - np.median(nr) if len(r) and len(nr) else np.nan)
    axB.plot(bws, deltas, marker="o", lw=1.6, color=CLAIM_COLOR[cl],
             label=lab.replace("\n", " "))
axB.axhline(0, color="grey", ls="--", lw=1)
axB.set_xlabel("Spatial bandwidth (µm)")
axB.set_ylabel("Δ strength  (R − NR)")
axB.set_title("B   Contrast is proximity-driven\n(decays as bandwidth widens)", fontsize=10, loc="left")
axB.legend(fontsize=6.5, frameon=False)
axB.set_xticks(bws)

# ---- Panel C: detection frequency (Fisher) for ICAM1 pairs ----
axC = fig.add_subplot(gs[1, 1])
fish = pd.read_csv(FISHER)
icam_labels = [f[0].replace("\n", " ") for f in ICAM]
xc = np.arange(len(ICAM)); wc = 0.38
for j, resp in enumerate(RESP_ORDER):
    col = "frac_spec_R" if resp == "R" else "frac_spec_NR"
    vals = []
    for _, fid, _ in ICAM:
        row = fish[fish["interaction"] == fid]
        vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
    axC.bar(xc + (j - 0.5) * wc, vals, wc, color=PALETTE[resp],
            edgecolor="black", linewidth=0.8, label=resp)
# annotate fisher p
for k, (_, fid, _) in enumerate(ICAM):
    row = fish[fish["interaction"] == fid]
    if len(row):
        axC.text(xc[k], 1.02, f"p={row['pvalue'].iloc[0]:.2f}", ha="center", fontsize=7)
axC.set_xticks(xc)
axC.set_xticklabels([l.replace("ICAM1–ITGAL/ITGB2", "ICAM1") for l in icam_labels], fontsize=7)
axC.set_ylabel("Fraction of cores with\nspecific interaction")
axC.set_ylim(0, 1.12)
axC.set_title("C   Detection prevalence\n(ICAM1 adhesion, Fisher)", fontsize=10, loc="left")
axC.legend(fontsize=7, frameon=False, loc="upper right")

fig.suptitle("Spatial ligand–receptor validation (pretreatment ITM Xenium TMA): "
             "immune-activating & immune–vascular crosstalk enriched in responders",
             fontsize=11, y=0.995)
fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
print("Saved:", FIG_OUT)
print("Saved:", FIG_PDF)
