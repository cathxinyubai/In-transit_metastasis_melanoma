# %% [markdown]
# Stage 3 - CODEX(SPACEc) distance x CellChat signalling correlation (R3 Q3c)
# ===========================================================================
# Reviewer 3 Q3c: does closer spatial distance (CODEX) correspond to stronger
# signalling (CITE-seq/CellChat)? e.g. are DCs closer to T cells in responders,
# and does that track stronger DC-T signalling?
#
# Refactor of itm18_distperm_cellchat_correlation_jun26.ipynb. Key changes:
#   1. RESPONDER-CENTRIC directionality (no more sign gymnastics):
#        proximity_delta  = (closer in Responder)  -> positive
#        cellchat_delta   = (stronger in Responder)-> positive
#        => positive Spearman rho = "closer in responders translates to stronger
#           signalling in responders" (directly answers Q3c).
#   2. PRIMARY contrast = PRE_Responsive vs PRE_Resistant only (the response axis).
#        The progression contrasts are a different question and are NOT pooled
#        into the same correlation (pooling re-uses pairs -> non-independence).
#   3. DC-T pairs are the LEAD result (quadrant plot + table); the per-CN
#        global Spearman is supporting context.
#   4. Per-CN Spearman reported with n, exact p, and a bootstrap 95% CI.
#
# Run on the workstation (needs the SPACEc per-CN distance_pvals files).

# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
# Config / paths  (edit to your machine)
# --------------------------------------------------------------------------- #
PER_CN_DIR = Path(
    "/Users/xbai6546/Documents/01_Research_projects/03_ITM_immunotherapy/"
    "ITM_spatial_analysis/distance_permutation/spacec_output_cellchat_matched/"
    "by_CN__ALL_timepoints")
CELLCHAT_CSV = Path("/Users/xbai6546/Desktop/ITM_rebuttal/CODEX_CellChat_correlation/"
                    "cellchat_pair_weight_collapsed_by_response_group.csv")
OUT_DIR = Path("/Users/xbai6546/Desktop/ITM_rebuttal/CODEX_CellChat_correlation/stage3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PRIMARY response contrast (responder vs resistant, pretreatment)
GROUP_R = "PRE_Responsive"     # responders
GROUP_NR = "PRE_Resistant"     # non-responders (resistant)

# Optional progression contrasts (run separately, NEVER pooled with primary)
EXTRA_CONTRASTS = [("PROG_Resistant", "PRE_Resistant"),
                   ("PROG_Resistant", "PRE_Responsive")]

CELLCHAT_VALUE = "cellchat_weight_sum"   # or "cellchat_weight_mean"
MIN_PAIRS_PER_CN = 8                      # min pairs to compute a CN correlation
N_BOOT = 2000
SEED = 0

# SPACEc per-row significance pre-filter. The delta correlation conditions on
# the outcome if you filter by per-group significance, so default OFF (use all
# pairs with a valid logFC). Toggle True only as a sensitivity check.
USE_SIG_FILTER = False
PVALUE, LOGF_ABS = 0.05, 0.10

CELLCHAT_TYPES = ["B_cell", "CD4_T", "CD8_T", "DC", "Endothelial",
                  "Lymph_endo", "Macrophage", "Melanocyte", "Treg"]
DC_T_PAIRS = ["CD4_T_DC", "CD8_T_DC", "DC_Treg"]   # alphabetical unordered

rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 1. SPACEc per-CN proximity per response group
# logfold_group: more NEGATIVE = closer / more spatially enriched.
# We define a per-group PROXIMITY = -logfold_group (higher = closer).

# %%
def parse_pair(interaction):
    s = str(interaction)
    if " --> " not in s:
        return (np.nan, np.nan, np.nan)
    src, tgt = s.split(" --> ", 1)
    if src not in CELLCHAT_TYPES or tgt not in CELLCHAT_TYPES:
        return (np.nan, np.nan, np.nan)
    a, b = sorted([src, tgt])
    return (src, tgt, f"{a}_{b}")


files = sorted(glob.glob(str(PER_CN_DIR / "*__response_group__distance_pvals_filt.csv")))
print(f"Found {len(files)} per-CN SPACEc files")

rows = []
for f in files:
    f = Path(f)
    cn = f.name.replace("__distance_pvals_filt.csv", "").replace("__response_group", "").replace("_", " ")
    df = pd.read_csv(f)
    df = df[df["logfold_group"].notna()].copy()
    if USE_SIG_FILTER:
        df = df[(df["pvalue"] < PVALUE) & (df["logfold_group"].abs() > LOGF_ABS)]
    if df.empty:
        continue
    df[["src", "tgt", "pair"]] = df["interaction"].apply(lambda x: pd.Series(parse_pair(x)))
    df = df.dropna(subset=["pair"])
    df["proximity"] = -df["logfold_group"]          # higher = closer
    df["CN"] = cn
    rows.append(df[["CN", "pair", "response_group", "proximity"]])

spacec = pd.concat(rows, ignore_index=True)
# collapse directional interactions to unordered pair (mean proximity)
spacec = (spacec.groupby(["CN", "pair", "response_group"], observed=True)["proximity"]
          .mean().reset_index())
prox = spacec.pivot_table(index=["CN", "pair"], columns="response_group",
                          values="proximity").reset_index()
print("SPACEc proximity table:", prox.shape, "| CNs:", prox['CN'].nunique())

# %% [markdown]
# ## 2. CellChat per-group signalling weight (global, no CN)

# %%
cc = pd.read_csv(CELLCHAT_CSV)
ccw = cc.pivot_table(index="pair", columns="response_group",
                     values=CELLCHAT_VALUE, aggfunc="first").reset_index()
print("CellChat weight table:", ccw.shape)


# %% [markdown]
# ## 3. Responder-centric deltas + merge (PRIMARY contrast)
#  proximity_delta = closer in Responder (positive)
#  cellchat_delta  = stronger in Responder (positive)
#
# IMPORTANT: CellChat tables list only *detected* interactions, so a pair absent
# in a group means no inferred communication = weight 0 (NOT missing data). We
# therefore fill missing CellChat group weights with 0 before the delta. In this
# cohort PRE_Resistant has only 15 detected pairs vs 36 in PRE_Responsive, and the
# DC-T pairs are present in responders / absent in resistant - dropping NaNs would
# have deleted exactly the lead result. SPACEc proximity, by contrast, is a
# measured distance and requires both groups present (dropna), since 0 is not a
# meaningful "no distance".

# %%
def build_merged(group_r, group_nr):
    # SPACEc: distance must be measured in both groups
    p = prox.dropna(subset=[group_r, group_nr]).copy()
    p["proximity_delta"] = p[group_r] - p[group_nr]          # + => closer in R
    # CellChat: absence = 0 communication
    c = ccw.copy()
    c[group_r] = c[group_r].fillna(0.0)
    c[group_nr] = c[group_nr].fillna(0.0)
    c = c[(c[group_r] > 0) | (c[group_nr] > 0)]              # drop all-zero pairs
    c["cellchat_delta"] = c[group_r] - c[group_nr]           # + => stronger in R
    c["cc_pattern"] = np.where(
        (c[group_r] > 0) & (c[group_nr] == 0), "R_only",
        np.where((c[group_r] == 0) & (c[group_nr] > 0), "NR_only", "both"))
    m = p[["CN", "pair", "proximity_delta"]].merge(
        c[["pair", "cellchat_delta", "cc_pattern"]], on="pair", how="inner")
    return m


merged = build_merged(GROUP_R, GROUP_NR)

# join cross-modal support tier (from stage3_support_qc.py) so we can separate
# confirmatory (CITE-seq + CODEX both adequately powered) from exploratory pairs.
SUPPORT_CSV = OUT_DIR / "pair_support_primary_contrast.csv"
if SUPPORT_CSV.exists():
    sup = pd.read_csv(SUPPORT_CSV)[["pair", "crossmodal_support_RvsNR", "tier"]]
    merged = merged.merge(sup, on="pair", how="left")
    merged["tier"] = merged["tier"].fillna("exploratory")
else:
    print("NOTE: run stage3_support_qc.py first to tier pairs; defaulting all to exploratory.")
    merged["tier"] = "exploratory"

merged.to_csv(OUT_DIR / "merged_proximity_cellchat_primary.csv", index=False)
print("Merged (primary):", merged.shape, "| pairs:", merged['pair'].nunique(),
      "| CNs:", merged['CN'].nunique())
print("Tier counts (pair×CN rows):", merged['tier'].value_counts().to_dict())


# %% [markdown]
# ## 3b. QC - does the R_only DC-T pattern hold, and do the pairs survive the merge?
# A pair enters `merged` only if BOTH are true:
#   (a) CellChat detects it in >=1 PRE group  (absence elsewhere -> weight 0), and
#   (b) SPACEc measured its proximity in BOTH PRE groups within that CN.
# Read the three printouts below in order:
#   1. CellChat pattern: confirms the DC-T pairs are R_only (weight>0 in R, 0 in NR).
#   2. SPACEc availability per CN: prox_in_both=True means the pair is testable
#      (has a distance in both groups) in that CN and will survive the merge.
#   3. Final merged DC-T rows: what actually enters the correlation/figure.
# If a DC-T pair is R_only (1) but prox_in_both=False in a CN (2), it is dropped
# there because proximity Delta is undefined - expected, not a bug.

# %%
def dc_t_qc(group_r, group_nr):
    c = ccw.copy()
    c[group_r] = c[group_r].fillna(0.0); c[group_nr] = c[group_nr].fillna(0.0)
    cpat = c[c["pair"].isin(DC_T_PAIRS)][["pair", group_r, group_nr]].copy()
    cpat["cc_pattern"] = np.where(
        (cpat[group_r] > 0) & (cpat[group_nr] == 0), "R_only",
        np.where((cpat[group_r] == 0) & (cpat[group_nr] > 0), "NR_only", "both"))
    print("1) CellChat DC-T pattern (primary contrast):")
    print(cpat.to_string(index=False))

    pp = prox[prox["pair"].isin(DC_T_PAIRS)][["CN", "pair", group_r, group_nr]].copy()
    pp["prox_in_both"] = pp[group_r].notna() & pp[group_nr].notna()
    print("\n2) SPACEc DC-T proximity availability per CN "
          "(prox_in_both=True => survives merge):")
    print(pp.sort_values(["pair", "CN"]).to_string(index=False))

    mm = merged[merged["pair"].isin(DC_T_PAIRS)]
    print("\n3) DC-T rows in the final merged primary table "
          f"({len(mm)} rows across {mm['CN'].nunique()} CNs):")
    print(mm.sort_values(["pair", "CN"]).to_string(index=False))

dc_t_qc(GROUP_R, GROUP_NR)


# %% [markdown]
# ## 4. Per-CN Spearman with bootstrap 95% CI

# %%
def spearman_ci(x, y, n_boot=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = spearmanr(x, y)
    n = len(x)
    if n < 4:
        return rho, p, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return rho, p, lo, hi


def correlate_by_cn(m, label):
    res = []
    for cn, sub in m.groupby("CN"):
        if len(sub) < MIN_PAIRS_PER_CN:
            continue
        rho, p, lo, hi = spearman_ci(sub["proximity_delta"], sub["cellchat_delta"])
        res.append({"contrast": label, "CN": cn, "n_pairs": len(sub),
                    "rho": rho, "pvalue": p, "ci_lo": lo, "ci_hi": hi})
    # also a global summary collapsing to one value per pair (mean across CNs)
    g = m.groupby("pair").agg(proximity_delta=("proximity_delta", "mean"),
                              cellchat_delta=("cellchat_delta", "first")).reset_index()
    if len(g) >= MIN_PAIRS_PER_CN:
        rho, p, lo, hi = spearman_ci(g["proximity_delta"], g["cellchat_delta"])
        res.append({"contrast": label, "CN": "ALL_CN_pairwise_mean", "n_pairs": len(g),
                    "rho": rho, "pvalue": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(res)


# PRIMARY: confirmatory pairs only (both modalities adequately powered)
conf = merged[merged["tier"] == "confirmatory"]
corr = correlate_by_cn(conf, f"{GROUP_R} vs {GROUP_NR} [confirmatory]")
corr = corr.sort_values("rho", ascending=False)
corr.to_csv(OUT_DIR / "correlation_per_CN_primary_confirmatory.csv", index=False)
print("CONFIRMATORY pairs only (headline):")
print(corr.to_string(index=False))

# SENSITIVITY: all pairs (incl. exploratory) - report but do not headline
corr_all = correlate_by_cn(merged, f"{GROUP_R} vs {GROUP_NR} [all pairs]")
corr_all.sort_values("rho", ascending=False).to_csv(
    OUT_DIR / "correlation_per_CN_primary_allpairs.csv", index=False)

# progression contrasts, run + saved SEPARATELY (not pooled)
extra = []
for gr, gnr in EXTRA_CONTRASTS:
    m2 = build_merged(gr, gnr)
    if len(m2):
        extra.append(correlate_by_cn(m2, f"{gr} vs {gnr}"))
if extra:
    pd.concat(extra, ignore_index=True).to_csv(
        OUT_DIR / "correlation_per_CN_progression.csv", index=False)


# %% [markdown]
# ## 5. DC-T lead result (the pair Q3c names)

# %%
dct = merged[merged["pair"].isin(DC_T_PAIRS)].copy()
dct.to_csv(OUT_DIR / "DC_T_proximity_cellchat_primary.csv", index=False)
print("\nDC-T pairs (primary contrast):")
print(dct.sort_values(["pair", "CN"]).to_string(index=False))


# %% [markdown]
# ## 6. Figure: (A) quadrant scatter w/ DC-T highlighted, (B) per-CN rho forest

# %%
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---- A: quadrant (confirmatory = filled dark, exploratory = open grey) ----
expl = merged[merged["tier"] != "confirmatory"]
conf_pts = merged[merged["tier"] == "confirmatory"]
axA.scatter(expl["proximity_delta"], expl["cellchat_delta"], s=16,
            facecolors="none", edgecolors="0.7", linewidth=0.7, alpha=0.7,
            label="exploratory (under-powered)", zorder=1)
axA.scatter(conf_pts["proximity_delta"], conf_pts["cellchat_delta"], s=22,
            color="#333", edgecolor="none", alpha=0.85,
            label="confirmatory pairs", zorder=2)
dc_color = {"CD8_T_DC": "#C44E52", "CD4_T_DC": "#4C72B0", "DC_Treg": "#55A868"}
for pair, g in dct.groupby("pair"):
    axA.scatter(g["proximity_delta"], g["cellchat_delta"], s=60,
                facecolors=dc_color.get(pair, "black"), edgecolor="black",
                linewidth=0.6, zorder=3, marker="D",
                label=pair.replace("_", "–") + " (exploratory)")
axA.axhline(0, color="grey", lw=0.8, ls="--"); axA.axvline(0, color="grey", lw=0.8, ls="--")
rho_all = corr[corr["CN"] == "ALL_CN_pairwise_mean"]
sub = f"ρ={rho_all['rho'].iloc[0]:.2f}, P={rho_all['pvalue'].iloc[0]:.3f}" if len(rho_all) else ""
axA.set_xlabel("Proximity Δ  (closer in responders →)")
axA.set_ylabel("CellChat signalling Δ  (stronger in responders →)")
axA.set_title("CODEX proximity vs CellChat signalling (R − Resistant)",
              fontsize=9.5, loc="left", pad=12)
axA.text(0.0, 1.13, "A", transform=axA.transAxes, fontsize=15,
         fontweight="bold", va="top", ha="left")
# quadrant annotation (top-right) with the colour legend directly beneath it
axA.text(0.98, 0.99, "closer & stronger\nin responders", transform=axA.transAxes,
         ha="right", va="top", fontsize=7.5, color="#444", style="italic")
axA.legend(fontsize=6.5, frameon=False, loc="upper right",
           bbox_to_anchor=(1.0, 0.88))

# ---- B: per-CN rho forest ----
cn_corr = corr[corr["CN"] != "ALL_CN_pairwise_mean"].sort_values("rho")
y = np.arange(len(cn_corr))
axB.errorbar(cn_corr["rho"], y,
             xerr=[cn_corr["rho"] - cn_corr["ci_lo"], cn_corr["ci_hi"] - cn_corr["rho"]],
             fmt="o", color="#333", ecolor="0.6", capsize=3, markersize=5)
axB.axvline(0, color="grey", ls="--", lw=0.8)
axB.set_yticks(y)
axB.set_yticklabels([f"{cn}  (n={n})" for cn, n in zip(cn_corr["CN"], cn_corr["n_pairs"])],
                    fontsize=7.5)
axB.set_xlabel("Spearman ρ (proximity Δ vs signalling Δ)")
axB.set_title("Per-CN correlation, confirmatory pairs (95% CI)",
              fontsize=9.5, loc="left", pad=12)
axB.text(0.0, 1.13, "B", transform=axB.transAxes, fontsize=15,
         fontweight="bold", va="top", ha="left")
axB.set_xlim(-1.05, 1.05)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_stage3_proximity_signalling_v2.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_stage3_proximity_signalling_v2.pdf", bbox_inches="tight")
print("\nSaved figure + tables to", OUT_DIR)
