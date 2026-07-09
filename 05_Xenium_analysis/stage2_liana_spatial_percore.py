"""
Stage 2 - ITM Xenium validation: spatial ligand-receptor inference (LIANA)
==========================================================================
Run on the WSL workstation in the `liana` env, on the .h5ad written by stage 1
(data_itm_xenium_stage1_for_liana.h5ad), which already contains:
  - obsm['spatial']  (x_centroid, y_centroid; microns)
  - obs['cell_type'], obs['response'], obs['region'], obs['sample'],
    obs['patient_id']
  - X = log-normalised expression

Addresses manuscript claims:
  (i)   decreased immune-activating interactions  (R should show MORE)
  (ii)  altered tumour-vasculature signalling
  (iii) impaired cDC1 trafficking & lack of immune-vascular crosstalk (R should
        show MORE DC<->T and immune<->endothelial signalling)

KEY METHODOLOGY
  * Proximity is computed PER CORE (loop over obs['sample']). TMA cores are
    independent tissues with separate coordinate frames - pooling would fabricate
    cross-core proximity. This is the single most important correction.
  * resource_name='consensus'  (HUMAN; not 'mouseconsensus').
  * spatial_key='spatial'; bandwidth in microns (default LIANA bandwidth is 250).
  * use_raw=False  -> uses X (log-normalised).
  * Statistical unit = patient: per-core scores are averaged to patient before
    R vs NR testing, mirroring stage 1. Reported with nominal + BH p-values.

NOTE: first rank_aggregate call is slow (numba compiles). Copy the .h5ad to the
Linux filesystem (e.g. ~/projects/itm/) rather than running off /mnt/c.
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import liana as li
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IN_H5AD = os.path.expanduser("~/projects/itm/data_itm_xenium_stage1_for_liana.h5ad")
OUT_DIR = os.path.expanduser("~/projects/itm/stage2_liana_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPBY = "cell_type"
RESOURCE = "consensus"          # human
SPATIAL_KEY = "spatial"
BANDWIDTH = 100                 # microns; sensitivity at 50/150/250 recommended
EXPR_PROP = 0.1                 # raised from 0.01 given sparse Xenium panel
MIN_CELLS = 10                  # min cells per cell-type within a core
N_PERMS = 1000
N_JOBS = 8                      # set to your core count
SEED = 1337

MIN_CELLS_PER_CORE = 200        # skip very sparse cores
MIN_CELLTYPES_PER_CORE = 2

RESPONSE_ORDER = ["R", "NR"]
REGION_ORDER = ["high_TILs", "peritumour", "high_tumour"]

# --------------------------------------------------------------------------- #
# Targeted hypotheses (mapped to claims). Filtering happens AFTER the global run,
# so the per-core table is preserved for any later exploration.
# Ligand/receptor names match the 'consensus' resource (complexes use '_').
# --------------------------------------------------------------------------- #
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
    # NB: LIANA's consensus DB encodes VEGFR1/2 as the complex 'FLT1_KDR', so
    # KDR/FLT1 are matched by complex-MEMBERSHIP (see targeted_mask). NRP1, PGF,
    # ANGPT2, TEK are absent from this 480-gene panel and cannot be tested.
    "ii_tumour_vasculature": [
        ("VEGFA", "KDR"), ("VEGFA", "FLT1"),   # -> matches VEGFA -> FLT1_KDR
        ("VEGFA", "CD44"), ("VEGFA", "EGFR"),
        ("ANGPT1", "TIE1"),
    ],
}
# Cell-type pairs of interest (source, target). Adjust to your exact cell_type
# strings if they differ. crosstalk is tested in both directions.
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
# Load
# --------------------------------------------------------------------------- #
print("Loading", IN_H5AD)
adata = sc.read_h5ad(IN_H5AD)
adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
print(adata)
print("\ncell_type values:", sorted(adata.obs["cell_type"].unique()))
print("samples (cores):", adata.obs["sample"].nunique())

# sample -> metadata map (one row per core)
core_meta = (adata.obs[["sample", "patient_id", "response", "region"]]
             .drop_duplicates().set_index("sample"))

# --------------------------------------------------------------------------- #
# 1. Per-core spatial rank_aggregate
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70, "\n1. PER-CORE LIANA\n", "=" * 70)
per_core = []
for core, sub_idx in adata.obs.groupby("sample", observed=True).groups.items():
    ad_c = adata[sub_idx].copy()
    n_ct = ad_c.obs["cell_type"].nunique()
    if ad_c.n_obs < MIN_CELLS_PER_CORE or n_ct < MIN_CELLTYPES_PER_CORE:
        print(f"  skip {core}: n={ad_c.n_obs}, n_celltypes={n_ct}")
        continue
    try:
        li.mt.rank_aggregate(
            ad_c,
            groupby=GROUPBY,
            resource_name=RESOURCE,
            spatial_key=SPATIAL_KEY,
            spatial_kwargs={"kernel": "gaussian", "bandwidth": BANDWIDTH},
            expr_prop=EXPR_PROP,
            min_cells=MIN_CELLS,
            use_raw=False,
            n_perms=N_PERMS,
            n_jobs=N_JOBS,
            seed=SEED,
            verbose=False,
        )
    except Exception as e:
        print(f"  FAILED {core}: {e}")
        continue
    df = ad_c.uns["liana_res"].copy()
    df["sample"] = core
    df["patient_id"] = core_meta.loc[core, "patient_id"]
    df["response"] = core_meta.loc[core, "response"]
    df["region"] = core_meta.loc[core, "region"]
    per_core.append(df)
    print(f"  ok   {core}: {len(df)} interactions  "
          f"({core_meta.loc[core,'response']}/{core_meta.loc[core,'region']})")

liana_all = pd.concat(per_core, ignore_index=True)
liana_all.to_csv(os.path.join(OUT_DIR, "liana_per_core_spatial.csv"), index=False)
print("\nSaved per-core table:", liana_all.shape)
print("Columns:", list(liana_all.columns))

# unique interaction id (cell-type pair + LR pair)
liana_all["ct_pair"] = liana_all["source"] + " -> " + liana_all["target"]
liana_all["lr_pair"] = (liana_all["ligand_complex"] + " -> "
                        + liana_all["receptor_complex"])
liana_all["interaction"] = liana_all["ct_pair"] + " | " + liana_all["lr_pair"]

# "strength" from magnitude_rank (lower rank = stronger); -log10 for an intuitive
# higher = stronger scale. specificity_rank handled the same way if preferred.
liana_all["magnitude_strength"] = -np.log10(
    liana_all["magnitude_rank"].clip(lower=1e-10))

# --------------------------------------------------------------------------- #
# 2. R vs NR testing on targeted interactions (patient-level)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70, "\n2. TARGETED R vs NR\n", "=" * 70)

def _members(complex_str):
    """Subunits of a LIANA complex, e.g. 'ITGAL_ITGB2' -> {'ITGAL','ITGB2'}."""
    return set(str(complex_str).split("_"))

def targeted_mask(df, claim):
    """Match on gene MEMBERSHIP within ligand/receptor complexes, so a target
    like ('VEGFA','KDR') matches the complex 'VEGFA -> FLT1_KDR'."""
    pairs = TARGET_LR[claim]
    cts = set(TARGET_CT_PAIRS[claim])

    def row_ok(r):
        if (r["source"], r["target"]) not in cts:
            return False
        lig, rec = _members(r["ligand_complex"]), _members(r["receptor_complex"])
        return any(lg in lig and rc in rec for lg, rc in pairs)

    return df[df.apply(row_ok, axis=1)].copy()

def patient_level_tests(df, value_col, by_region):
    """Average per-core score to patient, then Mann-Whitney NR vs R.
    by_region=True -> per region; False -> region-pooled (one value per patient)."""
    keys = ["interaction", "patient_id", "response"] + (["region"] if by_region else [])
    pat = df.groupby(keys, observed=True)[value_col].mean().reset_index()
    grp_keys = ["interaction"] + (["region"] if by_region else [])
    recs = []
    for key, sub in pat.groupby(grp_keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        g = {r: v[value_col].dropna().values for r, v in sub.groupby("response", observed=True)}
        if not {"NR", "R"}.issubset(g) or len(g["NR"]) < 3 or len(g["R"]) < 3:
            continue
        stat, p = mannwhitneyu(g["NR"], g["R"], alternative="two-sided")
        rec = dict(zip(grp_keys, key))
        rec.update({
            "n_NR": len(g["NR"]), "n_R": len(g["R"]),
            "median_NR": float(np.median(g["NR"])), "median_R": float(np.median(g["R"])),
            "delta_R_minus_NR": float(np.median(g["R"]) - np.median(g["NR"])),
            "pvalue": p,
        })
        recs.append(rec)
    out = pd.DataFrame(recs)
    if len(out):
        out["p_adj_BH"] = multipletests(out["pvalue"], method="fdr_bh")[1]
        out = out.sort_values("pvalue")
    return out

all_region, all_pooled = [], []
for claim in TARGET_LR:
    sub = targeted_mask(liana_all, claim)
    if sub.empty:
        print(f"  {claim}: no targeted interactions detected (check cell_type "
              f"strings / LR names).")
        continue
    sub.to_csv(os.path.join(OUT_DIR, f"targeted_percore_{claim}.csv"), index=False)
    r_reg = patient_level_tests(sub, "magnitude_strength", by_region=True)
    r_pool = patient_level_tests(sub, "magnitude_strength", by_region=False)
    for d in (r_reg, r_pool):
        if len(d):
            d.insert(0, "claim", claim)
    all_region.append(r_reg); all_pooled.append(r_pool)
    print(f"  {claim}: {sub['interaction'].nunique()} interactions tested")

if all_region:
    reg = pd.concat([d for d in all_region if len(d)], ignore_index=True)
    reg.to_csv(os.path.join(OUT_DIR, "targeted_tests_per_region.csv"), index=False)
    print("\nPer-region targeted tests (top):")
    print(reg.head(15).to_string(index=False))
if all_pooled:
    pool = pd.concat([d for d in all_pooled if len(d)], ignore_index=True)
    pool.to_csv(os.path.join(OUT_DIR, "targeted_tests_pooled.csv"), index=False)
    print("\nRegion-pooled targeted tests (top):")
    print(pool.head(15).to_string(index=False))

# --------------------------------------------------------------------------- #
# 3. Detection-frequency view (robust for sparse TMA cores)
#    fraction of cores where the interaction is specific (specificity_rank<=0.05)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70, "\n3. DETECTION FREQUENCY (R vs NR)\n", "=" * 70)
from scipy.stats import fisher_exact

liana_all["is_specific"] = liana_all["specificity_rank"] <= 0.05

# fraction of cores specific, per response (interpretation reference)
freq = (liana_all.groupby(["interaction", "response"], observed=True)["is_specific"]
        .mean().reset_index()
        .pivot(index="interaction", columns="response", values="is_specific")
        .reset_index())
freq.to_csv(os.path.join(OUT_DIR, "detection_frequency_by_response.csv"), index=False)
print("Saved detection-frequency table:", freq.shape)

# Per-interaction Fisher's exact: specific vs non-specific cores, R vs NR.
# 2x2 = [[R_spec, R_nonspec], [NR_spec, NR_nonspec]] over cores where the
# interaction was tested at all (i.e. cell types present in that core).
counts = (liana_all.groupby(["interaction", "response"], observed=True)["is_specific"]
          .agg(n_spec="sum", n_total="count").reset_index())
piv = counts.pivot(index="interaction", columns="response", values=["n_spec", "n_total"])
recs = []
for inter in piv.index:
    try:
        r_s = piv.loc[inter, ("n_spec", "R")];  r_t = piv.loc[inter, ("n_total", "R")]
        nr_s = piv.loc[inter, ("n_spec", "NR")]; nr_t = piv.loc[inter, ("n_total", "NR")]
    except KeyError:
        continue
    if pd.isna(r_t) or pd.isna(nr_t) or (r_t + nr_t) == 0:
        continue
    table = [[int(r_s), int(r_t - r_s)], [int(nr_s), int(nr_t - nr_s)]]
    # require the interaction specific in >=1 core somewhere, else skip
    if (r_s + nr_s) == 0:
        continue
    _, p = fisher_exact(table, alternative="two-sided")
    recs.append({
        "interaction": inter,
        "frac_spec_R": r_s / r_t if r_t else np.nan,
        "frac_spec_NR": nr_s / nr_t if nr_t else np.nan,
        "n_cores_R": int(r_t), "n_cores_NR": int(nr_t),
        "pvalue": p,
    })
fisher_df = pd.DataFrame(recs)
if len(fisher_df):
    fisher_df["p_adj_BH"] = multipletests(fisher_df["pvalue"], method="fdr_bh")[1]
    fisher_df["abs_frac_diff"] = (fisher_df["frac_spec_R"] - fisher_df["frac_spec_NR"]).abs()
    fisher_df = fisher_df.sort_values("pvalue")
    fisher_df.to_csv(os.path.join(OUT_DIR, "detection_frequency_fisher.csv"), index=False)
    print("\nTop differential-prevalence interactions (Fisher's exact):")
    print(fisher_df.head(15).to_string(index=False))

print("\nDONE. Outputs in:", OUT_DIR)
print("Reminder: report nominal + BH; lead on direction-of-effect consistency.")
