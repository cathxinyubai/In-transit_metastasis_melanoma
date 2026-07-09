"""
Stage 1 - ITM Xenium validation: inherent-result analysis
==========================================================
Pretreatment ITM Xenium TMA cohort (cores split into high_tumour / high_TILs /
peritumour in obs['region']). Validates three manuscript findings:
  (i)   decreased immune-activating interactions
  (ii)  altered tumour-vasculature signalling
  (iii) impaired cDC1 trafficking & lack of immune-vascular crosstalk

What this script does, in order:
  0. Load + QC the data structure (patient / core / region; pseudoreplication).
  1. Define + auto-filter marker signatures (drops genes absent from the panel).
  2. Score signatures on log-normalised X (sc.tl.score_genes).
  3. Cell-type composition per patient x region.
  4. PRIMARY: per-region patient-level NR vs R tests (Mann-Whitney, nominal + BH).
  5. POOLED: region-pooled patient-level NR vs R tests (more power, mixes biology).
  6. Prepare + save updated .h5ad for LIANA (adds obsm['spatial'], ID columns).

Statistical unit = patient (melpin). Cores are averaged within patient x region
to avoid pseudoreplication. Stats are reported with BOTH nominal and BH-adjusted
p-values, consistent with framing Xenium as a supportive validation cohort.

Run on laptop with: scanpy, pandas, numpy, scipy, statsmodels, seaborn.
"""

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
DATA_DIR = "/Users/xbai6546/Desktop/ITM_rebuttal/xenium"
IN_H5AD = os.path.join(DATA_DIR, "data_itm_xenium_20260514.h5ad")
OUT_H5AD = os.path.join(DATA_DIR, "data_itm_xenium_stage1_for_liana.h5ad")
OUT_DIR = os.path.join(DATA_DIR, "stage1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RESPONSE_ORDER = ["R", "NR"]
REGION_ORDER = ["high_TILs", "peritumour", "high_tumour"]
MIN_GENES = 2  # minimum present genes to score a signature

# --------------------------------------------------------------------------- #
# 0. Load + QC structure
# --------------------------------------------------------------------------- #
print("=" * 70, "\n0. LOAD + QC\n", "=" * 70)
adata = sc.read_h5ad(IN_H5AD)
print(adata)

obs = adata.obs
print("\nX is log-normalised (expected); raw counts in layers['raw_counts'].")
print("X sparse:", sparse.issparse(adata.X),
      "| raw_counts present:", "raw_counts" in adata.layers)

# ID columns (statistical units)
adata.obs["patient_id"] = adata.obs["melpin"].astype(str)
adata.obs["block_id"] = (adata.obs["melpin"].astype(str) + "_"
                         + adata.obs["pathref"].astype(str))
adata.obs["core_id"] = adata.obs["filename"].astype(str)
adata.obs["patient_region_id"] = (adata.obs["patient_id"] + "_"
                                  + adata.obs["region"].astype(str))

# --- structure summary (confirms pseudoreplication handling) ---
print("\nUnique patients (melpin):", obs["melpin"].nunique())
print("Unique cores (filename):", obs["filename"].nunique())
print("Unique samples:", obs["sample"].nunique())

print("\nPatients per response group:")
print(obs.drop_duplicates("melpin").groupby("response").size())

core_map = obs[["melpin", "response", "region", "core_id"]].drop_duplicates()
print("\nCores per response x region:")
print(core_map.groupby(["response", "region"]).size())

pr = core_map.drop_duplicates(["melpin", "response", "region"])
print("\nPATIENTS per response x region (= n used in primary tests):")
print(pr.groupby(["response", "region"]).size())

cores_per_pr = core_map.groupby(["melpin", "region"]).size()
print("\nMax cores per (patient, region):", int(cores_per_pr.max()),
      "| (patient,region) combos with >1 core:", int((cores_per_pr > 1).sum()))
print("(If >0, averaging cores within patient x region matters for "
      "pseudoreplication.)")

core_map.to_csv(os.path.join(OUT_DIR, "qc_core_patient_region_map.csv"), index=False)

# --------------------------------------------------------------------------- #
# 1. Marker signatures (auto-filtered to panel)
# --------------------------------------------------------------------------- #
print("\n", "=" * 70, "\n1. SIGNATURES\n", "=" * 70)
# NOTE on panel gaps confirmed earlier: PECAM1, CD34, SOX10, MLANA are NOT on
# this Xenium panel. 'endothelial' and 'melanoma' below will auto-drop to the
# available markers; obs already carries Endothelial_score / Melanoma_score
# (cell-type scores) if you prefer those for composition.
marker_sets = {
    "cdc1_activated_dc": ["CD1C", "ITGAX", "THBD", "BATF3", "IRF8", "FCER1A",
                          "CLEC10A", "CD83", "CD86", "CD80", "CD40", "HLA-B", "CD74"],
    "dc_migration_trafficking": ["CXCL16", "ICAM1", "CCL19", "CCR7", "CX3CL1"],
    "dc_tolerogenic_dysfunction": ["CD274", "IDO1", "TGFB1", "IL10", "VSIR",
                                   "HAVCR2", "PDCD1LG2", "NT5E"],
    # endothelial: PECAM1/CD34 absent -> add available vascular markers
    "endothelial": ["PECAM1", "CD34", "KDR", "FLT1", "EGFL7", "VWF", "CLDN5"],
    "lymphatic_endothelial": ["PDPN", "CCL21"],
    "angiogenesis_hypoxia": ["VEGFA", "HIF1A", "CA9", "CXCL12", "FGF2", "FN1",
                             "SPARC", "SPARCL1", "MMP9", "CXCL14"],
    "cytotoxic_tcell": ["CD8A", "CD8B", "PRF1", "GZMB", "GZMK", "GZMH", "GZMA",
                        "CTSW", "NKG7", "GNLY", "KLRD1", "KLRK1"],
    "tcell_activation": ["IFNG", "TNFRSF9", "ICOS", "CD27", "CXCR3", "CXCR6",
                         "CCL5", "CXCL9", "CXCL10", "CXCL11", "LTB"],
    "tcell_exhaustion_checkpoint": ["PDCD1", "TIGIT", "HAVCR2", "LAG3", "CTLA4",
                                    "TOX", "ENTPD1"],
    "melanoma": ["SOX10", "MLANA", "PMEL", "TYR", "DCT", "MITF"],
    "ereg_egfr_family": ["AREG", "EGFR", "ERBB2", "VEGFA", "STAT3", "AKT1"],
    # composite manuscript signatures
    "sig1_responder_immune_active_cdc1_cd8": ["CD8A", "GZMB", "IFNG", "CXCL9",
                                              "CXCL10", "CXCL11", "CCL5", "THBD",
                                              "ITGAX", "BATF3", "CD74"],
    "sig2_resistant_vascular_hypoxic": ["PECAM1", "CD34", "KDR", "FLT1", "VEGFA",
                                        "HIF1A", "CA9", "FN1", "SPARC", "MMP9"],
    "sig3_suppressive_TAM": ["CD68", "CD163", "TREM2", "APOE", "TGFB1", "IL10",
                             "S100A9", "VSIG4"],
    "sig4_exhausted_tcell": ["PDCD1", "TIGIT", "HAVCR2", "LAG3", "CTLA4", "TOX",
                             "ENTPD1"],
}

var_names = set(adata.var_names)
avail_records, score_cols = [], []
for name, genes in marker_sets.items():
    present = [g for g in genes if g in var_names]
    missing = [g for g in genes if g not in var_names]
    avail_records.append({
        "signature": name, "n_total": len(genes), "n_present": len(present),
        "n_missing": len(missing),
        "present_genes": ", ".join(present), "missing_genes": ", ".join(missing),
    })
    if len(present) < MIN_GENES:
        print(f"  SKIP {name}: only {len(present)} gene(s) present")
        continue
    sname = f"xenium_{name}"
    sc.tl.score_genes(adata, gene_list=present, score_name=sname, use_raw=False)
    score_cols.append(sname)

availability = pd.DataFrame(avail_records)
availability.to_csv(os.path.join(OUT_DIR, "signature_gene_availability.csv"), index=False)
print(availability[["signature", "n_present", "n_missing", "missing_genes"]].to_string(index=False))
print(f"\nScored {len(score_cols)} signatures.")

# --------------------------------------------------------------------------- #
# 2. Cell-type composition (patient x region)
# --------------------------------------------------------------------------- #
print("\n", "=" * 70, "\n2. COMPOSITION\n", "=" * 70)
counts = (adata.obs.groupby(["patient_id", "response", "region", "cell_type"],
                            observed=True).size().reset_index(name="n"))
totals = (counts.groupby(["patient_id", "response", "region"], observed=True)["n"]
          .sum().reset_index(name="total_n"))
comp = counts.merge(totals, on=["patient_id", "response", "region"])
comp["percent"] = comp["n"] / comp["total_n"] * 100
comp.to_csv(os.path.join(OUT_DIR, "celltype_composition_patient_region.csv"), index=False)
print("Composition table:", comp.shape)

# --------------------------------------------------------------------------- #
# helper: NR vs R Mann-Whitney over a long table
# --------------------------------------------------------------------------- #
def mwu_tests(summary, value_cols, group_cols):
    """Mann-Whitney NR vs R for each value_col within each group_cols stratum."""
    recs = []
    grp = summary.groupby(group_cols, observed=True) if group_cols else [((), summary)]
    for key, sub in grp:
        key = key if isinstance(key, tuple) else (key,)
        for col in value_cols:
            g = {r: v[col].dropna().values for r, v in sub.groupby("response", observed=True)}
            if not {"NR", "R"}.issubset(g) or len(g["NR"]) == 0 or len(g["R"]) == 0:
                continue
            stat, p = mannwhitneyu(g["NR"], g["R"], alternative="two-sided")
            rec = dict(zip(group_cols, key)) if group_cols else {}
            rec.update({
                "score": col, "comparison": "NR vs R",
                "n_NR": len(g["NR"]), "n_R": len(g["R"]),
                "median_NR": float(np.median(g["NR"])), "median_R": float(np.median(g["R"])),
                "delta_R_minus_NR": float(np.median(g["R"]) - np.median(g["NR"])),
                "pvalue": p,
            })
            recs.append(rec)
    out = pd.DataFrame(recs)
    if len(out):
        out["p_adj_BH_global"] = multipletests(out["pvalue"], method="fdr_bh")[1]
    return out

# --------------------------------------------------------------------------- #
# 3. PRIMARY analysis - per region, patient-level
# --------------------------------------------------------------------------- #
print("\n", "=" * 70, "\n3. PRIMARY: per-region patient-level tests\n", "=" * 70)
sig_summary = (adata.obs.groupby(["patient_id", "response", "region"], observed=True)[score_cols]
               .mean().reset_index())
sig_summary = sig_summary.dropna(subset=score_cols, how="all")
sig_summary.to_csv(os.path.join(OUT_DIR, "signature_scores_patient_region.csv"), index=False)

res_region = mwu_tests(sig_summary, score_cols, ["region"])
# BH correction *within* each region as well
res_region["p_adj_BH_within_region"] = np.nan
for region, idx in res_region.groupby("region").groups.items():
    res_region.loc[idx, "p_adj_BH_within_region"] = multipletests(
        res_region.loc[idx, "pvalue"], method="fdr_bh")[1]
res_region = res_region.sort_values("pvalue")
res_region.to_csv(os.path.join(OUT_DIR, "tests_per_region.csv"), index=False)
print(res_region.head(12).to_string(index=False))

# --------------------------------------------------------------------------- #
# 4. POOLED analysis - regions pooled per patient (more power, mixes biology)
# --------------------------------------------------------------------------- #
print("\n", "=" * 70, "\n4. POOLED: region-pooled patient-level tests\n", "=" * 70)
# one value per patient = mean across that patient's cells (all regions)
pooled_summary = (adata.obs.groupby(["patient_id", "response"], observed=True)[score_cols]
                  .mean().reset_index())
pooled_summary.to_csv(os.path.join(OUT_DIR, "signature_scores_patient_pooled.csv"), index=False)
print("Patients per group (pooled):")
print(pooled_summary.groupby("response").size())

res_pooled = mwu_tests(pooled_summary, score_cols, [])
res_pooled = res_pooled.sort_values("pvalue")
res_pooled.to_csv(os.path.join(OUT_DIR, "tests_region_pooled.csv"), index=False)
print(res_pooled.head(12).to_string(index=False))

# --------------------------------------------------------------------------- #
# 5. Prepare + save updated .h5ad for LIANA (stage 2)
# --------------------------------------------------------------------------- #
print("\n", "=" * 70, "\n5. SAVE FOR LIANA\n", "=" * 70)
# spatial coordinates for LIANA spatial weighting (Xenium centroids in microns)
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
# sanity: confirm cell_type is clean categorical for groupby
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
print("obsm keys:", list(adata.obsm.keys()))
print("cell_type categories:", list(adata.obs["cell_type"].cat.categories))
adata.write_h5ad(OUT_H5AD)
print("Saved:", OUT_H5AD)
print("\nDONE. Outputs in:", OUT_DIR)
