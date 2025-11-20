library(Seurat) 
library(ggplot2) 
library(ggpubr)
#library(dittoSeq) 
library(RColorBrewer)
library(miloR)
library(SingleCellExperiment)
library(scater)
library(scran)
library(dplyr)
library(tidyr)
library(patchwork)
library(scRNAtoolVis)
library(ggsc)
# library(clusterProfiler)
# library(enrichplot)

################################################################################

# Load the dataset
itm_obj <- readRDS("seurat_objects/ITM_n24_AllCells_PDEMUX_cleaned_filtered_CODEX_markers_9May25.rds")

# Define the cell type groups
groupings <- list(
  Melanocyte = c("Melanocyte"),
  Endothelial = c("Endothelial"),
  Lymph_endo = c("Endo_Lymph"),
  DC = c("DC1"),                                 ###removed "mregDC", "DC2", "pDC"
  CD8_T = c("CD8 Tcell", "Cycling T_ILC"),
  CD4_T = c("Tfh"),
  Treg = c("Treg"),
  B_cell = c("Bcell_naive", "Bcell_mem", "Cycling B"),
  Macrophage = c("SPP1_Macrophage", "Cycling M", "Monocyte", "CXCL10_Macrophage")
)


# Subset to include only the cells from specified groups
selected_cells <- itm_obj@meta.data$Minor_exp %in% unlist(groupings)
itm_codex_obj <- subset(itm_obj, cells = rownames(itm_obj@meta.data)[selected_cells])


# Add a new column for grouped labels
itm_codex_obj@meta.data$cath_grouped_Minor_exp <- NA
for (group in names(groupings)) {
  itm_codex_obj@meta.data$cath_grouped_Minor_exp[itm_codex_obj@meta.data$Minor_exp %in% groupings[[group]]] <- group
}


### Save object
saveRDS(itm_codex_obj, file = "seurat_objects/ITM_n24_AllCells_PDEMUX_cleaned_filtered_CODEX_markers_9May25.rds")

#itm_codex_obj <- readRDS("seurat_objects/ITM_n24_AllCells_PDEMUX_cleaned_filtered_CODEX_markers_cDC1only_18Nov24.rds")


################################################################################

# Check the number of cells
cell_counts <- itm_codex_obj@meta.data %>%
  group_by(SAMPLE_ID, cath_grouped_Minor_exp) %>%
  summarise(n_cells = n(), .groups = "drop")

cell_counts_wide <- cell_counts %>%
  tidyr::pivot_wider(names_from = cath_grouped_Minor_exp, values_from = n_cells, values_fill = 0)

head(cell_counts_wide)


# Plot faceted bar charts per cell type
# Convert wide to long format
cell_counts_long <- cell_counts_wide %>%
  pivot_longer(
    cols = -SAMPLE_ID,
    names_to = "cath_grouped_Minor_exp",
    values_to = "n_cells"
  )

ggplot(cell_counts_long, aes(x = SAMPLE_ID, y = n_cells)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  facet_wrap(~ cath_grouped_Minor_exp, scales = "free_y") +  # free_y so each plot has its own y-axis
  theme_minimal() +
  labs(
    x = "Patient (SAMPLE_ID)",
    y = "Number of cells"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold")
  )

################################################################################

codex_markers_genes <- c("SOX10",    # SOX10
                         "PECAM1",   # CD31
                         "CD34",     # CD34
                         "THBD",     # CD141
                         "ITGAX",    # CD11c
                         "ITGAM",    # CD11b
                         "CD68",     # CD68
                         "CD14",     # CD14
                         "CD163",    # CD163
                         "MRC1",     # CD206
                         "CD19",     # CD19
                         "MS4A1",    # CD20
                         "CD3E",     # CD3e
                         "CD4",      # CD4
                         "FOXP3",    # FoxP3
                         "CD8A")     # CD8



DefaultAssay(itm_codex_obj) <- "RNA"
Idents(itm_codex_obj) <- itm_codex_obj@meta.data$cath_grouped_Minor_exp

# Draw dot plot
cell_order <- c("Melanocyte", "Endothelial", "Lymph_endo", "Macrophage", "DC", "B_cell", "CD4_T", "Treg", "CD8_T")

# Generate dot plot with ordered identities
dotplot_codex <- DotPlot(
  itm_codex_obj,
  features = codex_markers_genes,
  idents = factor(Idents(itm_codex_obj), levels = cell_order)
) + 
  scale_color_gradient2(low = "deepskyblue4", mid = "beige", high = "coral") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggsave("plots/cath_obj_v3/itm_celltype_codex_markers_RNA_filtered.png", 
       plot = dotplot_codex, units = "cm", width = 17, height = 8, dpi = 300)


################################################################################

## immune cell proportion analysis

# Extract relevant metadata and filter out "Melanocyte"
meta_data <- itm_codex_obj@meta.data %>%
  filter(cath_grouped_Minor_exp != "Melanocyte")

# Calculate TME cell proportions
tme_cell_proportions <- meta_data %>%
  group_by(SAMPLE_ID, cath_timepoint_response, cath_grouped_Minor_exp) %>%
  summarise(count = n()) %>%
  group_by(SAMPLE_ID, cath_timepoint_response) %>%
  mutate(proportion = count / sum(count)) %>%
  ungroup()

# Save as CSV
write.csv(tme_cell_proportions, "codex-markers_singlecell_analysis/tme_cell_type_proportions.csv", row.names = FALSE)


tme_cell_proportions$cath_timepoint_response <- factor(
  tme_cell_proportions$cath_timepoint_response,
  levels = c("PRE_Responsive", "PRE_Resistant", "PROG_Resistant")
)

# Plot violin plot
ggplot(tme_cell_proportions, aes(
  x = cath_grouped_Minor_exp, 
  y = proportion, 
  fill = cath_timepoint_response
)) +
  geom_violin(scale = "width", trim = FALSE, position = position_dodge(0.8)) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8), size = 0.5, alpha = 0.7) +
  stat_compare_means(
    aes(group = cath_timepoint_response),
    method = "kruskal.test",
    label = "p.format",
    hide.ns = TRUE,
    label.y = 1.3
  )

## compare TME cell proportions between response groups - per cell type

response_colors <- c(
  "PRE_Responsive" = "deepskyblue4",
  "PRE_Resistant" = "burlywood3",
  "PROG_Resistant" = "coral4"
)

png("plots/cath_new_obj/ITM_CODEXcelltypes_response-compare.png", units = "in", width = 6, height = 6, res = 300)
ggplot(tme_cell_proportions, aes(
  x = cath_timepoint_response, 
  y = proportion, 
  fill = cath_timepoint_response
)) +
  geom_violin(trim = FALSE) +
  geom_jitter(
    size = 0.5, 
    alpha = 0.7, 
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)
  ) +
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.format", 
    hide.ns = TRUE,
    comparisons = list(
      c("PRE_Responsive", "PRE_Resistant"),
      c("PRE_Responsive", "PROG_Resistant"),
      c("PRE_Resistant", "PROG_Resistant")
    ),
    size = 2.5, # Adjust size of p-value labels
    vjust = -3 # Move the p-value labels upwards
  ) +
  facet_wrap(~ cath_grouped_Minor_exp, scales = "free_y") + # Facet by cell type
  labs(
    title = "TME cell type proportions by response group",
    x = "Response Group",
    y = "Proportion",
    fill = "Response Group"
  ) +
  scale_fill_manual(values = response_colors) + # Apply custom colors
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.4))) + # Add padding to y-axis (20% above max)
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )
dev.off()

################################################################################

### Chemokine cytokine gene expression

tme_obj <- subset(itm_obj, subset = (cath_grouped_Minor_exp == "Melanocyte"), invert = TRUE)
Idents(tme_obj) <- "cath_grouped_Minor_exp"
DefaultAssay(tme_obj) <- "RNA"

chemo_cyto_genes <- c("CCL2", "CCL3", "CCL4", "CCL5", "CCL17", "CCL22",
                      "CXCL2", "CXCL3", "CXCL8","CXCL13", "CXCL16",
                      "CXCR4", "XCR1",
                      "IL1B", "IL16", "IL32",
                      "IFNG", "TNF", "LTB", "TGFB1", "TNFSF9")
chemo_cyto_genes_df <- data.frame(gene = chemo_cyto_genes)

png("plots/cath_new_obj/ITM_CODEXcelltypes_chemo-cyto_dotplot.png", units = "in", width = 12, height = 5, res = 300)
jjDotPlot(object = tme_obj,
          gene = chemo_cyto_genes_df$gene,
          id = 'celltype',
          split.by = 'cath_timepoint_response',
          split.by.aesGroup = T,
          ytree = F)
dev.off()

################################################################################

# cDC1 phenotype markers

dc_obj <- subset(itm_obj, subset = (cath_grouped_Minor_exp == "DC"))

Idents(dc_obj) <- "cath_grouped_Minor_exp"
DefaultAssay(dc_obj) <- "RNA"

# maturation_genes <- c("CD40", "CD80", "CD86", "RELB", "CD83")
# regulatory_genes <- c("CD274", "PDCD1LG2", "CD200", "FAS", "ALDH1A2", "SOCS1", "SOCS2")
# migration_genes <- c("CCR7", "MYO1G", "CXCL16", "ADAM8", "ICAM1", "FSCN1", "MARCKS", "MARCKSL1")
# trl_genes <- c("MYD88", "MAVS", "TLR9", "TLR8", "TLR7", "TLR6", "TLR5", "TLR4", "TLR3", "TLR2", "TLR1")
# cross_presentation_genes <- c("MS4A7", "XCR1", "CLEC9A")

dc_genes <- c("CD40", "CD80", "CD86", "RELB", "CD83",
              "CCR7", "MYO1G", "CXCL16", "ADAM8", "ICAM1", "FSCN1", "MARCKS", "MARCKSL1",
              "CD274", "PDCD1LG2", "CD200", "FAS", "ALDH1A2", "SOCS1", "SOCS2",
              "MS4A7", "XCR1", "CLEC9A"
)

dc_genes_df <- data.frame(gene = dc_genes)

#group_order<- c("PROG_Resistant", "PRE_Resistant",  "PRE_Responsive")

png("plots/ITM_CODEXcelltypes_DC-genes_dotplot.png", units = "in", width = 10, height = 4, res = 300)
jjDotPlot(object = dc_obj,
          gene = dc_genes_df$gene,
          id = 'celltype',
          split.by = 'cath_timepoint_response',
          split.by.aesGroup = T,
          #cluster.order = group_order,
          dot.col = c("deepskyblue4", "beige", "coral1"),
          ytree = F)
dev.off()

################################################################################

# LEC tip-like signature expression

lec_obj <- subset(itm_obj, subset = cath_grouped_Minor_exp %in% c("Lymph_endo"))

Idents(lec_obj) <- "cath_timepoint_response"
DefaultAssay(lec_obj) <- "RNA"

tip_like <- c("HIF1A", "ICAM1", "CCL2", "CXCL1", "CXCL2", "FLT4", "DLL4", "NFKB1", "LDB2", "NOTCH1", "CD34", "PROX1")
ap_like <- c("CD74", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DRB1", "COL1A1", "COL3A1", "COL6A1", "TSC22D3", "IFNGR1")
lec_genes <- c(tip_like, ap_like)
lec_genes_df <- data.frame(gene = lec_genes)

png("plots/cath_obj_v3/itm_LEC_subtype_dotplot_by_timepointresponse_RNA.png", 
    units = "in", width = 12, height = 4, res = 300)
jjDotPlot(object = lec_obj,
          gene = lec_genes_df$gene,
          id = 'cath_grouped_Minor_exp',
          split.by = 'cath_timepoint_response',
          split.by.aesGroup = T,
          #cluster.order = group_order,
          dot.col = c("deepskyblue4", "beige", "coral1"),
          ytree = F)
dev.off()

# dotplot <- DotPlot(lec_obj, features = c(tip_like, ap_like)) +
#   scale_color_gradient2(low = "deepskyblue4", mid = "beige", high = "coral") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))
# 
# ggsave("plots/cath_obj_v3/itm_LEC_subtype_dotplot_by_timepointresponse_RNA.png", plot = dotplot, width = 9, height = 3, dpi = 300)

################################################################################

# TEC tip-like signature expression

itm_obj <- readRDS("seurat_objects/ITM_n24_AllCells_PDEMUX_cleaned_filtered_CODEX_markers_9May25.rds")

tec_obj <- subset(itm_obj, subset = cath_grouped_Minor_exp %in% c("Endothelial"))

Idents(tec_obj) <- "cath_timepoint_response"
DefaultAssay(tec_obj) <- "RNA"

tip_like <- c("HIF1A", "APLN", "DLL4", "NOTCH1", "NOTCH4")
stalk_like <- c("CXCL2", "CXCL3", "CXCL8", "TEK")
tec_genes <- c(tip_like, stalk_like)
tec_genes_df <- data.frame(gene = tec_genes)

png("plots/cath_obj_v3/itm_TEC_pheno_dotplot_by_timepointresponse_RNA.png", 
    units = "in", width = 8.5, height = 4, res = 300)
jjDotPlot(object = tec_obj,
          gene = tec_genes_df$gene,
          id = 'cath_grouped_Minor_exp',
          split.by = 'cath_timepoint_response',
          split.by.aesGroup = T,
          #cluster.order = group_order,
          dot.col = c("deepskyblue4", "beige", "coral1"),
          ytree = F)
dev.off()

# dotplot <- DotPlot(tec_obj, features = c(tip_like, stalk_like)) +
#   scale_color_gradient2(low = "deepskyblue4", mid = "beige", high = "coral") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))
# 
# ggsave("plots/cath_obj_v3/itm_TEC_pheno_dotplot_by_timepointresponse_RNA.png", 
#        plot = dotplot, width = 7, height = 3, dpi = 300)

### PRE
# tec_obj_pre <- subset(tec_obj, subset = timepoint %in% c("PRE"))
# 
# dotplot <- DotPlot(tec_obj_pre, features = c(tip_like, stalk_like)) +
#   scale_color_gradient2(low = "deepskyblue4", mid = "beige", high = "coral") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))
# 
# ggsave("plots/cath_obj_v3/itm_TEC_pheno_dotplot_PREresponse_RNA.png", 
#        plot = dotplot, width = 7, height = 3, dpi = 300)



################################################################################
library(scGSVA)

hsko<-buildAnnot(species="human",keytype="SYMBOL",anntype="KEGG")
hsgobp<-buildAnnot(species="human",keytype="SYMBOL",anntype="GOBP")

tumour_obj <- subset(itm_obj, subset = cath_grouped_Minor_exp %in% c("Melanocyte"))

res_tumour <-scgsva(tumour_obj, hsko, method="ssgsea")

pathways_tumour <- findPathway(res_tumour, group = "cath_timepoint_response")
sig_pathways_tumour <- sigPathway(res_tumour, group = "cath_timepoint_response")

write.csv(pathways_tumour, file = "results/scGSEA/ITM_tumour_scGSVA_pathways.csv", row.names = TRUE)
write.csv(sig_pathways_tumour, file = "results/scGSEA/ITM_tumour_scGSVA_pathways_significant.csv", row.names = TRUE)


# ridgePlot(res_tumour,features="VEGF.signaling.pathway",group_by="cath_timepoint_response", split.by = "cath_grouped_Minor_exp")+
#   scale_fill_manual(values = response_colors)

pathway_groups <- list(
  Immune_TME = c(
    "Antigen.processing.and.presentation",
    "Chemokine.signaling.pathway",
    "Complement.and.coagulation.cascades"
  ),
  ECM_Adhesion_Migration = c(
    "Cell.adhesion.molecules",
    "ECM.receptor.interaction",
    "Focal.adhesion"
  ),
  Angiogenesis_Vascular = c(
    "VEGF.signaling.pathway"
  ),
  Energy_Carbohydrate_Metabolism = c(
    "Glycolysis...Gluconeogenesis",
    "Citrate.cycle..TCA.cycle.",
    "Galactose.metabolism",
    "Pentose.and.glucuronate.interconversions",
    "Propanoate.metabolism"
  ),
  Lipid_Metabolism_PPAR = c(
    "PPAR.signaling.pathway",
    "Fatty.acid.degradation",
    "alpha.Linolenic.acid.metabolism",
    "Glycerophospholipid.metabolism"
  ),
  AminoAcid_Nitrogen_Metabolism = c(
    "Alanine..aspartate.and.glutamate.metabolism",
    "Lysine.degradation"
  ),
  Glycan_Biology = c(
    "Amino.sugar.and.nucleotide.sugar.metabolism",
    "Glycosaminoglycan.degradation",
    "Other.glycan.degradation"
  ),
  Vitamins_Cofactors_Redox = c(
    "Retinol.metabolism",
    "Ascorbate.and.aldarate.metabolism",
    "Biotin.metabolism",
    "Porphyrin.metabolism",
    "Vitamin.digestion.and.absorption",
    "Glutathione.metabolism"
  ),
  Proteostasis = c(
    "Proteasome",
    "Protein.digestion.and.absorption"
  ),
  DNA_Damage_CellFate = c(
    "p53.signaling.pathway",
    "Apoptosis",
    "Homologous.recombination",
    "Oocyte.meiosis"
  ),
  Transcriptional_Machinery = c(
    "RNA.polymerase"
  ),
  Ion_Homeostasis = c(
    "Aldosterone.regulated.sodium.reabsorption"
  )
)



# -----------------------------------
# Helper: safe file name for each group
# -----------------------------------
.safe_name <- function(x) gsub("[^A-Za-z0-9._-]+", "_", x)

# -----------------------------------
# Create output dir
# -----------------------------------
out_dir <- file.path("plots", "GSVA")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------
# Loop: one plot per group (all its pathways)
# -----------------------------------
library(ggplot2)

# --- per-subplot size (inches) ---
panel_w <- 4
panel_h <- 3

# --- colors you set ---
response_colors <- c(
  "PRE_Responsive" = "#F9E58A",
  "PRE_Resistant"  = "#7AB7BA",
  "PROG_Resistant" = "#8EA5AA"
)

# --- safe filename helper ---
.safe_name <- function(x) gsub("[^A-Za-z0-9._-]+", "_", x)

# --- output root ---
root_dir <- file.path("plots", "GSVA", "per_pathway")
if (!dir.exists(root_dir)) dir.create(root_dir, recursive = TRUE, showWarnings = FALSE)

for (grp in names(pathway_groups)) {
  feats <- unique(pathway_groups[[grp]])
  if (!length(feats)) next
  
  # make a folder per group
  out_dir_grp <- file.path(root_dir, .safe_name(grp))
  if (!dir.exists(out_dir_grp)) dir.create(out_dir_grp, recursive = TRUE, showWarnings = FALSE)
  
  for (ftr in feats) {
    # try each feature separately so one failure doesn't stop the loop
    try({
      p <- ridgePlot(
        res_tumour,
        features = ftr,
        group_by = "cath_timepoint_response",
        split.by = "cath_grouped_Minor_exp"
      ) +
        scale_fill_manual(values = response_colors, breaks = names(response_colors)) +
        labs(title = ftr, y = NULL) +                                # removes y-axis title
        theme(
          plot.margin = margin(3, 3, 3, 3),
          axis.title.y = element_blank()                              # (explicit) remove y-axis title
          # If you also want to remove y tick labels entirely, add:
          # axis.text.y  = element_blank(),
          # axis.ticks.y = element_blank()
        )
      p <- p +
        theme(
          plot.title = element_text(size = 8),  # was default ~11
          strip.text = element_text(size = 8),   # facet strip (e.g., "Melanocyte")
          axis.title.x = element_text(size = 8),
          axis.text.x  = element_text(size = 8),
          plot.margin  = margin(t = 8, r = 8, b = 10, l = 8)  # tiny padding, safe default
        )
      
      fout <- file.path(out_dir_grp, paste0(.safe_name(ftr), ".png"))
      ggplot2::ggsave(fout, p, width = panel_w, height = panel_h, dpi = 300, limitsize = FALSE)
    }, silent = TRUE)
  }
}


################################################################################



# hypoxia signature expression
Idents(tumour_obj) <- "cath_timepoint_response"
DefaultAssay(tumour_obj) <- "RNA"

hypo_genes <- c('VEGF', 'BNIP3', 'ADM', 'SLC16A3', 'DDIT4', 'HILPDA')
hypo_genes_df <- data.frame(gene = hypo_genes)

png("plots/cath_obj_v3/itm_TUMOUR_pheno_dotplot_by_timepointresponse_RNA.png", 
    units = "in", width = 8, height = 4, res = 300)
jjDotPlot(object = tumour_obj,
          gene = hypo_genes_df$gene,
          id = 'cath_grouped_Minor_exp',
          split.by = 'cath_timepoint_response',
          split.by.aesGroup = T,
          #cluster.order = group_order,
          dot.col = c("deepskyblue4", "beige", "coral1"),
          ytree = F)
dev.off()



################################################################################


library(escape)

# If your expression is log-normalized in RNA@data, use kcdf="Gaussian".
# If you're using raw counts (RNA@counts), use kcdf="Poisson" and set slot = "counts".
# Term–gene table
gobp_tbl <- msigdbr(species = "Homo sapiens", category = "C5", subcategory = "GO:BP") |>
  select(term = gs_name, gene = gene_symbol)

# Harmonize to your object and size filter
genes_obj <- rownames(tumour_obj)
annot_df <- gobp_tbl |> filter(gene %in% genes_obj) |>
  group_by(term) |> filter(dplyr::n() >= 10, dplyr::n() <= 500) |> ungroup()

# Run scGSVA (no gene.sets needed when annot is provided)
gobp_res_tumour <- scgsva(
  object      = tumour_obj,
  annot       = annot_df,          # <-- key fix
  assay       = "RNA",
  slot        = "data",
  method      = "ssgsea",
  kcdf        = "Gaussian",
  ssgsea.norm = TRUE,
  min.sz      = 10,
  max.sz      = 500,
  parallel.sz = max(1, parallel::detectCores()-1),
  return_obj  = TRUE
)

# scgsva() (escape) returns a Seurat object with pathway scores added to meta.data.
# You can confirm columns were added:
head(colnames(gobp_res_tumour@meta.data))

# Example: access one term’s scores
# gobp_res_tumour@meta.data[["GO_ASCORBATE_AND_ALDARATE_METABOLIC_PROCESS"]]



