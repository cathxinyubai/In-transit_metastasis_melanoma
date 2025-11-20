#load library
library(maftools)
library(RColorBrewer)

#load data
itm <- read.maf(maf = "itm.maf", clinicalData = "itm_clinvar.tsv")

#customised colours
vc_cols <- colorspace::qualitative_hcl(6, palette = "Harmonic")
names(vc_cols) = c(  'Translation_Start_Site',
  'Missense_Mutation',
  'Nonsense_Mutation',
  'Splice_Site',
  'In_Frame_Del')

respcolors = colorspace::sequential_hcl(2, palette = "BluGrn")
names(respcolors) <- c("Resistance", "Responsive")
anno_cols = list(Response_Status = respcolors)

#draw and save oncoplot
png("itm_oncoplot_selectedgenes_final1-anno.png", height=35, width=20, res=300, units="cm")
oncoplot(maf = itm, genes = itm_topgenes, colors = vc_cols, clinicalFeatures = "Response_Status", annotationColor=anno_cols, sortByAnnotation = TRUE, legendFontSize = 0.8, annotationFontSize = 0.8)
dev.off()
