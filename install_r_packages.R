#!/usr/bin/env Rscript
# Install R packages for CI-Lib Visualization Suite
# Run: Rscript install_r_packages.R

packages <- c(
  # Core ggplot2 ecosystem
  "ggplot2", "ggthemes", "scales", "patchwork", "ggpubr", "ggExtra",
  "ggridges", "ggforce", "ggrepel", "ggtext", "ggdist", "ggbeeswarm",
  "ggalluvial", "ggraph", "ggnetwork",

  # Interactive visualizations
  "plotly", "highcharter", "ggiraph", "rbokeh", "echarts4r",
  "leaflet", "leaflet.extras", "mapdeck",

  # Statistical visualizations
  "corrplot", "pheatmap", "ComplexHeatmap", "circlize", "UpSetR",
  "venn", "eulerr", "ggvenn", "ggmosaic", "ggparallel", "ggtern",
  "ggdendro", "ggdendroplot", "dendextend", "factoextra", "ggfortify",

  # Network/Graph
  "igraph", "network", "sna", "intergraph", "visNetwork",
  "DiagrammeR", "graphlayouts", "tidygraph", "ggraph",

  # Geospatial
  "sf", "sp", "raster", "terra", "stars", "tmap", "mapview",
  "osmdata", "rnaturalearth", "rnaturalearthdata",

  # Shiny ecosystem
  "shiny", "shinydashboard", "shinythemes", "shinyWidgets",
  "shinyjs", "bslib", "thematic", "fontawesome", "waiter", "fresh",

  # Data processing
  "dplyr", "tidyr", "readr", "data.table", "dtplyr", "arrow",
  "duckdb", "DBI", "RPostgres", "RMariaDB",

  # Publishing
  "rmarkdown", "quarto", "bookdown", "blogdown", "pkgdown",

  # Utilities
  "RColorBrewer", "viridis", "colorspace", "scico", "paletteer", "ggsci"
)

# Bioconductor packages
bioc_packages <- c("ComplexHeatmap", "circlize", "dendextend")

cat("Installing CRAN packages...\n")
install.packages(setdiff(packages, bioc_packages), repos = "https://cloud.r-project.org/")

cat("Installing Bioconductor packages...\n")
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org/")
}
BiocManager::install(bioc_packages, update = FALSE, ask = FALSE)

cat("All packages installed!\n")