# === CI-Lib R Visualization Templates ===
# Run from Python via: Rscript path/to/script.R [args]
# Or source this file in R and call any function

library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)
library(RColorBrewer)
library(scales)
library(patchwork)

`%||%` <- function(a, b) if (is.null(a)) b else a

theme_ci <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# --- Scatter plot with density contours ---
scatter_density <- function(data, x, y, color_col = NULL, title = "") {
  p <- ggplot(data, aes_string(x = x, y = y)) +
    geom_point(alpha = 0.6, size = 2, aes_string(color = color_col)) +
    geom_density_2d(alpha = 0.3, color = "gray40") +
    scale_color_viridis_d(option = "D") +
    labs(title = title, x = x, y = y) +
    theme_ci
  return(p)
}

# --- Correlation heatmap ---
correlation_heatmap <- function(data, title = "Correlation Matrix") {
  cor_mat <- cor(data %>% select_if(is.numeric))
  cor_df <- as.data.frame(as.table(cor_mat))
  names(cor_df) <- c("Var1", "Var2", "Correlation")

  ggplot(cor_df, aes(Var1, Var2, fill = Correlation)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "#4575B4", mid = "#FFFFBF", high = "#D73027",
                        midpoint = 0, limits = c(-1, 1)) +
    geom_text(aes(label = round(Correlation, 2)), size = 3, color = "black") +
    labs(title = title, x = "", y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold", hjust = 0.5))
}

# --- PCA biplot ---
pca_biplot <- function(data, groups = NULL, title = "PCA Biplot") {
  pca <- prcomp(data %>% select_if(is.numeric), scale. = TRUE)
  pca_df <- as.data.frame(pca$x[, 1:2])
  pca_df$Group <- groups

  var_explained <- round(100 * pca$sdev^2 / sum(pca$sdev^2), 1)

  loadings <- as.data.frame(pca$rotation[, 1:2])
  loadings$var <- rownames(loadings)

  p <- ggplot(pca_df, aes(PC1, PC2)) +
    geom_point(aes(color = Group), size = 3, alpha = 0.7) +
    geom_segment(data = loadings,
                aes(x = 0, y = 0, xend = PC1 * 5, yend = PC2 * 5),
                arrow = arrow(length = unit(0.3, "cm")),
                color = "gray40", alpha = 0.6) +
    geom_text(data = loadings,
             aes(x = PC1 * 5.5, y = PC2 * 5.5, label = var),
             size = 3.5, color = "darkred") +
    labs(title = title,
         x = paste0("PC1 (", var_explained[1], "%)"),
         y = paste0("PC2 (", var_explained[2], "%)")) +
    theme_ci
  return(p)
}

# --- Time series with forecast ---
ts_forecast_plot <- function(data, date_col, value_col,
                             forecast_values = NULL,
                             title = "Time Series") {
  p <- ggplot(data, aes_string(x = date_col, y = value_col)) +
    geom_line(color = "#1f77b4", size = 1) +
    geom_point(color = "#1f77b4", size = 1.5, alpha = 0.5)

  if (!is.null(forecast_values)) {
    forecast_df <- data.frame(
      x = tail(data[[date_col]], length(forecast_values)) + seq_len(length(forecast_values)),
      y = forecast_values
    )
    p <- p + geom_line(data = forecast_df, aes(x, y),
                       color = "#d62728", size = 1, linetype = "dashed")
  }

  p + labs(title = title, x = "Date", y = value_col) + theme_ci
}

# --- Boxplot with jitter ---
boxplot_jitter <- function(data, x_col, y_col, fill_col = NULL, title = "") {
  ggplot(data, aes_string(x = x_col, y = y_col, fill = fill_col %||% x_col)) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
    scale_fill_viridis_d(option = "D") +
    labs(title = title, x = x_col, y = y_col) +
    theme_ci +
    theme(legend.position = "none")
}

# --- Violin plot ---
violin_plot <- function(data, x_col, y_col, fill_col = NULL, title = "") {
  ggplot(data, aes_string(x = x_col, y = y_col, fill = fill_col %||% x_col)) +
    geom_violin(alpha = 0.7, trim = FALSE) +
    geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
    scale_fill_viridis_d(option = "D") +
    labs(title = title, x = x_col, y = y_col) +
    theme_ci +
    theme(legend.position = "none")
}

# --- Heatmap ---
heatmap_plot <- function(data, x_col, y_col, fill_col, title = "") {
  ggplot(data, aes_string(x = x_col, y = y_col, fill = fill_col)) +
    geom_tile(color = "white") +
    scale_fill_viridis_c(option = "D") +
    labs(title = title, x = "", y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold", hjust = 0.5))
}

# --- Ridge plot ---
ridge_plot <- function(data, x_col, y_col, fill_col = NULL, title = "") {
  if (!requireNamespace("ggridges", quietly = TRUE)) {
    return(ggplot() + labs(title = "Install ggridges: install.packages('ggridges')"))
  }
  library(ggridges)
  ggplot(data, aes_string(x = x_col, y = y_col, fill = fill_col %||% y_col)) +
    geom_density_ridges(alpha = 0.7, scale = 0.9) +
    scale_fill_viridis_d(option = "D") +
    labs(title = title) +
    theme_ci +
    theme(legend.position = "none")
}

# --- Dendrogram ---
dendrogram_plot <- function(dist_matrix, method = "ward.D2", title = "Dendrogram") {
  hc <- hclust(dist_matrix, method = method)
  plot(hc, main = title, xlab = "", sub = "", cex = 0.8)
}

# --- ROC curve ---
roc_curve <- function(roc_data, title = "ROC Curve") {
  ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
    geom_line(size = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", alpha = 0.5) +
    scale_color_viridis_d(option = "D") +
    labs(title = title, x = "False Positive Rate", y = "True Positive Rate") +
    coord_equal() +
    theme_ci
}