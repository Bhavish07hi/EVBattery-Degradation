
# ============================================================
# 08 - EXPLAINABLE AI: SHAP DEPENDENCE DATA
# ============================================================

library(dplyr)
library(ggplot2)

# Project paths
project_dir <- getwd()

processed_dir <- file.path(project_dir, "data", "processed")
models_dir <- file.path(project_dir, "models")
plots_dir <- file.path(project_dir, "plots")

# Create plots folder if needed
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE)
}

# ------------------------------------------------------------
# 1. LOAD EXISTING DATA
# ------------------------------------------------------------

test_data <- read.csv(
  file.path(processed_dir, "test_data.csv")
)

shap_values <- readRDS(
  file.path(models_dir, "shap_values.rds")
)

# Seven features used during model training
feature_names <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

# ------------------------------------------------------------
# 2. VALIDATE DATA
# ------------------------------------------------------------

missing_features <- setdiff(
  feature_names,
  names(test_data)
)

if (length(missing_features) > 0) {
  stop(
    "Missing features in test_data.csv: ",
    paste(missing_features, collapse = ", ")
  )
}

if (!"Severe Degradation" %in% names(shap_values)) {
  stop("Severe Degradation SHAP values were not found.")
}

# Extract original test features
X_test <- test_data[, feature_names, drop = FALSE]

# Convert features to numeric
X_test[] <- lapply(X_test, as.numeric)

# Extract SHAP values for Severe Degradation
severe_shap <- as.matrix(
  shap_values[["Severe Degradation"]]
)

# ------------------------------------------------------------
# 3. CHECK DIMENSIONS AND FEATURE ORDER
# ------------------------------------------------------------

if (nrow(X_test) != nrow(severe_shap)) {
  stop(
    "Number of test observations does not match SHAP observations."
  )
}

if (ncol(severe_shap) != length(feature_names)) {
  stop(
    "SHAP matrix does not contain exactly seven feature contributions."
  )
}

if (!is.null(colnames(severe_shap))) {
  
  if (all(feature_names %in% colnames(severe_shap))) {
    
    severe_shap <- severe_shap[
      ,
      feature_names,
      drop = FALSE
    ]
    
  } else {
    
    stop(
      "SHAP feature names do not match the training feature names."
    )
    
  }
  
} else {
  
  colnames(severe_shap) <- feature_names
  
}

# ------------------------------------------------------------
# 4. CREATE SHAP DEPENDENCE DATA
# ------------------------------------------------------------

dependence_data <- data.frame()

for (feature in feature_names) {
  
  feature_data <- data.frame(
    
    Sample_ID = seq_len(nrow(X_test)),
    
    Class = "Severe Degradation",
    
    Feature = feature,
    
    Feature_Value = X_test[[feature]],
    
    SHAP_Value = severe_shap[, feature],
    
    stringsAsFactors = FALSE
    
  )
  
  dependence_data <- rbind(
    dependence_data,
    feature_data
  )
}

# Remove invalid observations
dependence_data <- dependence_data %>%
  filter(
    is.finite(Feature_Value),
    is.finite(SHAP_Value)
  )

# ------------------------------------------------------------
# 5. SAVE DEPENDENCE DATA
# ------------------------------------------------------------

output_file <- file.path(
  processed_dir,
  "shap_individual_explanation.csv"
)

write.csv(
  dependence_data,
  output_file,
  row.names = FALSE
)

cat("\nSHAP DEPENDENCE DATA GENERATED SUCCESSFULLY\n")

cat(
  "\nTotal observations:",
  nrow(dependence_data)
)

cat(
  "\nFeatures:",
  paste(feature_names, collapse = ", ")
)

cat(
  "\nClass:",
  "Severe Degradation"
)

cat(
  "\nSaved to:",
  output_file,
  "\n"
)

# ------------------------------------------------------------
# 6. GENERATE SHAP DEPENDENCE PLOTS
# ------------------------------------------------------------

selected_features <- c(
  "Mean_Voltage",
  "Mean_Current",
  "Cycle_Number"
)

for (feature in selected_features) {
  
  plot_data <- dependence_data %>%
    filter(Feature == feature)
  
  p <- ggplot(
    plot_data,
    aes(
      x = Feature_Value,
      y = SHAP_Value
    )
  ) +
    
    geom_point(
      alpha = 0.65,
      size = 2
    ) +
    
    geom_smooth(
      method = "loess",
      formula = y ~ x,
      se = TRUE
    ) +
    
    labs(
      title = paste(
        "SHAP Dependence Plot -",
        feature
      ),
      subtitle = "Severe Degradation Class",
      x = feature,
      y = "SHAP Contribution"
    ) +
    
    theme_minimal()
  
  ggsave(
    filename = file.path(
      plots_dir,
      paste0("shap_dependence_", feature, ".png")
    ),
    plot = p,
    width = 8,
    height = 5,
    dpi = 300
  )
}

cat("\nThree SHAP dependence plots saved successfully.\n")
