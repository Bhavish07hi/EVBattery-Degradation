
# ============================================================
# 15 - BATTERY-SPECIFIC GENERALIZATION ANALYSIS
# Compare degradation patterns across battery cells
# ============================================================

library(dplyr)

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(
  project_dir,
  "data",
  "processed"
)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

data <- read.csv(
  file.path(
    processed_dir,
    "battery_classification.csv"
  ),
  stringsAsFactors = FALSE
)

predictions <- read.csv(
  file.path(
    processed_dir,
    "severe_threshold_probabilities.csv"
  ),
  stringsAsFactors = FALSE
)

class_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

features <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity",
  "SOH"
)

# ------------------------------------------------------------
# 3. VALIDATE DATA
# ------------------------------------------------------------

required_data_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "Degradation_Stage",
  features
)

required_prediction_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "Actual_Stage",
  "Probability_Healthy",
  "Probability_Early",
  "Probability_Moderate",
  "Probability_Severe"
)

if (
  length(
    setdiff(required_data_columns, names(data))
  ) > 0
) {
  stop("Required columns are missing from battery_classification.csv.")
}

if (
  length(
    setdiff(required_prediction_columns, names(predictions))
  ) > 0
) {
  stop("Required columns are missing from severe_threshold_probabilities.csv.")
}

data$Battery_ID <- as.character(data$Battery_ID)
predictions$Battery_ID <- as.character(predictions$Battery_ID)

data$Cycle_Number <- as.numeric(data$Cycle_Number)
predictions$Cycle_Number <- as.numeric(predictions$Cycle_Number)

data[features] <- lapply(
  data[features],
  as.numeric
)

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

predictions$Actual_Stage <- factor(
  predictions$Actual_Stage,
  levels = class_levels
)

probability_columns <- c(
  "Probability_Healthy",
  "Probability_Early",
  "Probability_Moderate",
  "Probability_Severe"
)

predictions[probability_columns] <- lapply(
  predictions[probability_columns],
  as.numeric
)

if (
  anyNA(data[, required_data_columns]) ||
  anyNA(predictions[, required_prediction_columns])
) {
  stop("Missing or invalid values found.")
}

# ------------------------------------------------------------
# 4. RECREATE STANDARD PREDICTIONS
# ------------------------------------------------------------

probability_matrix <- as.matrix(
  predictions[, probability_columns]
)

if (any(!is.finite(probability_matrix))) {
  stop("Invalid probability values found.")
}

predicted_indices <- max.col(
  probability_matrix,
  ties.method = "first"
)

predictions$Predicted_Stage <- factor(
  class_levels[predicted_indices],
  levels = class_levels
)

# ------------------------------------------------------------
# 5. MERGE PREDICTIONS WITH ORIGINAL MEASUREMENTS
# ------------------------------------------------------------

analysis_data <- predictions %>%
  left_join(
    data %>%
      select(
        Battery_ID,
        Cycle_Number,
        all_of(features),
        Degradation_Stage
      ),
    by = c(
      "Battery_ID",
      "Cycle_Number"
    )
  )

if (anyNA(analysis_data[, features])) {
  stop("Some prediction records could not be matched to measurements.")
}

if (
  any(
    as.character(analysis_data$Actual_Stage) !=
    as.character(analysis_data$Degradation_Stage)
  )
) {
  stop("Actual labels do not match the original dataset.")
}

# ------------------------------------------------------------
# 6. BATTERY-WISE DATASET DISTRIBUTION
# ------------------------------------------------------------

battery_stage_counts <- data %>%
  count(
    Battery_ID,
    Degradation_Stage,
    name = "Number_of_Cycles"
  ) %>%
  arrange(
    Battery_ID,
    Degradation_Stage
  )

cat("\n========================================\n")
cat("BATTERY-WISE DEGRADATION STAGE COUNTS\n")
cat("========================================\n")

print(battery_stage_counts)

battery_stage_wide <- as.data.frame(
  xtabs(
    ~ Battery_ID + Degradation_Stage,
    data = data
  )
)

names(battery_stage_wide) <- c(
  "Battery_ID",
  "Degradation_Stage",
  "Number_of_Cycles"
)

# ------------------------------------------------------------
# 7. BATTERY-WISE CLASS DISTRIBUTION
# ------------------------------------------------------------

battery_distribution <- data %>%
  group_by(Battery_ID) %>%
  summarise(
    Total_Cycles = n(),
    
    Healthy = sum(
      Degradation_Stage == "Healthy"
    ),
    
    Early_Degradation = sum(
      Degradation_Stage == "Early Degradation"
    ),
    
    Moderate_Degradation = sum(
      Degradation_Stage == "Moderate Degradation"
    ),
    
    Severe_Degradation = sum(
      Degradation_Stage == "Severe Degradation"
    ),
    
    Severe_Percentage = 100 *
      Severe_Degradation / Total_Cycles,
    
    .groups = "drop"
  )

cat("\nBATTERY CLASS DISTRIBUTION:\n")

print(battery_distribution)

# ------------------------------------------------------------
# 8. BATTERY-WISE SEVERE CASE COMPARISON
# ------------------------------------------------------------

severe_data <- data %>%
  filter(
    Degradation_Stage == "Severe Degradation"
  )

severe_battery_summary <- severe_data %>%
  group_by(Battery_ID) %>%
  summarise(
    Severe_Cases = n(),
    
    Mean_Cycle = mean(Cycle_Number),
    
    Mean_Voltage = mean(Mean_Voltage),
    
    Mean_Min_Voltage = mean(Min_Voltage),
    
    Mean_Max_Voltage = mean(Max_Voltage),
    
    Mean_Current = mean(Mean_Current),
    
    Mean_Temperature = mean(Mean_Temperature),
    
    Mean_Max_Temperature = mean(Max_Temperature),
    
    Mean_Capacity = mean(Discharge_Capacity),
    
    Mean_SOH = mean(SOH),
    
    .groups = "drop"
  )

cat("\nSEVERE CASES BY BATTERY:\n")

print(severe_battery_summary)

# ------------------------------------------------------------
# 9. COMPARE MODERATE AND SEVERE CASES BY BATTERY
# ------------------------------------------------------------

boundary_data <- data %>%
  filter(
    Degradation_Stage %in% c(
      "Moderate Degradation",
      "Severe Degradation"
    )
  )

boundary_summary <- boundary_data %>%
  group_by(
    Battery_ID,
    Degradation_Stage
  ) %>%
  summarise(
    Number_of_Cycles = n(),
    
    Mean_Cycle = mean(Cycle_Number),
    
    Mean_Voltage = mean(Mean_Voltage),
    
    Mean_Min_Voltage = mean(Min_Voltage),
    
    Mean_Max_Voltage = mean(Max_Voltage),
    
    Mean_Current = mean(Mean_Current),
    
    Mean_Temperature = mean(Mean_Temperature),
    
    Mean_Max_Temperature = mean(Max_Temperature),
    
    Mean_Capacity = mean(Discharge_Capacity),
    
    Mean_SOH = mean(SOH),
    
    .groups = "drop"
  )

cat("\nMODERATE VS SEVERE BY BATTERY:\n")

print(boundary_summary)

# ------------------------------------------------------------
# 10. HELD-OUT BATTERY PERFORMANCE
# ------------------------------------------------------------

calculate_metrics <- function(actual, predicted) {
  
  actual <- factor(
    actual,
    levels = class_levels
  )
  
  predicted <- factor(
    predicted,
    levels = class_levels
  )
  
  confusion <- table(
    Actual = actual,
    Predicted = predicted
  )
  
  class_metrics <- data.frame()
  
  for (class_name in class_levels) {
    
    tp <- confusion[class_name, class_name]
    
    fp <- sum(confusion[, class_name]) - tp
    
    fn <- sum(confusion[class_name, ]) - tp
    
    support <- sum(confusion[class_name, ])
    
    precision <- if (tp + fp > 0) {
      tp / (tp + fp)
    } else {
      0
    }
    
    recall <- if (tp + fn > 0) {
      tp / (tp + fn)
    } else {
      0
    }
    
    f1 <- if (precision + recall > 0) {
      2 * precision * recall /
        (precision + recall)
    } else {
      0
    }
    
    class_metrics <- rbind(
      class_metrics,
      data.frame(
        Class = class_name,
        Support = support,
        Precision = precision,
        Recall = recall,
        F1_Score = f1
      )
    )
  }
  
  list(
    confusion = confusion,
    class_metrics = class_metrics,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(class_metrics$F1_Score)
  )
}

battery_performance <- data.frame()

battery_class_metrics <- data.frame()

battery_confusion_matrices <- list()

battery_ids <- unique(analysis_data$Battery_ID)

for (battery in battery_ids) {
  
  fold_data <- analysis_data %>%
    filter(
      Battery_ID == battery
    )
  
  metrics <- calculate_metrics(
    fold_data$Actual_Stage,
    fold_data$Predicted_Stage
  )
  
  severe_metrics <- metrics$class_metrics %>%
    filter(
      Class == "Severe Degradation"
    )
  
  battery_performance <- rbind(
    battery_performance,
    data.frame(
      Battery_ID = battery,
      Test_Cycles = nrow(fold_data),
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1,
      Severe_Support = severe_metrics$Support,
      Severe_Precision = severe_metrics$Precision,
      Severe_Recall = severe_metrics$Recall,
      Severe_F1 = severe_metrics$F1_Score
    )
  )
  
  metrics$class_metrics$Battery_ID <- battery
  
  battery_class_metrics <- rbind(
    battery_class_metrics,
    metrics$class_metrics
  )
  
  battery_confusion_matrices[[battery]] <-
    metrics$confusion
}

cat("\n========================================\n")
cat("HELD-OUT BATTERY MODEL PERFORMANCE\n")
cat("========================================\n")

print(battery_performance)

cat("\nCLASS-WISE PERFORMANCE BY BATTERY:\n")

print(battery_class_metrics)

# ------------------------------------------------------------
# 11. BATTERY-WISE PREDICTED SEVERE PROBABILITY
# ------------------------------------------------------------

probability_summary <- analysis_data %>%
  group_by(
    Battery_ID,
    Actual_Stage
  ) %>%
  summarise(
    Number_of_Cycles = n(),
    
    Mean_Severe_Probability = mean(
      Probability_Severe
    ),
    
    Median_Severe_Probability = median(
      Probability_Severe
    ),
    
    Minimum_Severe_Probability = min(
      Probability_Severe
    ),
    
    Maximum_Severe_Probability = max(
      Probability_Severe
    ),
    
    .groups = "drop"
  )

cat("\nSEVERE PROBABILITY BY BATTERY AND ACTUAL CLASS:\n")

print(probability_summary)

# ------------------------------------------------------------
# 12. SPECIFIC B0005 AND B0006 SEVERE CASES
# ------------------------------------------------------------

b0005_b0006_severe <- analysis_data %>%
  filter(
    Battery_ID %in% c("B0005", "B0006"),
    Actual_Stage == "Severe Degradation"
  ) %>%
  arrange(
    Battery_ID,
    Cycle_Number
  ) %>%
  select(
    Battery_ID,
    Cycle_Number,
    Actual_Stage,
    Predicted_Stage,
    Probability_Severe,
    Mean_Voltage,
    Min_Voltage,
    Max_Voltage,
    Mean_Current,
    Mean_Temperature,
    Max_Temperature,
    Discharge_Capacity,
    SOH
  )

cat("\nB0005 AND B0006 SEVERE OBSERVATIONS:\n")

print(b0005_b0006_severe)

# ------------------------------------------------------------
# 13. FEATURE DISTRIBUTION BY BATTERY
# ------------------------------------------------------------

battery_feature_summary <- data %>%
  group_by(Battery_ID) %>%
  summarise(
    across(
      all_of(features),
      list(
        Mean = ~ mean(.x),
        SD = ~ sd(.x)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

cat("\nFEATURE DISTRIBUTION BY BATTERY:\n")

print(battery_feature_summary)

# ------------------------------------------------------------
# 14. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  battery_stage_counts,
  file.path(
    processed_dir,
    "battery_generalization_stage_counts.csv"
  ),
  row.names = FALSE
)

write.csv(
  battery_distribution,
  file.path(
    processed_dir,
    "battery_generalization_class_distribution.csv"
  ),
  row.names = FALSE
)

write.csv(
  severe_battery_summary,
  file.path(
    processed_dir,
    "battery_generalization_severe_summary.csv"
  ),
  row.names = FALSE
)

write.csv(
  boundary_summary,
  file.path(
    processed_dir,
    "battery_generalization_boundary_summary.csv"
  ),
  row.names = FALSE
)

write.csv(
  battery_performance,
  file.path(
    processed_dir,
    "battery_generalization_performance.csv"
  ),
  row.names = FALSE
)

write.csv(
  battery_class_metrics,
  file.path(
    processed_dir,
    "battery_generalization_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  probability_summary,
  file.path(
    processed_dir,
    "battery_generalization_probability_summary.csv"
  ),
  row.names = FALSE
)

write.csv(
  b0005_b0006_severe,
  file.path(
    processed_dir,
    "B0005_B0006_severe_comparison.csv"
  ),
  row.names = FALSE
)

write.csv(
  battery_feature_summary,
  file.path(
    processed_dir,
    "battery_generalization_feature_summary.csv"
  ),
  row.names = FALSE
)

for (battery in names(battery_confusion_matrices)) {
  
  write.csv(
    as.data.frame.matrix(
      battery_confusion_matrices[[battery]]
    ),
    file.path(
      processed_dir,
      paste0(
        "battery_confusion_",
        battery,
        ".csv"
      )
    )
  )
}

# ------------------------------------------------------------
# 15. FINAL MESSAGE
# ------------------------------------------------------------

cat("\n========================================\n")
cat("BATTERY GENERALIZATION ANALYSIS COMPLETED\n")
cat("========================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)
