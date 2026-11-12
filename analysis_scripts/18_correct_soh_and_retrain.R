# ============================================================
# EV BATTERY DEGRADATION
# Correct SOH, retrain XGBoost and evaluate held-out batteries
# ============================================================

library(dplyr)
library(ggplot2)
library(xgboost)

# 1. PROJECT PATHS

project_dir <- getwd()

data_dir <- file.path(project_dir, "data", "processed")
model_dir <- file.path(project_dir, "models")
plot_dir <- file.path(project_dir, "plots")

dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# 2. LOAD EXTRACTED NASA CYCLE DATA

input_file <- file.path(data_dir, "all_battery_cycles.csv")

if (!file.exists(input_file)) {
  stop("all_battery_cycles.csv not found in data/processed.")
}

cycles <- read.csv(input_file, stringsAsFactors = FALSE)

required_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity"
)

missing_columns <- setdiff(required_columns, names(cycles))

if (length(missing_columns) > 0) {
  stop(paste("Missing columns:", paste(missing_columns, collapse = ", ")))
}

# 3. REMOVE INVALID CAPACITY RECORDS

cycles <- cycles %>%
  mutate(
    Battery_ID = as.character(Battery_ID),
    Discharge_Capacity = as.numeric(Discharge_Capacity)
  ) %>%
  filter(
    is.finite(Discharge_Capacity),
    Discharge_Capacity >= 0.2,
    Discharge_Capacity <= 3.0,
    is.finite(Mean_Voltage),
    is.finite(Min_Voltage),
    is.finite(Max_Voltage),
    Min_Voltage >= 1,
    Max_Voltage <= 5
  ) %>%
  arrange(Battery_ID, Cycle_Number)

cat("Valid discharge records:", nrow(cycles), "\n")
cat("Batteries retained:", n_distinct(cycles$Battery_ID), "\n")

# 4. CALCULATE ROBUST INITIAL CAPACITY
# Median of the first five valid discharge capacities.
# This reduces the influence of anomalous first readings.

battery_baselines <- cycles %>%
  group_by(Battery_ID) %>%
  arrange(Cycle_Number, .by_group = TRUE) %>%
  slice_head(n = 5) %>%
  summarise(
    Initial_Capacity = median(Discharge_Capacity, na.rm = TRUE),
    .groups = "drop"
  )

cycles <- cycles %>%
  left_join(battery_baselines, by = "Battery_ID") %>%
  mutate(
    SOH = (Discharge_Capacity / Initial_Capacity) * 100
  ) %>%
  filter(
    is.finite(SOH),
    SOH > 0,
    SOH <= 110
  )

# 5. DEFINE DEGRADATION STAGES

cycles <- cycles %>%
  mutate(
    Degradation_Stage = case_when(
      SOH >= 90 ~ "Healthy",
      SOH >= 80 ~ "Early Degradation",
      SOH >= 70 ~ "Moderate Degradation",
      TRUE ~ "Severe Degradation"
    )
  )

stage_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

cycles$Degradation_Stage <- factor(
  cycles$Degradation_Stage,
  levels = stage_levels
)

# 6. SAVE CORRECTED DATA

write.csv(
  cycles,
  file.path(data_dir, "all_battery_classification_corrected.csv"),
  row.names = FALSE
)

write.csv(
  battery_baselines,
  file.path(data_dir, "battery_initial_capacity_corrected.csv"),
  row.names = FALSE
)

cat("\nSOH correction completed.\n")
print(table(cycles$Degradation_Stage))

# 7. TRAIN / EXTERNAL TEST BATTERY SPLIT

test_batteries <- c("B0045", "B0046", "B0047")

train_data <- cycles %>%
  filter(!Battery_ID %in% test_batteries)

test_data <- cycles %>%
  filter(Battery_ID %in% test_batteries)

if (nrow(train_data) == 0 || nrow(test_data) == 0) {
  stop("Training or external test dataset is empty.")
}

cat("\nTraining batteries:",
    n_distinct(train_data$Battery_ID), "\n")

cat("External test batteries:",
    paste(sort(unique(test_data$Battery_ID)), collapse = ", "), "\n")

# 8. MODEL FEATURES

feature_names <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

x_train <- as.matrix(train_data[, feature_names])
x_test <- as.matrix(test_data[, feature_names])

storage.mode(x_train) <- "double"
storage.mode(x_test) <- "double"

y_train <- as.integer(train_data$Degradation_Stage) - 1
y_test <- as.integer(test_data$Degradation_Stage) - 1

# 9. CLASS WEIGHTS

class_counts <- table(
  factor(y_train, levels = 0:3)
)

class_weights <- length(y_train) /
  (4 * as.numeric(class_counts))

class_weights <- pmin(class_weights, 3)

sample_weights <- class_weights[y_train + 1]

# 10. TRAIN XGBOOST

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train,
  weight = sample_weights
)

dtest <- xgb.DMatrix(data = x_test)

params <- list(
  objective = "multi:softprob",
  num_class = 4,
  eval_metric = "mlogloss",
  max_depth = 3,
  eta = 0.05,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(42)

final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 150,
  verbose = 0
)

# 11. EXTERNAL TEST PREDICTIONS

probabilities <- predict(final_model, dtest)

probabilities <- matrix(
  probabilities,
  ncol = 4,
  byrow = TRUE
)

predicted_class <- max.col(probabilities) - 1

stage_names <- stage_levels

predictions <- data.frame(
  Battery_ID = test_data$Battery_ID,
  Actual = stage_names[y_test + 1],
  Predicted = stage_names[predicted_class + 1],
  SOH = test_data$SOH
)

# 12. EVALUATION

accuracy <- mean(predicted_class == y_test)

macro_f1 <- function(actual, predicted) {
  
  scores <- sapply(0:3, function(class_id) {
    
    tp <- sum(actual == class_id & predicted == class_id)
    fp <- sum(actual != class_id & predicted == class_id)
    fn <- sum(actual == class_id & predicted != class_id)
    
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    
    ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall / (precision + recall)
    )
  })
  
  mean(scores)
}

f1 <- macro_f1(y_test, predicted_class)

metrics <- data.frame(
  Metric = c("Accuracy", "Macro F1"),
  Value = c(accuracy, f1)
)

confusion <- table(
  Actual = factor(y_test, levels = 0:3, labels = stage_levels),
  Predicted = factor(
    predicted_class,
    levels = 0:3,
    labels = stage_levels
  )
)

cat("\n========== FINAL EXTERNAL EVALUATION ==========\n")
print(metrics)
cat("\nConfusion Matrix:\n")
print(confusion)

# 13. SAVE EVALUATION RESULTS

write.csv(
  predictions,
  file.path(data_dir, "corrected_external_predictions.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(confusion),
  file.path(data_dir, "corrected_external_confusion_matrix.csv")
)

write.csv(
  metrics,
  file.path(data_dir, "corrected_external_metrics.csv"),
  row.names = FALSE
)

# 14. FEATURE IMPORTANCE

importance <- xgb.importance(
  feature_names = feature_names,
  model = final_model
)

write.csv(
  importance,
  file.path(data_dir, "corrected_xgboost_feature_importance.csv"),
  row.names = FALSE
)

png(
  file.path(plot_dir, "corrected_xgboost_feature_importance.png"),
  width = 1200,
  height = 800
)

print(
  xgb.plot.importance(
    importance_matrix = importance,
    top_n = length(feature_names)
  )
)

dev.off()

# 15. SAVE MODEL AND METADATA

model_bundle <- list(
  model = final_model,
  feature_names = feature_names,
  stage_levels = stage_levels,
  initial_capacity_method = "Median of first five valid discharge capacities",
  training_batteries = sort(unique(train_data$Battery_ID)),
  test_batteries = test_batteries,
  metrics = metrics
)

saveRDS(
  model_bundle,
  file.path(model_dir, "corrected_xgboost_model.rds")
)

cat("\n========================================\n")
cat("CORRECTED MODEL TRAINING COMPLETED\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Macro F1:", round(f1, 4), "\n")
cat("Model saved: models/corrected_xgboost_model.rds\n")
cat("========================================\n")