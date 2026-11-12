
# ============================================================
# FILE: R/21_battery_normalized_model.R
# Battery-normalised EV degradation prediction
# Models: Random Forest and XGBoost regression
# Validation: Battery-wise cross-validation
# External test: B0045, B0046, B0047
# ============================================================

cat("\nSTARTING BATTERY-NORMALISED MODEL\n")

# 1. PACKAGES
packages <- c("dplyr", "ggplot2", "xgboost", "randomForest")

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

library(dplyr)
library(ggplot2)
library(xgboost)
library(randomForest)

# 2. PROJECT PATHS
project_dir <- "C:/Users/bhavi/OneDrive/Documents/EV_Battery_Degradation"

data_file <- file.path(
  project_dir,
  "data/processed/all_battery_classification_corrected.csv"
)

results_dir <- file.path(project_dir, "data/processed")
models_dir <- file.path(project_dir, "models")
plots_dir <- file.path(project_dir, "plots")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_file)) {
  stop("Dataset not found: ", data_file)
}

data <- read.csv(data_file, stringsAsFactors = FALSE)

required <- c(
  "Battery_ID", "Cycle_Number", "SOH", "Degradation_Stage",
  "Mean_Voltage", "Min_Voltage", "Max_Voltage",
  "Mean_Current", "Mean_Temperature", "Max_Temperature"
)

missing_columns <- setdiff(required, names(data))

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# 3. CLEAN DATA
data$Battery_ID <- as.character(data$Battery_ID)

numeric_columns <- c(
  "Cycle_Number", "SOH",
  "Mean_Voltage", "Min_Voltage", "Max_Voltage",
  "Mean_Current", "Mean_Temperature", "Max_Temperature"
)

for (column in numeric_columns) {
  data[[column]] <- as.numeric(data[[column]])
}

data <- data %>%
  filter(
    !is.na(Battery_ID),
    is.finite(SOH),
    is.finite(Cycle_Number)
  ) %>%
  arrange(Battery_ID, Cycle_Number)

stage_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

data$Stage <- trimws(as.character(data$Degradation_Stage))

data <- data %>%
  filter(Stage %in% stage_levels)

data$Stage <- factor(data$Stage, levels = stage_levels)

cat("Rows:", nrow(data), "\n")
cat("Batteries:", length(unique(data$Battery_ID)), "\n")

# 4. BATTERY-WISE SENSOR BASELINE
# First five available cycles of each battery.
# This assumes these reference cycles are available at deployment.

sensor_columns <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

baseline <- data %>%
  group_by(Battery_ID) %>%
  arrange(Cycle_Number, .by_group = TRUE) %>%
  slice_head(n = 5) %>%
  summarise(
    across(
      all_of(sensor_columns),
      ~ median(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

names(baseline)[-1] <- paste0(
  "Baseline_",
  names(baseline)[-1]
)

data <- data %>%
  left_join(baseline, by = "Battery_ID")

# 5. CREATE NORMALISED FEATURES
# Relative change from each battery's own baseline.

for (sensor in sensor_columns) {
  
  baseline_name <- paste0("Baseline_", sensor)
  
  delta_name <- paste0(sensor, "_Change")
  ratio_name <- paste0(sensor, "_Relative")
  
  data[[delta_name]] <-
    data[[sensor]] - data[[baseline_name]]
  
  denominator <- abs(data[[baseline_name]])
  
  data[[ratio_name]] <- ifelse(
    denominator > 0.0001,
    data[[delta_name]] / denominator,
    NA_real_
  )
}

# Voltage range change
data$Voltage_Range_Change <-
  (data$Max_Voltage - data$Min_Voltage) -
  (data$Baseline_Max_Voltage - data$Baseline_Min_Voltage)

# Temperature spread change
data$Temperature_Spread_Change <-
  (data$Max_Temperature - data$Mean_Temperature) -
  (data$Baseline_Max_Temperature - data$Baseline_Mean_Temperature)

feature_names <- c(
  paste0(sensor_columns, "_Change"),
  paste0(sensor_columns, "_Relative"),
  "Voltage_Range_Change",
  "Temperature_Spread_Change"
)

# Remove unusable features
for (feature in feature_names) {
  data[[feature]][!is.finite(data[[feature]])] <- NA_real_
}

# Median imputation using the available training data will be
# performed separately inside each fold below.
# Remove rows with missing features here only if all values are missing.
data <- data[
  rowSums(!is.na(data[, feature_names, drop = FALSE])) > 0,
]

cat("Normalised features created:", length(feature_names), "\n")

# 6. EXTERNAL TEST SPLIT
external_ids <- c("B0045", "B0046", "B0047")

external_data <- data[
  data$Battery_ID %in% external_ids,
]

training_data <- data[
  !data$Battery_ID %in% external_ids,
]

if (nrow(external_data) == 0 || nrow(training_data) == 0) {
  stop("Training or external test data is empty.")
}

cat("Training rows:", nrow(training_data), "\n")
cat("External rows:", nrow(external_data), "\n")

# 7. HELPER FUNCTIONS

impute_features <- function(train, validation, features) {
  
  medians <- sapply(
    features,
    function(f) median(train[[f]], na.rm = TRUE)
  )
  
  medians[!is.finite(medians)] <- 0
  
  train_matrix <- as.matrix(train[, features, drop = FALSE])
  validation_matrix <- as.matrix(
    validation[, features, drop = FALSE]
  )
  
  for (j in seq_along(features)) {
    train_matrix[is.na(train_matrix[, j]), j] <- medians[j]
    validation_matrix[
      is.na(validation_matrix[, j]), j
    ] <- medians[j]
  }
  
  list(
    train = train_matrix,
    validation = validation_matrix,
    medians = medians
  )
}

soh_to_stage <- function(soh) {
  
  result <- ifelse(
    soh >= 90, "Healthy",
    ifelse(
      soh >= 80, "Early Degradation",
      ifelse(
        soh >= 70, "Moderate Degradation",
        "Severe Degradation"
      )
    )
  )
  
  factor(result, levels = stage_levels)
}

calculate_metrics <- function(actual, predicted) {
  
  actual <- factor(actual, levels = stage_levels)
  predicted <- factor(predicted, levels = stage_levels)
  
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )
  
  accuracy <- sum(diag(cm)) / sum(cm)
  
  class_results <- data.frame(
    Stage = stage_levels,
    Precision = 0,
    Recall = 0,
    F1 = 0
  )
  
  for (i in seq_along(stage_levels)) {
    
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    
    f1 <- ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall / (precision + recall)
    )
    
    class_results$Precision[i] <- precision
    class_results$Recall[i] <- recall
    class_results$F1[i] <- f1
  }
  
  list(
    accuracy = accuracy,
    macro_f1 = mean(class_results$F1),
    confusion = cm,
    class_metrics = class_results
  )
}

# 8. BATTERY-WISE CROSS-VALIDATION

set.seed(42)

battery_ids <- sample(unique(training_data$Battery_ID))

number_of_folds <- min(5, length(battery_ids))

fold_assignment <- rep(
  seq_len(number_of_folds),
  length.out = length(battery_ids)
)

folds <- split(battery_ids, fold_assignment)

model_names <- c("Random Forest", "XGBoost")

cv_results <- list()
cv_predictions <- list()

for (model_name in model_names) {
  
  cat("\nEvaluating:", model_name, "\n")
  
  all_actual <- character()
  all_predicted <- character()
  
  for (fold_id in seq_along(folds)) {
    
    validation_batteries <- folds[[fold_id]]
    
    fold_train <- training_data[
      !training_data$Battery_ID %in% validation_batteries,
    ]
    
    fold_valid <- training_data[
      training_data$Battery_ID %in% validation_batteries,
    ]
    
    prepared <- impute_features(
      fold_train,
      fold_valid,
      feature_names
    )
    
    if (model_name == "Random Forest") {
      
      set.seed(42)
      
      model <- randomForest(
        x = prepared$train,
        y = fold_train$SOH,
        ntree = 500,
        mtry = max(1, floor(sqrt(length(feature_names)))),
        importance = TRUE
      )
      
      predicted_soh <- predict(
        model,
        prepared$validation
      )
      
    } else {
      
      dtrain <- xgb.DMatrix(
        prepared$train,
        label = fold_train$SOH
      )
      
      dvalid <- xgb.DMatrix(
        prepared$validation
      )
      
      params <- list(
        objective = "reg:squarederror",
        eval_metric = "rmse",
        max_depth = 3,
        eta = 0.03,
        min_child_weight = 3,
        subsample = 0.85,
        colsample_bytree = 0.85,
        lambda = 3,
        alpha = 0.1,
        verbosity = 0
      )
      
      model <- xgb.train(
        params = params,
        data = dtrain,
        nrounds = 250,
        verbose = 0
      )
      
      predicted_soh <- predict(model, dvalid)
    }
    
    predicted_stage <- soh_to_stage(predicted_soh)
    
    all_actual <- c(
      all_actual,
      as.character(fold_valid$Stage)
    )
    
    all_predicted <- c(
      all_predicted,
      as.character(predicted_stage)
    )
  }
  
  metrics <- calculate_metrics(
    all_actual,
    all_predicted
  )
  
  cv_results[[model_name]] <- data.frame(
    Model = model_name,
    Accuracy = metrics$accuracy,
    Macro_F1 = metrics$macro_f1
  )
  
  cv_predictions[[model_name]] <- data.frame(
    Actual = all_actual,
    Predicted = all_predicted
  )
  
  cat(
    "Accuracy:", round(metrics$accuracy * 100, 2), "%",
    "| Macro F1:", round(metrics$macro_f1, 4), "\n"
  )
}

comparison <- bind_rows(cv_results) %>%
  arrange(desc(Accuracy), desc(Macro_F1))

write.csv(
  comparison,
  file.path(results_dir, "normalized_model_comparison.csv"),
  row.names = FALSE
)

cat("\n========== CROSS-VALIDATION COMPARISON ==========\n")
print(comparison)

best_model_name <- comparison$Model[1]

cat("\nSelected model:", best_model_name, "\n")

# 9. TRAIN SELECTED MODEL ON ALL TRAINING BATTERIES

prepared_final <- impute_features(
  training_data,
  external_data,
  feature_names
)

if (best_model_name == "Random Forest") {
  
  set.seed(42)
  
  final_model <- randomForest(
    x = prepared_final$train,
    y = training_data$SOH,
    ntree = 500,
    mtry = max(1, floor(sqrt(length(feature_names)))),
    importance = TRUE
  )
  
  external_predicted_soh <- predict(
    final_model,
    prepared_final$validation
  )
  
  importance_values <- importance(
    final_model,
    type = 1
  )
  
  importance_table <- data.frame(
    Feature = rownames(importance_values),
    Importance = importance_values[, 1]
  )
  
} else {
  
  final_train <- xgb.DMatrix(
    prepared_final$train,
    label = training_data$SOH
  )
  
  final_external <- xgb.DMatrix(
    prepared_final$validation
  )
  
  final_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 3,
    eta = 0.03,
    min_child_weight = 3,
    subsample = 0.85,
    colsample_bytree = 0.85,
    lambda = 3,
    alpha = 0.1,
    verbosity = 0
  )
  
  final_model <- xgb.train(
    params = final_params,
    data = final_train,
    nrounds = 250,
    verbose = 0
  )
  
  external_predicted_soh <- predict(
    final_model,
    final_external
  )
  
  importance_table <- xgb.importance(
    feature_names = feature_names,
    model = final_model
  ) %>%
    select(Feature = Feature, Importance = Gain)
}

external_predicted_stage <- soh_to_stage(
  external_predicted_soh
)

# 10. EXTERNAL TEST EVALUATION

external_metrics <- calculate_metrics(
  external_data$Stage,
  external_predicted_stage
)

cat("\n========== EXTERNAL TEST RESULTS ==========\n")

cat(
  "Accuracy:",
  round(external_metrics$accuracy * 100, 2),
  "%\n"
)

cat(
  "Macro F1:",
  round(external_metrics$macro_f1, 4),
  "\n"
)

cat("\nConfusion Matrix:\n")
print(external_metrics$confusion)

cat("\nClass Metrics:\n")
print(external_metrics$class_metrics)

external_output <- data.frame(
  Battery_ID = external_data$Battery_ID,
  Cycle_Number = external_data$Cycle_Number,
  Actual_SOH = external_data$SOH,
  Predicted_SOH = external_predicted_soh,
  Actual_Stage = external_data$Stage,
  Predicted_Stage = external_predicted_stage
)

write.csv(
  external_output,
  file.path(results_dir, "normalized_external_predictions.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(external_metrics$confusion),
  file.path(results_dir, "normalized_external_confusion_matrix.csv")
)

write.csv(
  external_metrics$class_metrics,
  file.path(results_dir, "normalized_external_class_metrics.csv"),
  row.names = FALSE
)

# 11. FEATURE IMPORTANCE

write.csv(
  importance_table,
  file.path(results_dir, "normalized_feature_importance.csv"),
  row.names = FALSE
)

importance_plot <- ggplot(
  importance_table,
  aes(
    x = reorder(Feature, Importance),
    y = Importance
  )
) +
  geom_col() +
  coord_flip() +
  theme_minimal() +
  labs(
    title = paste("Feature Importance -", best_model_name),
    x = "Normalised Feature",
    y = "Importance"
  )

ggsave(
  file.path(plots_dir, "normalized_feature_importance.png"),
  importance_plot,
  width = 10,
  height = 7
)

# 12. SAVE MODEL BUNDLE

model_bundle <- list(
  model = final_model,
  model_name = best_model_name,
  features = feature_names,
  imputation_medians = prepared_final$medians,
  baseline_method = "Median of first five available cycles per battery",
  stage_thresholds = c(
    Healthy = 90,
    Early_Degradation = 80,
    Moderate_Degradation = 70
  ),
  cv_results = comparison,
  external_accuracy = external_metrics$accuracy,
  external_macro_f1 = external_metrics$macro_f1,
  training_batteries = unique(training_data$Battery_ID),
  external_batteries = external_ids
)

saveRDS(
  model_bundle,
  file.path(models_dir, "battery_normalized_model.rds")
)

# 13. FINAL SUMMARY

summary_table <- data.frame(
  Metric = c(
    "Selected Model",
    "CV Accuracy",
    "CV Macro F1",
    "External Accuracy",
    "External Macro F1",
    "Training Batteries",
    "External Test Batteries"
  ),
  Result = c(
    best_model_name,
    paste0(round(comparison$Accuracy[1] * 100, 2), "%"),
    round(comparison$Macro_F1[1], 4),
    paste0(round(external_metrics$accuracy * 100, 2), "%"),
    round(external_metrics$macro_f1, 4),
    length(unique(training_data$Battery_ID)),
    paste(external_ids, collapse = ", ")
  )
)

write.csv(
  summary_table,
  file.path(results_dir, "normalized_model_summary.csv"),
  row.names = FALSE
)

cat("\n========== FINAL SUMMARY ==========\n")
print(summary_table)

cat("\nModel saved at:\n")
cat(file.path(models_dir, "battery_normalized_model.rds"), "\n")

cat("\nNORMALISED MODEL PIPELINE COMPLETE\n")
