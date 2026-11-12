
# ============================================================
# FILE: R/22_final_optimization.R
# FINAL EV BATTERY DEGRADATION MODEL OPTIMISATION
#
# Models:
# 1. Random Forest regression
# 2. XGBoost regression
# 3. XGBoost ordinal classification
# 4. XGBoost trend regression
# 5. Balanced XGBoost classification
#
# Validation: Battery-wise cross-validation
# External evaluation: B0045, B0046, B0047
# ============================================================

cat("\n============================================\n")
cat("FINAL MODEL OPTIMISATION STARTED\n")
cat("============================================\n")

# 1. PACKAGES

packages <- c(
  "dplyr",
  "ggplot2",
  "xgboost",
  "randomForest"
)

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

library(dplyr)
library(ggplot2)
library(xgboost)
library(randomForest)

# 2. PATHS

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

# 3. DATA VALIDATION

required_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "SOH",
  "Degradation_Stage",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

missing_columns <- setdiff(required_columns, names(data))

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

data$Battery_ID <- as.character(data$Battery_ID)

numeric_columns <- c(
  "Cycle_Number",
  "SOH",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

for (column in numeric_columns) {
  data[[column]] <- as.numeric(data[[column]])
}

stage_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

data$Stage <- trimws(as.character(data$Degradation_Stage))

data <- data %>%
  filter(
    !is.na(Battery_ID),
    is.finite(SOH),
    is.finite(Cycle_Number),
    Stage %in% stage_levels
  ) %>%
  arrange(Battery_ID, Cycle_Number)

data$Stage <- factor(data$Stage, levels = stage_levels)

external_ids <- c("B0045", "B0046", "B0047")

training_data <- data %>%
  filter(!Battery_ID %in% external_ids)

external_data <- data %>%
  filter(Battery_ID %in% external_ids)

if (nrow(training_data) == 0 || nrow(external_data) == 0) {
  stop("Training or external data is empty.")
}

cat("Total rows:", nrow(data), "\n")
cat("Training rows:", nrow(training_data), "\n")
cat("External rows:", nrow(external_data), "\n")

# 4. FEATURE ENGINEERING

sensor_columns <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

# Baselines are calculated separately for each battery.
# They use only the first five available cycles.

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
  left_join(baseline, by = "Battery_ID") %>%
  arrange(Battery_ID, Cycle_Number)

# Relative baseline features

for (sensor in sensor_columns) {
  
  base_name <- paste0("Baseline_", sensor)
  
  data[[paste0(sensor, "_Change")]] <-
    data[[sensor]] - data[[base_name]]
  
  data[[paste0(sensor, "_Relative")]] <-
    data[[paste0(sensor, "_Change")]] /
    pmax(abs(data[[base_name]]), 0.001)
}

# Physical differences

data$Voltage_Range <-
  data$Max_Voltage - data$Min_Voltage

data$Voltage_Range_Change <-
  data$Voltage_Range -
  (
    data$Baseline_Max_Voltage -
      data$Baseline_Min_Voltage
  )

data$Temperature_Spread <-
  data$Max_Temperature - data$Mean_Temperature

data$Temperature_Spread_Change <-
  data$Temperature_Spread -
  (
    data$Baseline_Max_Temperature -
      data$Baseline_Mean_Temperature
  )

# Cycle-to-cycle changes
# Lag features use only previous cycles.

data <- data %>%
  group_by(Battery_ID) %>%
  arrange(Cycle_Number, .by_group = TRUE) %>%
  mutate(
    Voltage_Trend = Mean_Voltage - lag(Mean_Voltage),
    Current_Trend = Mean_Current - lag(Mean_Current),
    Temperature_Trend =
      Mean_Temperature - lag(Mean_Temperature),
    Voltage_Trend_3 =
      Mean_Voltage - lag(Mean_Voltage, 3),
    Temperature_Trend_3 =
      Mean_Temperature - lag(Mean_Temperature, 3)
  ) %>%
  ungroup()

# Feature sets

normalized_features <- c(
  paste0(sensor_columns, "_Change"),
  paste0(sensor_columns, "_Relative"),
  "Voltage_Range_Change",
  "Temperature_Spread_Change"
)

trend_features <- c(
  normalized_features,
  "Voltage_Trend",
  "Current_Trend",
  "Temperature_Trend",
  "Voltage_Trend_3",
  "Temperature_Trend_3"
)

all_features <- unique(c(
  normalized_features,
  trend_features
))

# Split after feature engineering

training_data <- data %>%
  filter(!Battery_ID %in% external_ids)

external_data <- data %>%
  filter(Battery_ID %in% external_ids)

# 5. HELPER FUNCTIONS

soh_to_stage <- function(soh) {
  
  result <- ifelse(
    soh >= 90,
    "Healthy",
    ifelse(
      soh >= 80,
      "Early Degradation",
      ifelse(
        soh >= 70,
        "Moderate Degradation",
        "Severe Degradation"
      )
    )
  )
  
  factor(result, levels = stage_levels)
}

prepare_matrices <- function(train, validation, features) {
  
  train_matrix <- as.matrix(
    train[, features, drop = FALSE]
  )
  
  validation_matrix <- as.matrix(
    validation[, features, drop = FALSE]
  )
  
  medians <- apply(
    train_matrix,
    2,
    function(x) median(x, na.rm = TRUE)
  )
  
  medians[!is.finite(medians)] <- 0
  
  for (j in seq_along(features)) {
    
    train_matrix[
      !is.finite(train_matrix[, j]), j
    ] <- medians[j]
    
    validation_matrix[
      !is.finite(validation_matrix[, j]), j
    ] <- medians[j]
  }
  
  list(
    train = train_matrix,
    validation = validation_matrix,
    medians = medians
  )
}

calculate_metrics <- function(actual, predicted) {
  
  actual <- factor(actual, levels = stage_levels)
  predicted <- factor(predicted, levels = stage_levels)
  
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )
  
  accuracy <- sum(diag(cm)) / sum(cm)
  
  class_metrics <- data.frame(
    Stage = stage_levels,
    Precision = numeric(4),
    Recall = numeric(4),
    F1 = numeric(4)
  )
  
  for (i in seq_along(stage_levels)) {
    
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    
    p <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    r <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    
    f <- ifelse(
      p + r == 0,
      0,
      2 * p * r / (p + r)
    )
    
    class_metrics$Precision[i] <- p
    class_metrics$Recall[i] <- r
    class_metrics$F1[i] <- f
  }
  
  list(
    accuracy = accuracy,
    macro_f1 = mean(class_metrics$F1),
    confusion = cm,
    class_metrics = class_metrics
  )
}

# 6. ORDINAL CLASSIFICATION

# Four cumulative binary models:
# SOH < 90, SOH < 80, SOH < 70, etc.
# Cumulative probabilities are converted to stage probabilities.

fit_ordinal <- function(train_matrix, train_soh,
                        validation_matrix) {
  
  thresholds <- c(90, 80, 70)
  
  probabilities <- matrix(
    NA_real_,
    nrow = nrow(validation_matrix),
    ncol = 3
  )
  
  for (i in seq_along(thresholds)) {
    
    binary_label <- as.integer(
      train_soh < thresholds[i]
    )
    
    dtrain <- xgb.DMatrix(
      train_matrix,
      label = binary_label
    )
    
    dvalidation <- xgb.DMatrix(validation_matrix)
    
    model <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = 2,
        eta = 0.03,
        min_child_weight = 5,
        subsample = 0.85,
        colsample_bytree = 0.85,
        lambda = 5,
        verbosity = 0
      ),
      data = dtrain,
      nrounds = 150,
      verbose = 0
    )
    
    probabilities[, i] <- predict(
      model,
      dvalidation
    )
  }
  
  # Enforce monotonic cumulative probabilities.
  # P(SOH < 70) <= P(SOH < 80) <= P(SOH < 90)
  
  for (i in seq_len(nrow(probabilities))) {
    probabilities[i, ] <- cummax(probabilities[i, ])
  }
  
  # Class probabilities:
  # Healthy:       1 - P(SOH < 90)
  # Early:         P(SOH < 90) - P(SOH < 80)
  # Moderate:      P(SOH < 80) - P(SOH < 70)
  # Severe:        P(SOH < 70)
  
  class_probabilities <- cbind(
    1 - probabilities[, 1],
    probabilities[, 1] - probabilities[, 2],
    probabilities[, 2] - probabilities[, 3],
    probabilities[, 3]
  )
  
  class_probabilities <- pmax(
    class_probabilities,
    0
  )
  
  class_probabilities / rowSums(class_probabilities)
}

# 7. BATTERY-WISE CROSS-VALIDATION

set.seed(42)

battery_ids <- sample(unique(training_data$Battery_ID))

fold_count <- min(5, length(battery_ids))

fold_assignment <- rep(
  seq_len(fold_count),
  length.out = length(battery_ids)
)

folds <- split(battery_ids, fold_assignment)

model_names <- c(
  "Random Forest Regression",
  "XGBoost Regression",
  "XGBoost Trend Regression",
  "XGBoost Ordinal",
  "XGBoost Balanced Classification"
)

all_results <- list()
all_predictions <- list()

for (model_name in model_names) {
  
  cat("\n====================================\n")
  cat("MODEL:", model_name, "\n")
  cat("====================================\n")
  
  actual_all <- character()
  predicted_all <- character()
  
  for (fold_id in seq_along(folds)) {
    
    validation_batteries <- folds[[fold_id]]
    
    fold_train <- training_data[
      !training_data$Battery_ID %in% validation_batteries,
    ]
    
    fold_valid <- training_data[
      training_data$Battery_ID %in% validation_batteries,
    ]
    
    selected_features <- if (
      model_name == "XGBoost Trend Regression"
    ) {
      trend_features
    } else {
      normalized_features
    }
    
    prepared <- prepare_matrices(
      fold_train,
      fold_valid,
      selected_features
    )
    
    train_matrix <- prepared$train
    validation_matrix <- prepared$validation
    
    if (model_name == "Random Forest Regression") {
      
      set.seed(42)
      
      model <- randomForest(
        x = train_matrix,
        y = fold_train$SOH,
        ntree = 500,
        mtry = max(
          1,
          floor(sqrt(length(selected_features)))
        ),
        importance = TRUE
      )
      
      predicted_soh <- predict(
        model,
        validation_matrix
      )
      
      predicted_stage <- soh_to_stage(predicted_soh)
      
    } else if (
      model_name %in% c(
        "XGBoost Regression",
        "XGBoost Trend Regression"
      )
    ) {
      
      dtrain <- xgb.DMatrix(
        train_matrix,
        label = fold_train$SOH
      )
      
      dvalidation <- xgb.DMatrix(validation_matrix)
      
      model <- xgb.train(
        params = list(
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
        ),
        data = dtrain,
        nrounds = 250,
        verbose = 0
      )
      
      predicted_soh <- predict(
        model,
        dvalidation
      )
      
      predicted_stage <- soh_to_stage(predicted_soh)
      
    } else if (model_name == "XGBoost Ordinal") {
      
      class_probabilities <- fit_ordinal(
        train_matrix,
        fold_train$SOH,
        validation_matrix
      )
      
      predicted_stage <- factor(
        stage_levels[
          max.col(class_probabilities)
        ],
        levels = stage_levels
      )
      
    } else {
      
      train_labels <- as.integer(
        fold_train$Stage
      ) - 1
      
      class_counts <- table(
        factor(fold_train$Stage, levels = stage_levels)
      )
      
      weights <- length(train_labels) /
        (length(stage_levels) * as.numeric(class_counts))
      
      row_weights <- weights[train_labels + 1]
      
      dtrain <- xgb.DMatrix(
        train_matrix,
        label = train_labels,
        weight = row_weights
      )
      
      dvalidation <- xgb.DMatrix(validation_matrix)
      
      model <- xgb.train(
        params = list(
          objective = "multi:softprob",
          num_class = 4,
          eval_metric = "mlogloss",
          max_depth = 2,
          eta = 0.03,
          min_child_weight = 5,
          subsample = 0.85,
          colsample_bytree = 0.85,
          lambda = 5,
          verbosity = 0
        ),
        data = dtrain,
        nrounds = 180,
        verbose = 0
      )
      
      probs <- predict(model, dvalidation)
      
      probability_matrix <- matrix(
        probs,
        ncol = 4,
        byrow = TRUE
      )
      
      predicted_stage <- factor(
        stage_levels[max.col(probability_matrix)],
        levels = stage_levels
      )
    }
    
    actual_all <- c(
      actual_all,
      as.character(fold_valid$Stage)
    )
    
    predicted_all <- c(
      predicted_all,
      as.character(predicted_stage)
    )
  }
  
  metrics <- calculate_metrics(
    actual_all,
    predicted_all
  )
  
  all_results[[model_name]] <- data.frame(
    Model = model_name,
    Accuracy = metrics$accuracy,
    Macro_F1 = metrics$macro_f1
  )
  
  all_predictions[[model_name]] <- data.frame(
    Actual = actual_all,
    Predicted = predicted_all
  )
  
  cat(
    "CV Accuracy:",
    round(metrics$accuracy * 100, 2),
    "% | Macro F1:",
    round(metrics$macro_f1, 4),
    "\n"
  )
}

# 8. COMPARE MODELS

comparison <- bind_rows(all_results) %>%
  arrange(desc(Accuracy), desc(Macro_F1))

write.csv(
  comparison,
  file.path(results_dir, "final_optimization_comparison.csv"),
  row.names = FALSE
)

cat("\n========== MODEL COMPARISON ==========\n")
print(comparison)

best_model_name <- comparison$Model[1]

cat("\nSELECTED MODEL:", best_model_name, "\n")

# 9. FINAL MODEL TRAINING

selected_features <- if (
  best_model_name == "XGBoost Trend Regression"
) {
  trend_features
} else {
  normalized_features
}

prepared_final <- prepare_matrices(
  training_data,
  external_data,
  selected_features
)

train_matrix <- prepared_final$train
external_matrix <- prepared_final$validation

if (best_model_name == "Random Forest Regression") {
  
  set.seed(42)
  
  final_model <- randomForest(
    x = train_matrix,
    y = training_data$SOH,
    ntree = 500,
    mtry = max(
      1,
      floor(sqrt(length(selected_features)))
    ),
    importance = TRUE
  )
  
  external_predicted_soh <- predict(
    final_model,
    external_matrix
  )
  
  external_predictions <- soh_to_stage(
    external_predicted_soh
  )
  
  importance_values <- importance(
    final_model,
    type = 1
  )
  
  importance_table <- data.frame(
    Feature = rownames(importance_values),
    Importance = importance_values[, 1]
  )
  
} else if (
  best_model_name %in% c(
    "XGBoost Regression",
    "XGBoost Trend Regression"
  )
) {
  
  dtrain <- xgb.DMatrix(
    train_matrix,
    label = training_data$SOH
  )
  
  dexternal <- xgb.DMatrix(external_matrix)
  
  final_model <- xgb.train(
    params = list(
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
    ),
    data = dtrain,
    nrounds = 250,
    verbose = 0
  )
  
  external_predicted_soh <- predict(
    final_model,
    dexternal
  )
  
  external_predictions <- soh_to_stage(
    external_predicted_soh
  )
  
  importance_table <- xgb.importance(
    feature_names = selected_features,
    model = final_model
  ) %>%
    select(
      Feature = Feature,
      Importance = Gain
    )
  
} else if (best_model_name == "XGBoost Ordinal") {
  
  # Train three cumulative binary models and save all three.
  
  ordinal_models <- list()
  thresholds <- c(90, 80, 70)
  
  for (i in seq_along(thresholds)) {
    
    labels <- as.integer(
      training_data$SOH < thresholds[i]
    )
    
    dtrain <- xgb.DMatrix(
      train_matrix,
      label = labels
    )
    
    ordinal_models[[i]] <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = 2,
        eta = 0.03,
        min_child_weight = 5,
        subsample = 0.85,
        colsample_bytree = 0.85,
        lambda = 5,
        verbosity = 0
      ),
      data = dtrain,
      nrounds = 150,
      verbose = 0
    )
  }
  
  final_model <- ordinal_models
  
  cumulative <- sapply(
    seq_along(thresholds),
    function(i) {
      predict(
        ordinal_models[[i]],
        xgb.DMatrix(external_matrix)
      )
    }
  )
  
  if (is.null(dim(cumulative))) {
    cumulative <- matrix(cumulative, ncol = 3)
  }
  
  for (i in seq_len(nrow(cumulative))) {
    cumulative[i, ] <- cummax(cumulative[i, ])
  }
  
  probs <- cbind(
    1 - cumulative[, 1],
    cumulative[, 1] - cumulative[, 2],
    cumulative[, 2] - cumulative[, 3],
    cumulative[, 3]
  )
  
  probs <- pmax(probs, 0)
  probs <- probs / rowSums(probs)
  
  external_predictions <- factor(
    stage_levels[max.col(probs)],
    levels = stage_levels
  )
  
  external_predicted_soh <- NA_real_
  
  importance_table <- data.frame(
    Feature = selected_features,
    Importance = NA_real_
  )
  
} else {
  
  # Balanced multiclass XGBoost
  
  train_labels <- as.integer(
    training_data$Stage
  ) - 1
  
  class_counts <- table(
    factor(training_data$Stage, levels = stage_levels)
  )
  
  weights <- length(train_labels) /
    (length(stage_levels) * as.numeric(class_counts))
  
  dtrain <- xgb.DMatrix(
    train_matrix,
    label = train_labels,
    weight = weights[train_labels + 1]
  )
  
  dexternal <- xgb.DMatrix(external_matrix)
  
  final_model <- xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 4,
      eval_metric = "mlogloss",
      max_depth = 2,
      eta = 0.03,
      min_child_weight = 5,
      subsample = 0.85,
      colsample_bytree = 0.85,
      lambda = 5,
      verbosity = 0
    ),
    data = dtrain,
    nrounds = 180,
    verbose = 0
  )
  
  probs <- predict(final_model, dexternal)
  
  probability_matrix <- matrix(
    probs,
    ncol = 4,
    byrow = TRUE
  )
  
  external_predictions <- factor(
    stage_levels[max.col(probability_matrix)],
    levels = stage_levels
  )
  
  external_predicted_soh <- NA_real_
  
  importance_table <- xgb.importance(
    feature_names = selected_features,
    model = final_model
  ) %>%
    select(
      Feature = Feature,
      Importance = Gain
    )
}

# 10. EXTERNAL EVALUATION

external_metrics <- calculate_metrics(
  external_data$Stage,
  external_predictions
)

cat("\n========== EXTERNAL TEST RESULTS ==========\n")

cat(
  "External Accuracy:",
  round(external_metrics$accuracy * 100, 2),
  "%\n"
)

cat(
  "External Macro F1:",
  round(external_metrics$macro_f1, 4),
  "\n"
)

cat("\nConfusion Matrix:\n")
print(external_metrics$confusion)

cat("\nClass-wise Metrics:\n")
print(external_metrics$class_metrics)

# 11. SAVE PREDICTIONS AND RESULTS

external_output <- data.frame(
  Battery_ID = external_data$Battery_ID,
  Cycle_Number = external_data$Cycle_Number,
  Actual_SOH = external_data$SOH,
  Predicted_SOH = external_predicted_soh,
  Actual_Stage = external_data$Stage,
  Predicted_Stage = external_predictions
)

write.csv(
  external_output,
  file.path(results_dir, "final_optimization_predictions.csv"),
  row.names = FALSE
)

write.csv(
  external_metrics$class_metrics,
  file.path(results_dir, "final_optimization_class_metrics.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(external_metrics$confusion),
  file.path(results_dir, "final_optimization_confusion_matrix.csv")
)

write.csv(
  importance_table,
  file.path(results_dir, "final_optimization_feature_importance.csv"),
  row.names = FALSE
)

# 12. SAVE FEATURE IMPORTANCE PLOT

if (all(is.finite(importance_table$Importance))) {
  
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
      title = paste(
        "Feature Importance -",
        best_model_name
      ),
      x = "Feature",
      y = "Importance"
    )
  
  ggsave(
    file.path(
      plots_dir,
      "final_optimization_feature_importance.png"
    ),
    importance_plot,
    width = 10,
    height = 7
  )
}

# 13. SAVE WINNING MODEL

model_bundle <- list(
  model = final_model,
  model_name = best_model_name,
  features = selected_features,
  imputation_medians = prepared_final$medians,
  baseline_method = "Median of first five available cycles",
  thresholds = c(
    Healthy = 90,
    Early_Degradation = 80,
    Moderate_Degradation = 70
  ),
  stage_levels = stage_levels,
  cv_results = comparison,
  cv_accuracy = comparison$Accuracy[1],
  cv_macro_f1 = comparison$Macro_F1[1],
  external_accuracy = external_metrics$accuracy,
  external_macro_f1 = external_metrics$macro_f1,
  training_batteries = unique(training_data$Battery_ID),
  external_batteries = external_ids
)

saveRDS(
  model_bundle,
  file.path(
    models_dir,
    "final_optimized_ev_battery_model.rds"
  )
)

# 14. FINAL SUMMARY

summary_table <- data.frame(
  Metric = c(
    "Selected Model",
    "CV Accuracy",
    "CV Macro F1",
    "External Accuracy",
    "External Macro F1",
    "Training Batteries",
    "External Batteries"
  ),
  Result = c(
    best_model_name,
    paste0(
      round(comparison$Accuracy[1] * 100, 2),
      "%"
    ),
    round(comparison$Macro_F1[1], 4),
    paste0(
      round(external_metrics$accuracy * 100, 2),
      "%"
    ),
    round(external_metrics$macro_f1, 4),
    length(unique(training_data$Battery_ID)),
    paste(external_ids, collapse = ", ")
  )
)

write.csv(
  summary_table,
  file.path(results_dir, "final_optimization_summary.csv"),
  row.names = FALSE
)

cat("\n============================================\n")
cat("FINAL OPTIMISATION COMPLETED\n")
cat("============================================\n")

print(summary_table)

cat("\nSaved model:\n")
cat(
  file.path(
    models_dir,
    "final_optimized_ev_battery_model.rds"
  ),
  "\n"
)

cat("\nAll comparison and evaluation files saved.\n")

