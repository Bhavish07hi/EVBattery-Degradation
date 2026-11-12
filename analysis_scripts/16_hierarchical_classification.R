
# ============================================================
# 16 - HIERARCHICAL XGBOOST CLASSIFICATION
# Compare with standard 4-class XGBoost
# Leave-One-Battery-Out Validation
# ============================================================

library(dplyr)
library(xgboost)

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(
  project_dir,
  "data",
  "processed"
)

class_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

features <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

# ------------------------------------------------------------
# 2. LOAD DATA AND BASELINE PREDICTIONS
# ------------------------------------------------------------

data <- read.csv(
  file.path(
    processed_dir,
    "battery_classification.csv"
  ),
  stringsAsFactors = FALSE
)

baseline <- read.csv(
  file.path(
    processed_dir,
    "severe_threshold_probabilities.csv"
  ),
  stringsAsFactors = FALSE
)

data$Battery_ID <- as.character(data$Battery_ID)
baseline$Battery_ID <- as.character(baseline$Battery_ID)

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

baseline$Actual_Stage <- factor(
  baseline$Actual_Stage,
  levels = class_levels
)

data[features] <- lapply(
  data[features],
  as.numeric
)

if (
  anyNA(data[, features]) ||
  anyNA(data$Degradation_Stage)
) {
  stop("Missing values found in the input dataset.")
}

data <- data %>%
  arrange(Battery_ID, Cycle_Number)

# ------------------------------------------------------------
# 3. METRIC FUNCTION
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
  
  results <- data.frame()
  
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
    
    results <- rbind(
      results,
      data.frame(
        Class = class_name,
        Support = support,
        Precision = precision,
        Recall = recall,
        F1 = f1
      )
    )
  }
  
  list(
    confusion = confusion,
    class_metrics = results,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(results$F1)
  )
}

# ------------------------------------------------------------
# 4. GENERAL BINARY XGBOOST TRAINING FUNCTION
# ------------------------------------------------------------

train_binary_model <- function(
    train_data,
    positive_class,
    negative_class,
    severe_weight = FALSE
) {
  
  selected_data <- train_data %>%
    filter(
      Degradation_Stage %in% c(
        negative_class,
        positive_class
      )
    )
  
  y <- as.integer(
    selected_data$Degradation_Stage == positive_class
  )
  
  class_counts <- table(
    factor(y, levels = c(0, 1))
  )
  
  if (any(class_counts == 0)) {
    stop(
      "A binary training stage is missing a class: ",
      negative_class,
      " or ",
      positive_class
    )
  }
  
  X <- as.matrix(
    selected_data[, features, drop = FALSE]
  )
  
  storage.mode(X) <- "double"
  
  # Inverse-frequency sample weighting
  sample_weights <- 1 / as.numeric(
    class_counts[as.character(y)]
  )
  
  sample_weights <- sample_weights / mean(
    sample_weights
  )
  
  # Additional severe-class weight in Stage 3
  if (severe_weight) {
    
    sample_weights[y == 1] <-
      sample_weights[y == 1] * 2
    
    sample_weights <- sample_weights / mean(
      sample_weights
    )
  }
  
  dtrain <- xgb.DMatrix(
    data = X,
    label = y,
    weight = sample_weights
  )
  
  parameters <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 3,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 2,
    seed = 123
  )
  
  set.seed(123)
  
  xgb.train(
    params = parameters,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
}

# ------------------------------------------------------------
# 5. LEAVE-ONE-BATTERY-OUT VALIDATION
# ------------------------------------------------------------

battery_ids <- unique(data$Battery_ID)

hierarchical_predictions <- data.frame()

for (held_out_battery in battery_ids) {
  
  cat(
    "\n========================================\n"
  )
  
  cat(
    "HELD-OUT BATTERY:",
    held_out_battery,
    "\n"
  )
  
  train_fold <- data %>%
    filter(
      Battery_ID != held_out_battery
    )
  
  test_fold <- data %>%
    filter(
      Battery_ID == held_out_battery
    )
  
  X_test <- as.matrix(
    test_fold[, features, drop = FALSE]
  )
  
  storage.mode(X_test) <- "double"
  
  dtest <- xgb.DMatrix(
    data = X_test
  )
  
  # ----------------------------------------------------------
  # STAGE 1: HEALTHY VS DEGRADED
  # ----------------------------------------------------------
  
  cat("Training Stage 1...\n")
  
  model_stage1 <- train_binary_model(
    train_data = train_fold,
    negative_class = "Healthy",
    positive_class = "Early Degradation"
  )
  
  # Stage 1 training function uses only the two supplied
  # classes. Build a separate model for Healthy vs all degraded.
  stage1_train <- train_fold %>%
    mutate(
      Binary_Label = ifelse(
        Degradation_Stage == "Healthy",
        "Healthy",
        "Degraded"
      )
    )
  
  stage1_train$Degradation_Stage <- factor(
    stage1_train$Binary_Label,
    levels = c("Healthy", "Degraded")
  )
  
  model_stage1 <- train_binary_model(
    train_data = stage1_train,
    negative_class = "Healthy",
    positive_class = "Degraded"
  )
  
  p_degraded <- predict(
    model_stage1,
    dtest
  )
  
  # ----------------------------------------------------------
  # STAGE 2: EARLY VS ADVANCED DEGRADATION
  # ----------------------------------------------------------
  
  cat("Training Stage 2...\n")
  
  stage2_train <- train_fold %>%
    filter(
      Degradation_Stage != "Healthy"
    ) %>%
    mutate(
      Binary_Label = ifelse(
        Degradation_Stage == "Early Degradation",
        "Early",
        "Advanced"
      )
    )
  
  stage2_train$Degradation_Stage <- factor(
    stage2_train$Binary_Label,
    levels = c("Early", "Advanced")
  )
  
  model_stage2 <- train_binary_model(
    train_data = stage2_train,
    negative_class = "Early",
    positive_class = "Advanced"
  )
  
  p_advanced <- predict(
    model_stage2,
    dtest
  )
  
  # ----------------------------------------------------------
  # STAGE 3: MODERATE VS SEVERE
  # ----------------------------------------------------------
  
  cat("Training Stage 3...\n")
  
  stage3_train <- train_fold %>%
    filter(
      Degradation_Stage %in% c(
        "Moderate Degradation",
        "Severe Degradation"
      )
    )
  
  model_stage3 <- train_binary_model(
    train_data = stage3_train,
    negative_class = "Moderate Degradation",
    positive_class = "Severe Degradation",
    severe_weight = TRUE
  )
  
  p_severe <- predict(
    model_stage3,
    dtest
  )
  
  # ----------------------------------------------------------
  # HIERARCHICAL DECISION
  # ----------------------------------------------------------
  
  predicted_classes <- character(
    nrow(test_fold)
  )
  
  for (i in seq_len(nrow(test_fold))) {
    
    if (p_degraded[i] < 0.5) {
      
      predicted_classes[i] <- "Healthy"
      
    } else if (p_advanced[i] < 0.5) {
      
      predicted_classes[i] <- "Early Degradation"
      
    } else if (p_severe[i] >= 0.5) {
      
      predicted_classes[i] <- "Severe Degradation"
      
    } else {
      
      predicted_classes[i] <- "Moderate Degradation"
      
    }
  }
  
  fold_predictions <- data.frame(
    Battery_ID = held_out_battery,
    Cycle_Number = test_fold$Cycle_Number,
    Actual_Stage = as.character(
      test_fold$Degradation_Stage
    ),
    Predicted_Stage = predicted_classes,
    Probability_Degraded = p_degraded,
    Probability_Advanced = p_advanced,
    Probability_Severe_Conditional = p_severe
  )
  
  hierarchical_predictions <- rbind(
    hierarchical_predictions,
    fold_predictions
  )
}

# ------------------------------------------------------------
# 6. EVALUATE HIERARCHICAL MODEL
# ------------------------------------------------------------

hierarchical_metrics <- calculate_metrics(
  hierarchical_predictions$Actual_Stage,
  hierarchical_predictions$Predicted_Stage
)

cat("\n========================================\n")
cat("HIERARCHICAL MODEL RESULTS\n")
cat("========================================\n")

print(
  data.frame(
    Accuracy = hierarchical_metrics$accuracy,
    Macro_F1 = hierarchical_metrics$macro_f1
  )
)

cat("\nHIERARCHICAL CONFUSION MATRIX:\n")

print(hierarchical_metrics$confusion)

cat("\nHIERARCHICAL CLASS METRICS:\n")

print(hierarchical_metrics$class_metrics)

# ------------------------------------------------------------
# 7. LOAD AND EVALUATE BASELINE MODEL
# ------------------------------------------------------------

baseline$Predicted_Stage <- class_levels[
  max.col(
    as.matrix(
      baseline[, c(
        "Probability_Healthy",
        "Probability_Early",
        "Probability_Moderate",
        "Probability_Severe"
      )]
    ),
    ties.method = "first"
  )
]

# Match predictions by battery and cycle
baseline_matched <- baseline %>%
  select(
    Battery_ID,
    Cycle_Number,
    Actual_Stage,
    Predicted_Stage
  )

hierarchical_matched <- hierarchical_predictions %>%
  select(
    Battery_ID,
    Cycle_Number,
    Actual_Stage,
    Predicted_Stage
  )

comparison_data <- baseline_matched %>%
  inner_join(
    hierarchical_matched,
    by = c(
      "Battery_ID",
      "Cycle_Number"
    ),
    suffix = c("_Baseline", "_Hierarchical")
  )

if (nrow(comparison_data) != nrow(data)) {
  stop("Baseline and hierarchical predictions do not match.")
}

baseline_metrics <- calculate_metrics(
  comparison_data$Actual_Stage_Baseline,
  comparison_data$Predicted_Stage_Baseline
)

hierarchical_metrics <- calculate_metrics(
  comparison_data$Actual_Stage_Hierarchical,
  comparison_data$Predicted_Stage_Hierarchical
)

baseline_severe <- baseline_metrics$class_metrics %>%
  filter(Class == "Severe Degradation")

hierarchical_severe <- hierarchical_metrics$class_metrics %>%
  filter(Class == "Severe Degradation")

# ------------------------------------------------------------
# 8. FINAL MODEL COMPARISON
# ------------------------------------------------------------

final_comparison <- data.frame(
  Model = c(
    "Standard XGBoost",
    "Hierarchical XGBoost"
  ),
  
  Accuracy = c(
    baseline_metrics$accuracy,
    hierarchical_metrics$accuracy
  ),
  
  Macro_F1 = c(
    baseline_metrics$macro_f1,
    hierarchical_metrics$macro_f1
  ),
  
  Severe_Precision = c(
    baseline_severe$Precision,
    hierarchical_severe$Precision
  ),
  
  Severe_Recall = c(
    baseline_severe$Recall,
    hierarchical_severe$Recall
  ),
  
  Severe_F1 = c(
    baseline_severe$F1,
    hierarchical_severe$F1
  )
)

cat("\n========================================\n")
cat("FINAL MODEL COMPARISON\n")
cat("========================================\n")

print(final_comparison)

# ------------------------------------------------------------
# 9. BATTERY-WISE PERFORMANCE
# ------------------------------------------------------------

battery_results <- data.frame()

for (battery in battery_ids) {
  
  fold <- comparison_data %>%
    filter(
      Battery_ID == battery
    )
  
  base_metrics <- calculate_metrics(
    fold$Actual_Stage_Baseline,
    fold$Predicted_Stage_Baseline
  )
  
  hier_metrics <- calculate_metrics(
    fold$Actual_Stage_Hierarchical,
    fold$Predicted_Stage_Hierarchical
  )
  
  battery_results <- rbind(
    battery_results,
    data.frame(
      Battery_ID = battery,
      Baseline_Accuracy = base_metrics$accuracy,
      Hierarchical_Accuracy = hier_metrics$accuracy,
      Baseline_Macro_F1 = base_metrics$macro_f1,
      Hierarchical_Macro_F1 = hier_metrics$macro_f1
    )
  )
}

cat("\nBATTERY-WISE MODEL COMPARISON:\n")

print(battery_results)

# ------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  hierarchical_predictions,
  file.path(
    processed_dir,
    "hierarchical_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  final_comparison,
  file.path(
    processed_dir,
    "hierarchical_model_comparison.csv"
  ),
  row.names = FALSE
)

write.csv(
  hierarchical_metrics$class_metrics,
  file.path(
    processed_dir,
    "hierarchical_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(
    hierarchical_metrics$confusion
  ),
  file.path(
    processed_dir,
    "hierarchical_confusion_matrix.csv"
  )
)

write.csv(
  battery_results,
  file.path(
    processed_dir,
    "hierarchical_battery_comparison.csv"
  ),
  row.names = FALSE
)

cat("\n========================================\n")
cat("HIERARCHICAL CLASSIFICATION COMPLETED\n")
cat("========================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)
