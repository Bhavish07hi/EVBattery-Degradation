
# ============================================================
# 13 - SEVERE DEGRADATION THRESHOLD OPTIMIZATION
# Leave-One-Battery-Out Validation
# ============================================================

library(dplyr)
library(xgboost)

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(
  project_dir, "data", "processed"
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
# 2. LOAD DATA
# ------------------------------------------------------------

data <- read.csv(
  file.path(
    processed_dir,
    "battery_classification.csv"
  ),
  stringsAsFactors = FALSE
)

required_columns <- c(
  "Battery_ID",
  "Degradation_Stage",
  features
)

missing_columns <- setdiff(
  required_columns,
  names(data)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

data$Battery_ID <- as.character(data$Battery_ID)

data[features] <- lapply(
  data[features],
  as.numeric
)

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

if (
  anyNA(data[, required_columns]) ||
  anyNA(data$Degradation_Stage)
) {
  stop("Missing or invalid input values found.")
}

data <- data %>%
  arrange(Battery_ID, Cycle_Number)

battery_ids <- unique(data$Battery_ID)

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
  
  class_results <- data.frame()
  
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
    
    class_results <- rbind(
      class_results,
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
    class_metrics = class_results,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(class_results$F1_Score)
  )
}

# ------------------------------------------------------------
# 4. TRAINING FUNCTION
# Same model configuration as Stage 9
# ------------------------------------------------------------

train_model <- function(train_data) {
  
  y_train <- factor(
    train_data$Degradation_Stage,
    levels = class_levels
  )
  
  class_counts <- table(y_train)
  
  if (any(class_counts == 0)) {
    stop("A training fold is missing a class.")
  }
  
  X_train <- as.matrix(
    train_data[, features, drop = FALSE]
  )
  
  storage.mode(X_train) <- "double"
  
  # Inverse-frequency weights
  sample_weights <- 1 / as.numeric(
    class_counts[as.character(y_train)]
  )
  
  sample_weights <- sample_weights / mean(
    sample_weights
  )
  
  # 2x Severe Degradation weight
  severe_rows <- y_train == "Severe Degradation"
  
  sample_weights[severe_rows] <-
    sample_weights[severe_rows] * 2
  
  sample_weights <- sample_weights / mean(
    sample_weights
  )
  
  dtrain <- xgb.DMatrix(
    data = X_train,
    label = as.integer(y_train) - 1,
    weight = sample_weights
  )
  
  parameters <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
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
# 5. LEAVE-ONE-BATTERY-OUT PREDICTIONS
# ------------------------------------------------------------

oof_predictions <- data.frame()

for (held_out_battery in battery_ids) {
  
  cat(
    "\nTesting unseen battery:",
    held_out_battery,
    "\n"
  )
  
  train_fold <- data[
    data$Battery_ID != held_out_battery,
    ,
    drop = FALSE
  ]
  
  test_fold <- data[
    data$Battery_ID == held_out_battery,
    ,
    drop = FALSE
  ]
  
  model <- train_model(train_fold)
  
  X_test <- as.matrix(
    test_fold[, features, drop = FALSE]
  )
  
  storage.mode(X_test) <- "double"
  
  dtest <- xgb.DMatrix(
    data = X_test
  )
  
  probabilities <- predict(
    model,
    dtest
  )
  
  if (is.matrix(probabilities)) {
    
    probability_matrix <- probabilities
    
  } else {
    
    probability_matrix <- matrix(
      probabilities,
      ncol = length(class_levels),
      byrow = TRUE
    )
  }
  
  if (
    nrow(probability_matrix) != nrow(test_fold) ||
    ncol(probability_matrix) != length(class_levels)
  ) {
    stop("Unexpected prediction probability dimensions.")
  }
  
  fold_results <- data.frame(
    Battery_ID = held_out_battery,
    Cycle_Number = test_fold$Cycle_Number,
    Actual_Stage = as.character(
      test_fold$Degradation_Stage
    ),
    Probability_Healthy = probability_matrix[, 1],
    Probability_Early = probability_matrix[, 2],
    Probability_Moderate = probability_matrix[, 3],
    Probability_Severe = probability_matrix[, 4]
  )
  
  oof_predictions <- rbind(
    oof_predictions,
    fold_results
  )
}

# ------------------------------------------------------------
# 6. SAVE OUT-OF-FOLD PROBABILITIES
# ------------------------------------------------------------

write.csv(
  oof_predictions,
  file.path(
    processed_dir,
    "severe_threshold_probabilities.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. STANDARD ARGMAX BASELINE
# ------------------------------------------------------------

probability_matrix <- as.matrix(
  oof_predictions[, c(
    "Probability_Healthy",
    "Probability_Early",
    "Probability_Moderate",
    "Probability_Severe"
  )]
)

baseline_indices <- max.col(
  probability_matrix,
  ties.method = "first"
)

baseline_predictions <- class_levels[
  baseline_indices
]

baseline_metrics <- calculate_metrics(
  oof_predictions$Actual_Stage,
  baseline_predictions
)

cat("\nSTANDARD ARGMAX BASELINE:\n")

print(
  data.frame(
    Accuracy = baseline_metrics$accuracy,
    Macro_F1 = baseline_metrics$macro_f1
  )
)

# ------------------------------------------------------------
# 8. TEST SEVERE CLASS THRESHOLDS
# ------------------------------------------------------------

thresholds <- seq(
  0.20,
  0.70,
  by = 0.05
)

threshold_results <- data.frame()

threshold_predictions <- data.frame()

for (threshold in thresholds) {
  
  predicted_classes <- character(
    nrow(oof_predictions)
  )
  
  for (i in seq_len(nrow(oof_predictions))) {
    
    severe_probability <-
      probability_matrix[i, 4]
    
    if (severe_probability >= threshold) {
      
      predicted_classes[i] <-
        "Severe Degradation"
      
    } else {
      
      non_severe_probabilities <-
        probability_matrix[i, 1:3]
      
      best_non_severe <- which.max(
        non_severe_probabilities
      )
      
      predicted_classes[i] <-
        class_levels[best_non_severe]
    }
  }
  
  metrics <- calculate_metrics(
    oof_predictions$Actual_Stage,
    predicted_classes
  )
  
  severe_metrics <- metrics$class_metrics[
    metrics$class_metrics$Class ==
      "Severe Degradation",
  ]
  
  threshold_results <- rbind(
    threshold_results,
    data.frame(
      Threshold = threshold,
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1,
      Severe_Precision = severe_metrics$Precision,
      Severe_Recall = severe_metrics$Recall,
      Severe_F1 = severe_metrics$F1_Score
    )
  )
  
  threshold_predictions <- rbind(
    threshold_predictions,
    data.frame(
      Threshold = threshold,
      Actual_Stage = oof_predictions$Actual_Stage,
      Predicted_Stage = predicted_classes
    )
  )
  
  cat(
    "\nThreshold:",
    threshold,
    "\n"
  )
  
  print(
    data.frame(
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1,
      Severe_Precision = severe_metrics$Precision,
      Severe_Recall = severe_metrics$Recall,
      Severe_F1 = severe_metrics$F1_Score
    )
  )
}

# ------------------------------------------------------------
# 9. IDENTIFY BEST THRESHOLD
# ------------------------------------------------------------

# Select the threshold with the highest Severe F1.
# If tied, prefer higher Macro F1.

ranked_results <- threshold_results[
  order(
    -threshold_results$Severe_F1,
    -threshold_results$Macro_F1
  ),
]

best_threshold <- ranked_results[1, ]

cat("\n========================================\n")
cat("BEST SEVERE THRESHOLD\n")
cat("========================================\n")

print(best_threshold)

# ------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  threshold_results,
  file.path(
    processed_dir,
    "severe_threshold_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  ranked_results,
  file.path(
    processed_dir,
    "severe_threshold_ranked.csv"
  ),
  row.names = FALSE
)

write.csv(
  threshold_predictions,
  file.path(
    processed_dir,
    "severe_threshold_predictions.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 11. BEST THRESHOLD CONFUSION MATRIX
# ------------------------------------------------------------

best_predictions <- threshold_predictions %>%
  filter(
    Threshold == best_threshold$Threshold
  )

best_metrics <- calculate_metrics(
  best_predictions$Actual_Stage,
  best_predictions$Predicted_Stage
)

cat("\nBEST THRESHOLD CONFUSION MATRIX:\n")

print(best_metrics$confusion)

cat("\nBEST THRESHOLD CLASS METRICS:\n")

print(best_metrics$class_metrics)

# ------------------------------------------------------------
# 12. FINAL COMPARISON
# ------------------------------------------------------------

final_comparison <- data.frame(
  Method = c(
    "Standard Argmax",
    paste0(
      "Severe Threshold ",
      best_threshold$Threshold
    )
  ),
  
  Accuracy = c(
    baseline_metrics$accuracy,
    best_metrics$accuracy
  ),
  
  Macro_F1 = c(
    baseline_metrics$macro_f1,
    best_metrics$macro_f1
  ),
  
  Severe_Precision = c(
    baseline_metrics$class_metrics$Precision[4],
    best_metrics$class_metrics$Precision[4]
  ),
  
  Severe_Recall = c(
    baseline_metrics$class_metrics$Recall[4],
    best_metrics$class_metrics$Recall[4]
  ),
  
  Severe_F1 = c(
    baseline_metrics$class_metrics$F1_Score[4],
    best_metrics$class_metrics$F1_Score[4]
  )
)

cat("\n========================================\n")
cat("FINAL THRESHOLD COMPARISON\n")
cat("========================================\n")

print(final_comparison)

write.csv(
  final_comparison,
  file.path(
    processed_dir,
    "severe_threshold_final_comparison.csv"
  ),
  row.names = FALSE
)

cat("\n========================================\n")
cat("THRESHOLD OPTIMIZATION COMPLETED\n")
cat("========================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)
