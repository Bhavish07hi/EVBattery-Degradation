
# ============================================================
# 10 - SEVERE DEGRADATION CLASS WEIGHT OPTIMIZATION
# ============================================================

library(xgboost)

# ------------------------------------------------------------
# 1. PATHS AND DATA
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(project_dir, "data", "processed")
models_dir <- file.path(project_dir, "models")

data <- read.csv(
  file.path(processed_dir, "battery_classification.csv"),
  stringsAsFactors = FALSE
)

feature_names <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

class_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

data$Battery_ID <- as.character(data$Battery_ID)

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

data[feature_names] <- lapply(
  data[feature_names],
  as.numeric
)

if (anyNA(data[, c(feature_names, "Degradation_Stage")])) {
  stop("Missing or invalid values found in the dataset.")
}

battery_ids <- sort(unique(data$Battery_ID))

# Severe-class weighting multipliers
severe_multipliers <- c(1, 1.5, 2, 3, 4)

# ------------------------------------------------------------
# 2. METRIC FUNCTION
# ------------------------------------------------------------

calculate_class_metrics <- function(actual, predicted) {
  
  actual <- factor(actual, levels = class_levels)
  predicted <- factor(predicted, levels = class_levels)
  
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
      NA_real_
    }
    
    recall <- if (tp + fn > 0) {
      tp / (tp + fn)
    } else {
      NA_real_
    }
    
    f1 <- if (
      !is.na(precision) &&
      !is.na(recall) &&
      precision + recall > 0
    ) {
      2 * precision * recall / (precision + recall)
    } else if (
      !is.na(precision) &&
      !is.na(recall)
    ) {
      0
    } else {
      NA_real_
    }
    
    results <- rbind(
      results,
      data.frame(
        Class = class_name,
        Support = support,
        Precision = precision,
        Recall = recall,
        F1_Score = f1
      )
    )
  }
  
  results
}

# ------------------------------------------------------------
# 3. EXPERIMENT LOOP
# ------------------------------------------------------------

all_results <- data.frame()
all_predictions <- data.frame()
all_class_results <- data.frame()

for (multiplier in severe_multipliers) {
  
  cat("\n========================================\n")
  cat("SEVERE CLASS WEIGHT:", multiplier, "x\n")
  cat("========================================\n")
  
  experiment_predictions <- data.frame()
  
  for (held_out_battery in battery_ids) {
    
    cat("Testing battery:", held_out_battery, "\n")
    
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
    
    y_train <- factor(
      train_fold$Degradation_Stage,
      levels = class_levels
    )
    
    y_test <- factor(
      test_fold$Degradation_Stage,
      levels = class_levels
    )
    
    class_counts <- table(y_train)
    
    if (any(class_counts == 0)) {
      stop("A training fold is missing a class.")
    }
    
    X_train <- as.matrix(
      train_fold[, feature_names, drop = FALSE]
    )
    
    X_test <- as.matrix(
      test_fold[, feature_names, drop = FALSE]
    )
    
    storage.mode(X_train) <- "double"
    storage.mode(X_test) <- "double"
    
    # --------------------------------------------------------
    # BASE CLASS WEIGHTS
    # --------------------------------------------------------
    
    sample_weights <- 1 / as.numeric(
      class_counts[as.character(y_train)]
    )
    
    sample_weights <- sample_weights / mean(sample_weights)
    
    # Increase Severe Degradation weight
    severe_rows <- y_train == "Severe Degradation"
    
    sample_weights[severe_rows] <-
      sample_weights[severe_rows] * multiplier
    
    # Normalize weights
    sample_weights <- sample_weights / mean(sample_weights)
    
    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------
    
    dtrain <- xgb.DMatrix(
      data = X_train,
      label = as.integer(y_train) - 1,
      weight = sample_weights
    )
    
    dtest <- xgb.DMatrix(
      data = X_test
    )
    
    parameters <- list(
      objective = "multi:softprob",
      num_class = 4,
      eval_metric = "mlogloss",
      eta = 0.05,
      max_depth = 3,
      subsample = 0.8,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      nthread = 2,
      seed = 123
    )
    
    set.seed(123)
    
    model <- xgb.train(
      params = parameters,
      data = dtrain,
      nrounds = 100,
      verbose = 0
    )
    
    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------
    
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
      stop("Unexpected prediction dimensions.")
    }
    
    predicted_indices <- max.col(
      probability_matrix,
      ties.method = "first"
    )
    
    predicted_classes <- factor(
      class_levels[predicted_indices],
      levels = class_levels
    )
    
    fold_predictions <- data.frame(
      Weight_Multiplier = multiplier,
      Battery_ID = held_out_battery,
      Actual_Stage = as.character(y_test),
      Predicted_Stage = as.character(predicted_classes)
    )
    
    experiment_predictions <- rbind(
      experiment_predictions,
      fold_predictions
    )
  }
  
  # ----------------------------------------------------------
  # POOLED PERFORMANCE
  # ----------------------------------------------------------
  
  class_metrics <- calculate_class_metrics(
    experiment_predictions$Actual_Stage,
    experiment_predictions$Predicted_Stage
  )
  
  accuracy <- mean(
    experiment_predictions$Actual_Stage ==
      experiment_predictions$Predicted_Stage
  )
  
  macro_f1 <- mean(
    class_metrics$F1_Score,
    na.rm = TRUE
  )
  
  severe_metrics <- class_metrics[
    class_metrics$Class == "Severe Degradation",
  ]
  
  result_row <- data.frame(
    Severe_Weight = multiplier,
    Accuracy = accuracy,
    Macro_F1 = macro_f1,
    Severe_Precision = severe_metrics$Precision,
    Severe_Recall = severe_metrics$Recall,
    Severe_F1 = severe_metrics$F1_Score
  )
  
  all_results <- rbind(
    all_results,
    result_row
  )
  
  class_metrics$Severe_Weight <- multiplier
  
  all_class_results <- rbind(
    all_class_results,
    class_metrics
  )
  
  all_predictions <- rbind(
    all_predictions,
    experiment_predictions
  )
  
  cat("\nExperiment results:\n")
  print(result_row)
  
  cat("\nClass-wise results:\n")
  print(class_metrics)
}

# ------------------------------------------------------------
# 4. DISPLAY COMPARISON
# ------------------------------------------------------------

cat("\n========================================\n")
cat("FINAL CLASS WEIGHT COMPARISON\n")
cat("========================================\n")

print(all_results)

# ------------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  all_results,
  file.path(
    processed_dir,
    "severe_weight_comparison.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_class_results,
  file.path(
    processed_dir,
    "severe_weight_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(
    processed_dir,
    "severe_weight_predictions.csv"
  ),
  row.names = FALSE
)

# Select candidate based on severe recall,
# with macro F1 as a tie-breaker.
ranked_results <- all_results[
  order(
    -all_results$Severe_Recall,
    -all_results$Macro_F1
  ),
]

write.csv(
  ranked_results,
  file.path(
    processed_dir,
    "severe_weight_ranked_results.csv"
  ),
  row.names = FALSE
)

cat("\nAll experiments completed.\n")
cat("Results saved in data/processed/.\n")
