
# ============================================================
# 09 - UNSEEN BATTERY VALIDATION
# Leave-One-Battery-Out Cross-Validation using XGBoost
# ============================================================

library(xgboost)

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(
  project_dir,
  "data",
  "processed"
)

models_dir <- file.path(
  project_dir,
  "models"
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

# ------------------------------------------------------------
# 3. DEFINE FEATURES AND CLASSES
# ------------------------------------------------------------

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

required_columns <- c(
  "Battery_ID",
  "Degradation_Stage",
  feature_names
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

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

data[feature_names] <- lapply(
  data[feature_names],
  as.numeric
)

if (anyNA(data[, c(feature_names, "Degradation_Stage")])) {
  stop("Missing or invalid feature values found.")
}

battery_ids <- sort(unique(data$Battery_ID))

cat("\nBATTERIES AVAILABLE:\n")
print(battery_ids)

cat("\nCLASS DISTRIBUTION:\n")
print(table(data$Degradation_Stage))

# ------------------------------------------------------------
# 4. METRIC FUNCTION
# ------------------------------------------------------------

calculate_metrics <- function(actual, predicted, battery_name) {
  
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
  
  accuracy <- sum(diag(confusion)) / sum(confusion)
  
  class_metrics <- data.frame()
  
  for (class_name in class_levels) {
    
    true_positive <- confusion[
      class_name,
      class_name
    ]
    
    false_positive <- sum(
      confusion[, class_name]
    ) - true_positive
    
    false_negative <- sum(
      confusion[class_name, ]
    ) - true_positive
    
    support <- sum(
      confusion[class_name, ]
    )
    
    precision <- if (
      true_positive + false_positive > 0
    ) {
      true_positive / (
        true_positive + false_positive
      )
    } else {
      NA_real_
    }
    
    recall <- if (
      true_positive + false_negative > 0
    ) {
      true_positive / (
        true_positive + false_negative
      )
    } else {
      NA_real_
    }
    
    f1 <- if (
      !is.na(precision) &&
      !is.na(recall) &&
      precision + recall > 0
    ) {
      2 * precision * recall / (
        precision + recall
      )
    } else if (
      !is.na(precision) &&
      !is.na(recall)
    ) {
      0
    } else {
      NA_real_
    }
    
    class_metrics <- rbind(
      class_metrics,
      data.frame(
        Battery_Test = battery_name,
        Class = class_name,
        Support = support,
        Precision = precision,
        Recall = recall,
        F1_Score = f1
      )
    )
  }
  
  # Macro F1 over classes actually present in the test battery
  observed_classes <- class_metrics$Support > 0
  
  macro_f1 <- mean(
    class_metrics$F1_Score[observed_classes],
    na.rm = TRUE
  )
  
  list(
    accuracy = accuracy,
    macro_f1 = macro_f1,
    confusion = confusion,
    class_metrics = class_metrics
  )
}

# ------------------------------------------------------------
# 5. LEAVE-ONE-BATTERY-OUT VALIDATION
# ------------------------------------------------------------

fold_results <- data.frame()
all_predictions <- data.frame()
all_class_metrics <- data.frame()
all_confusion <- matrix(
  0,
  nrow = length(class_levels),
  ncol = length(class_levels),
  dimnames = list(
    Actual = class_levels,
    Predicted = class_levels
  )
)

for (held_out_battery in battery_ids) {
  
  cat(
    "\n====================================\n"
  )
  
  cat(
    "Testing unseen battery:",
    held_out_battery,
    "\n"
  )
  
  # Training uses all batteries except the held-out battery
  train_fold <- data %>%
    dplyr::filter(
      Battery_ID != held_out_battery
    )
  
  # Testing uses only the unseen battery
  test_fold <- data %>%
    dplyr::filter(
      Battery_ID == held_out_battery
    )
  
  y_train <- factor(
    train_fold$Degradation_Stage,
    levels = class_levels
  )
  
  y_test <- factor(
    test_fold$Degradation_Stage,
    levels = class_levels
  )
  
  # Check that all classes are represented in training
  train_class_counts <- table(y_train)
  
  if (any(train_class_counts == 0)) {
    stop(
      "A training fold is missing a degradation class. ",
      "Cannot safely train this multiclass model."
    )
  }
  
  X_train <- as.matrix(
    train_fold[, feature_names]
  )
  
  X_test <- as.matrix(
    test_fold[, feature_names]
  )
  
  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"
  
  # ----------------------------------------------------------
  # CLASS WEIGHTS
  # ----------------------------------------------------------
  
  class_weights <- 1 / as.numeric(
    train_class_counts[as.character(y_train)]
  )
  
  class_weights <- class_weights / mean(class_weights)
  
  # ----------------------------------------------------------
  # XGBOOST TRAINING
  # ----------------------------------------------------------
  
  dtrain <- xgb.DMatrix(
    data = X_train,
    label = as.integer(y_train) - 1,
    weight = class_weights
  )
  
  dtest <- xgb.DMatrix(
    data = X_test
  )
  
  parameters <- list(
    objective = "multi:softprob",
    num_class = length(class_levels),
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
  
  # ----------------------------------------------------------
  # PREDICTION
  # ----------------------------------------------------------
  
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
  
  predicted_indices <- max.col(
    probability_matrix,
    ties.method = "first"
  )
  
  predicted_classes <- factor(
    class_levels[predicted_indices],
    levels = class_levels
  )
  
  # ----------------------------------------------------------
  # EVALUATION
  # ----------------------------------------------------------
  
  metrics <- calculate_metrics(
    actual = y_test,
    predicted = predicted_classes,
    battery_name = held_out_battery
  )
  
  cat(
    "Accuracy:",
    round(metrics$accuracy * 100, 2),
    "%\n"
  )
  
  cat(
    "Macro F1:",
    round(metrics$macro_f1, 4),
    "\n"
  )
  
  cat("\nConfusion Matrix:\n")
  print(metrics$confusion)
  
  fold_results <- rbind(
    fold_results,
    data.frame(
      Held_Out_Battery = held_out_battery,
      Training_Records = nrow(train_fold),
      Testing_Records = nrow(test_fold),
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1
    )
  )
  
  all_class_metrics <- rbind(
    all_class_metrics,
    metrics$class_metrics
  )
  
  all_confusion <- all_confusion +
    as.matrix(metrics$confusion)
  
  fold_predictions <- data.frame(
    Battery_ID = held_out_battery,
    Cycle_Number = test_fold$Cycle_Number,
    Actual_Stage = as.character(y_test),
    Predicted_Stage = as.character(predicted_classes),
    stringsAsFactors = FALSE
  )
  
  all_predictions <- rbind(
    all_predictions,
    fold_predictions
  )
}

# ------------------------------------------------------------
# 6. OVERALL RESULTS
# ------------------------------------------------------------

overall_accuracy <- mean(
  all_predictions$Actual_Stage ==
    all_predictions$Predicted_Stage
)

overall_metrics <- calculate_metrics(
  actual = all_predictions$Actual_Stage,
  predicted = all_predictions$Predicted_Stage,
  battery_name = "Overall"
)

overall_summary <- data.frame(
  Evaluation = "Leave-One-Battery-Out",
  Total_Batteries = length(battery_ids),
  Total_Observations = nrow(all_predictions),
  Overall_Accuracy = overall_accuracy,
  Macro_F1 = overall_metrics$macro_f1
)

cat("\n====================================\n")
cat("FINAL UNSEEN BATTERY VALIDATION\n")
cat("====================================\n")

print(fold_results)

cat("\nOVERALL RESULTS:\n")
print(overall_summary)

cat("\nOVERALL CONFUSION MATRIX:\n")
print(all_confusion)

# ------------------------------------------------------------
# 7. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  fold_results,
  file.path(
    processed_dir,
    "unseen_battery_fold_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  overall_summary,
  file.path(
    processed_dir,
    "unseen_battery_summary.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(
    processed_dir,
    "unseen_battery_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_class_metrics,
  file.path(
    processed_dir,
    "unseen_battery_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(all_confusion),
  file.path(
    processed_dir,
    "unseen_battery_confusion_matrix.csv"
  )
)

saveRDS(
  overall_metrics,
  file.path(
    models_dir,
    "unseen_battery_validation.rds"
  )
)

cat("\nVALIDATION COMPLETED SUCCESSFULLY.\n")
cat("All results saved in data/processed/.\n")
