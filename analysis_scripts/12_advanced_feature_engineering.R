
# ============================================================
# 12 - ADVANCED FEATURE ENGINEERING
# Baseline vs Engineered XGBoost
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

models_dir <- file.path(
  project_dir, "models"
)

if (!dir.exists(processed_dir)) {
  stop("Processed data folder not found.")
}

if (!dir.exists(models_dir)) {
  dir.create(models_dir, recursive = TRUE)
}

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

original_features <- c(
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
  "Cycle_Number",
  "Discharge_Capacity",
  "Degradation_Stage",
  original_features
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

data[original_features] <- lapply(
  data[original_features],
  as.numeric
)

data$Discharge_Capacity <- as.numeric(
  data$Discharge_Capacity
)

data$Degradation_Stage <- factor(
  data$Degradation_Stage,
  levels = class_levels
)

if (anyNA(data[, required_columns])) {
  stop("Missing or invalid values found in the input data.")
}

# ------------------------------------------------------------
# 3. SORT BATTERY OBSERVATIONS
# ------------------------------------------------------------

data <- data %>%
  arrange(
    Battery_ID,
    Cycle_Number
  )

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

engineered_data <- data

# Feature 1: Voltage Range
engineered_data$Voltage_Range <-
  engineered_data$Max_Voltage -
  engineered_data$Min_Voltage

# Feature 2: Temperature Rise
engineered_data$Temperature_Rise <-
  engineered_data$Max_Temperature -
  engineered_data$Mean_Temperature

# Initialize historical features
engineered_data$Previous_Capacity_Change <- 0

engineered_data$Previous_Capacity_Slope <- 0

# Calculate historical features separately for each battery
battery_ids <- unique(engineered_data$Battery_ID)

for (battery in battery_ids) {
  
  battery_rows <- which(
    engineered_data$Battery_ID == battery
  )
  
  capacities <- engineered_data$Discharge_Capacity[
    battery_rows
  ]
  
  for (i in seq_along(battery_rows)) {
    
    current_row <- battery_rows[i]
    
    # --------------------------------------------------------
    # Feature 3: Previous Capacity Change
    # Uses the two observations BEFORE the current observation
    # --------------------------------------------------------
    
    if (i >= 3) {
      
      previous_capacity <- capacities[i - 1]
      
      previous_previous_capacity <- capacities[i - 2]
      
      if (
        is.finite(previous_capacity) &&
        is.finite(previous_previous_capacity) &&
        previous_previous_capacity != 0
      ) {
        
        engineered_data$Previous_Capacity_Change[
          current_row
        ] <- (
          previous_capacity /
            previous_previous_capacity
        ) - 1
        
      }
    }
    
    # --------------------------------------------------------
    # Feature 4: Previous Capacity Slope
    # Uses up to five preceding observations only
    # --------------------------------------------------------
    
    if (i >= 4) {
      
      start_index <- max(1, i - 5)
      
      historical_capacities <- capacities[
        start_index:(i - 1)
      ]
      
      if (
        length(historical_capacities) >= 3 &&
        all(is.finite(historical_capacities))
      ) {
        
        historical_cycles <- seq_along(
          historical_capacities
        )
        
        slope <- tryCatch(
          coef(
            lm(
              historical_capacities ~ historical_cycles
            )
          )[2],
          error = function(e) 0
        )
        
        if (is.finite(slope)) {
          
          engineered_data$Previous_Capacity_Slope[
            current_row
          ] <- as.numeric(slope)
          
        }
      }
    }
  }
}

# ------------------------------------------------------------
# 5. DEFINE FEATURE SETS
# ------------------------------------------------------------

engineered_features <- c(
  original_features,
  "Voltage_Range",
  "Temperature_Rise",
  "Previous_Capacity_Change",
  "Previous_Capacity_Slope"
)

cat("\nORIGINAL FEATURE COUNT:", length(original_features))

cat(
  "\nENGINEERED FEATURE COUNT:",
  length(engineered_features),
  "\n"
)

cat("\nENGINEERED FEATURE SUMMARY:\n")

print(
  summary(
    engineered_data[, engineered_features]
  )
)

# ------------------------------------------------------------
# 6. SAVE ENGINEERED DATASET
# ------------------------------------------------------------

write.csv(
  engineered_data,
  file.path(
    processed_dir,
    "battery_engineered_features.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. METRIC FUNCTION
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
      
      2 * precision * recall /
        (precision + recall)
      
    } else if (
      !is.na(precision) &&
      !is.na(recall)
    ) {
      
      0
      
    } else {
      
      NA_real_
      
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
  
  observed_classes <- class_results$Support > 0
  
  list(
    confusion = confusion,
    class_metrics = class_results,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(
      class_results$F1_Score[observed_classes],
      na.rm = TRUE
    )
  )
}

# ------------------------------------------------------------
# 8. XGBOOST TRAINING FUNCTION
# ------------------------------------------------------------

train_model <- function(
    train_data,
    selected_features
) {
  
  y_train <- factor(
    train_data$Degradation_Stage,
    levels = class_levels
  )
  
  class_counts <- table(y_train)
  
  if (any(class_counts == 0)) {
    stop("A training fold is missing a class.")
  }
  
  X_train <- as.matrix(
    train_data[, selected_features, drop = FALSE]
  )
  
  storage.mode(X_train) <- "double"
  
  # Inverse-frequency class weights
  sample_weights <- 1 / as.numeric(
    class_counts[as.character(y_train)]
  )
  
  sample_weights <- sample_weights / mean(
    sample_weights
  )
  
  # Use the selected 2x severe weighting
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
# 9. LEAVE-ONE-BATTERY-OUT VALIDATION
# ------------------------------------------------------------

feature_sets <- list(
  Baseline = original_features,
  Engineered = engineered_features
)

all_results <- data.frame()

all_predictions <- data.frame()

all_class_metrics <- data.frame()

all_fold_results <- data.frame()

all_confusion_matrices <- list()

for (feature_set_name in names(feature_sets)) {
  
  selected_features <- feature_sets[[feature_set_name]]
  
  cat("\n========================================\n")
  
  cat(
    "FEATURE SET:",
    feature_set_name,
    "\n"
  )
  
  cat(
    "Number of features:",
    length(selected_features),
    "\n"
  )
  
  experiment_predictions <- data.frame()
  
  # ----------------------------------------------------------
  # TEST EACH UNSEEN BATTERY
  # ----------------------------------------------------------
  
  for (held_out_battery in battery_ids) {
    
    cat(
      "\nTesting unseen battery:",
      held_out_battery,
      "\n"
    )
    
    train_fold <- engineered_data[
      engineered_data$Battery_ID != held_out_battery,
      ,
      drop = FALSE
    ]
    
    test_fold <- engineered_data[
      engineered_data$Battery_ID == held_out_battery,
      ,
      drop = FALSE
    ]
    
    # Train only on the other three batteries
    model <- train_model(
      train_data = train_fold,
      selected_features = selected_features
    )
    
    X_test <- as.matrix(
      test_fold[, selected_features, drop = FALSE]
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
    
    predicted_indices <- max.col(
      probability_matrix,
      ties.method = "first"
    )
    
    predicted_classes <- class_levels[
      predicted_indices
    ]
    
    fold_predictions <- data.frame(
      Feature_Set = feature_set_name,
      Battery_ID = held_out_battery,
      Cycle_Number = test_fold$Cycle_Number,
      Actual_Stage = as.character(
        test_fold$Degradation_Stage
      ),
      Predicted_Stage = predicted_classes,
      stringsAsFactors = FALSE
    )
    
    experiment_predictions <- rbind(
      experiment_predictions,
      fold_predictions
    )
    
    fold_metrics <- calculate_metrics(
      test_fold$Degradation_Stage,
      predicted_classes
    )
    
    all_fold_results <- rbind(
      all_fold_results,
      data.frame(
        Feature_Set = feature_set_name,
        Battery_ID = held_out_battery,
        Accuracy = fold_metrics$accuracy,
        Macro_F1 = fold_metrics$macro_f1
      )
    )
  }
  
  # ----------------------------------------------------------
  # POOLED PERFORMANCE
  # ----------------------------------------------------------
  
  metrics <- calculate_metrics(
    experiment_predictions$Actual_Stage,
    experiment_predictions$Predicted_Stage
  )
  
  severe_metrics <- metrics$class_metrics[
    metrics$class_metrics$Class ==
      "Severe Degradation",
  ]
  
  result <- data.frame(
    Feature_Set = feature_set_name,
    Feature_Count = length(selected_features),
    Accuracy = metrics$accuracy,
    Macro_F1 = metrics$macro_f1,
    Severe_Precision = severe_metrics$Precision,
    Severe_Recall = severe_metrics$Recall,
    Severe_F1 = severe_metrics$F1_Score
  )
  
  all_results <- rbind(
    all_results,
    result
  )
  
  metrics$class_metrics$Feature_Set <- feature_set_name
  
  all_class_metrics <- rbind(
    all_class_metrics,
    metrics$class_metrics
  )
  
  all_confusion_matrices[[feature_set_name]] <-
    metrics$confusion
  
  all_predictions <- rbind(
    all_predictions,
    experiment_predictions
  )
  
  cat("\nFEATURE SET RESULTS:\n")
  
  print(result)
  
  cat("\nCLASS-WISE RESULTS:\n")
  
  print(metrics$class_metrics)
  
  cat("\nCONFUSION MATRIX:\n")
  
  print(metrics$confusion)
}

# ------------------------------------------------------------
# 10. COMPARE BASELINE AND ENGINEERED FEATURES
# ------------------------------------------------------------

cat("\n========================================\n")
cat("FINAL FEATURE ENGINEERING COMPARISON\n")
cat("========================================\n")

print(all_results)

# Calculate changes relative to baseline
baseline_result <- all_results[
  all_results$Feature_Set == "Baseline",
]

engineered_result <- all_results[
  all_results$Feature_Set == "Engineered",
]

comparison <- data.frame(
  Metric = c(
    "Accuracy",
    "Macro F1",
    "Severe Precision",
    "Severe Recall",
    "Severe F1"
  ),
  
  Baseline = c(
    baseline_result$Accuracy,
    baseline_result$Macro_F1,
    baseline_result$Severe_Precision,
    baseline_result$Severe_Recall,
    baseline_result$Severe_F1
  ),
  
  Engineered = c(
    engineered_result$Accuracy,
    engineered_result$Macro_F1,
    engineered_result$Severe_Precision,
    engineered_result$Severe_Recall,
    engineered_result$Severe_F1
  )
)

comparison$Change <- comparison$Engineered -
  comparison$Baseline

cat("\nMETRIC IMPROVEMENT COMPARISON:\n")

print(comparison)

# ------------------------------------------------------------
# 11. SAVE ALL RESULTS
# ------------------------------------------------------------

write.csv(
  all_results,
  file.path(
    processed_dir,
    "feature_engineering_model_comparison.csv"
  ),
  row.names = FALSE
)

write.csv(
  comparison,
  file.path(
    processed_dir,
    "feature_engineering_metric_changes.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(
    processed_dir,
    "feature_engineering_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_class_metrics,
  file.path(
    processed_dir,
    "feature_engineering_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_fold_results,
  file.path(
    processed_dir,
    "feature_engineering_fold_results.csv"
  ),
  row.names = FALSE
)

for (feature_set_name in names(all_confusion_matrices)) {
  
  write.csv(
    as.data.frame.matrix(
      all_confusion_matrices[[feature_set_name]]
    ),
    file.path(
      processed_dir,
      paste0(
        "feature_engineering_confusion_",
        feature_set_name,
        ".csv"
      )
    )
  )
}

# ------------------------------------------------------------
# 12. SAVE FEATURE DEFINITIONS
# ------------------------------------------------------------

saveRDS(
  list(
    baseline_features = original_features,
    engineered_features = engineered_features,
    engineered_dataset = engineered_data
  ),
  file.path(
    models_dir,
    "engineered_feature_definitions.rds"
  )
)

cat("\n========================================\n")
cat("FEATURE ENGINEERING EXPERIMENT COMPLETED\n")
cat("========================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)
