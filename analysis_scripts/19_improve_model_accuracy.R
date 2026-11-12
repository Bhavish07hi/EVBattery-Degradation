
# ============================================================
# FILE: R/19_improve_model_accuracy.R
# Purpose:
# 1. Compare multiclass XGBoost with SOH-regression XGBoost
# 2. Use battery-wise cross-validation
# 3. Keep B0045, B0046 and B0047 out of model selection
# 4. Save the best model and final evaluation
# ============================================================

cat("\nSTARTING MODEL IMPROVEMENT\n")

# -------------------- 1. PACKAGES ----------------------------

required_packages <- c("xgboost", "dplyr", "ggplot2")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(xgboost)
library(dplyr)
library(ggplot2)

# -------------------- 2. PATHS ------------------------------

project_dir <- "C:/Users/bhavi/OneDrive/Documents/EV_Battery_Degradation"

data_file <- file.path(
  project_dir,
  "data/processed/all_battery_classification_corrected.csv"
)

model_dir <- file.path(project_dir, "models")
results_dir <- file.path(project_dir, "data/processed")
plots_dir <- file.path(project_dir, "plots")

dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_file)) {
  stop("Corrected dataset not found: ", data_file)
}

data <- read.csv(data_file, stringsAsFactors = FALSE)

cat("Dataset loaded:", nrow(data), "rows\n")
cat("Columns:", paste(names(data), collapse = ", "), "\n")

# -------------------- 3. IDENTIFY COLUMNS -------------------

find_column <- function(possible_names, available_names) {
  match_index <- match(
    tolower(possible_names),
    tolower(available_names)
  )
  
  match_index <- match_index[!is.na(match_index)]
  
  if (length(match_index) == 0) {
    return(NULL)
  }
  
  available_names[match_index[1]]
}

battery_col <- find_column(
  c("Battery_ID", "Battery", "BatteryID", "battery_id"),
  names(data)
)

soh_col <- find_column(
  c("SOH", "State_of_Health", "StateOfHealth"),
  names(data)
)

stage_col <- find_column(
  c("Degradation_Stage", "Stage", "SOH_Stage", "Class"),
  names(data)
)

if (is.null(battery_col) || is.null(soh_col) || is.null(stage_col)) {
  stop(
    "Could not identify Battery_ID, SOH and Degradation_Stage columns. ",
    "Check the dataset column names."
  )
}

data$Battery_ID <- as.character(data[[battery_col]])
data$SOH_Value <- as.numeric(data[[soh_col]])

# Standardise stage names
stage_mapping <- c(
  "healthy" = "Healthy",
  "early degradation" = "Early Degradation",
  "moderate degradation" = "Moderate Degradation",
  "severe degradation" = "Severe Degradation"
)

data$Stage <- trimws(as.character(data[[stage_col]]))
data$Stage <- unname(
  stage_mapping[tolower(data$Stage)]
)

stage_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

data <- data[
  !is.na(data$Battery_ID) &
    !is.na(data$SOH_Value) &
    !is.na(data$Stage) &
    data$Stage %in% stage_levels,
]

data$Stage <- factor(data$Stage, levels = stage_levels)

# -------------------- 4. FEATURES ---------------------------

# Cycle_Number is deliberately excluded.
base_features <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

available_features <- base_features[
  base_features %in% names(data)
]

if (length(available_features) < 4) {
  stop("Not enough expected sensor features were found in the dataset.")
}

for (feature in available_features) {
  data[[feature]] <- as.numeric(data[[feature]])
}

# Additional physically meaningful features
if (all(c("Max_Voltage", "Min_Voltage") %in% names(data))) {
  data$Voltage_Range <- data$Max_Voltage - data$Min_Voltage
}

if (all(c("Max_Temperature", "Mean_Temperature") %in% names(data))) {
  data$Temperature_Rise <-
    data$Max_Temperature - data$Mean_Temperature
}

if ("Mean_Current" %in% names(data)) {
  data$Absolute_Current <- abs(data$Mean_Current)
}

if (all(c("Mean_Voltage", "Mean_Current") %in% names(data))) {
  data$Voltage_Current_Ratio <-
    data$Mean_Voltage / (abs(data$Mean_Current) + 0.001)
}

feature_names <- c(
  available_features,
  "Voltage_Range",
  "Temperature_Rise",
  "Absolute_Current",
  "Voltage_Current_Ratio"
)

feature_names <- feature_names[
  feature_names %in% names(data)
]

# Remove rows with missing or infinite model inputs
for (feature in feature_names) {
  data[[feature]][!is.finite(data[[feature]])] <- NA
}

data <- data[
  complete.cases(data[, feature_names]) &
    is.finite(data$SOH_Value),
]

cat("\nValid rows:", nrow(data), "\n")
cat("Batteries:", length(unique(data$Battery_ID)), "\n")
cat("Features:", paste(feature_names, collapse = ", "), "\n")

# -------------------- 5. TRAIN / EXTERNAL TEST SPLIT --------

external_batteries <- c("B0045", "B0046", "B0047")

external_data <- data[
  data$Battery_ID %in% external_batteries,
]

training_data <- data[
  !data$Battery_ID %in% external_batteries,
]

if (nrow(training_data) == 0 || nrow(external_data) == 0) {
  stop("Training or external test dataset is empty. Check battery IDs.")
}

cat("\nTraining rows:", nrow(training_data), "\n")
cat("External test rows:", nrow(external_data), "\n")

# -------------------- 6. METRICS ----------------------------

calculate_metrics <- function(actual, predicted) {
  
  actual <- factor(actual, levels = stage_levels)
  predicted <- factor(predicted, levels = stage_levels)
  
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )
  
  accuracy <- sum(diag(cm)) / sum(cm)
  
  precision <- numeric(length(stage_levels))
  recall <- numeric(length(stage_levels))
  f1 <- numeric(length(stage_levels))
  
  for (i in seq_along(stage_levels)) {
    
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    
    precision[i] <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall[i] <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    
    f1[i] <- ifelse(
      precision[i] + recall[i] == 0,
      0,
      2 * precision[i] * recall[i] /
        (precision[i] + recall[i])
    )
  }
  
  list(
    accuracy = accuracy,
    macro_f1 = mean(f1),
    confusion_matrix = cm,
    class_metrics = data.frame(
      Stage = stage_levels,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  )
}

# Convert predicted continuous SOH into project-defined stages
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

# -------------------- 7. BATTERY-WISE FOLDS -----------------

set.seed(42)

battery_ids <- sample(unique(training_data$Battery_ID))

number_of_folds <- min(5, length(battery_ids))

fold_assignment <- rep(
  seq_len(number_of_folds),
  length.out = length(battery_ids)
)

battery_folds <- split(
  battery_ids,
  fold_assignment
)

cat("\nBattery-wise folds:", number_of_folds, "\n")

# -------------------- 8. MODEL CONFIGURATIONS ---------------

configurations <- list(
  list(depth = 2, eta = 0.03, rounds = 150, weight = FALSE),
  list(depth = 3, eta = 0.03, rounds = 200, weight = FALSE),
  list(depth = 4, eta = 0.03, rounds = 200, weight = FALSE),
  list(depth = 2, eta = 0.05, rounds = 150, weight = TRUE),
  list(depth = 3, eta = 0.05, rounds = 200, weight = TRUE),
  list(depth = 4, eta = 0.05, rounds = 200, weight = TRUE)
)

results <- list()
result_counter <- 1

# -------------------- 9. CROSS-VALIDATION -------------------

for (config_id in seq_along(configurations)) {
  
  config <- configurations[[config_id]]
  
  for (model_type in c("classification", "regression")) {
    
    cat(
      "\nTesting configuration", config_id,
      "| Model:", model_type, "\n"
    )
    
    fold_actual <- character()
    fold_predicted <- character()
    
    for (fold_id in seq_along(battery_folds)) {
      
      validation_batteries <- battery_folds[[fold_id]]
      
      fold_train <- training_data[
        !training_data$Battery_ID %in% validation_batteries,
      ]
      
      fold_validation <- training_data[
        training_data$Battery_ID %in% validation_batteries,
      ]
      
      x_train <- as.matrix(
        fold_train[, feature_names, drop = FALSE]
      )
      
      x_validation <- as.matrix(
        fold_validation[, feature_names, drop = FALSE]
      )
      
      dtrain <- xgb.DMatrix(data = x_train)
      dvalidation <- xgb.DMatrix(data = x_validation)
      
      if (model_type == "classification") {
        
        y_train <- as.integer(fold_train$Stage) - 1
        
        if (config$weight) {
          counts <- table(
            factor(fold_train$Stage, levels = stage_levels)
          )
          
          class_weights <- length(y_train) /
            (length(stage_levels) * as.numeric(counts))
          
          row_weights <- class_weights[y_train + 1]
          setinfo(dtrain, "weight", row_weights)
        }
        
        setinfo(dtrain, "label", y_train)
        
        params <- list(
          objective = "multi:softprob",
          num_class = 4,
          eval_metric = "mlogloss",
          max_depth = config$depth,
          eta = config$eta,
          min_child_weight = 3,
          subsample = 0.85,
          colsample_bytree = 0.85,
          lambda = 2,
          alpha = 0.1,
          verbosity = 0
        )
        
        model <- xgb.train(
          params = params,
          data = dtrain,
          nrounds = config$rounds,
          verbose = 0
        )
        
        probabilities <- predict(model, dvalidation)
        
        probability_matrix <- matrix(
          probabilities,
          ncol = 4,
          byrow = TRUE
        )
        
        predictions <- stage_levels[
          max.col(probability_matrix)
        ]
        
      } else {
        
        setinfo(dtrain, "label", fold_train$SOH_Value)
        
        params <- list(
          objective = "reg:squarederror",
          eval_metric = "rmse",
          max_depth = config$depth,
          eta = config$eta,
          min_child_weight = 3,
          subsample = 0.85,
          colsample_bytree = 0.85,
          lambda = 2,
          alpha = 0.1,
          verbosity = 0
        )
        
        model <- xgb.train(
          params = params,
          data = dtrain,
          nrounds = config$rounds,
          verbose = 0
        )
        
        predicted_soh <- predict(model, dvalidation)
        predictions <- soh_to_stage(predicted_soh)
      }
      
      fold_actual <- c(
        fold_actual,
        as.character(fold_validation$Stage)
      )
      
      fold_predicted <- c(
        fold_predicted,
        as.character(predictions)
      )
    }
    
    metrics <- calculate_metrics(
      fold_actual,
      fold_predicted
    )
    
    results[[result_counter]] <- data.frame(
      Configuration = config_id,
      Model = model_type,
      Depth = config$depth,
      Eta = config$eta,
      Rounds = config$rounds,
      Class_Weights = config$weight,
      Accuracy = metrics$accuracy,
      Macro_F1 = metrics$macro_f1
    )
    
    result_counter <- result_counter + 1
    
    cat(
      "CV Accuracy:", round(metrics$accuracy * 100, 2), "%",
      "| Macro F1:", round(metrics$macro_f1, 4), "\n"
    )
  }
}

# -------------------- 10. SELECT BEST MODEL -----------------

results_table <- bind_rows(results)

results_table <- results_table %>%
  arrange(desc(Accuracy), desc(Macro_F1))

write.csv(
  results_table,
  file.path(results_dir, "model_comparison_results.csv"),
  row.names = FALSE
)

cat("\n================ MODEL COMPARISON ================\n")
print(results_table)

best_result <- results_table[1, ]
best_config <- configurations[[best_result$Configuration]]
best_type <- best_result$Model

cat("\nBEST MODEL:", best_type, "\n")
cat("CV Accuracy:", round(best_result$Accuracy * 100, 2), "%\n")
cat("CV Macro F1:", round(best_result$Macro_F1, 4), "\n")

# -------------------- 11. TRAIN FINAL MODEL -----------------

x_train <- as.matrix(
  training_data[, feature_names, drop = FALSE]
)

x_external <- as.matrix(
  external_data[, feature_names, drop = FALSE]
)

dtrain <- xgb.DMatrix(x_train)
dexternal <- xgb.DMatrix(x_external)

if (best_type == "classification") {
  
  y_train <- as.integer(training_data$Stage) - 1
  
  if (best_config$weight) {
    
    counts <- table(
      factor(training_data$Stage, levels = stage_levels)
    )
    
    class_weights <- length(y_train) /
      (length(stage_levels) * as.numeric(counts))
    
    setinfo(
      dtrain,
      "weight",
      class_weights[y_train + 1]
    )
  }
  
  setinfo(dtrain, "label", y_train)
  
  final_params <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    max_depth = best_config$depth,
    eta = best_config$eta,
    min_child_weight = 3,
    subsample = 0.85,
    colsample_bytree = 0.85,
    lambda = 2,
    alpha = 0.1,
    verbosity = 0
  )
  
  final_model <- xgb.train(
    params = final_params,
    data = dtrain,
    nrounds = best_config$rounds,
    verbose = 0
  )
  
  external_probabilities <- predict(
    final_model,
    dexternal
  )
  
  external_matrix <- matrix(
    external_probabilities,
    ncol = 4,
    byrow = TRUE
  )
  
  external_predictions <- factor(
    stage_levels[max.col(external_matrix)],
    levels = stage_levels
  )
  
} else {
  
  setinfo(dtrain, "label", training_data$SOH_Value)
  
  final_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = best_config$depth,
    eta = best_config$eta,
    min_child_weight = 3,
    subsample = 0.85,
    colsample_bytree = 0.85,
    lambda = 2,
    alpha = 0.1,
    verbosity = 0
  )
  
  final_model <- xgb.train(
    params = final_params,
    data = dtrain,
    nrounds = best_config$rounds,
    verbose = 0
  )
  
  external_predicted_soh <- predict(
    final_model,
    dexternal
  )
  
  external_predictions <- soh_to_stage(
    external_predicted_soh
  )
}

# -------------------- 12. EXTERNAL EVALUATION ---------------

external_metrics <- calculate_metrics(
  external_data$Stage,
  external_predictions
)

cat("\n================ EXTERNAL TEST RESULTS ================\n")

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
print(external_metrics$confusion_matrix)

cat("\nClass-wise Metrics:\n")
print(external_metrics$class_metrics)

write.csv(
  as.data.frame.matrix(external_metrics$confusion_matrix),
  file.path(results_dir, "improved_external_confusion_matrix.csv")
)

write.csv(
  external_metrics$class_metrics,
  file.path(results_dir, "improved_external_class_metrics.csv"),
  row.names = FALSE
)

external_output <- data.frame(
  Battery_ID = external_data$Battery_ID,
  Actual_Stage = external_data$Stage,
  Predicted_Stage = external_predictions
)

if (best_type == "regression") {
  external_output$Predicted_SOH <- external_predicted_soh
}

write.csv(
  external_output,
  file.path(results_dir, "improved_external_predictions.csv"),
  row.names = FALSE
)

# -------------------- 13. FEATURE IMPORTANCE ----------------

importance <- xgb.importance(
  feature_names = feature_names,
  model = final_model
)

write.csv(
  importance,
  file.path(results_dir, "improved_feature_importance.csv"),
  row.names = FALSE
)

png(
  file.path(plots_dir, "improved_feature_importance.png"),
  width = 1200,
  height = 800
)

xgb.plot.importance(
  importance,
  top_n = length(feature_names),
  main = "Feature Importance - Improved EV Battery Model"
)

dev.off()

# -------------------- 14. SAVE MODEL ------------------------

model_bundle <- list(
  model = final_model,
  model_type = best_type,
  features = feature_names,
  stage_levels = stage_levels,
  thresholds = c(
    healthy = 90,
    early_degradation = 80,
    moderate_degradation = 70
  ),
  configuration = best_config,
  cv_accuracy = best_result$Accuracy,
  cv_macro_f1 = best_result$Macro_F1,
  external_accuracy = external_metrics$accuracy,
  external_macro_f1 = external_metrics$macro_f1,
  training_batteries = unique(training_data$Battery_ID),
  external_batteries = external_batteries
)

saveRDS(
  model_bundle,
  file.path(model_dir, "improved_ev_battery_model.rds")
)

# -------------------- 15. FINAL SUMMARY ---------------------

summary_table <- data.frame(
  Metric = c(
    "Selected Model",
    "Cross-validation Accuracy",
    "Cross-validation Macro F1",
    "External Accuracy",
    "External Macro F1",
    "Training Batteries",
    "External Test Batteries"
  ),
  Result = c(
    best_type,
    paste0(round(best_result$Accuracy * 100, 2), "%"),
    round(best_result$Macro_F1, 4),
    paste0(round(external_metrics$accuracy * 100, 2), "%"),
    round(external_metrics$macro_f1, 4),
    length(unique(training_data$Battery_ID)),
    paste(external_batteries, collapse = ", ")
  )
)

write.csv(
  summary_table,
  file.path(results_dir, "improved_model_summary.csv"),
  row.names = FALSE
)

cat("\n================ COMPLETE ================\n")
print(summary_table)

cat(
  "\nSaved model:",
  file.path(model_dir, "improved_ev_battery_model.rds"),
  "\n"
)

cat(
  "\nSaved comparison:",
  file.path(results_dir, "model_comparison_results.csv"),
  "\n"
)
