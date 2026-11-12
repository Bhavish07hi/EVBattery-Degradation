
# ============================================================
# 11 - XGBOOST HYPERPARAMETER OPTIMIZATION
# Leave-One-Battery-Out Validation
# Severe Degradation Weight = 2x
# ============================================================

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
  stop("Missing or invalid values found.")
}

battery_ids <- sort(unique(data$Battery_ID))

# ------------------------------------------------------------
# 3. DEFINE HYPERPARAMETER CONFIGURATIONS
# ------------------------------------------------------------

# Configuration 1 reproduces the previous 2x-weight
# model with 100 boosting rounds.

configurations <- data.frame(
  Config_ID = paste0("Config_", 1:9),
  
  Max_Depth = c(
    3, 2, 2, 3, 3, 4, 4, 2, 3
  ),
  
  Eta = c(
    0.05, 0.03, 0.05, 0.03, 0.10,
    0.03, 0.05, 0.10, 0.05
  ),
  
  Min_Child_Weight = c(
    1, 1, 3, 3, 1, 1, 3, 3, 1
  ),
  
  N_Rounds = c(
    100, 150, 150, 150, 100,
    150, 150, 100, 200
  ),
  
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# 4. METRIC FUNCTION
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
  
  list(
    confusion = confusion,
    class_metrics = results,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(results$F1_Score, na.rm = TRUE)
  )
}

# ------------------------------------------------------------
# 5. TRAINING FUNCTION
# ------------------------------------------------------------

train_xgboost_model <- function(
    train_data,
    config
) {
  
  y_train <- factor(
    train_data$Degradation_Stage,
    levels = class_levels
  )
  
  class_counts <- table(y_train)
  
  if (any(class_counts == 0)) {
    stop("Training data is missing a degradation class.")
  }
  
  X_train <- as.matrix(
    train_data[, feature_names, drop = FALSE]
  )
  
  storage.mode(X_train) <- "double"
  
  # Inverse-frequency class weighting
  sample_weights <- 1 / as.numeric(
    class_counts[as.character(y_train)]
  )
  
  sample_weights <- sample_weights / mean(sample_weights)
  
  # Severe Degradation receives 2x additional weight
  severe_rows <- y_train == "Severe Degradation"
  
  sample_weights[severe_rows] <-
    sample_weights[severe_rows] * 2
  
  sample_weights <- sample_weights / mean(sample_weights)
  
  dtrain <- xgb.DMatrix(
    data = X_train,
    label = as.integer(y_train) - 1,
    weight = sample_weights
  )
  
  parameters <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    eta = config$Eta,
    max_depth = config$Max_Depth,
    min_child_weight = config$Min_Child_Weight,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 2,
    seed = 123
  )
  
  set.seed(123)
  
  xgb.train(
    params = parameters,
    data = dtrain,
    nrounds = config$N_Rounds,
    verbose = 0
  )
}

# ------------------------------------------------------------
# 6. RUN FOUR-BATTERY VALIDATION
# ------------------------------------------------------------

all_results <- data.frame()
all_predictions <- data.frame()
all_class_results <- data.frame()
all_fold_results <- data.frame()

for (config_index in seq_len(nrow(configurations))) {
  
  config <- configurations[config_index, ]
  
  cat("\n====================================\n")
  
  cat(
    "RUNNING:",
    config$Config_ID,
    "\n"
  )
  
  cat(
    "Depth:", config$Max_Depth,
    "| Eta:", config$Eta,
    "| Child Weight:", config$Min_Child_Weight,
    "| Rounds:", config$N_Rounds,
    "\n"
  )
  
  experiment_predictions <- data.frame()
  
  # ----------------------------------------------------------
  # LEAVE-ONE-BATTERY-OUT
  # ----------------------------------------------------------
  
  for (held_out_battery in battery_ids) {
    
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
    
    cat(
      "Testing unseen battery:",
      held_out_battery,
      "\n"
    )
    
    model <- train_xgboost_model(
      train_data = train_fold,
      config = config
    )
    
    X_test <- as.matrix(
      test_fold[, feature_names, drop = FALSE]
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
      stop("Unexpected prediction dimensions.")
    }
    
    predicted_indices <- max.col(
      probability_matrix,
      ties.method = "first"
    )
    
    predicted_classes <- class_levels[
      predicted_indices
    ]
    
    fold_predictions <- data.frame(
      Config_ID = config$Config_ID,
      Battery_ID = held_out_battery,
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
        Config_ID = config$Config_ID,
        Battery_ID = held_out_battery,
        Accuracy = fold_metrics$accuracy,
        Macro_F1 = fold_metrics$macro_f1
      )
    )
  }
  
  # ----------------------------------------------------------
  # POOLED EXPERIMENT METRICS
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
    Config_ID = config$Config_ID,
    Max_Depth = config$Max_Depth,
    Eta = config$Eta,
    Min_Child_Weight = config$Min_Child_Weight,
    N_Rounds = config$N_Rounds,
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
  
  metrics$class_metrics$Config_ID <- config$Config_ID
  
  all_class_results <- rbind(
    all_class_results,
    metrics$class_metrics
  )
  
  experiment_predictions$Max_Depth <- config$Max_Depth
  experiment_predictions$Eta <- config$Eta
  experiment_predictions$Min_Child_Weight <-
    config$Min_Child_Weight
  experiment_predictions$N_Rounds <- config$N_Rounds
  
  all_predictions <- rbind(
    all_predictions,
    experiment_predictions
  )
  
  cat("\nExperiment performance:\n")
  print(result)
}

# ------------------------------------------------------------
# 7. RANK CONFIGURATIONS
# ------------------------------------------------------------

ranked_results <- all_results[
  order(
    -all_results$Severe_Recall,
    -all_results$Macro_F1,
    -all_results$Accuracy
  ),
]

cat("\n====================================\n")
cat("FINAL HYPERPARAMETER COMPARISON\n")
cat("====================================\n")

print(ranked_results)

# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  all_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  ranked_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_ranked.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_class_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_fold_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_fold_results.csv"
  ),
  row.names = FALSE
)

cat("\n====================================\n")
cat("OPTIMIZATION COMPLETED\n")
cat("====================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)

# ============================================================
# 11 - XGBOOST HYPERPARAMETER OPTIMIZATION
# Leave-One-Battery-Out Validation
# Severe Degradation Weight = 2x
# ============================================================

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
  stop("Missing or invalid values found.")
}

battery_ids <- sort(unique(data$Battery_ID))

# ------------------------------------------------------------
# 3. DEFINE HYPERPARAMETER CONFIGURATIONS
# ------------------------------------------------------------

# Configuration 1 reproduces the previous 2x-weight
# model with 100 boosting rounds.

configurations <- data.frame(
  Config_ID = paste0("Config_", 1:9),
  
  Max_Depth = c(
    3, 2, 2, 3, 3, 4, 4, 2, 3
  ),
  
  Eta = c(
    0.05, 0.03, 0.05, 0.03, 0.10,
    0.03, 0.05, 0.10, 0.05
  ),
  
  Min_Child_Weight = c(
    1, 1, 3, 3, 1, 1, 3, 3, 1
  ),
  
  N_Rounds = c(
    100, 150, 150, 150, 100,
    150, 150, 100, 200
  ),
  
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# 4. METRIC FUNCTION
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
  
  list(
    confusion = confusion,
    class_metrics = results,
    accuracy = sum(diag(confusion)) / sum(confusion),
    macro_f1 = mean(results$F1_Score, na.rm = TRUE)
  )
}

# ------------------------------------------------------------
# 5. TRAINING FUNCTION
# ------------------------------------------------------------

train_xgboost_model <- function(
    train_data,
    config
) {
  
  y_train <- factor(
    train_data$Degradation_Stage,
    levels = class_levels
  )
  
  class_counts <- table(y_train)
  
  if (any(class_counts == 0)) {
    stop("Training data is missing a degradation class.")
  }
  
  X_train <- as.matrix(
    train_data[, feature_names, drop = FALSE]
  )
  
  storage.mode(X_train) <- "double"
  
  # Inverse-frequency class weighting
  sample_weights <- 1 / as.numeric(
    class_counts[as.character(y_train)]
  )
  
  sample_weights <- sample_weights / mean(sample_weights)
  
  # Severe Degradation receives 2x additional weight
  severe_rows <- y_train == "Severe Degradation"
  
  sample_weights[severe_rows] <-
    sample_weights[severe_rows] * 2
  
  sample_weights <- sample_weights / mean(sample_weights)
  
  dtrain <- xgb.DMatrix(
    data = X_train,
    label = as.integer(y_train) - 1,
    weight = sample_weights
  )
  
  parameters <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    eta = config$Eta,
    max_depth = config$Max_Depth,
    min_child_weight = config$Min_Child_Weight,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 2,
    seed = 123
  )
  
  set.seed(123)
  
  xgb.train(
    params = parameters,
    data = dtrain,
    nrounds = config$N_Rounds,
    verbose = 0
  )
}

# ------------------------------------------------------------
# 6. RUN FOUR-BATTERY VALIDATION
# ------------------------------------------------------------

all_results <- data.frame()
all_predictions <- data.frame()
all_class_results <- data.frame()
all_fold_results <- data.frame()

for (config_index in seq_len(nrow(configurations))) {
  
  config <- configurations[config_index, ]
  
  cat("\n====================================\n")
  
  cat(
    "RUNNING:",
    config$Config_ID,
    "\n"
  )
  
  cat(
    "Depth:", config$Max_Depth,
    "| Eta:", config$Eta,
    "| Child Weight:", config$Min_Child_Weight,
    "| Rounds:", config$N_Rounds,
    "\n"
  )
  
  experiment_predictions <- data.frame()
  
  # ----------------------------------------------------------
  # LEAVE-ONE-BATTERY-OUT
  # ----------------------------------------------------------
  
  for (held_out_battery in battery_ids) {
    
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
    
    cat(
      "Testing unseen battery:",
      held_out_battery,
      "\n"
    )
    
    model <- train_xgboost_model(
      train_data = train_fold,
      config = config
    )
    
    X_test <- as.matrix(
      test_fold[, feature_names, drop = FALSE]
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
      stop("Unexpected prediction dimensions.")
    }
    
    predicted_indices <- max.col(
      probability_matrix,
      ties.method = "first"
    )
    
    predicted_classes <- class_levels[
      predicted_indices
    ]
    
    fold_predictions <- data.frame(
      Config_ID = config$Config_ID,
      Battery_ID = held_out_battery,
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
        Config_ID = config$Config_ID,
        Battery_ID = held_out_battery,
        Accuracy = fold_metrics$accuracy,
        Macro_F1 = fold_metrics$macro_f1
      )
    )
  }
  
  # ----------------------------------------------------------
  # POOLED EXPERIMENT METRICS
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
    Config_ID = config$Config_ID,
    Max_Depth = config$Max_Depth,
    Eta = config$Eta,
    Min_Child_Weight = config$Min_Child_Weight,
    N_Rounds = config$N_Rounds,
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
  
  metrics$class_metrics$Config_ID <- config$Config_ID
  
  all_class_results <- rbind(
    all_class_results,
    metrics$class_metrics
  )
  
  experiment_predictions$Max_Depth <- config$Max_Depth
  experiment_predictions$Eta <- config$Eta
  experiment_predictions$Min_Child_Weight <-
    config$Min_Child_Weight
  experiment_predictions$N_Rounds <- config$N_Rounds
  
  all_predictions <- rbind(
    all_predictions,
    experiment_predictions
  )
  
  cat("\nExperiment performance:\n")
  print(result)
}

# ------------------------------------------------------------
# 7. RANK CONFIGURATIONS
# ------------------------------------------------------------

ranked_results <- all_results[
  order(
    -all_results$Severe_Recall,
    -all_results$Macro_F1,
    -all_results$Accuracy
  ),
]

cat("\n====================================\n")
cat("FINAL HYPERPARAMETER COMPARISON\n")
cat("====================================\n")

print(ranked_results)

# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  all_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_results.csv"
  ),
  row.names = FALSE
)

write.csv(
  ranked_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_ranked.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_predictions,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_class_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  all_fold_results,
  file.path(
    processed_dir,
    "xgboost_hyperparameter_fold_results.csv"
  ),
  row.names = FALSE
)

cat("\n====================================\n")
cat("OPTIMIZATION COMPLETED\n")
cat("====================================\n")

cat(
  "\nResults saved in:",
  processed_dir,
  "\n"
)
