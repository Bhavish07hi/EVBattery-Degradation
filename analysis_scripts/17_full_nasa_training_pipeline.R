
# ============================================================
# EV BATTERY DEGRADATION - EXPANDED NASA MASTER PIPELINE
# File: R/17_full_nasa_training_pipeline.R
# ==============================================================

project_dir <- "C:/Users/bhavi/OneDrive/Documents/EV_Battery_Degradation"
setwd(project_dir)

raw_dir <- "data/raw"
processed_dir <- "data/processed"
models_dir <- "models"
plots_dir <- "plots"

for (p in c(processed_dir, models_dir, plots_dir)) {
  dir.create(p, recursive = TRUE, showWarnings = FALSE)
}

packages <- c("R.matlab", "dplyr", "ggplot2", "xgboost")

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

library(R.matlab)
library(dplyr)
library(ggplot2)
library(xgboost)

set.seed(42)

cat("\nNASA EXPANDED BATTERY PIPELINE STARTED\n")

# 1. DISCOVER FILES
files <- list.files(
  raw_dir,
  pattern = "\\.mat$",
  full.names = TRUE,
  ignore.case = TRUE
)

if (length(files) == 0) stop("No MAT files found.")

cat("MAT files found:", length(files), "\n")

# 2. EXTRACTION FUNCTIONS
get_field <- function(data, field_name) {
  field_names <- attr(data, "dimnames")
  
  if (is.null(field_names) || length(field_names) == 0) {
    return(NA_real_)
  }
  
  field_index <- match(field_name, field_names[[1]])
  
  if (is.na(field_index)) return(NA_real_)
  
  values <- tryCatch(
    unlist(data[[field_index]], use.names = FALSE),
    error = function(e) numeric(0)
  )
  
  suppressWarnings(as.numeric(values))
}

safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_min <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  min(x, na.rm = TRUE)
}

safe_max <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  max(x, na.rm = TRUE)
}

# 3. EXTRACT ALL BATTERIES
all_batteries <- list()
extraction_report <- list()

for (file in files) {
  
  id <- tools::file_path_sans_ext(basename(file))
  cat("\nProcessing:", id, "\n")
  
  result <- tryCatch({
    
    mat <- readMat(file)
    records <- mat[[1]][[1]]
    
    if (length(records) %% 4 != 0) {
      stop("Unexpected cycle structure")
    }
    
    total_cycles <- length(records) / 4
    battery_records <- list()
    
    for (i in seq_len(total_cycles)) {
      
      base <- (i - 1) * 4
      
      cycle_type <- tryCatch(
        as.character(unlist(records[[base + 1]])),
        error = function(e) ""
      )
      
      if (length(cycle_type) == 0 ||
          is.na(cycle_type[1]) ||
          cycle_type[1] != "discharge") next
      
      cycle <- records[[base + 4]]
      
      voltage <- get_field(cycle, "Voltage.measured")
      current <- get_field(cycle, "Current.measured")
      temperature <- get_field(cycle, "Temperature.measured")
      capacity <- get_field(cycle, "Capacity")
      
      battery_records[[length(battery_records) + 1]] <-
        data.frame(
          Battery_ID = id,
          Cycle_Number = i,
          Mean_Voltage = safe_mean(voltage),
          Min_Voltage = safe_min(voltage),
          Max_Voltage = safe_max(voltage),
          Mean_Current = safe_mean(current),
          Mean_Temperature = safe_mean(temperature),
          Max_Temperature = safe_max(temperature),
          Discharge_Capacity = if (length(capacity) > 0) {
            capacity[1]
          } else {
            NA_real_
          }
        )
    }
    
    dplyr::bind_rows(battery_records)
    
  }, error = function(e) {
    cat("Extraction failed:", conditionMessage(e), "\n")
    NULL
  })
  
  if (!is.null(result) && nrow(result) > 0) {
    all_batteries[[id]] <- result
    extraction_report[[id]] <- data.frame(
      Battery_ID = id,
      Status = "Successful",
      Cycles = nrow(result)
    )
    cat("Extracted cycles:", nrow(result), "\n")
  } else {
    extraction_report[[id]] <- data.frame(
      Battery_ID = id,
      Status = "Failed or no discharge cycles",
      Cycles = 0
    )
  }
  
  rm(result)
  invisible(gc())
}

if (length(all_batteries) == 0) {
  stop("No batteries were successfully extracted.")
}

# 4. COMBINE DATA
cycles <- bind_rows(all_batteries) %>%
  arrange(Battery_ID, Cycle_Number)

write.csv(
  cycles,
  file.path(processed_dir, "all_battery_cycles.csv"),
  row.names = FALSE
)

write.csv(
  bind_rows(extraction_report),
  file.path(processed_dir, "all_battery_extraction_report.csv"),
  row.names = FALSE
)

cat("\nBatteries extracted:", length(all_batteries), "\n")
cat("Total discharge cycles:", nrow(cycles), "\n")

# 5. CALCULATE SOH AND LABELS
stage_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

classified <- cycles %>%
  group_by(Battery_ID) %>%
  arrange(Cycle_Number, .by_group = TRUE) %>%
  mutate(
    Initial_Capacity = first(Discharge_Capacity),
    SOH = ifelse(
      is.finite(Initial_Capacity) & Initial_Capacity > 0,
      Discharge_Capacity / Initial_Capacity * 100,
      NA_real_
    ),
    Degradation_Stage = case_when(
      SOH >= 90 ~ "Healthy",
      SOH >= 80 ~ "Early Degradation",
      SOH >= 70 ~ "Moderate Degradation",
      SOH < 70 ~ "Severe Degradation",
      TRUE ~ NA_character_
    )
  ) %>%
  ungroup()

classified$Degradation_Stage <- factor(
  classified$Degradation_Stage,
  levels = stage_levels
)

write.csv(
  classified,
  file.path(processed_dir, "all_battery_classification.csv"),
  row.names = FALSE
)

battery_summary <- classified %>%
  group_by(Battery_ID) %>%
  summarise(
    Cycles = n(),
    Initial_Capacity = first(Discharge_Capacity),
    Final_Capacity = last(Discharge_Capacity),
    Initial_SOH = first(SOH),
    Final_SOH = last(SOH),
    Severe_Cycles = sum(
      Degradation_Stage == "Severe Degradation",
      na.rm = TRUE
    ),
    .groups = "drop"
  )

write.csv(
  battery_summary,
  file.path(processed_dir, "all_battery_summary.csv"),
  row.names = FALSE
)

cat("\nStage distribution:\n")
print(table(classified$Degradation_Stage, useNA = "ifany"))

# 6. MODEL FEATURES
features <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

model_data <- classified %>%
  filter(!is.na(Degradation_Stage))

for (f in features) {
  model_data[[f]] <- suppressWarnings(
    as.numeric(model_data[[f]])
  )
}

valid <- complete.cases(model_data[, features])

valid <- valid & apply(
  as.matrix(model_data[, features]),
  1,
  function(x) all(is.finite(x))
)

model_data <- model_data[valid, ]

if (nrow(model_data) == 0) {
  stop("No valid model rows.")
}

# 7. HOLD OUT EXTERNAL BATTERIES
external_ids <- c("B0045", "B0046", "B0047")

external_test <- model_data %>%
  filter(Battery_ID %in% external_ids)

training_pool <- model_data %>%
  filter(!Battery_ID %in% external_ids)

if (n_distinct(training_pool$Battery_ID) < 5) {
  stop("Not enough training batteries for grouped validation.")
}

cat("\nTraining batteries:",
    n_distinct(training_pool$Battery_ID), "\n")

cat("External test batteries:",
    paste(intersect(external_ids, unique(external_test$Battery_ID)),
          collapse = ", "), "\n")

if (nrow(external_test) == 0) {
  warning("No external test rows found for B0045-B0047.")
}

# 8. BATTERY-WISE FOLDS
battery_ids <- unique(training_pool$Battery_ID)

n_folds <- min(5, length(battery_ids))

fold_map <- data.frame(
  Battery_ID = battery_ids,
  Fold = sample(
    rep(seq_len(n_folds), length.out = length(battery_ids))
  )
)

write.csv(
  fold_map,
  file.path(processed_dir, "expanded_battery_folds.csv"),
  row.names = FALSE
)

# 9. METRIC FUNCTION
calculate_metrics <- function(actual, predicted) {
  
  cm <- table(
    Actual = factor(actual, levels = stage_levels),
    Predicted = factor(predicted, levels = stage_levels)
  )
  
  precision <- recall <- f1 <- numeric(4)
  
  for (i in 1:4) {
    
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    
    precision[i] <- if (tp + fp == 0) 0 else tp / (tp + fp)
    recall[i] <- if (tp + fn == 0) 0 else tp / (tp + fn)
    
    f1[i] <- if (precision[i] + recall[i] == 0) {
      0
    } else {
      2 * precision[i] * recall[i] /
        (precision[i] + recall[i])
    }
  }
  
  list(
    Accuracy = sum(diag(cm)) / sum(cm),
    Macro_F1 = mean(f1),
    Confusion = cm,
    Class_Metrics = data.frame(
      Stage = stage_levels,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  )
}

# 10. CLASS WEIGHTS
make_weights <- function(y) {
  
  counts <- table(factor(y, levels = 0:3))
  class_weights <- length(y) / (4 * pmax(as.numeric(counts), 1))
  
  # Mild additional emphasis for severe degradation
  class_weights[4] <- class_weights[4] * 1.5
  
  row_weights <- class_weights[y + 1]
  row_weights / mean(row_weights)
}

# 11. MODEL TRAINING FUNCTION
train_model <- function(df, config) {
  
  x <- as.matrix(df[, features])
  
  y <- as.integer(
    factor(df$Degradation_Stage, levels = stage_levels)
  ) - 1
  
  dtrain <- xgb.DMatrix(
    data = x,
    label = y,
    weight = make_weights(y)
  )
  
  params <- list(
    objective = "multi:softprob",
    num_class = 4,
    eval_metric = "mlogloss",
    max_depth = config$max_depth,
    eta = config$eta,
    min_child_weight = config$min_child_weight,
    subsample = 0.9,
    colsample_bytree = 0.9,
    nthread = max(1, parallel::detectCores() - 1)
  )
  
  xgb.train(
    params = params,
    data = dtrain,
    nrounds = config$nrounds,
    verbose = 0
  )
}

# 12. PROBABILITY PREDICTION
predict_probabilities <- function(model, df) {
  
  dtest <- xgb.DMatrix(
    data = as.matrix(df[, features])
  )
  
  raw <- predict(model, dtest)
  
  if (is.matrix(raw)) {
    probs <- raw
  } else {
    probs <- matrix(
      raw,
      ncol = 4,
      byrow = TRUE
    )
  }
  
  if (nrow(probs) != nrow(df) || ncol(probs) != 4) {
    stop("Unexpected XGBoost probability dimensions.")
  }
  
  colnames(probs) <- stage_levels
  probs
}

# 13. CONFIGURATIONS
configs <- list(
  Config_1 = list(
    max_depth = 3,
    eta = 0.05,
    min_child_weight = 1,
    nrounds = 100
  ),
  Config_2 = list(
    max_depth = 4,
    eta = 0.03,
    min_child_weight = 2,
    nrounds = 150
  ),
  Config_3 = list(
    max_depth = 5,
    eta = 0.03,
    min_child_weight = 3,
    nrounds = 150
  )
)

# 14. BATTERY-WISE CROSS VALIDATION
cv_results <- list()
cv_predictions <- list()

for (config_name in names(configs)) {
  
  cat("\nTesting", config_name, "\n")
  
  predictions_by_fold <- list()
  
  for (fold in seq_len(n_folds)) {
    
    validation_ids <- fold_map$Battery_ID[
      fold_map$Fold == fold
    ]
    
    train_df <- training_pool %>%
      filter(!Battery_ID %in% validation_ids)
    
    valid_df <- training_pool %>%
      filter(Battery_ID %in% validation_ids)
    
    if (nrow(train_df) == 0 || nrow(valid_df) == 0) next
    
    model <- train_model(
      train_df,
      configs[[config_name]]
    )
    
    probs <- predict_probabilities(model, valid_df)
    
    predicted <- stage_levels[
      max.col(probs, ties.method = "first")
    ]
    
    predictions_by_fold[[length(predictions_by_fold) + 1]] <-
      data.frame(
        Battery_ID = valid_df$Battery_ID,
        Actual = as.character(valid_df$Degradation_Stage),
        Predicted = predicted
      )
    
    rm(model)
    invisible(gc())
  }
  
  if (length(predictions_by_fold) == 0) next
  
  combined <- bind_rows(predictions_by_fold)
  
  metrics <- calculate_metrics(
    combined$Actual,
    combined$Predicted
  )
  
  cv_results[[config_name]] <- data.frame(
    Configuration = config_name,
    Accuracy = metrics$Accuracy,
    Macro_F1 = metrics$Macro_F1
  )
  
  cv_predictions[[config_name]] <- combined
  
  cat(
    "Accuracy:", round(metrics$Accuracy * 100, 2), "%",
    "| Macro F1:", round(metrics$Macro_F1, 4), "\n"
  )
}

if (length(cv_results) == 0) {
  stop("Cross-validation did not complete.")
}

# 15. SELECT BEST CONFIGURATION
comparison <- bind_rows(cv_results) %>%
  arrange(desc(Macro_F1), desc(Accuracy))

best_name <- comparison$Configuration[1]

# Correct double-bracket indexing
best_configuration <- configs[[best_name]]
best_cv_predictions <- cv_predictions[[best_name]]

write.csv(
  comparison,
  file.path(processed_dir, "expanded_xgboost_cv_comparison.csv"),
  row.names = FALSE
)

write.csv(
  best_cv_predictions,
  file.path(processed_dir, "expanded_xgboost_cv_predictions.csv"),
  row.names = FALSE
)

cat("\nSelected configuration:", best_name, "\n")

# 16. TRAIN FINAL MODEL
final_model <- train_model(
  training_pool,
  best_configuration
)

saveRDS(
  final_model,
  file.path(models_dir, "expanded_xgboost_model.rds")
)

saveRDS(
  list(
    model = final_model,
    features = features,
    stages = stage_levels,
    configuration = best_configuration,
    training_batteries = unique(training_pool$Battery_ID),
    external_batteries = external_ids
  ),
  file.path(models_dir, "expanded_xgboost_complete.rds")
)

# 17. EXTERNAL EVALUATION
if (nrow(external_test) > 0) {
  
  external_probs <- predict_probabilities(
    final_model,
    external_test
  )
  
  external_predictions <- stage_levels[
    max.col(external_probs, ties.method = "first")
  ]
  
  external_results <- external_test %>%
    mutate(Predicted_Stage = external_predictions)
  
  for (i in seq_along(stage_levels)) {
    external_results[[
      paste0("Probability_", make.names(stage_levels[i]))
    ]] <- external_probs[, i]
  }
  
  write.csv(
    external_results,
    file.path(processed_dir, "expanded_external_predictions.csv"),
    row.names = FALSE
  )
  
  external_metrics <- calculate_metrics(
    external_results$Degradation_Stage,
    external_results$Predicted_Stage
  )
  
  write.csv(
    data.frame(
      Model = "Expanded XGBoost",
      Accuracy = external_metrics$Accuracy,
      Macro_F1 = external_metrics$Macro_F1
    ),
    file.path(processed_dir, "expanded_external_metrics.csv"),
    row.names = FALSE
  )
  
  write.csv(
    as.data.frame.matrix(external_metrics$Confusion),
    file.path(processed_dir, "expanded_external_confusion_matrix.csv")
  )
  
  write.csv(
    external_metrics$Class_Metrics,
    file.path(processed_dir, "expanded_external_class_metrics.csv"),
    row.names = FALSE
  )
  
  external_by_battery <- lapply(
    unique(external_results$Battery_ID),
    function(id) {
      
      z <- external_results %>%
        filter(Battery_ID == id)
      
      m <- calculate_metrics(
        z$Degradation_Stage,
        z$Predicted_Stage
      )
      
      data.frame(
        Battery_ID = id,
        Cycles = nrow(z),
        Accuracy = m$Accuracy,
        Macro_F1 = m$Macro_F1
      )
    }
  )
  
  write.csv(
    bind_rows(external_by_battery),
    file.path(processed_dir, "expanded_external_battery_metrics.csv"),
    row.names = FALSE
  )
  
  cat("\nEXTERNAL TEST RESULTS\n")
  print(external_metrics$Accuracy)
  print(external_metrics$Macro_F1)
  print(external_metrics$Class_Metrics)
}

# 18. FEATURE IMPORTANCE
importance <- xgb.importance(
  feature_names = features,
  model = final_model
)

write.csv(
  importance,
  file.path(processed_dir, "expanded_xgboost_feature_importance.csv"),
  row.names = FALSE
)

png(
  file.path(plots_dir, "expanded_xgboost_feature_importance.png"),
  width = 1200,
  height = 800,
  res = 120
)

xgb.plot.importance(
  importance_matrix = importance,
  top_n = length(features),
  measure = "Gain"
)

dev.off()

# 19. SOH PLOT
soh_plot <- ggplot(
  classified,
  aes(x = Cycle_Number, y = SOH, color = Battery_ID)
) +
  geom_line() +
  geom_hline(
    yintercept = c(90, 80, 70),
    linetype = "dashed",
    alpha = 0.5
  ) +
  labs(
    title = "NASA Battery SOH Degradation",
    x = "Cycle Number",
    y = "SOH (%)"
  ) +
  theme_minimal()

ggsave(
  file.path(plots_dir, "all_nasa_battery_soh.png"),
  soh_plot,
  width = 12,
  height = 7
)

# 20. FINAL SUMMARY
summary_result <- data.frame(
  MAT_Files = length(files),
  Extracted_Batteries = length(all_batteries),
  Total_Discharge_Cycles = nrow(cycles),
  Valid_Model_Rows = nrow(model_data),
  Training_Batteries = n_distinct(training_pool$Battery_ID),
  External_Test_Batteries = n_distinct(external_test$Battery_ID),
  Best_Configuration = best_name
)

write.csv(
  summary_result,
  file.path(processed_dir, "expanded_pipeline_summary.csv"),
  row.names = FALSE
)

cat("\n============================================\n")
cat("EXPANDED NASA PIPELINE COMPLETED\n")
cat("============================================\n")

print(summary_result)
print(comparison)

cat("\nModel saved in models/expanded_xgboost_complete.rds\n")
cat("Results saved in data/processed/\n")
cat("Plots saved in plots/\n")
