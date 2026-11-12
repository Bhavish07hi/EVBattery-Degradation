
# ==========================================
# PHASE 8: MODEL EVALUATION
# ==========================================

library(tidyverse)
library(caret)
library(ggplot2)

# 1. Load saved models
models <- readRDS(
  "models/all_classification_models.rds"
)

# 2. Load testing dataset
test_data <- read.csv(
  "data/processed/test_data.csv"
)

# 3. Define features
features <- c(
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

x_test <- as.matrix(
  test_data[, features]
)

actual <- factor(
  test_data$Degradation_Stage,
  levels = class_levels
)

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================

predictions <- list()

# Decision Tree
predictions$Decision_Tree <- predict(
  models$Decision_Tree,
  newdata = as.data.frame(x_test)
)

# Random Forest
predictions$Random_Forest <- predict(
  models$Random_Forest,
  newdata = as.data.frame(x_test)
)

# SVM
predictions$SVM <- predict(
  models$SVM,
  newdata = as.data.frame(x_test)
)

# Logistic Regression
predictions$Logistic_Regression <- predict(
  models$Logistic_Regression,
  newdata = as.data.frame(x_test)
)

# XGBoost
xgb_prob <- predict(
  models$XGBoost,
  newdata = x_test
)

# Convert XGBoost probabilities to class labels
xgb_prob <- matrix(
  xgb_prob,
  ncol = length(class_levels),
  byrow = FALSE
)

xgb_class <- max.col(xgb_prob)

predictions$XGBoost <- factor(
  class_levels[xgb_class],
  levels = class_levels
)

# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

evaluate_model <- function(actual, predicted) {
  
  predicted <- factor(
    predicted,
    levels = class_levels
  )
  
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )
  
  # Calculate metrics for each class
  tp <- diag(cm)
  
  precision <- tp / colSums(cm)
  recall <- tp / rowSums(cm)
  
  precision[!is.finite(precision)] <- NA
  recall[!is.finite(recall)] <- NA
  
  f1 <- 2 * precision * recall /
    (precision + recall)
  
  f1[!is.finite(f1)] <- NA
  
  # Only classes represented in the test data
  supported <- rowSums(cm) > 0
  
  macro_f1 <- mean(
    f1[supported],
    na.rm = TRUE
  )
  
  if (!is.finite(macro_f1)) {
    macro_f1 <- NA_real_
  }
  
  accuracy <- sum(tp) / sum(cm)
  
  list(
    accuracy = accuracy,
    macro_f1 = macro_f1,
    precision = precision,
    recall = recall,
    f1 = f1,
    confusion_matrix = cm
  )
}

# ==========================================
# 6. EVALUATE ALL MODELS
# ==========================================

results <- list()
comparison <- data.frame()

for (model_name in names(predictions)) {
  
  cat("\n============================\n")
  cat("Evaluating:", model_name, "\n")
  cat("============================\n")
  
  result <- evaluate_model(
    actual,
    predictions[[model_name]]
  )
  
  results[[model_name]] <- result
  
  comparison <- rbind(
    comparison,
    data.frame(
      Model = model_name,
      Accuracy = result$accuracy,
      Macro_F1 = result$macro_f1
    )
  )
  
  print(result$confusion_matrix)
  
  cat("\nAccuracy:",
      round(result$accuracy * 100, 2), "%\n")
  
  cat("Macro F1:",
      round(result$macro_f1, 4), "\n")
  
  cat("\nClass-wise metrics:\n")
  
  print(
    data.frame(
      Class = class_levels,
      Precision = result$precision,
      Recall = result$recall,
      F1 = result$f1,
      Support = as.numeric(
        rowSums(result$confusion_matrix)
      )
    )
  )
}

# ==========================================
# 7. MODEL COMPARISON
# ==========================================

comparison <- comparison %>%
  arrange(desc(Macro_F1))

cat("\nFINAL MODEL COMPARISON:\n")

print(comparison)

# Save comparison report
write.csv(
  comparison,
  "data/processed/model_comparison.csv",
  row.names = FALSE
)

# Save predictions
prediction_report <- data.frame(
  Actual = actual
)

for (model_name in names(predictions)) {
  prediction_report[[model_name]] <-
    predictions[[model_name]]
}

write.csv(
  prediction_report,
  "data/processed/model_predictions.csv",
  row.names = FALSE
)

# ==========================================
# 8. ACCURACY COMPARISON GRAPH
# ==========================================

p1 <- ggplot(
  comparison,
  aes(
    x = reorder(Model, Accuracy),
    y = Accuracy
  )
) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(
    labels = scales::percent_format()
  ) +
  labs(
    title = "Classification Model Accuracy",
    x = "Machine Learning Algorithm",
    y = "Accuracy"
  ) +
  theme_minimal()

print(p1)

ggsave(
  "plots/model_accuracy_comparison.png",
  p1,
  width = 9,
  height = 6
)

# ==========================================
# 9. MACRO F1 COMPARISON GRAPH
# ==========================================

p2 <- ggplot(
  comparison,
  aes(
    x = reorder(Model, Macro_F1),
    y = Macro_F1
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Macro F1 Score Comparison",
    x = "Machine Learning Algorithm",
    y = "Macro F1 Score"
  ) +
  theme_minimal()

print(p2)

ggsave(
  "plots/model_f1_comparison.png",
  p2,
  width = 9,
  height = 6
)

# ==========================================
# 10. SAVE CONFUSION MATRICES
# ==========================================

for (model_name in names(results)) {
  
  cm_data <- as.data.frame(
    results[[model_name]]$confusion_matrix
  )
  
  write.csv(
    cm_data,
    paste0(
      "data/processed/confusion_",
      model_name,
      ".csv"
    ),
    row.names = FALSE
  )
}

cat("\n================================\n")
cat("MODEL EVALUATION COMPLETED!\n")
cat("================================\n")
