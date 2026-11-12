
# ==========================================
# PHASE 7: CLASSIFICATION MODEL TRAINING
# CORRECTED VERSION
# ==========================================

library(caret)
library(randomForest)
library(rpart)
library(e1071)
library(nnet)
library(xgboost)
library(tidyverse)

set.seed(123)

# 1. Load datasets
train_data <- read.csv(
  "data/processed/train_data.csv"
)

test_data <- read.csv(
  "data/processed/test_data.csv"
)

# 2. Define predictors
features <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature"
)

# 3. Define class levels
class_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

# 4. Prepare training data
x_train <- as.matrix(
  train_data[, features]
)

x_test <- as.matrix(
  test_data[, features]
)

y_train <- factor(
  train_data$Degradation_Stage,
  levels = class_levels
)

y_test <- factor(
  test_data$Degradation_Stage,
  levels = class_levels
)

# ==========================================
# CROSS VALIDATION
# ==========================================

control <- trainControl(
  method = "cv",
  number = 3,
  sampling = "up"
)

# ==========================================
# MODEL 1: DECISION TREE
# ==========================================

cat("\nTraining Decision Tree...\n")

model_dt <- train(
  x = x_train,
  y = y_train,
  method = "rpart",
  trControl = control,
  tuneLength = 5
)

# ==========================================
# MODEL 2: RANDOM FOREST
# ==========================================

cat("\nTraining Random Forest...\n")

model_rf <- train(
  x = x_train,
  y = y_train,
  method = "rf",
  trControl = control,
  ntree = 500,
  tuneLength = 3
)

# ==========================================
# MODEL 3: SUPPORT VECTOR MACHINE
# ==========================================

cat("\nTraining SVM...\n")

model_svm <- train(
  x = x_train,
  y = y_train,
  method = "svmRadial",
  trControl = control,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

# ==========================================
# MODEL 4: LOGISTIC REGRESSION
# ==========================================

cat("\nTraining Logistic Regression...\n")

model_lr <- train(
  x = x_train,
  y = y_train,
  method = "multinom",
  trControl = control,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    decay = c(0.01, 0.1, 0.5)
  ),
  trace = FALSE
)

# ==========================================
# MODEL 5: XGBOOST
# NATIVE XGBOOST IMPLEMENTATION
# ==========================================

cat("\nTraining XGBoost...\n")

# Convert target labels into integers
y_xgb <- as.integer(y_train) - 1

# Calculate class weights
class_counts <- table(y_train)

sample_weights <- as.numeric(
  1 / class_counts[as.character(y_train)]
)

# Normalize weights
sample_weights <- sample_weights /
  mean(sample_weights)

# Create XGBoost training matrix
dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_xgb,
  weight = sample_weights
)

# XGBoost parameters
xgb_params <- list(
  objective = "multi:softprob",
  num_class = 4,
  eval_metric = "mlogloss",
  eta = 0.05,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  nthread = 2
)

# Train XGBoost
model_xgb <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

# ==========================================
# SAVE ALL FIVE MODELS
# ==========================================

models <- list(
  Decision_Tree = model_dt,
  Random_Forest = model_rf,
  SVM = model_svm,
  Logistic_Regression = model_lr,
  XGBoost = model_xgb
)

dir.create(
  "models",
  showWarnings = FALSE
)

saveRDS(
  models,
  "models/all_classification_models.rds"
)

# ==========================================
# DISPLAY MODEL RESULTS
# ==========================================

cat("\n================================\n")
cat("ALL FIVE MODELS TRAINED!\n")
cat("================================\n")

for (model_name in names(models)) {
  
  cat("\nModel:", model_name, "\n")
  
  if (model_name == "XGBoost") {
    print(model_xgb)
  } else {
    print(models[[model_name]])
  }
}

cat("\nModels saved successfully!\n")
