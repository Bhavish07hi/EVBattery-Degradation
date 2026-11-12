
# ==========================================
# PHASE 6: TRAIN TEST SPLIT
# ==========================================

library(tidyverse)

# Load classification dataset
battery_data <- read.csv(
  "data/processed/battery_classification.csv"
)

# Arrange battery records chronologically
battery_data <- battery_data %>%
  arrange(Battery_ID, Cycle_Number) %>%
  group_by(Battery_ID) %>%
  mutate(
    Observation_Number = row_number(),
    Training_Limit = floor(n() * 0.70),
    
    Dataset_Type = if_else(
      Observation_Number <= Training_Limit,
      "Train",
      "Test"
    )
  ) %>%
  ungroup()

# Separate datasets
train_data <- battery_data %>%
  filter(Dataset_Type == "Train")

test_data <- battery_data %>%
  filter(Dataset_Type == "Test")

# Remove helper columns
train_data <- train_data %>%
  select(-Observation_Number, -Training_Limit, -Dataset_Type)

test_data <- test_data %>%
  select(-Observation_Number, -Training_Limit, -Dataset_Type)

# Display dimensions
cat("\nTraining dimensions:\n")
print(dim(train_data))

cat("\nTesting dimensions:\n")
print(dim(test_data))

# Class distribution
cat("\nTraining class distribution:\n")
print(table(train_data$Degradation_Stage))

cat("\nTesting class distribution:\n")
print(table(test_data$Degradation_Stage))

# Battery-wise split verification
cat("\nTraining records per battery:\n")
print(table(train_data$Battery_ID))

cat("\nTesting records per battery:\n")
print(table(test_data$Battery_ID))

# Save datasets
write.csv(
  train_data,
  "data/processed/train_data.csv",
  row.names = FALSE
)

write.csv(
  test_data,
  "data/processed/test_data.csv",
  row.names = FALSE
)

cat("\nTrain-test split completed successfully!\n")
