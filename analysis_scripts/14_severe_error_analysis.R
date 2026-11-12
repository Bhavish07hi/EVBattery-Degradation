
# ============================================================
# 14 - SEVERE DEGRADATION ERROR ANALYSIS
# Analysis of missed and correctly detected severe cases
# Uses Leave-One-Battery-Out predictions
# ============================================================

library(dplyr)

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

project_dir <- getwd()

processed_dir <- file.path(
  project_dir,
  "data",
  "processed"
)

# ------------------------------------------------------------
# 2. LOAD SAVED PREDICTIONS AND ORIGINAL DATA
# ------------------------------------------------------------

prediction_file <- file.path(
  processed_dir,
  "severe_threshold_probabilities.csv"
)

data_file <- file.path(
  processed_dir,
  "battery_classification.csv"
)

if (!file.exists(prediction_file)) {
  stop("severe_threshold_probabilities.csv not found.")
}

if (!file.exists(data_file)) {
  stop("battery_classification.csv not found.")
}

predictions <- read.csv(
  prediction_file,
  stringsAsFactors = FALSE
)

battery_data <- read.csv(
  data_file,
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

prediction_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "Actual_Stage",
  "Probability_Healthy",
  "Probability_Early",
  "Probability_Moderate",
  "Probability_Severe"
)

feature_columns <- c(
  "Battery_ID",
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity",
  "SOH",
  "Degradation_Stage"
)

missing_prediction_columns <- setdiff(
  prediction_columns,
  names(predictions)
)

missing_feature_columns <- setdiff(
  feature_columns,
  names(battery_data)
)

if (length(missing_prediction_columns) > 0) {
  stop(
    "Missing prediction columns: ",
    paste(missing_prediction_columns, collapse = ", ")
  )
}

if (length(missing_feature_columns) > 0) {
  stop(
    "Missing dataset columns: ",
    paste(missing_feature_columns, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 4. PREPARE DATA TYPES
# ------------------------------------------------------------

predictions$Battery_ID <- as.character(
  predictions$Battery_ID
)

battery_data$Battery_ID <- as.character(
  battery_data$Battery_ID
)

predictions$Cycle_Number <- as.numeric(
  predictions$Cycle_Number
)

battery_data$Cycle_Number <- as.numeric(
  battery_data$Cycle_Number
)

probability_columns <- c(
  "Probability_Healthy",
  "Probability_Early",
  "Probability_Moderate",
  "Probability_Severe"
)

predictions[probability_columns] <- lapply(
  predictions[probability_columns],
  as.numeric
)

numeric_features <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity",
  "SOH"
)

battery_data[numeric_features] <- lapply(
  battery_data[numeric_features],
  as.numeric
)

# ------------------------------------------------------------
# 5. CHECK FOR DUPLICATE BATTERY CYCLE RECORDS
# ------------------------------------------------------------

if (
  anyDuplicated(
    battery_data[c("Battery_ID", "Cycle_Number")]
  ) > 0
) {
  stop("Duplicate battery-cycle records found in original data.")
}

if (
  anyDuplicated(
    predictions[c("Battery_ID", "Cycle_Number")]
  ) > 0
) {
  stop("Duplicate battery-cycle records found in predictions.")
}

# ------------------------------------------------------------
# 6. RECREATE STANDARD ARGMAX PREDICTIONS
# ------------------------------------------------------------

probability_matrix <- as.matrix(
  predictions[, probability_columns]
)

if (any(!is.finite(probability_matrix))) {
  stop("Invalid prediction probabilities found.")
}

if (
  any(probability_matrix < 0) ||
  any(probability_matrix > 1)
) {
  stop("Prediction probabilities are outside the 0 to 1 range.")
}

predicted_indices <- max.col(
  probability_matrix,
  ties.method = "first"
)

class_levels <- c(
  "Healthy",
  "Early Degradation",
  "Moderate Degradation",
  "Severe Degradation"
)

predictions$Predicted_Stage <- class_levels[
  predicted_indices
]

predictions$Severe_Probability <-
  predictions$Probability_Severe

# ------------------------------------------------------------
# 7. MERGE PREDICTIONS WITH BATTERY MEASUREMENTS
# ------------------------------------------------------------

analysis_data <- predictions %>%
  left_join(
    battery_data %>%
      select(
        Battery_ID,
        Cycle_Number,
        all_of(numeric_features)
      ),
    by = c(
      "Battery_ID",
      "Cycle_Number"
    )
  )

if (
  anyNA(
    analysis_data[, numeric_features]
  )
) {
  stop(
    "Some predictions could not be matched to the original dataset."
  )
}

# ------------------------------------------------------------
# 8. FILTER ACTUAL SEVERE DEGRADATION CASES
# ------------------------------------------------------------

severe_cases <- analysis_data %>%
  filter(
    Actual_Stage == "Severe Degradation"
  ) %>%
  mutate(
    Detection_Status = ifelse(
      Predicted_Stage == "Severe Degradation",
      "Correctly Detected",
      "Missed Severe Case"
    )
  )

correctly_detected <- severe_cases %>%
  filter(
    Detection_Status == "Correctly Detected"
  )

missed_severe <- severe_cases %>%
  filter(
    Detection_Status == "Missed Severe Case"
  )

# ------------------------------------------------------------
# 9. OVERALL ERROR ANALYSIS
# ------------------------------------------------------------

total_severe <- nrow(severe_cases)

correct_count <- nrow(correctly_detected)

missed_count <- nrow(missed_severe)

detection_rate <- correct_count / total_severe

miss_rate <- missed_count / total_severe

overview <- data.frame(
  Total_Severe_Cases = total_severe,
  Correctly_Detected = correct_count,
  Missed_Severe_Cases = missed_count,
  Severe_Detection_Rate = detection_rate,
  Severe_Miss_Rate = miss_rate
)

cat("\n========================================\n")
cat("SEVERE DEGRADATION ERROR OVERVIEW\n")
cat("========================================\n")

print(overview)

# ------------------------------------------------------------
# 10. BATTERY-WISE ERROR ANALYSIS
# ------------------------------------------------------------

battery_summary <- severe_cases %>%
  group_by(Battery_ID) %>%
  summarise(
    Total_Severe = n(),
    
    Correctly_Detected = sum(
      Detection_Status == "Correctly Detected"
    ),
    
    Missed = sum(
      Detection_Status == "Missed Severe Case"
    ),
    
    Detection_Rate = Correctly_Detected /
      Total_Severe,
    
    .groups = "drop"
  )

cat("\nBATTERY-WISE SEVERE DETECTION:\n")

print(battery_summary)

# ------------------------------------------------------------
# 11. COMPARE MEASUREMENTS
# CORRECTLY DETECTED VS MISSED SEVERE CASES
# ------------------------------------------------------------

comparison_features <- c(
  "Cycle_Number",
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity",
  "SOH",
  "Severe_Probability"
)

feature_summary <- data.frame()

for (feature in comparison_features) {
  
  for (status in c(
    "Correctly Detected",
    "Missed Severe Case"
  )) {
    
    group_data <- severe_cases %>%
      filter(
        Detection_Status == status
      )
    
    values <- group_data[[feature]]
    
    feature_summary <- rbind(
      feature_summary,
      data.frame(
        Feature = feature,
        Detection_Status = status,
        Count = length(values),
        Mean = mean(values, na.rm = TRUE),
        Median = median(values, na.rm = TRUE),
        Standard_Deviation = sd(values, na.rm = TRUE),
        Minimum = min(values, na.rm = TRUE),
        Maximum = max(values, na.rm = TRUE)
      )
    )
  }
}

cat("\nFEATURE COMPARISON:\n")

print(feature_summary)

# ------------------------------------------------------------
# 12. PREDICTED CLASS OF MISSED SEVERE CASES
# ------------------------------------------------------------

missed_prediction_summary <- missed_severe %>%
  group_by(Predicted_Stage) %>%
  summarise(
    Number_of_Cases = n(),
    .groups = "drop"
  )

cat("\nHOW WERE SEVERE CASES MISCLASSIFIED?\n")

print(missed_prediction_summary)

# ------------------------------------------------------------
# 13. RANK MISSED CASES BY SEVERE PROBABILITY
# ------------------------------------------------------------

missed_ranked <- missed_severe %>%
  arrange(
    Severe_Probability
  ) %>%
  select(
    Battery_ID,
    Cycle_Number,
    Actual_Stage,
    Predicted_Stage,
    Severe_Probability,
    Mean_Voltage,
    Min_Voltage,
    Max_Voltage,
    Mean_Current,
    Mean_Temperature,
    Max_Temperature,
    Discharge_Capacity,
    SOH
  )

cat("\nMISSED SEVERE CASES WITH LOWEST SEVERE PROBABILITY:\n")

print(
  head(missed_ranked, 15)
)

# ------------------------------------------------------------
# 14. CORRECTLY DETECTED SEVERE CASES
# ------------------------------------------------------------

detected_ranked <- correctly_detected %>%
  arrange(
    desc(Severe_Probability)
  ) %>%
  select(
    Battery_ID,
    Cycle_Number,
    Actual_Stage,
    Predicted_Stage,
    Severe_Probability,
    Mean_Voltage,
    Min_Voltage,
    Max_Voltage,
    Mean_Current,
    Mean_Temperature,
    Max_Temperature,
    Discharge_Capacity,
    SOH
  )

# ------------------------------------------------------------
# 15. SAVE ANALYSIS RESULTS
# ------------------------------------------------------------

write.csv(
  overview,
  file.path(
    processed_dir,
    "severe_error_overview.csv"
  ),
  row.names = FALSE
)

write.csv(
  battery_summary,
  file.path(
    processed_dir,
    "severe_error_by_battery.csv"
  ),
  row.names = FALSE
)

write.csv(
  feature_summary,
  file.path(
    processed_dir,
    "severe_error_feature_summary.csv"
  ),
  row.names = FALSE
)

write.csv(
  missed_ranked,
  file.path(
    processed_dir,
    "missed_severe_cases.csv"
  ),
  row.names = FALSE
)

write.csv(
  detected_ranked,
  file.path(
    processed_dir,
    "correctly_detected_severe_cases.csv"
  ),
  row.names = FALSE
)

write.csv(
  missed_prediction_summary,
  file.path(
    processed_dir,
    "missed_severe_prediction_summary.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 16. FINAL SUMMARY
# ------------------------------------------------------------

cat("\n========================================\n")
cat("SEVERE ERROR ANALYSIS COMPLETED\n")
cat("========================================\n")

cat(
  "\nTotal severe cases:",
  total_severe
)

cat(
  "\nCorrectly detected:",
  correct_count
)

cat(
  "\nMissed severe cases:",
  missed_count
)

cat(
  "\nSevere detection rate:",
  round(detection_rate * 100, 2),
  "%"
)

cat(
  "\nSevere miss rate:",
  round(miss_rate * 100, 2),
  "%"
)

cat(
  "\n\nResults saved in:",
  processed_dir,
  "\n"
)
