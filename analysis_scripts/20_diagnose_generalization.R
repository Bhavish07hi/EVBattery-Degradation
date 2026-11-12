
# ============================================================
# FILE: R/20_diagnose_generalization.R
# Diagnose poor external battery generalisation
# ============================================================

library(dplyr)
library(ggplot2)
library(tidyr)

project_dir <- "C:/Users/bhavi/OneDrive/Documents/EV_Battery_Degradation"

data_file <- file.path(
  project_dir,
  "data/processed/all_battery_classification_corrected.csv"
)

output_dir <- file.path(project_dir, "data/processed")
plots_dir <- file.path(project_dir, "plots")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

data <- read.csv(data_file, stringsAsFactors = FALSE)

external_ids <- c("B0045", "B0046", "B0047")

data$Dataset_Group <- ifelse(
  data$Battery_ID %in% external_ids,
  "External Test",
  "Training"
)

# 1. BATTERY-WISE SUMMARY

battery_summary <- data %>%
  group_by(Battery_ID, Dataset_Group) %>%
  summarise(
    Cycles = n(),
    Mean_SOH = mean(SOH, na.rm = TRUE),
    Min_SOH = min(SOH, na.rm = TRUE),
    Max_SOH = max(SOH, na.rm = TRUE),
    Mean_Voltage = mean(Mean_Voltage, na.rm = TRUE),
    Mean_Current = mean(Mean_Current, na.rm = TRUE),
    Mean_Temperature = mean(Mean_Temperature, na.rm = TRUE),
    Healthy = sum(Degradation_Stage == "Healthy"),
    Early = sum(Degradation_Stage == "Early Degradation"),
    Moderate = sum(Degradation_Stage == "Moderate Degradation"),
    Severe = sum(Degradation_Stage == "Severe Degradation"),
    .groups = "drop"
  )

write.csv(
  battery_summary,
  file.path(output_dir, "diagnostic_battery_summary.csv"),
  row.names = FALSE
)

cat("\n========== BATTERY-WISE SUMMARY ==========\n")
print(battery_summary, n = Inf)

# 2. TRAIN VS TEST DISTRIBUTION

features <- c(
  "Mean_Voltage",
  "Min_Voltage",
  "Max_Voltage",
  "Mean_Current",
  "Mean_Temperature",
  "Max_Temperature",
  "Discharge_Capacity",
  "SOH"
)

features <- features[features %in% names(data)]

distribution_summary <- data %>%
  group_by(Dataset_Group) %>%
  summarise(
    across(
      all_of(features),
      list(
        Mean = ~mean(.x, na.rm = TRUE),
        SD = ~sd(.x, na.rm = TRUE),
        Min = ~min(.x, na.rm = TRUE),
        Max = ~max(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

cat("\n========== TRAIN VS EXTERNAL DISTRIBUTION ==========\n")
print(distribution_summary)

write.csv(
  distribution_summary,
  file.path(output_dir, "diagnostic_train_test_distribution.csv"),
  row.names = FALSE
)

# 3. STAGE DISTRIBUTION

stage_distribution <- data %>%
  count(Dataset_Group, Degradation_Stage) %>%
  group_by(Dataset_Group) %>%
  mutate(
    Percentage = round(n / sum(n) * 100, 2)
  ) %>%
  ungroup()

cat("\n========== STAGE DISTRIBUTION ==========\n")
print(stage_distribution)

write.csv(
  stage_distribution,
  file.path(output_dir, "diagnostic_stage_distribution.csv"),
  row.names = FALSE
)

# 4. SOH DISTRIBUTION BY BATTERY

soh_plot <- ggplot(
  data,
  aes(x = Battery_ID, y = SOH, fill = Dataset_Group)
) +
  geom_boxplot(outlier.size = 0.5) +
  labs(
    title = "SOH Distribution Across Batteries",
    x = "Battery",
    y = "State of Health (%)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1)
  )

ggsave(
  filename = file.path(plots_dir, "diagnostic_soh_by_battery.png"),
  plot = soh_plot,
  width = 14,
  height = 7
)

# 5. FEATURE DISTRIBUTION

feature_long <- data %>%
  select(Battery_ID, Dataset_Group, all_of(features)) %>%
  pivot_longer(
    cols = all_of(features),
    names_to = "Feature",
    values_to = "Value"
  )

feature_plot <- ggplot(
  feature_long,
  aes(x = Dataset_Group, y = Value, fill = Dataset_Group)
) +
  geom_boxplot(outlier.size = 0.4) +
  facet_wrap(~Feature, scales = "free", ncol = 2) +
  labs(
    title = "Training vs External Sensor Distributions",
    x = "",
    y = "Value"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  filename = file.path(plots_dir, "diagnostic_feature_distributions.png"),
  plot = feature_plot,
  width = 12,
  height = 10
)

# 6. SENSOR / SOH CORRELATIONS

correlation_results <- data %>%
  group_by(Dataset_Group) %>%
  group_modify(~ {
    numeric_data <- .x %>%
      select(
        SOH,
        any_of(c(
          "Mean_Voltage",
          "Min_Voltage",
          "Max_Voltage",
          "Mean_Current",
          "Mean_Temperature",
          "Max_Temperature",
          "Discharge_Capacity"
        ))
      )
    
    correlations <- sapply(
      names(numeric_data)[-1],
      function(feature) {
        cor(
          numeric_data$SOH,
          numeric_data[[feature]],
          use = "complete.obs"
        )
      }
    )
    
    data.frame(
      Feature = names(correlations),
      Correlation_With_SOH = as.numeric(correlations)
    )
  }) %>%
  ungroup()

cat("\n========== SENSOR / SOH CORRELATIONS ==========\n")
print(correlation_results)

write.csv(
  correlation_results,
  file.path(output_dir, "diagnostic_soh_correlations.csv"),
  row.names = FALSE
)

# 7. INITIAL CAPACITY CHECK

capacity_summary <- data %>%
  group_by(Battery_ID, Dataset_Group) %>%
  summarise(
    Initial_Capacity = first(Initial_Capacity),
    Mean_Discharge_Capacity = mean(Discharge_Capacity, na.rm = TRUE),
    Min_Discharge_Capacity = min(Discharge_Capacity, na.rm = TRUE),
    Max_Discharge_Capacity = max(Discharge_Capacity, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n========== CAPACITY BASELINE CHECK ==========\n")
print(capacity_summary, n = Inf)

write.csv(
  capacity_summary,
  file.path(output_dir, "diagnostic_capacity_summary.csv"),
  row.names = FALSE
)

cat("\n========== DIAGNOSTICS COMPLETE ==========\n")
cat("CSV files saved in:", output_dir, "\n")
cat("Plots saved in:", plots_dir, "\n")
