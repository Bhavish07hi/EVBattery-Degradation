
# ==========================================
# PHASE 5: FEATURE ENGINEERING
# EV BATTERY DEGRADATION CLASSIFICATION
# ==========================================

library(tidyverse)

# Load extracted dataset
battery_data <- read.csv(
  "data/processed/battery_cycles.csv"
)

# Arrange records battery-wise
battery_data <- battery_data %>%
  arrange(Battery_ID, Cycle_Number)

# Calculate battery SOH
battery_data <- battery_data %>%
  group_by(Battery_ID) %>%
  mutate(
    Initial_Capacity = first(Discharge_Capacity),
    
    SOH = (
      Discharge_Capacity / Initial_Capacity
    ) * 100
  ) %>%
  ungroup()

# Create degradation classification labels
battery_data <- battery_data %>%
  mutate(
    Degradation_Stage = case_when(
      
      SOH >= 90 ~ "Healthy",
      
      SOH >= 80 & SOH < 90 ~ "Early Degradation",
      
      SOH >= 70 & SOH < 80 ~ "Moderate Degradation",
      
      SOH < 70 ~ "Severe Degradation"
      
    )
  )

# Convert classification into ordered factor
battery_data$Degradation_Stage <- factor(
  battery_data$Degradation_Stage,
  
  levels = c(
    "Healthy",
    "Early Degradation",
    "Moderate Degradation",
    "Severe Degradation"
  )
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

cat("\nSOH SUMMARY:\n")

print(summary(battery_data$SOH))

cat("\nCLASS DISTRIBUTION:\n")

print(table(battery_data$Degradation_Stage))

cat("\nBATTERY-WISE CLASS DISTRIBUTION:\n")

print(
  table(
    battery_data$Battery_ID,
    battery_data$Degradation_Stage
  )
)

# ==========================================
# VISUALIZE DEGRADATION STAGES
# ==========================================

p <- ggplot(
  battery_data,
  aes(
    x = Cycle_Number,
    y = SOH,
    color = Degradation_Stage
  )
) +
  
  geom_line(linewidth = 0.8) +
  
  geom_hline(
    yintercept = c(90, 80, 70),
    linetype = "dashed",
    color = "grey40"
  ) +
  
  facet_wrap(~Battery_ID, scales = "free_x") +
  
  labs(
    title = "EV Battery State of Health",
    subtitle = "Classification of battery degradation stages",
    x = "Experimental Cycle Number",
    y = "State of Health (%)",
    color = "Degradation Stage"
  ) +
  
  theme_minimal()

print(p)

# Save visualization
ggsave(
  "plots/battery_soh_classification.png",
  p,
  width = 11,
  height = 7
)

# ==========================================
# SAVE FINAL CLASSIFICATION DATASET
# ==========================================

write.csv(
  battery_data,
  "data/processed/battery_classification.csv",
  row.names = FALSE
)

cat("\nClassification dataset created successfully!\n")
