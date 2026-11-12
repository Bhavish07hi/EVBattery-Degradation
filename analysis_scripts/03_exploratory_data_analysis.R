
# ==========================================
# PHASE 4: EXPLORATORY DATA ANALYSIS
# ==========================================

library(tidyverse)

# Load processed dataset
battery_data <- read.csv(
  "data/processed/battery_cycles.csv"
)

# 1. Dataset overview
cat("Dataset dimensions:\n")
print(dim(battery_data))

cat("\nDataset summary:\n")
print(summary(battery_data))

# 2. Missing value analysis
cat("\nMissing values:\n")
print(colSums(is.na(battery_data)))

# 3. Battery-wise capacity summary
capacity_summary <- battery_data %>%
  group_by(Battery_ID) %>%
  summarise(
    Initial_Capacity = first(Discharge_Capacity),
    Final_Capacity = last(Discharge_Capacity),
    Capacity_Loss = Initial_Capacity - Final_Capacity,
    Total_Cycles = n(),
    .groups = "drop"
  )

print(capacity_summary)

# Capacity degradation graph

p1 <- ggplot(
  battery_data,
  aes(
    x = Cycle_Number,
    y = Discharge_Capacity,
    color = Battery_ID
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  labs(
    title = "EV Battery Capacity Degradation",
    subtitle = "Capacity changes across discharge cycles",
    x = "Experimental Cycle Number",
    y = "Discharge Capacity (Ah)",
    color = "Battery"
  ) +
  theme_minimal()

print(p1)

ggsave(
  "plots/capacity_degradation.png",
  p1,
  width = 10,
  height = 6
)

p2 <- ggplot(
  battery_data,
  aes(
    x = Cycle_Number,
    y = Discharge_Capacity
  )
) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~Battery_ID, scales = "free_x") +
  labs(
    title = "Battery-wise Degradation Comparison",
    x = "Cycle Number",
    y = "Discharge Capacity (Ah)"
  ) +
  theme_minimal()

print(p2)

ggsave(
  "plots/battery_comparison.png",
  p2,
  width = 10,
  height = 7
)

p3 <- ggplot(
  battery_data,
  aes(
    x = Cycle_Number,
    y = Mean_Temperature,
    color = Battery_ID
  )
) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Battery Temperature Across Cycles",
    x = "Cycle Number",
    y = "Average Temperature (°C)"
  ) +
  theme_minimal()

print(p3)

ggsave(
  "plots/temperature_analysis.png",
  p3,
  width = 10,
  height = 6
)
