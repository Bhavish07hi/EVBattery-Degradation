
# ==========================================
# EV BATTERY DEGRADATION CLASSIFICATION
# PHASE 2: DATA INSPECTION
# ==========================================

library(R.matlab)
library(tidyverse)

# 1. Set dataset directory
data_path <- "data/raw"

# 2. List battery files
battery_files <- c(
  "B0005.mat",
  "B0006.mat",
  "B0007.mat",
  "B0018.mat"
)

# 3. Read and inspect each file
for (file in battery_files) {
  
  cat("\n================================\n")
  cat("Reading:", file, "\n")
  cat("================================\n")
  
  file_path <- file.path(data_path, file)
  
  # Check file existence
  if (!file.exists(file_path)) {
    cat("File not found:", file, "\n")
    next
  }
  
  # Read MATLAB file
  battery <- readMat(file_path)
  
  # Display top-level variable names
  cat("\nVariable names:\n")
  print(names(battery))
  
  # Display internal structure
  cat("\nBattery structure:\n")
  str(
    battery,
    max.level = 4,
    list.len = 10
  )
}

cat("\nInspection completed!\n")

