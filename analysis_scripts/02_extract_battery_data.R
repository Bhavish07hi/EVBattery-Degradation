
# ==========================================
# EV BATTERY DEGRADATION CLASSIFICATION
# PHASE 3: DATA EXTRACTION
# ==========================================

library(R.matlab)
library(tidyverse)

# Dataset location
data_path <- "data/raw"

# Battery files
battery_files <- c(
  "B0005.mat",
  "B0006.mat",
  "B0007.mat",
  "B0018.mat"
)

# Function to extract a MATLAB data field
get_field <- function(data, field_name) {
  
  field_names <- attr(data, "dimnames")[[1]]
  
  field_index <- match(field_name, field_names)
  
  if (is.na(field_index)) {
    return(NA_real_)
  }
  
  values <- unlist(
    data[[field_index]],
    use.names = FALSE
  )
  
  return(as.numeric(values))
}

# Function for safe average
safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

# Function for safe maximum
safe_max <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA_real_)
  }
  max(x, na.rm = TRUE)
}

# Function for safe minimum
safe_min <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA_real_)
  }
  min(x, na.rm = TRUE)
}

# Store extracted battery records
all_batteries <- list()

# Process each battery
for (file in battery_files) {
  
  cat("\nProcessing:", file, "\n")
  
  # Read MATLAB file
  mat_data <- readMat(
    file.path(data_path, file)
  )
  
  # Extract cycle records
  records <- mat_data[[1]][[1]]
  
  # Number of experimental cycles
  total_cycles <- length(records) / 4
  
  battery_id <- tools::file_path_sans_ext(file)
  
  battery_records <- list()
  
  # Loop through cycles
  for (i in seq_len(total_cycles)) {
    
    base <- (i - 1) * 4
    
    # Extract cycle type
    cycle_type <- as.character(
      unlist(records[[base + 1]])
    )
    
    # Process discharge cycles only
    if (cycle_type != "discharge") {
      next
    }
    
    # Extract experimental data
    cycle_data <- records[[base + 4]]
    
    # Extract measurements
    voltage <- get_field(
      cycle_data, "Voltage.measured"
    )
    
    current <- get_field(
      cycle_data, "Current.measured"
    )
    
    temperature <- get_field(
      cycle_data, "Temperature.measured"
    )
    
    capacity <- get_field(
      cycle_data, "Capacity"
    )
    
    # Extract features
    battery_records[[length(battery_records) + 1]] <-
      data.frame(
        
        Battery_ID = battery_id,
        
        Cycle_Number = i,
        
        Mean_Voltage = safe_mean(voltage),
        
        Min_Voltage = safe_min(voltage),
        
        Max_Voltage = safe_max(voltage),
        
        Mean_Current = safe_mean(current),
        
        Mean_Temperature = safe_mean(temperature),
        
        Max_Temperature = safe_max(temperature),
        
        Discharge_Capacity = if (
          length(capacity) > 0
        ) capacity[1] else NA_real_
        
      )
  }
  
  # Combine records for this battery
  all_batteries[[battery_id]] <-
    bind_rows(battery_records)
  
  cat(
    "Discharge cycles extracted:",
    nrow(all_batteries[[battery_id]]),
    "\n"
  )
}

# Combine all batteries
battery_dataset <- bind_rows(all_batteries)

# Save processed dataset
dir.create(
  "data/processed",
  recursive = TRUE,
  showWarnings = FALSE
)

write.csv(
  battery_dataset,
  "data/processed/battery_cycles.csv",
  row.names = FALSE
)

# Display results
cat("\n================================\n")
cat("DATA EXTRACTION COMPLETED\n")
cat("================================\n")

print(dim(battery_dataset))

print(head(battery_dataset))

print(summary(battery_dataset))

cat("\nCSV successfully created!\n")
