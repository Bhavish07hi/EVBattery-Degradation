
# ============================================================
# STEP 0: INSPECT ALL NASA BATTERY MATLAB FILES
# Project: EV Battery Degradation Classification
# File: R/00_inspect_all_batteries.R
# ============================================================

# 1. Project directory
project_dir <- "C:/Users/bhavi/OneDrive/Documents/EV_Battery_Degradation"

raw_dir <- file.path(project_dir, "data", "raw")
processed_dir <- file.path(project_dir, "data", "processed")

dir.create(
  processed_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

setwd(project_dir)

# 2. Install and load required package
if (!requireNamespace("R.matlab", quietly = TRUE)) {
  install.packages("R.matlab")
}

library(R.matlab)

# 3. Find all MATLAB files
mat_files <- list.files(
  raw_dir,
  pattern = "\\.mat$",
  full.names = TRUE,
  ignore.case = TRUE
)

if (length(mat_files) == 0) {
  stop("No MATLAB files found in data/raw.")
}

cat("\n====================================\n")
cat("NASA BATTERY DATASET INSPECTION\n")
cat("====================================\n")

cat("Total MATLAB files:", length(mat_files), "\n\n")

# 4. Function to describe MATLAB objects
describe_object <- function(x) {
  
  if (is.null(x)) {
    return("NULL")
  }
  
  object_class <- paste(class(x), collapse = "/")
  
  object_dimensions <- if (is.null(dim(x))) {
    paste("Length:", length(x))
  } else {
    paste(dim(x), collapse = " x ")
  }
  
  object_fields <- if (is.list(x)) {
    paste(head(names(x), 15), collapse = ", ")
  } else {
    ""
  }
  
  paste(
    "Class:", object_class,
    "| Dimensions:", object_dimensions,
    "| Fields:", object_fields
  )
}

# 5. Inspect every battery file
inventory <- data.frame(
  File = character(),
  Top_Level_Variable = character(),
  Object_Structure = character(),
  stringsAsFactors = FALSE
)

for (file in mat_files) {
  
  battery_id <- tools::file_path_sans_ext(
    basename(file)
  )
  
  cat("------------------------------------\n")
  cat("Inspecting:", basename(file), "\n")
  
  result <- tryCatch(
    R.matlab::readMat(file),
    error = function(e) {
      cat("ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
  
  if (is.null(result)) {
    inventory <- rbind(
      inventory,
      data.frame(
        File = basename(file),
        Top_Level_Variable = NA,
        Object_Structure = "READ ERROR"
      )
    )
    next
  }
  
  variable_names <- names(result)
  
  if (length(variable_names) == 0) {
    cat("No named MATLAB variables found.\n")
    next
  }
  
  for (variable_name in variable_names) {
    
    object <- result[[variable_name]]
    
    structure_description <- describe_object(object)
    
    cat("\nVariable:", variable_name, "\n")
    cat(structure_description, "\n")
    
    # Show the first level of nested fields
    if (is.list(object) && length(object) > 0) {
      
      cat("Nested structure:\n")
      
      for (i in seq_len(min(length(object), 5))) {
        
        child <- object[[i]]
        
        child_name <- if (
          !is.null(names(object)) &&
          nzchar(names(object)[i])
        ) {
          names(object)[i]
        } else {
          paste0("Element_", i)
        }
        
        cat(
          "  ",
          child_name,
          ": ",
          describe_object(child),
          "\n",
          sep = ""
        )
      }
    }
    
    inventory <- rbind(
      inventory,
      data.frame(
        File = basename(file),
        Top_Level_Variable = variable_name,
        Object_Structure = structure_description,
        stringsAsFactors = FALSE
      )
    )
  }
  
  rm(result)
  invisible(gc())
}

# 6. Save inspection results
inventory_path <- file.path(
  processed_dir,
  "nasa_mat_inventory.csv"
)

write.csv(
  inventory,
  inventory_path,
  row.names = FALSE
)

# 7. Final summary
cat("\n====================================\n")
cat("INSPECTION COMPLETED\n")
cat("====================================\n")

cat("Files inspected:", length(mat_files), "\n")
cat("Inventory saved to:\n")
cat(inventory_path, "\n")

print(inventory)

cat("\nNext: review the inventory before extracting all batteries.\n")
