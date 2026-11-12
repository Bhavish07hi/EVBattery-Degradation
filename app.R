# EV Battery Health Analytics - Professional Shiny Dashboard
# Replace the existing app.R with this complete file.
# Run from the EV_Battery_Degradation project root.

packages <- c("shiny", "shinydashboard", "ggplot2", "dplyr", "DT", "scales", "xgboost", "tidyr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install required packages: ", paste(missing, collapse = ", "))

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DT)
library(scales)

# Consistent plotting language across every dashboard chart.
theme_minimal <- function(base_size = 12, ...) {
  ggplot2::theme_minimal(base_size = base_size, ...) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = NA, colour = NA),
      panel.background = ggplot2::element_rect(fill = NA, colour = NA),
      panel.grid.major = ggplot2::element_line(colour = "#E8EFF1", linewidth = .4),
      panel.grid.minor = ggplot2::element_blank(),
      axis.text = ggplot2::element_text(colour = "#71838E", size = base_size - 1),
      axis.title = ggplot2::element_text(colour = "#49636F", face = "bold", size = base_size - 1),
      legend.position = "bottom",
      legend.title = ggplot2::element_text(colour = "#49636F", face = "bold"),
      legend.text = ggplot2::element_text(colour = "#71838E"),
      plot.caption = ggplot2::element_text(colour = "#82939B", size = base_size - 2, hjust = 0),
      plot.margin = ggplot2::margin(7, 10, 7, 7)
    )
}

root <- getwd()
data_path <- file.path(root, "data", "processed", "all_battery_classification_corrected.csv")
model_path <- file.path(root, "models", "final_optimized_ev_battery_model.rds")
importance_path <- file.path(root, "data", "processed", "final_optimization_feature_importance.csv")

if (!file.exists(data_path)) stop("Missing corrected dataset: ", data_path)
if (!file.exists(model_path)) stop("Missing final model: ", model_path)

battery_data <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
model_bundle <- readRDS(model_path)

required <- c("Battery_ID", "Cycle_Number", "Discharge_Capacity", "SOH",
              "Degradation_Stage", "Mean_Voltage", "Min_Voltage", "Max_Voltage",
              "Mean_Current", "Mean_Temperature", "Max_Temperature")
missing_cols <- setdiff(required, names(battery_data))
if (length(missing_cols)) stop("Dataset missing columns: ", paste(missing_cols, collapse = ", "))
if (is.null(model_bundle$model) || is.null(model_bundle$features)) {
  stop("The final model RDS does not contain the expected model/features.")
}

num_cols <- setdiff(required, c("Battery_ID", "Degradation_Stage"))
battery_data$Battery_ID <- as.character(battery_data$Battery_ID)
for (nm in num_cols) battery_data[[nm]] <- suppressWarnings(as.numeric(battery_data[[nm]]))
stages <- c("Healthy", "Early Degradation", "Moderate Degradation", "Severe Degradation")
battery_data$Degradation_Stage <- factor(battery_data$Degradation_Stage, levels = stages)
battery_data <- battery_data %>%
  filter(!is.na(Battery_ID), is.finite(Cycle_Number), is.finite(SOH),
         is.finite(Discharge_Capacity), !is.na(Degradation_Stage)) %>%
  arrange(Battery_ID, Cycle_Number)
if (!nrow(battery_data)) stop("No valid records found in the corrected dataset.")

# Softer status colours improve readability while retaining the warm brand palette.
stage_cols <- c("Healthy" = "#23866D", "Early Degradation" = "#D39A45",
                "Moderate Degradation" = "#D8794F", "Severe Degradation" = "#B94E55")
model_features <- as.character(model_bundle$features)
medians <- model_bundle$imputation_medians

engineer_features <- function(df) {
  sensors <- c("Mean_Voltage", "Min_Voltage", "Max_Voltage",
               "Mean_Current", "Mean_Temperature", "Max_Temperature")
  needed <- c("Battery_ID", "Cycle_Number", sensors)
  absent <- setdiff(needed, names(df))
  if (length(absent)) stop("Missing CSV columns: ", paste(absent, collapse = ", "))
  
  df$Battery_ID <- as.character(df$Battery_ID)
  for (nm in c("Cycle_Number", sensors)) df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
  if (any(!is.finite(as.matrix(df[, c("Cycle_Number", sensors)])))) {
    stop("Cycle history contains missing or non-numeric measurements.")
  }
  
  out <- df %>% arrange(Battery_ID, Cycle_Number) %>%
    group_by(Battery_ID) %>%
    group_modify(function(x, key) {
      for (nm in sensors) {
        base <- median(head(x[[nm]], 5), na.rm = TRUE)
        x[[paste0(nm, "_Change")]] <- x[[nm]] - base
        x[[paste0(nm, "_Relative")]] <- if (is.finite(base) && abs(base) > 1e-12) {
          (x[[nm]] - base) / abs(base)
        } else 0
      }
      x$Voltage_Range <- x$Max_Voltage - x$Min_Voltage
      x$Temperature_Spread <- x$Max_Temperature - x$Mean_Temperature
      x$Voltage_Range_Change <- x$Voltage_Range -
        median(head(x$Voltage_Range, 5), na.rm = TRUE)
      x$Temperature_Spread_Change <- x$Temperature_Spread -
        median(head(x$Temperature_Spread, 5), na.rm = TRUE)
      x$Voltage_Trend <- c(NA_real_, diff(x$Mean_Voltage))
      x$Current_Trend <- c(NA_real_, diff(x$Mean_Current))
      x$Temperature_Trend <- c(NA_real_, diff(x$Mean_Temperature))
      x$Voltage_Trend_3 <- x$Mean_Voltage - dplyr::lag(x$Mean_Voltage, 3)
      x$Temperature_Trend_3 <- x$Mean_Temperature - dplyr::lag(x$Mean_Temperature, 3)
      x
    }) %>% ungroup()
  
  absent_features <- setdiff(model_features, names(out))
  if (length(absent_features)) stop("Could not construct model features: ", paste(absent_features, collapse = ", "))
  x <- out[, model_features, drop = FALSE]
  
  for (nm in model_features) {
    x[[nm]] <- suppressWarnings(as.numeric(x[[nm]]))
    bad <- !is.finite(x[[nm]])
    if (any(bad)) {
      replacement <- 0
      if (!is.null(medians)) {
        if (!is.null(names(medians)) && nm %in% names(medians)) {
          replacement <- as.numeric(medians[[nm]])
        } else if (length(medians) >= match(nm, model_features)) {
          replacement <- as.numeric(medians[[match(nm, model_features)]])
        }
      }
      if (length(replacement) != 1 || !is.finite(replacement)) replacement <- 0
      x[[nm]][bad] <- replacement
    }
  }
  list(raw = out, x = x)
}

stage_from_soh <- function(x) ifelse(x >= 90, stages[1],
                                     ifelse(x >= 80, stages[2], ifelse(x >= 70, stages[3], stages[4])))

performance <- data.frame(
  Evaluation = c("Battery-wise cross-validation", "External battery evaluation"),
  Accuracy = c(0.6183, 0.5220),
  Macro_F1 = c(0.4868, 0.3989)
)

# Actual rows and predicted columns from the recorded external evaluation.
external_confusion <- matrix(c(33, 9, 0, 0,
                               9, 42, 13, 0,
                               26, 37, 32, 0,
                               0, 4, 0, 0), nrow = 4, byrow = TRUE,
                             dimnames = list(Actual = stages, Predicted = stages))
class_metrics <- data.frame(
  Stage = stages,
  Precision = c(0.4853, 0.4565, 0.7111, 0),
  Recall = c(0.7857, 0.65625, 0.33684, 0),
  F1 = c(0.6000, 0.53846, 0.45714, 0)
)

ui <- dashboardPage(
  skin = "black",
  dashboardHeader(title = span(class = "brand-lockup", icon("car-battery"), span("VOLT", class = "brand-accent"), " / BATTERY LAB"), titleWidth = 280),
  dashboardSidebar(
    width = 280,
    tags$div(class = "sidebar-brand",
             tags$div(class = "sidebar-brand-mark", icon("car-battery")),
             tags$div(class = "sidebar-brand-copy",
                      tags$span("VOLT / BATTERY LAB"),
                      tags$small("EV HEALTH ANALYTICS")
             )
    ),
    sidebarMenu(
      id = "tabs",
      menuItem("Overview", tabName = "overview", icon = icon("chart-line")),
      menuItem("Battery Insights", tabName = "battery", icon = icon("battery-half")),
      menuItem("Prediction Centre", tabName = "prediction", icon = icon("crosshairs")),
      menuItem("Model Performance", tabName = "performance", icon = icon("bullseye")),
      menuItem("Explainability", tabName = "explain", icon = icon("lightbulb")),
      menuItem("Data Explorer", tabName = "data", icon = icon("table"))
    ),
    tags$div(class = "side-foot", "EV BATTERY ANALYTICS", tags$small("Research decision-support system"))
  ),
  dashboardBody(
    tags$head(
      tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
      tags$link(rel = "preconnect", href = "https://fonts.gstatic.com", crossorigin = "anonymous"),
      tags$link(rel = "stylesheet", href = "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap"),
      tags$style(HTML(r"(
        :root{
          --ink:#172B3A;
          --navy:#183B50;
          --navy-deep:#102C3D;
          --teal:#138A83;
          --teal-deep:#0D716C;
          --mint:#DFF3EF;
          --canvas:#F3F7F8;
          --paper:#FFFFFF;
          --line:#E3EBEF;
          --muted:#71838E;
          --amber:#D49A4F;
          --coral:#C96F62;
          --sage:#638D79;
          --shadow:0 8px 24px rgba(24,59,80,.055);
          --shadow-hover:0 16px 36px rgba(24,59,80,.11);
        }
        html,body,.content-wrapper,.right-side{
          font-family:'DM Sans','Segoe UI',Arial,sans-serif!important;
          background:var(--canvas)!important;color:var(--ink);
          -webkit-font-smoothing:antialiased
        }
        body{font-size:13px}
        .wrapper{background:var(--canvas)}
        .main-header{max-height:66px}
        .main-header .logo,.main-header .navbar{
          background:var(--navy-deep)!important;min-height:66px;border:0
        }
        .main-header .logo{
          height:66px;line-height:66px;font-family:Manrope,sans-serif;
          font-size:14px;font-weight:800;letter-spacing:1.25px;
          text-align:left;padding:0 22px;color:#F6FBFC!important
        }
        .brand-lockup .fa{color:#69D0C2;margin-right:10px;font-size:18px}
        .brand-accent{color:#70D7C9;margin-left:2px}
        .main-header .navbar .sidebar-toggle{
          color:#EAF5F6!important;padding-top:21px;height:66px;
          transition:background .2s ease
        }
        .main-header .navbar .sidebar-toggle:hover{background:rgba(112,215,201,.12)!important}
        .main-sidebar{
          background:linear-gradient(165deg,#183B50 0%,#102C3D 100%)!important;
          padding-top:0;box-shadow:8px 0 30px rgba(16,44,61,.09)
        }
        .sidebar-brand{
          display:flex;align-items:center;gap:12px;padding:23px 20px 21px;
          margin:0 0 12px;border-bottom:1px solid rgba(226,243,245,.12)
        }
        .sidebar-brand-mark{
          width:39px;height:39px;border-radius:12px;display:flex;
          align-items:center;justify-content:center;background:linear-gradient(145deg,#4BC3B2,#168A83);
          color:#FFFFFF;font-size:18px;box-shadow:0 6px 16px rgba(19,138,131,.25)
        }
        .sidebar-brand-copy{display:flex;flex-direction:column;gap:3px}
        .sidebar-brand-copy span{font-family:Manrope,sans-serif;font-size:11px;font-weight:800;letter-spacing:1.1px;color:#F3FAFA}
        .sidebar-brand-copy small{font-size:9px;letter-spacing:1.55px;color:#8FB4BA;font-weight:700}
        .sidebar{padding-bottom:85px}
        .sidebar-menu{padding:4px 0}
        .sidebar-menu>li{margin:4px 12px}
        .sidebar-menu>li>a{
          color:#BDD0D5!important;padding:14px 15px;font-size:12.5px;
          font-weight:600;letter-spacing:.12px;border-radius:10px!important;
          margin:0!important;border-left:0!important;background:transparent!important;
          transition:all .2s ease
        }
        .sidebar-menu>li>a>i{
          color:#83AAB0!important;
          display:inline-flex!important;
          align-items:center!important;
          justify-content:center!important;
          flex:0 0 26px!important;
          width:26px!important;
          min-width:26px!important;
          margin:0 4px 0 0!important;
          padding:0!important;
          font-size:15px!important;
          line-height:1!important;
          text-align:center!important;
          vertical-align:middle!important;
          transition:color .2s ease
        }
        .sidebar-menu>li.active>a{
          color:#FFFFFF!important;
          background:linear-gradient(100deg,rgba(36,164,151,.28),rgba(36,164,151,.08))!important;
          box-shadow:inset 3px 0 0 #61D1C2,0 5px 14px rgba(5,25,36,.12);
          font-weight:700
        }
        .sidebar-menu>li.active>a>.fa,.sidebar-menu>li.active>a>.fas{color:#78E0D0!important}
        .sidebar-menu>li:hover>a{
          color:#FFFFFF!important;background:rgba(255,255,255,.065)!important
        }
        .sidebar-menu>li:hover>a>.fa,.sidebar-menu>li:hover>a>.fas{color:#78E0D0!important}
        .side-foot{
          position:absolute;bottom:18px;left:21px;right:18px;padding-top:13px;
          border-top:1px solid rgba(226,245,241,.13);color:#B6CDD0;
          font-size:9px;letter-spacing:1.35px;line-height:1.5
        }
        .side-foot small{display:block;letter-spacing:.1px;margin-top:5px;color:#789FA6;font-size:10px}
        .content-wrapper{min-height:100vh}
        .content{padding:28px 32px 38px;max-width:1680px}
        .content-header{display:none}
        .page-hero{
          position:relative;overflow:hidden;display:flex;align-items:center;
          justify-content:space-between;gap:20px;min-height:122px;
          padding:23px 28px;margin:0 0 23px;border:1px solid #E1ECEF;
          border-radius:18px;background:linear-gradient(110deg,#FFFFFF 0%,#F7FBFC 63%,#EAF7F4 100%);
          box-shadow:0 8px 24px rgba(24,59,80,.045)
        }
        .page-hero:after{
          content:"";position:absolute;width:220px;height:220px;right:-52px;top:-116px;
          border-radius:50%;border:1px solid rgba(19,138,131,.10);
          box-shadow:0 0 0 22px rgba(19,138,131,.025),0 0 0 46px rgba(19,138,131,.018)
        }
        .page-hero-copy{position:relative;z-index:1;max-width:780px}
        .eyebrow{
          color:var(--teal);font-size:9px;font-weight:800;letter-spacing:1.8px;
          text-transform:uppercase;margin-bottom:8px
        }
        .heading{
          font-family:Manrope,'DM Sans',sans-serif;font-size:26px;
          line-height:1.2;font-weight:800;color:var(--navy);
          letter-spacing:-.8px;margin:0 0 7px
        }
        .subheading{color:var(--muted);font-size:12.5px;margin:0;line-height:1.65}
        .hero-chip{
          position:relative;z-index:1;display:inline-flex;align-items:center;gap:8px;
          white-space:nowrap;padding:9px 13px;border-radius:30px;
          background:#E5F4F1;color:#27766F;border:1px solid #D4EBE5;
          font-size:10px;font-weight:700;letter-spacing:.25px
        }
        .hero-chip .fa{font-size:11px}
        .metric{
          position:relative;overflow:hidden;background:var(--paper);
          border:1px solid var(--line);border-radius:16px;padding:20px 21px;
          min-height:132px;margin-bottom:19px;box-shadow:var(--shadow);
          transition:transform .22s ease,box-shadow .22s ease
        }
        .metric:before{
          content:"";position:absolute;left:0;top:0;bottom:0;width:4px;
          background:linear-gradient(180deg,#168F88,#71D0BE)
        }
        .metric:after{
          content:"";position:absolute;width:118px;height:118px;border-radius:50%;
          right:-46px;top:-51px;background:radial-gradient(circle,rgba(19,138,131,.09),rgba(19,138,131,0) 70%)
        }
        .metric:hover{transform:translateY(-4px);box-shadow:var(--shadow-hover)}
        .metric-label{font-size:9px;font-weight:800;letter-spacing:1.25px;color:#7B8E97;text-transform:uppercase}
        .metric-value{
          font-family:Manrope,sans-serif;font-size:29px;font-weight:800;
          color:var(--navy);margin-top:11px;letter-spacing:-.9px;
          font-variant-numeric:tabular-nums
        }
        .metric-foot{font-size:10.5px;color:#82939B;margin-top:6px;line-height:1.5}
        .panel{
          background:var(--paper);border:1px solid var(--line);border-radius:16px;
          padding:21px 22px;margin-bottom:19px;box-shadow:var(--shadow);
          transition:box-shadow .2s ease
        }
        .panel:hover{box-shadow:var(--shadow-hover)}
        .panel-title{
          font-family:Manrope,sans-serif;font-size:13px;font-weight:800;
          color:var(--navy);margin:0 0 16px;padding-bottom:13px;
          border-bottom:1px solid #EDF2F4;display:flex;align-items:center;gap:9px
        }
        .panel-title:before{
          content:"";width:5px;height:17px;border-radius:4px;
          background:linear-gradient(180deg,#1A9B91,#75D2C2)
        }
        .box{
          border:1px solid var(--line)!important;border-radius:16px!important;
          overflow:hidden;box-shadow:var(--shadow)!important;background:#FFFFFF
        }
        .box.box-solid>.box-header{
          background:#FFFFFF!important;color:var(--navy);
          border-bottom:1px solid #EDF2F4
        }
        .box.box-solid>.box-header .box-title{font-family:Manrope,sans-serif;font-weight:800;font-size:13px}
        .note{
          background:linear-gradient(100deg,#E8F5F2,#F4FAF9);
          border:1px solid #D9ECE7;border-left:4px solid var(--teal);
          padding:14px 17px;border-radius:11px;color:#315E61;font-size:11.5px;
          margin:12px 0 19px;line-height:1.65
        }
        label{font-size:11px;font-weight:700;color:#4E6873;margin-bottom:6px}
        .form-control,.selectize-input{
          border:1px solid #DCE7EB!important;border-radius:9px!important;
          color:var(--ink);background:#FFFFFF;min-height:39px;box-shadow:none!important;
          transition:border-color .15s,box-shadow .15s
        }
        .form-control:focus,.selectize-input.focus{
          border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(19,138,131,.12)!important
        }
        .btn,.btn-primary,.btn-default{
          border-radius:9px!important;font-weight:700;padding:9px 15px;
          transition:all .18s ease
        }
        .btn-primary{
          background:linear-gradient(100deg,#168F88,#11766F)!important;
          border:0!important;color:#FFFFFF!important;
          box-shadow:0 5px 13px rgba(19,138,131,.19)
        }
        .btn-primary:hover,.btn-primary:focus{
          background:linear-gradient(100deg,#11766F,#0D625D)!important;
          transform:translateY(-1px);box-shadow:0 8px 18px rgba(19,138,131,.24)
        }
        .btn-default{background:#EEF5F6;border-color:#DCE8EA;color:#315766}
        .btn-default:hover{background:#E1F0ED;border-color:#CBE5DE;color:#173B50}
        .btn-file{background:#EEF5F6!important;border-color:#DCE8EA!important;color:#315766!important}
        .progress-bar{background:var(--teal)}
        .nav-tabs{border-bottom:1px solid #E1EAED}
        .nav-tabs>li>a{
          color:#71858E;font-size:11.5px;font-weight:700;
          border-radius:9px 9px 0 0;border:1px solid transparent;margin-right:5px
        }
        .nav-tabs>li>a:hover{background:#EFF7F6;border-color:transparent}
        .nav-tabs>li.active>a,.nav-tabs>li.active>a:focus,.nav-tabs>li.active>a:hover{
          color:var(--teal-deep);border-color:#DCE9E9 #DCE9E9 #FFFFFF;background:#FFFFFF
        }
        .dataTables_wrapper,.table{color:var(--ink);font-size:11.5px}
        table.dataTable{border-collapse:separate!important;border-spacing:0!important}
        table.dataTable thead th{
          background:#EDF5F5;color:#31535D;border-bottom:1px solid #DCE9E9!important;
          font-weight:800;font-size:10.5px;letter-spacing:.25px;padding:11px 10px
        }
        table.dataTable tbody td{padding:9px 10px;border-top:1px solid #EFF3F4}
        table.dataTable tbody tr:nth-child(even){background:#FAFCFC}
        table.dataTable tbody tr:hover{background:#EDF7F5!important}
        .dataTables_filter input,.dataTables_length select{
          border:1px solid #DCE7EB;border-radius:8px;padding:5px 8px
        }
        .pagination>.active>a,.pagination>.active>a:hover{
          background:var(--teal);border-color:var(--teal)
        }
        .pagination>li>a{color:var(--teal-deep)}
        a{color:var(--teal-deep)} a:hover{color:var(--navy)}
        hr{border-top:1px solid #EAF0F2}
        .small-box{border-radius:15px;box-shadow:var(--shadow)}
        .plot-container{background:transparent}
        .shiny-output-error{color:#A6534C;background:#FFF0ED;padding:11px;border-radius:9px}
        .shiny-output-error-validation{color:var(--muted);font-style:italic}
        .recalculating{opacity:.45;transition:opacity .25s}
        ::-webkit-scrollbar{width:9px;height:9px}
        ::-webkit-scrollbar-thumb{background:#C9D9DD;border-radius:8px}

        /* Final refinement layer: tighter visual rhythm and a more premium product feel */
        .main-header .logo{
          position:relative;text-transform:uppercase;
          background:linear-gradient(105deg,#102C3D 0%,#173D50 72%,#164A55 100%)!important
        }
        .main-header .logo:after{
          content:"";position:absolute;right:0;top:14px;height:38px;width:1px;
          background:rgba(221,243,241,.13)
        }
        .main-header .navbar{
          background:linear-gradient(105deg,#102C3D,#14394B)!important;
          box-shadow:0 2px 10px rgba(13,38,53,.08)
        }
        .main-sidebar:before{
          content:"";position:absolute;top:0;right:0;width:1px;height:100%;
          background:rgba(255,255,255,.035);pointer-events:none
        }
        .sidebar-menu>li>a{
          display:flex;align-items:center;min-height:45px
        }
        /* Use one fixed icon column for every navigation item, including regular (far) icons. */
        .sidebar-menu>li>a>i{
          box-sizing:border-box!important;
          height:18px!important;
          position:relative!important;
          top:0!important;
          transform:none!important
        }
        .sidebar-menu>li.active>a:after{
          content:"";position:absolute;right:13px;top:50%;width:5px;height:5px;
          margin-top:-2.5px;border-radius:50%;background:#78E0D0;
          box-shadow:0 0 0 4px rgba(120,224,208,.10)
        }
        .page-hero{
          border-color:#E2ECEF;
          background:
            radial-gradient(circle at 94% 22%,rgba(31,159,145,.075),transparent 22%),
            linear-gradient(112deg,#FFFFFF 0%,#FBFDFD 58%,#EFF8F6 100%);
        }
        .page-hero:before{
          content:"";position:absolute;left:0;top:0;bottom:0;width:4px;
          background:linear-gradient(180deg,#1AA397,#70D3C3)
        }
        .page-hero:after{
          width:250px;height:250px;right:-76px;top:-142px;
          border-color:rgba(19,138,131,.12);
          box-shadow:0 0 0 26px rgba(19,138,131,.025),0 0 0 54px rgba(19,138,131,.018)
        }
        .eyebrow{
          display:flex;align-items:center;gap:8px;
          letter-spacing:2px
        }
        .eyebrow:before{
          content:"";width:17px;height:2px;border-radius:2px;background:#25A89B
        }
        .heading{letter-spacing:-1px}
        .subheading{max-width:700px}
        .hero-chip{
          background:rgba(224,244,239,.85);
          box-shadow:0 3px 10px rgba(30,112,105,.045);
          border-color:#D5EDE6
        }
        .metric{
          border-color:#E5EDF0;
          background:linear-gradient(145deg,#FFFFFF 0%,#FFFFFF 76%,#FAFDFC 100%)
        }
        .metric:before{width:3px}
        .metric-label{letter-spacing:1.35px}
        .metric-value{line-height:1.12}
        .panel{
          border-color:#E5EDF0;
          background:linear-gradient(155deg,#FFFFFF 0%,#FFFFFF 82%,#FCFDFD 100%)
        }
        .panel-title{
          letter-spacing:.05px;
          font-size:13.2px
        }
        .panel-title:before{
          width:4px;height:16px;
          box-shadow:0 2px 5px rgba(19,138,131,.18)
        }
        .note{
          box-shadow:0 4px 14px rgba(19,138,131,.035)
        }
        .form-control,.selectize-input,.selectize-input.input-active{
          min-height:41px
        }
        .selectize-dropdown{
          border:1px solid #DCE8EA!important;border-radius:9px!important;
          box-shadow:0 12px 28px rgba(24,59,80,.12)!important;overflow:hidden
        }
        .selectize-dropdown .active{
          background:#EAF6F3!important;color:#176D68!important
        }
        .btn-primary{letter-spacing:.1px}
        .btn-primary .fa{margin-right:5px}
        .dataTables_wrapper .dataTables_filter input:focus,
        .dataTables_wrapper .dataTables_length select:focus{
          outline:0;border-color:#168F88;box-shadow:0 0 0 3px rgba(19,138,131,.10)
        }
        table.dataTable tbody tr{transition:background .15s ease}
        .plot-container svg text{font-family:'DM Sans','Segoe UI',Arial,sans-serif}
        .plot-container .modebar{opacity:.18;transition:opacity .2s ease}
        .plot-container:hover .modebar{opacity:.85}
        .content{padding-top:30px}
        @media(min-width:1200px){
          .content{padding-left:36px;padding-right:36px}
          .page-hero{padding:25px 31px}
          .panel{padding:23px 24px}
        }
        @media(max-width:767px){
          .content{padding:16px 12px 26px}
          .page-hero{min-height:0;padding:19px 18px;align-items:flex-start}
          .page-hero:after{right:-120px}
          .heading{font-size:22px}
          .hero-chip{display:none}
          .metric{min-height:108px;padding:16px}
          .metric-value{font-size:25px}
          .panel{padding:16px}
          .sidebar-menu>li>a{padding:14px 18px}
          .main-header .logo{font-size:12px;padding:0 12px}
        }
        @media(prefers-reduced-motion:reduce){
          *,*:before,*:after{transition:none!important;animation:none!important}
        }
      )"))
    ),
    tabItems(
      tabItem(tabName = "overview",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("OVERVIEW", class = "eyebrow"),
                      h2("Battery Health Overview", class = "heading"),
                      p("A decision-focused view of health condition, degradation patterns and capacity behaviour.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "Activity summary")
              ),
              fluidRow(
                column(3, uiOutput("m_soh")),
                column(3, uiOutput("m_healthy")),
                column(3, uiOutput("m_attention")),
                column(3, uiOutput("m_severe"))
              ),
              fluidRow(
                column(7, div(class = "panel", h4("Degradation Stage Distribution", class = "panel-title"),
                              plotOutput("stage_plot", height = 300))),
                column(5, div(class = "panel", h4("SOH Distribution", class = "panel-title"),
                              plotOutput("soh_plot", height = 300)))
              ),
              fluidRow(column(12, div(class = "panel", h4("Capacity Fade Across Cycles", class = "panel-title"),
                                      plotOutput("capacity_plot", height = 330)))),
              div(class = "note", strong("Interpretation: "),
                  "Stages use this project's capacity-derived SOH thresholds. This is a research decision-support tool, not a certified battery safety assessment.")
      ),
      tabItem(tabName = "battery",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("BATTERY ANALYTICS", class = "eyebrow"),
                      h2("Battery Insights", class = "heading"),
                      p("Inspect an individual battery's health trajectory and measured operating behaviour.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "Battery-level view")
              ),
              fluidRow(
                column(4, div(class = "panel", h4("Select Battery", class = "panel-title"),
                              selectInput("battery_id", "Battery identifier", sort(unique(battery_data$Battery_ID))),
                              uiOutput("battery_summary"))),
                column(8, div(class = "panel", h4("State of Health Trajectory", class = "panel-title"),
                              plotOutput("battery_soh", height = 300)))
              ),
              fluidRow(
                column(6, div(class = "panel", h4("Discharge Capacity", class = "panel-title"),
                              plotOutput("battery_capacity", height = 270))),
                column(6, div(class = "panel", h4("Operating Temperature", class = "panel-title"),
                              plotOutput("battery_temp", height = 270)))
              ),
              fluidRow(column(12, div(class = "panel", h4("Short-Term SOH Trend Estimate", class = "panel-title"),
                                      plotOutput("battery_forecast", height = 280),
                                      p("Illustrative linear trend extrapolation from this battery's recorded SOH history; not a validated remaining-useful-life model.", class = "metric-foot")))),
              fluidRow(column(12, box(title = "Cycle-level Measurements", width = 12,
                                      DTOutput("battery_table"))))
      ),
      tabItem(tabName = "prediction",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("PREDICTIVE ANALYTICS", class = "eyebrow"),
                      h2("Prediction Centre", class = "heading"),
                      p("Estimate state of health and degradation stage from chronological cycle-history measurements.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "SOH estimation")
              ),
              div(class = "note", strong("Required input: "),
                  "Upload cycle history, not a single isolated reading. The model uses each battery's first five cycles as a reference baseline and uses previous-cycle trends. Required columns: Battery_ID, Cycle_Number, Mean_Voltage, Min_Voltage, Max_Voltage, Mean_Current, Mean_Temperature, Max_Temperature."),
              tabsetPanel(
                tabPanel("CSV Upload",
                         br(),
                         fluidRow(
                           column(5, div(class = "panel", h4("Cycle History Upload", class = "panel-title"),
                                         fileInput("pred_file", "Choose CSV file", accept = ".csv"),
                                         actionButton("predict", "Generate Health Estimates", icon = icon("play"), class = "btn-primary"),
                                         br(), br(), uiOutput("pred_status"), downloadButton("download_pred", "Download results"))),
                           column(7, div(class = "panel", h4("Predicted Degradation Profile", class = "panel-title"),
                                         plotOutput("pred_plot", height = 270)))
                         ),
                         fluidRow(column(12, box(title = "CSV Prediction Results", width = 12, DTOutput("pred_table"))))
                ),
                tabPanel("Manual Entry",
                         br(),
                         div(class = "note", strong("Quick manual prediction: "),
                             "Enter one cycle's sensor readings. The model uses a reference baseline calculated from the project dataset, so this is a quick indicative estimate rather than a battery-specific trend analysis."),
                         fluidRow(
                           column(5, div(class = "panel", h4("Battery Measurements", class = "panel-title"),
                                         textInput("manual_battery_id", "Battery ID / label", value = "New Battery"),
                                         numericInput("manual_cycle", "Cycle number", value = 10, min = 1, step = 1),
                                         fluidRow(
                                           column(6, numericInput("manual_mean_voltage", "Mean voltage (V)", value = NA_real_)),
                                           column(6, numericInput("manual_min_voltage", "Minimum voltage (V)", value = NA_real_))
                                         ),
                                         fluidRow(
                                           column(6, numericInput("manual_max_voltage", "Maximum voltage (V)", value = NA_real_)),
                                           column(6, numericInput("manual_mean_current", "Mean current (A)", value = NA_real_))
                                         ),
                                         fluidRow(
                                           column(6, numericInput("manual_mean_temp", "Mean temperature (°C)", value = NA_real_)),
                                           column(6, numericInput("manual_max_temp", "Maximum temperature (°C)", value = NA_real_))
                                         ),
                                         actionButton("manual_predict", "Predict Degradation", icon = icon("bolt"), class = "btn-primary"),
                                         br(), br(), uiOutput("manual_status"),
                                         downloadButton("download_manual_report", "Download PDF report")
                           )),
                           column(7, div(class = "panel", h4("Manual Prediction Result", class = "panel-title"),
                                         uiOutput("manual_result_ui")))
                         )
                )
              )
      ),
      tabItem(tabName = "performance",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("MODEL VALIDATION", class = "eyebrow"),
                      h2("Model Performance", class = "heading"),
                      p("Evaluation results for the final XGBoost trend-regression model.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "Evaluation metrics")
              ),
              fluidRow(
                column(6, div(class = "metric", div("BATTERY-WISE CROSS-VALIDATION ACCURACY", class = "metric-label"),
                              div("61.83%", class = "metric-value"), div("Macro F1-score: 0.4868", class = "metric-foot"))),
                column(6, div(class = "metric", div("EXTERNAL BATTERY EVALUATION ACCURACY", class = "metric-label"),
                              div("52.20%", class = "metric-value"), div("Macro F1-score: 0.3989", class = "metric-foot")))
              ),
              div(class = "panel", h4("Evaluation Comparison", class = "panel-title"), plotOutput("performance_plot", height = 300)),
              div(class = "panel", h4("Evaluation Metrics", class = "panel-title"), DTOutput("performance_table")),
              fluidRow(
                column(7, div(class = "panel", h4("External Evaluation Confusion Matrix", class = "panel-title"), plotOutput("confusion_plot", height = 300))),
                column(5, div(class = "panel", h4("Class-wise External Metrics", class = "panel-title"), DTOutput("class_metrics_table")))
              ),
              div(class = "note", strong("Evaluation caveat: "),
                  "The external battery group was examined during earlier model-development experiments; it is not a pristine blind test. The model did not correctly identify the Severe Degradation class in that evaluation. Performance is moderate and predictions must not be used as the sole basis for safety decisions.")
      ),
      tabItem(tabName = "explain",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("MODEL TRANSPARENCY", class = "eyebrow"),
                      h2("Model Explainability", class = "heading"),
                      p("Inspect which engineered signals are most influential in the trained model.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "Feature insights")
              ),
              div(class = "panel", h4("Final Model Feature Importance", class = "panel-title"),
                  plotOutput("importance_plot", height = 360), uiOutput("importance_note")),
              div(class = "panel", h4("Local Explanation for Latest Manual Prediction", class = "panel-title"),
                  plotOutput("local_explain_plot", height = 320), uiOutput("local_explain_note")),
              div(class = "note", strong("Interpretation: "),
                  "Global feature importance describes overall model usage. Local contribution scores explain how input features contribute to an individual XGBoost prediction; they are not causal effects.")
      ),
      tabItem(tabName = "data",
              div(class = "page-hero",
                  div(class = "page-hero-copy",
                      div("DATA WORKSPACE", class = "eyebrow"),
                      h2("Data Explorer", class = "heading"),
                      p("Search and inspect the corrected cycle-level dataset.", class = "subheading")
                  ),
                  div(class = "hero-chip", icon("circle-check"), "Cycle-level records")
              ),
              fluidRow(
                column(3, div(class = "panel", h4("Filters", class = "panel-title"),
                              selectInput("data_battery", "Battery", c("All batteries", sort(unique(battery_data$Battery_ID)))),
                              selectInput("data_stage", "Degradation stage", c("All stages", stages)))),
                column(9, div(class = "panel", h4("Filtered Observations", class = "panel-title"),
                              uiOutput("data_count"), DTOutput("data_table")))
              )
      )
    )
  )
)

server <- function(input, output, session) {
  `%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x
  metric_ui <- function(label, value, foot) {
    div(class = "metric", div(label, class = "metric-label"),
        div(value, class = "metric-value"), div(foot, class = "metric-foot"))
  }
  
  output$m_soh <- renderUI(metric_ui("AVERAGE STATE OF HEALTH",
                                     paste0(round(mean(battery_data$SOH, na.rm = TRUE), 1), "%"), "Across valid cycle observations"))
  output$m_healthy <- renderUI(metric_ui("HEALTHY OBSERVATION SHARE",
                                         percent(mean(battery_data$Degradation_Stage == stages[1], na.rm = TRUE), accuracy = 0.1),
                                         "SOH at or above 90%"))
  output$m_attention <- renderUI(metric_ui("OBSERVATIONS NEEDING ATTENTION",
                                           percent(mean(battery_data$Degradation_Stage %in% stages[3:4], na.rm = TRUE), accuracy = 0.1),
                                           "Moderate or severe stage"))
  output$m_severe <- renderUI(metric_ui("SEVERE-STAGE OBSERVATIONS",
                                        comma(sum(battery_data$Degradation_Stage == stages[4], na.rm = TRUE)),
                                        "SOH below 70% in this project"))
  
  output$stage_plot <- renderPlot({
    d <- battery_data %>% count(Degradation_Stage) %>% mutate(p = n / sum(n))
    ggplot(d, aes(Degradation_Stage, p, fill = Degradation_Stage)) +
      geom_col(width = .68, show.legend = FALSE) +
      geom_text(aes(label = percent(p, accuracy = .1)), vjust = -.5, size = 4) +
      scale_fill_manual(values = stage_cols, drop = FALSE) +
      scale_y_continuous(labels = percent, expand = expansion(mult = c(0, .15))) +
      labs(x = NULL, y = "Share of observations") + theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 15, hjust = 1), panel.grid.minor = element_blank())
  })
  output$soh_plot <- renderPlot({
    ggplot(battery_data, aes(SOH)) +
      geom_histogram(bins = 28, fill = "#168F88", color = "#DDF1EC") +
      geom_vline(xintercept = c(70, 80, 90), linetype = "dashed", color = "#183B50") +
      labs(x = "State of Health (%)", y = "Cycle observations") + theme_minimal(base_size = 12)
  })
  output$capacity_plot <- renderPlot({
    d <- battery_data %>% group_by(Cycle_Number) %>%
      summarise(mid = median(Discharge_Capacity, na.rm = TRUE),
                lo = quantile(Discharge_Capacity, .25, na.rm = TRUE),
                hi = quantile(Discharge_Capacity, .75, na.rm = TRUE), .groups = "drop")
    ggplot(d, aes(Cycle_Number, mid)) +
      geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#B9DED7", alpha = .55) +
      geom_line(color = "#183B50", linewidth = 1.1) +
      labs(x = "Cycle number", y = "Median discharge capacity (Ah)") + theme_minimal(base_size = 12)
  })
  
  battery_selected <- reactive({
    req(input$battery_id)
    battery_data %>% filter(Battery_ID == input$battery_id) %>% arrange(Cycle_Number)
  })
  output$battery_summary <- renderUI({
    d <- battery_selected(); last <- tail(d, 1); s <- as.character(last$Degradation_Stage[1])
    tagList(tags$p(strong("Latest health stage")),
            tags$span(style = paste0("display:inline-block;padding:6px 11px;border-radius:20px;background:",
                                     stage_cols[[s]], "20;color:", stage_cols[[s]], ";font-weight:700"), s),
            tags$hr(),
            tags$p(strong("Latest SOH: "), paste0(round(last$SOH[1], 2), "%")),
            tags$p(strong("Initial recorded SOH: "), paste0(round(d$SOH[1], 2), "%")),
            tags$p(strong("SOH change: "), paste0(round(last$SOH[1] - d$SOH[1], 2), " percentage points")),
            tags$p(strong("Latest capacity: "), paste0(round(last$Discharge_Capacity[1], 3), " Ah")))
  })
  output$battery_soh <- renderPlot({
    ggplot(battery_selected(), aes(Cycle_Number, SOH)) +
      geom_hline(yintercept = c(70, 80, 90), linetype = "dashed", color = "#C7D9DE") +
      geom_line(color = "#183B50", linewidth = 1) +
      geom_point(aes(color = Degradation_Stage), size = 1.7) +
      scale_color_manual(values = stage_cols, drop = FALSE) +
      labs(x = "Cycle number", y = "SOH (%)", color = "Stage") + theme_minimal(base_size = 12)
  })
  output$battery_capacity <- renderPlot({
    ggplot(battery_selected(), aes(Cycle_Number, Discharge_Capacity)) +
      geom_line(color = "#168F88", linewidth = 1) +
      labs(x = "Cycle number", y = "Discharge capacity (Ah)") + theme_minimal(base_size = 12)
  })
  output$battery_temp <- renderPlot({
    ggplot(battery_selected(), aes(Cycle_Number, Mean_Temperature)) +
      geom_line(color = "#168F88", linewidth = 1) +
      labs(x = "Cycle number", y = "Mean temperature") + theme_minimal(base_size = 12)
  })
  output$battery_forecast <- renderPlot({
    d <- battery_selected() %>% filter(is.finite(SOH))
    validate(need(nrow(d) >= 3, "At least three recorded cycles are needed to estimate a trend."))
    fit <- lm(SOH ~ Cycle_Number, data = d)
    future <- data.frame(Cycle_Number = seq(max(d$Cycle_Number) + 1, max(d$Cycle_Number) + 20, length.out = 20))
    future$SOH_Estimate <- pmax(0, pmin(110, predict(fit, newdata = future)))
    ggplot() + geom_line(data = d, aes(Cycle_Number, SOH), color = "#183B50", linewidth = 1) +
      geom_line(data = future, aes(Cycle_Number, SOH_Estimate), color = "#168F88", linewidth = 1, linetype = "dashed") +
      geom_hline(yintercept = c(70, 80, 90), linetype = "dotted", color = "#C7D9DE") +
      labs(x = "Cycle number", y = "SOH (%)", caption = "Dashed segment: simple linear extrapolation, not a validated RUL prediction") +
      theme_minimal(base_size = 12)
  })
  output$battery_table <- renderDT(datatable(battery_selected(), filter = "top",
                                             options = list(pageLength = 8, scrollX = TRUE), rownames = FALSE))
  
  # Simple single-record manual prediction using dataset-level reference baselines.
  manual_result <- eventReactive(input$manual_predict, {
    values <- c(
      Cycle_Number = input$manual_cycle,
      Mean_Voltage = input$manual_mean_voltage,
      Min_Voltage = input$manual_min_voltage,
      Max_Voltage = input$manual_max_voltage,
      Mean_Current = input$manual_mean_current,
      Mean_Temperature = input$manual_mean_temp,
      Max_Temperature = input$manual_max_temp
    )
    if (any(!is.finite(suppressWarnings(as.numeric(values))))) {
      return(list(error = "Please enter all six sensor measurements and a valid cycle number."))
    }
    values <- as.numeric(values)
    names(values) <- c("Cycle_Number", "Mean_Voltage", "Min_Voltage", "Max_Voltage",
                       "Mean_Current", "Mean_Temperature", "Max_Temperature")
    if (values["Cycle_Number"] < 1) return(list(error = "Cycle number must be at least 1."))
    if (values["Min_Voltage"] > values["Max_Voltage"]) return(list(error = "Minimum voltage cannot exceed maximum voltage."))
    if (values["Mean_Temperature"] > values["Max_Temperature"]) return(list(error = "Mean temperature cannot exceed maximum temperature."))
    
    sensors <- c("Mean_Voltage", "Min_Voltage", "Max_Voltage", "Mean_Current",
                 "Mean_Temperature", "Max_Temperature")
    # Use median of the first five observations per training battery as the reference.
    reference_rows <- battery_data %>% arrange(Battery_ID, Cycle_Number) %>%
      group_by(Battery_ID) %>% slice_head(n = 5) %>% ungroup()
    baseline <- vapply(sensors, function(nm) median(reference_rows[[nm]], na.rm = TRUE), numeric(1))
    feature_values <- as.list(rep(0, length(model_features)))
    names(feature_values) <- model_features
    if (!is.null(medians)) {
      for (nm in model_features) {
        if (!is.null(names(medians)) && nm %in% names(medians)) feature_values[[nm]] <- as.numeric(medians[[nm]])
        else if (length(medians) >= match(nm, model_features)) feature_values[[nm]] <- as.numeric(medians[[match(nm, model_features)]])
      }
    }
    for (nm in sensors) {
      feature_values[[paste0(nm, "_Change")]] <- values[[nm]] - baseline[[nm]]
      feature_values[[paste0(nm, "_Relative")]] <- if (abs(baseline[[nm]]) > 1e-12) (values[[nm]] - baseline[[nm]]) / abs(baseline[[nm]]) else 0
    }
    base_range <- median(reference_rows$Max_Voltage - reference_rows$Min_Voltage, na.rm = TRUE)
    base_spread <- median(reference_rows$Max_Temperature - reference_rows$Mean_Temperature, na.rm = TRUE)
    feature_values[["Voltage_Range_Change"]] <- (values[["Max_Voltage"]] - values[["Min_Voltage"]]) - base_range
    feature_values[["Temperature_Spread_Change"]] <- (values[["Max_Temperature"]] - values[["Mean_Temperature"]]) - base_spread
    x <- as.data.frame(feature_values, check.names = FALSE)[, model_features, drop = FALSE]
    x[] <- lapply(x, function(z) { z <- as.numeric(z); z[!is.finite(z)] <- 0; z })
    
    tryCatch({
      predicted_soh <- as.numeric(predict(model_bundle$model, xgboost::xgb.DMatrix(as.matrix(x))))[1]
      predicted_soh <- max(0, min(100, predicted_soh))
      result <- data.frame(Battery_ID = trimws(input$manual_battery_id %||% "New Battery"),
                           Cycle_Number = values[["Cycle_Number"]], Predicted_SOH = predicted_soh,
                           Predicted_Stage = stage_from_soh(predicted_soh))
      list(data = result, x = x)
    }, error = function(e) list(error = conditionMessage(e)))
  })
  output$manual_status <- renderUI({
    r <- manual_result(); req(r)
    if (!is.null(r$error)) div(class = "note", strong("Prediction not completed: "), r$error)
    else div(class = "note", strong("Prediction generated successfully."))
  })
  output$manual_result_ui <- renderUI({
    r <- manual_result(); req(r)
    if (!is.null(r$error)) return(tags$p("Enter the measurements and select Predict Degradation."))
    d <- r$data
    col <- stage_cols[[as.character(d$Predicted_Stage)]]
    tagList(
      div(class = "metric", div("PREDICTED STATE OF HEALTH", class = "metric-label"),
          div(paste0(round(d$Predicted_SOH, 2), "%"), class = "metric-value"),
          div(paste("Battery:", d$Battery_ID, "| Cycle:", d$Cycle_Number), class = "metric-foot")),
      tags$span(style = paste0("display:inline-block;padding:7px 12px;border-radius:18px;background:", col,
                               "20;color:", col, ";font-weight:700"), as.character(d$Predicted_Stage)),
      tags$p(style = "font-size:11px;color:#168F88;margin-top:12px",
             "Quick estimate using the dataset-level reference baseline. For battery-specific trend analysis, use CSV cycle-history upload." )
    )
  })
  
  local_explanation_data <- reactive({
    r <- manual_result()
    if (is.null(r) || !is.null(r$error) || is.null(r$x) || nrow(r$x) == 0) return(NULL)
    
    x <- r$x[, model_features, drop = FALSE]
    dm <- xgboost::xgb.DMatrix(as.matrix(x))
    
    # First choice: genuine per-row XGBoost contribution values (includes a bias term).
    contrib <- tryCatch(
      predict(model_bundle$model, dm, predcontrib = TRUE),
      error = function(e) NULL
    )
    
    if (!is.null(contrib)) {
      cm <- as.matrix(contrib)
      if (nrow(cm) >= 1 && ncol(cm) >= 2) {
        vals <- as.numeric(cm[1, , drop = TRUE])
        cn <- colnames(cm)
        if (!is.null(cn) && length(cn) == length(vals)) {
          keep <- !toupper(cn) %in% c("BIAS", "BIAS_TERM", "(BIAS)")
          vals <- vals[keep]
          features <- cn[keep]
          features <- sub("^.*=", "", features)
          matched <- match(model_features, features)
          if (all(!is.na(matched))) vals <- vals[matched]
          else if (length(vals) == length(model_features)) features <- model_features
          else features <- character(0)
        } else {
          # XGBoost commonly returns features followed by the BIAS column.
          if (length(vals) == length(model_features) + 1) vals <- vals[-length(vals)]
          features <- if (length(vals) == length(model_features)) model_features else character(0)
        }
        if (length(features) == length(model_features) && all(is.finite(vals))) {
          return(list(data = data.frame(Feature = features, Contribution = vals),
                      method = "XGBoost per-observation contribution scores"))
        }
      }
    }
    
    # Fallback: create a clearly labelled approximation from input deviation and global gain.
    # This is NOT SHAP and is not presented as a causal explanation.
    imp <- tryCatch({
      if (file.exists(importance_path)) read.csv(importance_path, check.names = FALSE)
      else xgboost::xgb.importance(model = model_bundle$model)
    }, error = function(e) NULL)
    if (is.null(imp) || !nrow(imp)) {
      imp <- tryCatch(xgboost::xgb.importance(model = model_bundle$model), error = function(e) NULL)
    }
    if (is.null(imp) || !nrow(imp)) return(NULL)
    
    feature_col <- intersect(c("Feature", "feature", "Feature_Name"), names(imp))[1]
    score_col <- intersect(c("Gain", "Importance", "importance", "Frequency", "Cover"), names(imp))[1]
    if (is.na(feature_col) || is.na(score_col)) return(NULL)
    imp_names <- as.character(imp[[feature_col]])
    scores <- suppressWarnings(as.numeric(imp[[score_col]]))
    names(scores) <- imp_names
    med <- medians
    baseline <- vapply(model_features, function(nm) {
      if (!is.null(med) && !is.null(names(med)) && nm %in% names(med)) as.numeric(med[[nm]])
      else if (!is.null(med) && length(med) >= match(nm, model_features)) as.numeric(med[[match(nm, model_features)]])
      else 0
    }, numeric(1))
    observed <- as.numeric(x[1, model_features, drop = TRUE])
    delta <- observed - baseline
    gain <- scores[model_features]
    gain[!is.finite(gain)] <- 0
    approx <- delta * gain
    if (!any(is.finite(approx)) || all(approx == 0)) approx <- gain
    if (!any(is.finite(approx)) || all(approx == 0)) return(NULL)
    list(data = data.frame(Feature = model_features, Contribution = as.numeric(approx)),
         method = "Approximation: input deviation weighted by global feature importance (not SHAP)")
  })
  
  output$local_explain_plot <- renderPlot({
    ex <- local_explanation_data()
    validate(need(!is.null(ex) && nrow(ex$data) > 0,
                  "Local explanation could not be calculated. Check that the saved XGBoost model and feature-importance file are available."))
    d <- ex$data %>%
      filter(is.finite(Contribution)) %>%
      arrange(desc(abs(Contribution))) %>%
      slice_head(n = 10)
    validate(need(nrow(d) > 0, "No usable feature contribution values were returned."))
    ggplot(d, aes(reorder(Feature, Contribution), Contribution, fill = Contribution > 0)) +
      geom_col(show.legend = FALSE, width = 0.72) + coord_flip() +
      scale_fill_manual(values = c("TRUE" = "#168F88", "FALSE" = "#C7D9DE")) +
      labs(x = NULL, y = "Contribution to predicted SOH (model scale)") +
      theme_minimal(base_size = 11) +
      theme(axis.text.y = element_text(size = 9), panel.grid.major.y = element_blank())
  })
  
  output$local_explain_note <- renderUI({
    r <- manual_result()
    if (is.null(r)) return(tags$p("Enter the sensor readings in Prediction Centre and click Predict Degradation. The local explanation will appear here.", class = "metric-foot"))
    if (!is.null(r$error)) return(tags$p(paste("No explanation: prediction failed —", r$error), class = "metric-foot"))
    ex <- local_explanation_data()
    if (is.null(ex)) return(tags$p("Prediction was generated, but the saved model did not return usable local contribution values. Check model feature names and the feature-importance CSV.", class = "metric-foot"))
    tags$p(paste0("Explanation method: ", ex$method,
                  ". Positive bars push the model output upward; negative bars push it downward. These values describe model behaviour, not causal effects."),
           class = "metric-foot")
  })
  
  output$download_manual_report <- downloadHandler(
    filename = function() paste0("battery_prediction_report_", Sys.Date(), ".pdf"),
    content = function(file) {
      r <- manual_result(); req(r); if (!is.null(r$error)) stop(r$error)
      d <- r$data
      grDevices::pdf(file, width = 8.27, height = 11.69, paper = "a4")
      on.exit(grDevices::dev.off(), add = TRUE)
      grid::grid.newpage(); grid::grid.text("EV BATTERY DEGRADATION REPORT", y = .92, gp = grid::gpar(fontsize = 18, fontface = "bold", col = "#183B50"))
      lines <- c(paste("Battery:", d$Battery_ID), paste("Cycle:", d$Cycle_Number),
                 paste0("Predicted SOH: ", round(d$Predicted_SOH, 2), "%"),
                 paste("Predicted stage:", as.character(d$Predicted_Stage)),
                 "", "Model evaluation context",
                 "Battery-wise CV accuracy: 61.83% | Macro F1: 0.4868",
                 "External evaluation accuracy: 52.20% | Macro F1: 0.3989",
                 "", "Important limitations",
                 "This manual estimate uses a dataset-level reference baseline, not this battery's own first-five-cycle baseline.",
                 "External evaluation is not a pristine blind test. Severe degradation was not correctly identified in that evaluation.",
                 "This research prototype is not a certified battery safety assessment.")
      grid::grid.text(paste(lines, collapse = "\n"), x = .10, y = .78, just = c("left", "top"),
                      gp = grid::gpar(fontsize = 11, col = "#183B50"))
    })
  
  pred_result <- eventReactive(input$predict, {
    req(input$pred_file)
    raw <- tryCatch(read.csv(input$pred_file$datapath, stringsAsFactors = FALSE,
                             check.names = FALSE), error = function(e) NULL)
    if (is.null(raw) || !nrow(raw)) return(list(error = "The uploaded CSV is empty or unreadable."))
    if (nrow(raw) > 20000) return(list(error = "Upload no more than 20,000 cycle records at a time."))
    tryCatch({
      engineered <- engineer_features(raw)
      soh <- as.numeric(predict(model_bundle$model,
                                xgboost::xgb.DMatrix(as.matrix(engineered$x))))
      result <- engineered$raw %>% mutate(Predicted_SOH = soh,
                                          Predicted_Stage = stage_from_soh(soh))
      list(data = result)
    }, error = function(e) list(error = conditionMessage(e)))
  })
  output$pred_status <- renderUI({
    r <- pred_result()
    if (!is.null(r$error)) div(class = "note", strong("Prediction not completed: "), r$error)
    else div(class = "note", strong("Prediction completed: "), nrow(r$data), " cycle records processed.")
  })
  output$pred_table <- renderDT({
    r <- pred_result(); req(r)
    if (!is.null(r$error)) return(datatable(data.frame(Status = r$error), options = list(dom = "t")))
    cols <- intersect(c("Battery_ID", "Cycle_Number", "Predicted_SOH", "Predicted_Stage",
                        "Mean_Voltage", "Mean_Current", "Mean_Temperature"), names(r$data))
    datatable(r$data[, cols, drop = FALSE], filter = "top",
              options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
      formatRound("Predicted_SOH", 2)
  })
  output$pred_plot <- renderPlot({
    r <- pred_result(); req(r); validate(need(is.null(r$error), r$error))
    d <- r$data %>% count(Predicted_Stage) %>%
      mutate(Predicted_Stage = factor(Predicted_Stage, levels = stages))
    ggplot(d, aes(Predicted_Stage, n, fill = Predicted_Stage)) +
      geom_col(width = .68, show.legend = FALSE) +
      scale_fill_manual(values = stage_cols, drop = FALSE) +
      labs(x = NULL, y = "Predicted cycle records") + theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 15, hjust = 1))
  })
  output$download_pred <- downloadHandler(
    filename = function() paste0("ev_battery_predictions_", Sys.Date(), ".csv"),
    content = function(file) {
      r <- pred_result(); req(r)
      if (!is.null(r$error)) stop(r$error)
      write.csv(r$data, file, row.names = FALSE)
    })
  
  output$performance_plot <- renderPlot({
    d <- tidyr::pivot_longer(performance, c(Accuracy, Macro_F1),
                             names_to = "Metric", values_to = "Score")
    ggplot(d, aes(Evaluation, Score, fill = Metric)) +
      geom_col(position = position_dodge(.72), width = .62) +
      scale_y_continuous(limits = c(0, 1), labels = percent) +
      scale_fill_manual(values = c(Accuracy = "#183B50", Macro_F1 = "#168F88")) +
      labs(x = NULL, y = "Score", fill = "Metric") + theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 10, hjust = 1))
  })
  output$performance_table <- renderDT({
    d <- performance %>% mutate(Accuracy = percent(Accuracy, accuracy = .01),
                                Macro_F1 = round(Macro_F1, 4))
    datatable(d, options = list(dom = "t", paging = FALSE), rownames = FALSE)
  })
  
  output$confusion_plot <- renderPlot({
    cm <- as.data.frame(as.table(external_confusion))
    names(cm) <- c("Actual", "Predicted", "Count")
    ggplot(cm, aes(Predicted, Actual, fill = Count)) + geom_tile(color = "white") +
      geom_text(aes(label = Count), color = "#183B50", fontface = "bold", size = 4) +
      scale_fill_gradient(low = "#DDF1EC", high = "#168F88") +
      labs(x = "Predicted class", y = "Actual class", fill = "Count") +
      theme_minimal(base_size = 10) + theme(axis.text.x = element_text(angle = 25, hjust = 1))
  })
  output$class_metrics_table <- renderDT({
    d <- class_metrics %>% mutate(across(c(Precision, Recall, F1), ~round(.x, 3)))
    datatable(d, options = list(dom = "t", paging = FALSE), rownames = FALSE)
  })
  
  output$importance_plot <- renderPlot({
    validate(need(file.exists(importance_path),
                  "Final feature-importance output not found. Run R/22_final_optimization.R."))
    imp <- read.csv(importance_path, stringsAsFactors = FALSE, check.names = FALSE)
    f <- intersect(c("Feature", "feature"), names(imp))[1]
    v <- intersect(c("Importance", "Gain", "Frequency", "Cover", "Gain_Importance"), names(imp))[1]
    validate(need(!is.na(f) && !is.na(v), "Could not identify feature and importance columns in the importance file."))
    d <- data.frame(Feature = as.character(imp[[f]]),
                    Importance = suppressWarnings(as.numeric(imp[[v]]))) %>%
      filter(is.finite(Importance)) %>% arrange(desc(Importance)) %>% slice_head(n = 15)
    validate(need(nrow(d) > 0, "No valid feature importance values found."))
    ggplot(d, aes(reorder(Feature, Importance), Importance)) +
      geom_col(fill = "#168F88", width = .7) + coord_flip() +
      labs(x = NULL, y = "Model importance score") + theme_minimal(base_size = 12)
  })
  output$importance_note <- renderUI(tags$p(style = "font-size:12px;color:#168F88",
                                            "Importance reflects how the trained model uses features; it is not causal evidence or a per-record SHAP explanation."))
  
  filtered <- reactive({
    d <- battery_data
    if (!is.null(input$data_battery) && input$data_battery != "All batteries")
      d <- d %>% filter(Battery_ID == input$data_battery)
    if (!is.null(input$data_stage) && input$data_stage != "All stages")
      d <- d %>% filter(as.character(Degradation_Stage) == input$data_stage)
    d
  })
  output$data_count <- renderUI(tags$p(style = "font-size:12px;color:#168F88",
                                       paste(format(nrow(filtered()), big.mark = ","), "matching observations")))
  output$data_table <- renderDT(datatable(filtered(), filter = "top",
                                          options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE))
}

shinyApp(ui, server)
