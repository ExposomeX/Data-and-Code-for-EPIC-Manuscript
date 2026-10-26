library(data.table)
library(ggplot2)
library(fixest)

output_dir <- "code_5_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

input_file <- "code_4_results/2_merged_yearly_data_new.csv"
if (!file.exists(input_file)) stop("文件未找到: ", input_file)

cat("\n>>> Loading data...\n")
df_year <- fread(input_file)
df_year <- df_year[B_year >= 2017 & B_year <= 2023]

exposure_cols <- names(df_year)[grep("^CEI_", names(df_year))]
if(length(exposure_cols) == 0) stop("Error")

covariate_vars <- c('C_BEDS_1000', 'C_GDP_PC', 'C_AGING', 'C_Temp', 'C_SHR_SECONDARY')
cov_str <- paste(covariate_vars, collapse = " + ")

all_results_list <- list()

for (target_exposure in exposure_cols) {
disease_name <- sub("^CEI_", "", target_exposure)
target_outcome <- paste0("Y_", disease_name)

if (!(target_outcome %in% names(df_year))) {
cat(sprintf("Warning，Skip %s\n", target_outcome, target_exposure))
next
}

cat(sprintf("Regression: %s  =>  %s \n", target_exposure, target_outcome))


df_clean <- df_year[!is.na(get(target_exposure)) & !is.na(get(target_outcome)) & C_POP > 0]

if (nrow(df_clean) < 100) next

zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
if (zero_ratio > 0.3) next

epd_iqr <- IQR(df_clean[[target_exposure]], na.rm = TRUE)
if (epd_iqr == 0) next

df_clean[, X_scaled := get(target_exposure) / epd_iqr]

if (sum(df_clean[[target_outcome]], na.rm = TRUE) < 50) next

formula_str <- paste0(target_outcome, " ~ X_scaled + ", cov_str, " + offset(log(C_POP)) | B_county_id + B_year")

fit <- tryCatch({
feglm(as.formula(formula_str), family = "quasipoisson", data = df_clean, nthreads = 4)
}, error = function(e) return(NULL))

if (!is.null(fit)) {
coef_summary <- coeftable(fit, cluster = ~B_county_id)

if ("X_scaled" %in% rownames(coef_summary)) {
  coef_val <- coef_summary["X_scaled", "Estimate"]
  se_val   <- coef_summary["X_scaled", "Std. Error"]
  p_val    <- coef_summary["X_scaled", 4]
  
  if (se_val < 1.0 && abs(coef_val) < 5) { 
    all_results_list[[length(all_results_list) + 1]] <- data.table(
      Disease  = disease_name,
      Outcome  = target_outcome,
      Exposure = target_exposure,
      RR       = exp(coef_val),
      RR_low   = exp(coef_val - 1.96 * se_val),
      RR_high  = exp(coef_val + 1.96 * se_val),
      LogRR    = coef_val,
      SE       = se_val,
      P_value  = p_val,
      Samples  = nrow(df_clean)
    )
  }
}

}
}

res_all <- rbindlist(all_results_list)

res_all[, Significance := ifelse(P_value < 0.05,
ifelse(RR > 1, "Significant Harm", "Significant Protect"),
"Not Significant")]


csv_out <- file.path(output_dir, "Targeted_CEI_Association_Results.csv")
fwrite(res_all, csv_out)


if (nrow(res_all) > 0) {

color_map <- c("Significant Harm" = "#cb181d",
"Significant Protect" = "#2166ac",
"Not Significant" = "grey50")

res_all[, Text_Label := sprintf("%.3f [%.3f-%.3f]", RR, RR_low, RR_high)]


x_max_limit <- max(res_all$RR_high, na.rm = TRUE) * 1.15

p_forest <- ggplot(res_all, aes(x = RR, y = Disease, color = Significance)) +
geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
geom_errorbarh(aes(xmin = RR_low, xmax = RR_high), height = 0.2, linewidth = 1.2) +
geom_point(size = 4) +
geom_text(aes(x = x_max_limit, label = Text_Label), 
          hjust = 0, size = 4, color = "black", fontface = "bold") +

scale_color_manual(values = color_map) +
coord_cartesian(xlim = c(min(c(res_all$RR_low, 0.95), na.rm=T), x_max_limit), clip = "off") +

labs(title = "Health Effects of Composite Exposure Index (CEI)",
     subtitle = "Targeted Association between Disease-Specific CEI and Clinical Outcomes",
     x = "Relative Risk (RR per IQR increase, 95% CI)",
     y = "Disease Category") +

theme_classic(base_size = 14) +
theme(
  plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
  plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
  axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
  axis.title.y = element_text(face = "bold", margin = margin(r = 15)),
  axis.text.y = element_text(face = "bold", color = "black", size = 12),
  legend.position = "bottom",
  legend.title = element_blank(),
  panel.grid.minor = element_blank(),
  panel.grid.major.y = element_blank(),
  plot.margin = margin(t = 10, r = 120, b = 10, l = 10) 
)

plot_out <- file.path(output_dir, "Targeted_CEI_Forest_Plot.png")
ggsave(plot_out, plot = p_forest, width = 10, height = max(6, nrow(res_all)*0.4), dpi = 300)
} else {
cat("⚠️ Warning\n")
}


cat("\n>>> Begin construct dose-response curve...\n")
all_dose_response <- list()

for (target_exposure in exposure_cols) {
  disease_name <- sub("^CEI_", "", target_exposure)
  target_outcome <- paste0("Y_", disease_name)
  
  if (!(target_outcome %in% names(df_year))) next
  cat(sprintf("Fitting: %s  =>  %s \n", target_exposure, target_outcome))
  
  df_clean <- df_year[!is.na(get(target_exposure)) & !is.na(get(target_outcome)) & C_POP > 0]
  
  if (nrow(df_clean) < 100) next
  zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
  if (zero_ratio > 0.3) next
  
  x_var <- df_clean[[target_exposure]]
  ref_val <- quantile(x_var, 0.05, na.rm = TRUE)
  min_val <- quantile(x_var, 0.01, na.rm = TRUE)
  max_val <- quantile(x_var, 0.90, na.rm = TRUE)
  x_seq <- seq(min_val, max_val, length.out = 100)

  b_basis <- ns(x_var, df = 3)
  
  df_clean[, c("spline_1", "spline_2", "spline_3") := as.data.table(b_basis)]
  formula_str <- paste0(target_outcome, " ~ spline_1 + spline_2 + spline_3 + ", cov_str, " + offset(log(C_POP)) | B_county_id + B_year")
  
  fit <- tryCatch({
    feglm(as.formula(formula_str), family = "quasipoisson", data = df_clean, nthreads = 4)
  }, error = function(e) return(NULL))
  
  if (!is.null(fit)) {
    spline_vars <- c("spline_1", "spline_2", "spline_3")
    coefs <- coef(fit)[spline_vars]
    vcov_mat <- vcov(fit, cluster = ~B_county_id)[spline_vars, spline_vars]
    pred_basis <- predict(b_basis, newx = x_seq)
    ref_basis <- predict(b_basis, newx = ref_val)
  
    diff_basis <- sweep(pred_basis, 2, ref_basis, "-")
    
    log_rr <- as.vector(diff_basis %*% coefs)
    se_log_rr <- sqrt(rowSums((diff_basis %*% vcov_mat) * diff_basis))
    
    plot_dt <- data.table(
      Disease = disease_name,
      Exposure_Value = x_seq,
      RR = exp(log_rr),
      RR_low = exp(log_rr - 1.96 * se_log_rr),
      RR_high = exp(log_rr + 1.96 * se_log_rr),
      Ref_Value = ref_val
    )
    
    all_dose_response[[length(all_dose_response) + 1]] <- plot_dt
  }
}

dt_all_curves <- rbindlist(all_dose_response)

csv_out <- file.path(output_dir, "Dose_Response_CEI_Results.csv")
fwrite(dt_all_curves, csv_out)


if (nrow(dt_all_curves) > 0) {
  
  p_curve <- ggplot(dt_all_curves, aes(x = Exposure_Value)) +
    geom_ribbon(aes(ymin = RR_low, ymax = RR_high), fill = "#cb181d", alpha = 0.2) +
    geom_line(aes(y = RR), color = "#cb181d", linewidth = 1.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
    facet_wrap(~ Disease, scales = "free") +
    labs(title = "Dose-Response Relationships of Disease-Specific CEI",
         subtitle = "Relative Risk (RR) with 95% CI (Reference at 5th percentile)",
         x = "Composite Exposure Index (CEI)",
         y = "Relative Risk (RR)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
      strip.background = element_rect(fill = "#f0f0f0", color = "black"),
      strip.text = element_text(face = "bold", size = 12),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
  
  plot_out <- file.path(output_dir, "Dose_Response_CEI_Curves.png")
  
  n_diseases <- length(unique(dt_all_curves$Disease))
  n_cols <- ceiling(sqrt(n_diseases))
  n_rows <- ceiling(n_diseases / n_cols)
  
  ggsave(plot_out, plot = p_curve, width = 4 * n_cols, height = 3.5 * n_rows, dpi = 300)
  
  
} else {
  cat("⚠️ Warning\n")
}





cat("\n>>> Begin construct dose-response curve based on IQR...\n")
all_dose_response <- list()

for (target_exposure in exposure_cols) {
  disease_name <- sub("^CEI_", "", target_exposure)
  target_outcome <- paste0("Y_", disease_name)
  
  if (!(target_outcome %in% names(df_year))) next
  cat(sprintf("Fitting: %s  =>  %s \n", target_exposure, target_outcome))
  
  df_clean <- df_year[!is.na(get(target_exposure)) & !is.na(get(target_outcome)) & C_POP > 0]
  
  if (nrow(df_clean) < 100) next
  zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
  if (zero_ratio > 0.3) next
  
  epd_iqr <- IQR(df_clean[[target_exposure]], na.rm = TRUE)
  if (epd_iqr == 0) next
  df_clean[, X_scaled := get(target_exposure) / epd_iqr]
  
  x_var <- df_clean$X_scaled
  
  ref_val <- quantile(x_var, 0.05, na.rm = TRUE)  # 5th percentile as reference (RR=1)
  min_val <- quantile(x_var, 0.01, na.rm = TRUE)  # 1st percentile as plot start
  max_val <- quantile(x_var, 0.95, na.rm = TRUE)  # 90th percentile as plot end
  x_seq <- seq(min_val, max_val, length.out = 100)
  
  inner_knots <- quantile(x_var, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)
  b_knots <- quantile(x_var, probs = c(0.01, 0.99), na.rm = TRUE) 
  
  b_basis <- ns(x_var, knots = inner_knots, Boundary.knots = b_knots)
  
  spline_cols <- paste0("spline_", 1:ncol(b_basis))
  df_clean[, (spline_cols) := as.data.table(b_basis)]
  
  spline_str <- paste(spline_cols, collapse = " + ")
  formula_str <- paste0(target_outcome, " ~ ", spline_str, " + ", cov_str, " + offset(log(C_POP)) | B_county_id + B_year")
  
  fit <- tryCatch({
    feglm(as.formula(formula_str), family = "quasipoisson", data = df_clean, nthreads = 4)
  }, error = function(e) return(NULL))
  
  if (!is.null(fit)) {

    coefs <- coef(fit)[spline_cols]
    vcov_mat <- vcov(fit, cluster = ~B_county_id)[spline_cols, spline_cols]
    
    pred_basis <- predict(b_basis, newx = x_seq)
    ref_basis <- predict(b_basis, newx = ref_val)
    
    diff_basis <- sweep(pred_basis, 2, ref_basis, "-")
    
    log_rr <- as.vector(diff_basis %*% coefs)
    se_log_rr <- sqrt(rowSums((diff_basis %*% vcov_mat) * diff_basis))
    
    plot_dt <- data.table(
      Disease = disease_name,
      Exposure_Value = x_seq,
      RR = exp(log_rr),
      RR_low = exp(log_rr - 1.96 * se_log_rr),
      RR_high = exp(log_rr + 1.96 * se_log_rr),
      Ref_Value = ref_val,
      IQR = epd_iqr
    )
    
    all_dose_response[[length(all_dose_response) + 1]] <- plot_dt
  }
}

dt_all_curves <- rbindlist(all_dose_response)

csv_out <- file.path(output_dir, "Dose_Response_CEI_IQR_Results.csv")
fwrite(dt_all_curves, csv_out)


if (nrow(dt_all_curves) > 0) {
  
  p_curve <- ggplot(dt_all_curves, aes(x = Exposure_Value)) +
    geom_ribbon(aes(ymin = RR_low, ymax = RR_high), fill = "#cb181d", alpha = 0.2) +
    geom_line(aes(y = RR), color = "#cb181d", linewidth = 1.2) +
  
    geom_hline(yintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
    
    facet_wrap(~ Disease, scales = "free_y") +
 
    labs(title = "Dose-Response Relationships of Disease-Specific CEI",
         subtitle = "Relative Risk (RR) with 95% CI (Reference at 5th percentile, X-axis in IQR units)",
         x = "Composite Exposure Index (IQR units)",
         y = "Relative Risk") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
      strip.background = element_rect(fill = "#f0f0f0", color = "black"),
      strip.text = element_text(face = "bold", size = 12),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
  
  plot_out <- file.path(output_dir, "Dose_Response_CEI_IQR_Curves.png")
  
  n_diseases <- length(unique(dt_all_curves$Disease))
  n_cols <- ceiling(sqrt(n_diseases))
  n_rows <- ceiling(n_diseases / n_cols)
  
  ggsave(plot_out, plot = p_curve, width = 4 * n_cols, height = 3.5 * n_rows, dpi = 300)
  
  
} else {
  cat("⚠️ Warning\n")
}






