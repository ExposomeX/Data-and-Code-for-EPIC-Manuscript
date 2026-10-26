library(data.table)
library(ggplot2)
library(fixest)
library(splines)

output_dir <- "code_2_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n>>> Stage 1: Begin EWAS...\n")
input_file <- "data_raw/df_urban_hospitalization_for_association.csv"
if (!file.exists(input_file)) stop("no file")

df_year <- fread(input_file)
df_year <- df_year[B_year >= 2017 & B_year <= 2023]
  
exposure_cols <- names(df_year)[grep("^X_", names(df_year))]
if(length(exposure_cols) == 0) stop("Error：no exposure variables found starting with 'X_'")
cat("Successful loading", length(exposure_cols), "exposure variables\n")

outcome_vars <- c("Y_All_cause", "Y_Blood", "Y_Cancer", "Y_Circulation", 
                  "Y_Congenital", "Y_Digestive", "Y_Endocrine", "Y_Genitourinary", 
                  "Y_Mental", "Y_Neural", "Y_Perinatal", "Y_Pregnancy", 
                  "Y_Respiration", "Y_Sensory", "Y_Skin")

covariate_vars <- c('C_BEDS_1000','C_GDP_PC','C_AGING','C_Temp','C_SHR_SECONDARY')
cov_str <- paste(covariate_vars, collapse = " + ")

urban_cols <- c('ratio0.5','ratio0.3','ratio0.7')
for (urban_col in urban_cols) {
  df_year[, Urban_Category := ifelse(df_year[[urban_col]], "Urban", "Rural")]
  strata_vars <- c("Urban" = "Urban", "Rural" = "Rural") 

  all_results_list <- list()

  for (target_exposure in exposure_cols) {
    cat("Processing: ", target_exposure, "\n")
    
    df_clean <- df_year[!is.na(get(target_exposure)) & C_POP > 0]
    
    zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
    if (zero_ratio > 0.3) next
    
    epd_iqr <- IQR(df_clean[[target_exposure]], na.rm = TRUE)
    if (epd_iqr == 0) next


    epd_sd <- sd(df_clean[[target_exposure]], na.rm = TRUE)
    if (is.na(epd_sd) || epd_sd == 0) next
    
    df_clean[, X_scaled := get(target_exposure) / epd_iqr]
    
    for (stratum_name in names(strata_vars)) {
      df_strat <- df_clean[Urban_Category == stratum_name] 
      
      for (oc in outcome_vars) {
        if (sum(df_strat[[oc]], na.rm = TRUE) < 50) next 
        formula_str <- paste0(oc, " ~ X_scaled + ", cov_str, " + offset(log(C_POP)) | B_county_id + B_year")
        
        fit <- tryCatch({
          feglm(as.formula(formula_str), family = "poisson", data = df_strat, nthreads = 4)
        }, error = function(e) return(NULL))
        
        if (!is.null(fit)) {
          coef_summary <- coeftable(fit, cluster = ~B_county_id)
          
          if ("X_scaled" %in% rownames(coef_summary)) {
            coef_val <- coef_summary["X_scaled", "Estimate"]
            se_val <- coef_summary["X_scaled", "Std. Error"]
            p_val <- coef_summary["X_scaled", 4]
            
            if (se_val < 1.0 && abs(coef_val) < 5) { 
              all_results_list[[length(all_results_list) + 1]] <- data.table(
                Outcome = oc,
                Exposure = target_exposure,
                Stratum = stratum_name, 
                RR = exp(coef_val),
                RR_low = exp(coef_val - 1.96 * se_val),
                RR_high = exp(coef_val + 1.96 * se_val),
                LogRR = coef_val,
                SE = se_val,         
                P_value = p_val,
                Samples = nrow(df_strat)
              )
            }
          }
        }
      }
    }
  }

  res_all <- rbindlist(all_results_list)

  res_wide <- dcast(
    res_all, 
    Exposure + Outcome ~ Stratum, 
    value.var = c("RR", "RR_low", "RR_high", "LogRR", "SE",'P_value')
  )
  res_wide[, Label := gsub("_", " ", gsub("^Y_", "", Outcome))]
  res_wide <- na.omit(res_wide)

  exposure_cols_for_adjust <- unique(res_wide$Exposure)
 
  cat("\n>>> Performing effective independent test (Me) correction...\n")

  exposure_matrix <- df_year[, ..exposure_cols_for_adjust]
  cor_matrix <- cor(exposure_matrix, use = "pairwise.complete.obs")

  eigen_vals <- eigen(cor_matrix)$values
  M_total <- length(exposure_cols)
  lambda_greater_than_1 <- eigen_vals[eigen_vals > 1]

  Me_exposure <- M_total - sum(lambda_greater_than_1 - 1)

  Me_global <- Me_exposure * length(outcome_vars)
  Me_global <- Me_exposure
  alpha_threshold <- 0.05 / Me_global

  res_wide[, Significance_adjusted_Rural:= ifelse(P_value_Rural < alpha_threshold & RR_Rural > 1, "Significant Harm",
                            ifelse(P_value_Rural < alpha_threshold & RR_Rural < 1, "Significant Protect", "Not Significant"))]

  cat(">>> Plotting volcano plot...\n")
  p_volcano <- ggplot(res_wide, aes(x = LogRR_Rural, y = -log10(P_value_Rural), color = Significance_adjusted_Rural)) +
    geom_point(alpha = 0.6, size = 1.2) +
    # 注意：火山图的基准线也自动更新为你的新阈值
    geom_hline(yintercept = -log10(alpha_threshold), linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant Harm" = "#d73027", 
                                  "Significant Protect" = "#4575b4", 
                                  "Not Significant" = "grey80")) +
    facet_wrap(~ Label, scales = "free", ncol = 5) +
    labs(title = "Environment-Wide Association Study (EWAS) Volcano Plot",
        subtitle = sprintf("Significance Threshold based on Effective Number of Tests (Me = %.1f, P < %.2e)", Me_global, alpha_threshold),
        x = "Effect Size (Log Relative Risk)", y = "-log10(P-value)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  fig_path <- paste0("results_association_hospital/fixest/urban_full/01_EWAS_Volcano_Plot_Me_Corrected_Rural_",urban_col,'.png')
  ggsave(fig_path, plot = p_volcano, width = 16, height = 10, dpi = 300)

  # 定义显著性标签
  res_wide[, Significance_raw_Rural:= ifelse(P_value_Rural < 0.05 & RR_Rural > 1, "Significant Harm",
                            ifelse(P_value_Rural < 0.05 & RR_Rural < 1, "Significant Protect", "Not Significant"))]

  p_volcano <- ggplot(res_wide, aes(x = LogRR_Rural, y = -log10(P_value_Rural), color = Significance_raw_Rural)) +
    geom_point(alpha = 0.6, size = 1.2) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant Harm" = "#d73027", 
                                  "Significant Protect" = "#4575b4", 
                                  "Not Significant" = "grey80")) +
    facet_wrap(~ Label, scales = "free", ncol = 5) +
    labs(title = "Environment-Wide Association Study (EWAS) Volcano Plot",
        subtitle = "Raw P-Values",
        x = "Effect Size (Log Relative Risk)", y = "-log10(P-value)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  fig_path <- paste0("results_association_hospital/fixest/urban_full/01_EWAS_Volcano_Plot_Raw_Rural_",urban_col,'.png')
  ggsave(fig_path, plot = p_volcano, width = 16, height = 10, dpi = 300)

  res_wide[, Significance_adjusted_Urban:= ifelse(P_value_Urban < alpha_threshold & RR_Urban > 1, "Significant Harm",
                            ifelse(P_value_Urban < alpha_threshold & RR_Urban < 1, "Significant Protect", "Not Significant"))]


  p_volcano <- ggplot(res_wide, aes(x = LogRR_Urban, y = -log10(P_value_Urban), color = Significance_adjusted_Urban)) +
    geom_point(alpha = 0.6, size = 1.2) +
    # 注意：火山图的基准线也自动更新为你的新阈值
    geom_hline(yintercept = -log10(alpha_threshold), linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant Harm" = "#d73027", 
                                  "Significant Protect" = "#4575b4", 
                                  "Not Significant" = "grey80")) +
    facet_wrap(~ Label, scales = "free", ncol = 5) +
    labs(title = "Environment-Wide Association Study (EWAS) Volcano Plot",
        subtitle = sprintf("Significance Threshold based on Effective Number of Tests (Me = %.1f, P < %.2e)", Me_global, alpha_threshold),
        x = "Effect Size (Log Relative Risk)", y = "-log10(P-value)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  fig_path <- paste0("results_association_hospital/fixest/urban_full/01_EWAS_Volcano_Plot_Me_Corrected_Urban_",urban_col,'.png')
  ggsave(fig_path, plot = p_volcano, width = 16, height = 10, dpi = 300)

  res_wide[, Significance_raw_Urban:= ifelse(P_value_Urban < 0.05 & RR_Urban > 1, "Significant Harm",
                            ifelse(P_value_Urban < 0.05 & RR_Urban < 1, "Significant Protect", "Not Significant"))]

  p_volcano <- ggplot(res_wide, aes(x = LogRR_Urban, y = -log10(P_value_Urban), color = Significance_raw_Urban)) +
    geom_point(alpha = 0.6, size = 1.2) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant Harm" = "#d73027", 
                                  "Significant Protect" = "#4575b4", 
                                  "Not Significant" = "grey80")) +
    facet_wrap(~ Label, scales = "free", ncol = 5) +
    labs(title = "Environment-Wide Association Study (EWAS) Volcano Plot",
        subtitle = "Raw P-Values",
        x = "Effect Size (Log Relative Risk)", y = "-log10(P-value)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  fig_path <- paste0("results_association_hospital/fixest/urban_full/01_EWAS_Volcano_Plot_Raw_Urban_",urban_col,'.png')
  ggsave(fig_path, plot = p_volcano, width = 16, height = 10, dpi = 300)

