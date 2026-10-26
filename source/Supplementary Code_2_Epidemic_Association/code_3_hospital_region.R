library(data.table)
library(ggplot2)
library(fixest)
library(splines)

output_dir <- "code_3_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n>>> Stage 1: Begin EWAS...\n")

input_file <- "data_raw/df_region_hospitalization_for_association.csv"
if (!file.exists(input_file)) stop("no file: ", input_file)

df_year <- fread(input_file)
df_year <- df_year[B_year >= 2017 & B_year <= 2023]

exposure_cols <- names(df_year)[grep("^X_", names(df_year))]
if(length(exposure_cols) == 0) stop("Error：no exposure variables found starting with 'X_'")
cat(length(exposure_cols))

outcome_vars <- c("Y_All_cause", "Y_Blood", "Y_Cancer", "Y_Circulation", 
                  "Y_Congenital", "Y_Digestive", "Y_Endocrine", "Y_Genitourinary", 
                  "Y_Mental", "Y_Neural", "Y_Perinatal", "Y_Pregnancy", 
                  "Y_Respiration", "Y_Sensory", "Y_Skin")

covariate_vars <- c('C_BEDS_1000','C_GDP_PC','C_AGING','C_Temp','C_SHR_SECONDARY')
cov_str <- paste(covariate_vars, collapse = " + ")

strata_vars <- c("North", "South", "Plateau") 
all_results_list <- list()

for (target_exposure in exposure_cols) {
  cat("Processing: ", target_exposure, "\r")
  
  df_clean <- df_year[!is.na(get(target_exposure)) & C_POP > 0 & !is.na(Region)]
  
  zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
  if (zero_ratio > 0.3) next
  
  epd_iqr <- IQR(df_clean[[target_exposure]], na.rm = TRUE)
  epd_sd <- sd(df_clean[[target_exposure]], na.rm = TRUE)
  if (is.na(epd_sd) || epd_sd == 0 || epd_iqr == 0) next
  
  df_clean[, X_scaled := get(target_exposure) / epd_iqr]
  
  for (stratum_name in strata_vars) {
    df_strat <- df_clean[Region == stratum_name] 
    if (nrow(df_strat) < 100) next 
    
    for (oc in outcome_vars) {
    
      if (sum(df_strat[[oc]], na.rm = TRUE) < 300) next 
      
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
          
          if (se_val < 1.0 && abs(coef_val) < 2) { 
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

exposure_cols_for_adjust <- unique(res_all$Exposure)

exposure_matrix <- df_year[, ..exposure_cols_for_adjust]
cor_matrix <- cor(exposure_matrix, use = "pairwise.complete.obs")
eigen_vals <- eigen(cor_matrix)$values
lambda_greater_than_1 <- eigen_vals[eigen_vals > 1]
Me_exposure <- length(exposure_cols_for_adjust) - sum(lambda_greater_than_1 - 1)
Me_global <- Me_exposure 
alpha_threshold <- 0.05 / Me_global

res_all[, Significance_adjusted := ifelse(P_value < alpha_threshold & RR > 1, "Significant Harm",
                                   ifelse(P_value < alpha_threshold & RR < 1, "Significant Protect", "Not Significant"))]
res_all[, Significance_raw := ifelse(P_value < 0.05 & RR > 1, "Significant Harm",
                              ifelse(P_value < 0.05 & RR < 1, "Significant Protect", "Not Significant"))]

res_all[, Label := gsub("_", " ", gsub("^Y_", "", Outcome))]

for (reg in strata_vars) {
  plot_data <- res_all[Stratum == reg]
  if(nrow(plot_data) == 0) next
  
  p_volcano <- ggplot(plot_data, aes(x = LogRR, y = -log10(P_value), color = Significance_adjusted)) +
    geom_point(alpha = 0.6, size = 1.2) +
    geom_hline(yintercept = -log10(alpha_threshold), linetype = "dashed", color = "black") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant Harm" = "#d73027", "Significant Protect" = "#4575b4", "Not Significant" = "grey80")) +
    facet_wrap(~ Label, scales = "free", ncol = 5) +
    labs(title = paste("EWAS Volcano Plot -", reg, "Region"),
         subtitle = sprintf("Threshold based on Me (Me = %.1f, P < %.2e)", Me_global, alpha_threshold),
         x = "Effect Size (Log Relative Risk)", y = "-log10(P-value)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  
  ggsave(file.path(output_dir, paste0("01_EWAS_Volcano_Plot_Me_Corrected_", reg, ".png")), plot = p_volcano, width = 16, height = 10, dpi = 300)
}

res_wide <- dcast(res_all, Exposure + Outcome + Label ~ Stratum, 
                  value.var = c("RR", "RR_low", "RR_high", "LogRR", "SE", "P_value", "Significance_adjusted","Significance_raw"))

res_wide[, Z_diff_NS := (LogRR_North - LogRR_South) / sqrt(SE_North^2 + SE_South^2)]
res_wide[, P_diff_NS := 2 * (1 - pnorm(abs(Z_diff_NS)))]
res_wide[, Abs_Z_NS := abs(Z_diff_NS)]

top_diff_combos <- head(res_wide[!is.na(Z_diff_NS)][order(-Abs_Z_NS)], 10)

plot_forest_data <- res_all[paste(Exposure, Outcome) %in% paste(top_diff_combos$Exposure, top_diff_combos$Outcome) & Stratum %in% c("North", "South")]
plot_forest_data[, Combo_Label := paste0(gsub("^X_", "", Exposure), " - ", Label)]

order_mapping <- top_diff_combos[, .(Combo = paste0(gsub("^X_", "", Exposure), " - ", Label), Abs_Z_NS)]
plot_forest_data[, Combo_Label := factor(Combo_Label, levels = order_mapping[order(Abs_Z_NS), Combo])]

p_forest <- ggplot(plot_forest_data, aes(x = Combo_Label, y = RR, color = Stratum)) +
  geom_point(position = position_dodge(width = 0.5), size = 3.5) +
  geom_errorbar(aes(ymin = RR_low, ymax = RR_high), width = 0.2, position = position_dodge(width = 0.5)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.8) +
  coord_flip() +
  scale_color_manual(values = c("North" = "#d73027", "South" = "#4575b4")) +
  labs(title = "Top 10 Most Significant North-South Differences",
       subtitle = "Ranked by absolute Interaction Z-score", x = "", y = "Relative Risk (RR)", color = "Region") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", panel.grid.major.y = element_blank(), plot.title = element_text(face = "bold", size = 14))

ggsave(file.path(output_dir, "02_Top10_North_South_Diff_ForestPlot.png"), p_forest, width = 12, height = 8, dpi = 300)
fwrite(res_wide, file = file.path(output_dir, "EWAS_Results_Wide_Significance.csv"), row.names = FALSE)

