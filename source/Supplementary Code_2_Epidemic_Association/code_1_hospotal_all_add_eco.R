library(data.table)
library(ggplot2)
library(fixest)

out_dir <- Sys.getenv("EWAS_OUTPUT_DIR", unset = "code_1_results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n>>> loading data...\n")
input_file <- Sys.getenv(
  "EWAS_INPUT_FILE",
  unset = "demo_data/df_hospitalization_for_association_demo.csv"
)
if (!file.exists(input_file)) stop("File not found: ", input_file)

df_year <- fread(input_file)
df_year <- df_year[B_year >= 2017 & B_year <= 2023]

exposure_cols <- names(df_year)[grep("^X_", names(df_year))]
if(length(exposure_cols) == 0) stop("Error：no exposure variables found starting with 'X_'")
cat(length(exposure_cols))

outcome_vars <- c("Y_All_cause", "Y_Blood", "Y_Cancer", "Y_Circulation", 
                  "Y_Congenital", "Y_Digestive", "Y_Endocrine", "Y_Genitourinary", 
                  "Y_Mental", "Y_Neural", "Y_Perinatal", "Y_Pregnancy", 
                  "Y_Respiration", "Y_Sensory", "Y_Skin")
cat(length(outcome_vars), "success outcome variables loaded.\n")

model_formulas <- list(
  "M1_Base"       = "C_Temp",                                                 
  "M2_SocioEcon"  = "C_Temp + C_GDP_PC + C_SHR_SECONDARY",                   
  "M3_Demographic"= "C_Temp + C_GDP_PC + C_SHR_SECONDARY + C_AGING",          
  "M4_Full"       = "C_Temp + C_GDP_PC + C_SHR_SECONDARY + C_AGING + C_BEDS_1000" 
)

all_covariates <- c("C_Temp", "C_GDP_PC", "C_SHR_SECONDARY", "C_AGING", "C_BEDS_1000")

all_results_list <- list()
total_models <- length(exposure_cols) * length(outcome_vars) * length(model_formulas)
current_model <- 0

for (target_exposure in exposure_cols) {
  cat(sprintf("Processing exposure: %s \r", target_exposure))
  
  cols_to_check <- c(target_exposure, outcome_vars, all_covariates, "C_POP")
  existing_cols <- intersect(cols_to_check, names(df_year))
  
  df_clean <- df_year[complete.cases(df_year[, ..existing_cols])]
  df_clean <- df_clean[C_POP > 0]
  
  if (nrow(df_clean) < 100) next
  
  zero_ratio <- sum(df_clean[[target_exposure]] == 0, na.rm = TRUE) / nrow(df_clean)
  if (zero_ratio > 0.3) next
  
  epd_iqr <- IQR(df_clean[[target_exposure]], na.rm = TRUE)
  if (epd_iqr == 0) next
  
  df_clean[, X_scaled := get(target_exposure) / epd_iqr]
  
  for (oc in outcome_vars) {
    if (sum(df_clean[[oc]] > 0, na.rm = TRUE) < (nrow(df_clean) * 0.1)) {
      current_model <- current_model + length(model_formulas)
      next
    }
    
    for (mod_name in names(model_formulas)) {
      current_model <- current_model + 1
      cov_str <- model_formulas[[mod_name]]
      formula_str <- paste0(oc, " ~ X_scaled + ", cov_str, " + offset(log(C_POP)) | B_county_id + B_year")
      
      fit <- tryCatch({
        feglm(as.formula(formula_str), family = "poisson", data = df_clean, nthreads = 4)
      }, error = function(e) return(NULL))
      
      if (!is.null(fit)) {
        coef_summary <- coeftable(fit, cluster = ~B_county_id)
        if ("X_scaled" %in% rownames(coef_summary)) {
          coef_val <- coef_summary["X_scaled", "Estimate"]
          se_val   <- coef_summary["X_scaled", "Std. Error"]
          p_val    <- coef_summary["X_scaled", 4]
          
          if (se_val < 5 && abs(coef_val) < 5) {
            all_results_list[[length(all_results_list) + 1]] <- data.table(
              Model_Type = mod_name,
              Outcome    = oc,
              Exposure   = target_exposure,
              RR         = exp(coef_val),
              RR_low     = exp(coef_val - 1.96 * se_val),
              RR_high    = exp(coef_val + 1.96 * se_val),
              LogRR      = coef_val,
              P_value    = p_val,
              Samples    = nrow(df_clean)
            )
          }
        }
      }
    }
  }
}

res_all <- rbindlist(all_results_list)
res_all[, Label := gsub("_", " ", gsub("^Y_", "", Outcome))]
cat("\nScan complete, generated", nrow(res_all), "sensitivity regression results.\n")

cat("\n>>> Performing effective independent test (Me) correction...\n")
exposure_cols_for_adjust <- unique(res_all$Exposure)
exposure_matrix <- df_year[, ..exposure_cols_for_adjust]
cor_matrix <- cor(exposure_matrix, use = "pairwise.complete.obs")

cor_matrix[is.na(cor_matrix)] <- 0 
eigen_vals <- eigen(cor_matrix)$values
lambda_greater_than_1 <- eigen_vals[eigen_vals > 1]

Me_exposure <- length(exposure_cols_for_adjust) - sum(lambda_greater_than_1 - 1)
Me_global <- Me_exposure 
alpha_threshold <- 0.05 / Me_global

cat(sprintf(" - Global effective independent test count (Me_global): %.2f\n", Me_global))
cat(sprintf(" - Final significance P-value threshold: %.2e\n", alpha_threshold))

res_all[, Significance_adjusted := ifelse(P_value < alpha_threshold & RR > 1, "Significant Harm",
                                   ifelse(P_value < alpha_threshold & RR < 1, "Significant Protect", "Not Significant"))]

res_all[, Significance_raw := ifelse(P_value < 0.05 & RR > 1, "Significant Harm",
                                   ifelse(P_value < 0.05 & RR < 1, "Significant Protect", "Not Significant"))]


output_file <- file.path(out_dir, "EWAS_Sensitivity_Results_All.csv")
fwrite(res_all, file = output_file, row.names = FALSE)
cat(">>> All sensitivity results exported to:", output_file, "\n")






res_all <- fread(file.path(out_dir, "EWAS_Sensitivity_Results_All.csv"))
cat("\n>>> Plotting sensitivity analysis forest plot...\n")
top_exposures <- res_all[Model_Type == "M4_Full" & Significance_adjusted == "Significant Harm"][order(-RR)]

if(nrow(top_exposures) > 25) {
  sig_pairs <- head(top_exposures[, .(Exposure, Outcome)], 25)
} else {
  sig_pairs <- top_exposures[, .(Exposure, Outcome)]
}

if(nrow(sig_pairs) > 0) {
  plot_data <- merge(res_all, sig_pairs, by = c("Exposure", "Outcome"))
  plot_data[, Combo_Label := paste0(gsub("^X_", "", Exposure), " - ", Label)]
  
  order_mapping <- plot_data[Model_Type == "M4_Full"][order(RR), .(Combo_Label)]
  plot_data[, Combo_Label := factor(Combo_Label, levels = order_mapping$Combo_Label)]
  
  plot_data[, Model_Type := factor(Model_Type, levels = c("M1_Base", "M2_SocioEcon", "M3_Demographic", "M4_Full"))]
  
  model_colors <- c("M1_Base" = "#fee5d9", "M2_SocioEcon" = "#fcae91", "M3_Demographic" = "#fb6a4a", "M4_Full" = "#cb181d")
  
  p_sens <- ggplot(plot_data, aes(x = Combo_Label, y = RR, color = Model_Type)) +
    geom_point(position = position_dodge(width = 0.75), size = 2.5) +
    geom_errorbar(aes(ymin = RR_low, ymax = RR_high), width = 0.4, position = position_dodge(width = 0.75), linewidth = 0.7) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.8) +
    coord_flip() +
    scale_color_manual(values = model_colors) +
    labs(title = "Sensitivity Analysis: Stepwise Covariate Adjustment",
         subtitle = "Tracking RR stability across Base, Socioeconomic, Demographic, and Full Resource models",
         x = "Exposure - Disease Outcome", y = "Relative Risk (RR)") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.y = element_text(face = "bold", size = 10))
  
  ggsave(file.path(out_dir, "01_Sensitivity_ForestPlot_Stepwise.png"), plot = p_sens, width = 12, height = max(8, nrow(sig_pairs)*0.4), dpi = 300)
  cat("Sensitivity forest plot completed! Saved as 01_Sensitivity_ForestPlot_Stepwise.png\n")
  
  robust_check <- plot_data[P_value < alpha_threshold & RR > 1, .N, by = .(Combo_Label)]
  robust_targets <- robust_check[N == 4, Combo_Label] 
  
  cat("\n🏆 Found", length(robust_targets), "absolutely robust target-outcome pairs:\n")
  if (length(robust_targets) > 0) print(robust_targets)
  
} else {
  cat("\nNo significant harmful targets found in the main model, cannot plot forest plot.\n")
}







cat("\n>>> Plotting sensitivity analysis forest plot...\n")
top_exposures <- res_all[Model_Type == "M4_Full" & Significance_raw == "Significant Harm"][order(-RR)]

if(nrow(top_exposures) > 25) {
  sig_pairs <- head(top_exposures[, .(Exposure, Outcome)], 25)
} else {
  sig_pairs <- top_exposures[, .(Exposure, Outcome)]
}

if(nrow(sig_pairs) > 0) {
 
  plot_data <- merge(res_all, sig_pairs, by = c("Exposure", "Outcome"))
  plot_data[, Combo_Label := paste0(gsub("^X_", "", Exposure), " - ", Label)]
  
  order_mapping <- plot_data[Model_Type == "M4_Full"][order(RR), .(Combo_Label)]
  plot_data[, Combo_Label := factor(Combo_Label, levels = order_mapping$Combo_Label)]
  
  plot_data[, Model_Type := factor(Model_Type, levels = c("M1_Base", "M2_SocioEcon", "M3_Demographic", "M4_Full"))]
  model_colors <- c("M1_Base" = "#fee5d9", "M2_SocioEcon" = "#fcae91", "M3_Demographic" = "#fb6a4a", "M4_Full" = "#cb181d")
  
  p_sens <- ggplot(plot_data, aes(x = Combo_Label, y = RR, color = Model_Type)) +
    geom_point(position = position_dodge(width = 0.75), size = 2.5) +
    geom_errorbar(aes(ymin = RR_low, ymax = RR_high), width = 0.4, position = position_dodge(width = 0.75), linewidth = 0.7) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.8) +
    coord_flip() +
    scale_color_manual(values = model_colors) +
    labs(title = "Sensitivity Analysis: Stepwise Covariate Adjustment",
         subtitle = "Tracking RR stability across Base, Socioeconomic, Demographic, and Full Resource models",
         x = "Exposure - Disease Outcome", y = "Relative Risk (RR)") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          panel.grid.minor = element_blank(),
          axis.text.y = element_text(face = "bold", size = 10))
  
  ggsave(file.path(out_dir, "01_Sensitivity_ForestPlot_Stepwise_raw.png"), plot = p_sens, width = 12, height = max(8, nrow(sig_pairs)*0.4), dpi = 300)
  cat("Sensitivity forest plot completed! Saved as 01_Sensitivity_ForestPlot_Stepwise_raw.png\n")
  
  robust_targets <- plot_data[Model_Type == "M4_Full" & Significance_raw == "Significant Harm", Combo_Label]

  cat("\n🏆 Found", length(robust_targets), "absolutely robust target-outcome pairs:\n")
if (length(robust_targets) > 0) print(robust_targets}





cat(">>> Plotting global RR distribution violin plots...\n")
plot_data <- res_compare[Model_Type %in% c("M1_Base", "M2_SocioEcon", "M3_Demographic", "M4_Full")]
plot_data[, Model_Type := factor(Model_Type, levels = c("M1_Base", "M2_SocioEcon", "M3_Demographic", "M4_Full"))]
plot_data[, Combo_Label := paste0(gsub("^X_", "", Exposure), " - ", Label)]

p_violin <- ggplot(plot_data, aes(x = Model_Type, y = RR, fill = Model_Type)) +
  geom_violin(alpha = 0.5, trim = FALSE) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_manual(values = c("M1_Base" = "#fee5d9", "M2_SocioEcon" = "#fcae91", 
                               "M3_Demographic" = "#fb6a4a", "M4_Full" = "#cb181d")) +
  labs(title = "Global Distribution of Relative Risks Across Adjustment Steps",
       subtitle = sprintf("Overall median RR shifts toward the null hypothesis (RR=1)"),
       x = "Model Specification", y = "Relative Risk (RR)") +
  theme_bw(base_size = 14) +
  theme(legend.position = "none",
        axis.text.x = element_text(face = "bold", size = 11))

ggsave(file.path(out_dir, "02_RR_Global_Shift_Violin.png"), plot = p_violin, width = 10, height = 7, dpi = 300)






cat(">>> Plotting violin plots by disease outcome...\n")

p_violin_by_disease <- ggplot(plot_data, aes(x = Model_Type, y = RR, fill = Model_Type)) +
  geom_violin(alpha = 0.6, trim = FALSE, scale = "width") + 
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.8) + 
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.8) +
  
  scale_fill_manual(values = c("M1_Base" = "#fee5d9", "M2_SocioEcon" = "#fcae91", 
                               "M3_Demographic" = "#fb6a4a", "M4_Full" = "#cb181d")) +
  
  facet_wrap(~ Outcome, scales = "free_y", ncol = 3) + 
  
  labs(title = "Distribution of Relative Risks by Disease Outcome",
       subtitle = "Comparison across adjustment models (Base to Full)",
       x = "Model Specification", 
       y = "Relative Risk (RR)") +
  
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom",    
        strip.text = element_text(face = "bold", size = 12), 
        axis.text.x = element_text(angle = 45, hjust = 1))  

ggsave(file.path(out_dir, "02_RR_By_Disease_Violin.png"), 
       plot = p_violin_by_disease, 
       width = 14, height = 10, dpi = 300)

cat(">>> Violin plots by disease outcome completed!\n")






library(data.table)
library(ggplot2)

res_all <- fread(file.path(out_dir, "EWAS_Sensitivity_Results_All.csv"))

if (!file.exists(input_file)) {
  stop("no input file found. Please ensure you have run the full sensitivity analysis code!")
}

res_all <- fread(input_file)
res_m4 <- res_all[Model_Type == "M4_Full" & Significance_raw == "Significant Harm"]

res_m4[, Clean_Exposure := gsub("^X_", "", Exposure)]

res_m4[, Pollutant := sapply(strsplit(Clean_Exposure, "_"), `[`, 1)]
res_m4[, Sector := sapply(strsplit(Clean_Exposure, "_"), function(x) {
  if(length(x) > 1) paste(x[-1], collapse="_") else x[1]
})]

if(nrow(res_m4) == 0) {
  stop("No significant harmful targets found in the M4_Full model, cannot generate plots!")
}

setorder(res_m4, Label, Sector, -RR)
top3_per_disease <- res_m4[, head(.SD, 1), by = .(Label, Sector)]
top3_per_disease[, Estimate_Text := sprintf("%.4f (%.4f - %.4f)", RR, RR_low, RR_high)]
top3_per_disease[, Display_Text := Estimate_Text]
top3_per_disease[, Y_Label := paste0(Pollutant, "  [", Sector, "]")]

top3_per_disease[, Plot_ID := paste(Label, Y_Label, sep = "|||")]

setorder(top3_per_disease, Label, RR) 
top3_per_disease[, Plot_ID := factor(Plot_ID, levels = Plot_ID)]

cat(sprintf("\n>>> Successfully extracted %d core targets by disease category, preparing to plot %d forest plot segments.\n",
            length(unique(top3_per_disease$Label)), nrow(top3_per_disease)))

nature_colors <- c("#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148")


plot_x_min <- min(c(top3_per_disease$RR_low, 1.0), na.rm = TRUE)
plot_x_max <- max(top3_per_disease$RR_high, na.rm = TRUE)
x_span <- plot_x_max - plot_x_min

axis_limit_min <- plot_x_min - x_span * 0.05
axis_limit_max <- plot_x_max + x_span * 0.05

text_start_x <- axis_limit_max + x_span * 0.08

cat(">>> Plotting advanced faceted forest plots...\n")

p_forest <- ggplot(top3_per_disease, aes(x = RR, y = Plot_ID, color = Sector)) +
  geom_errorbar(aes(xmin = RR_low, xmax = RR_high), width = 0.2, linewidth = 0.8) +
  geom_point(size = 3.5) +
  geom_text(aes(x = text_start_x, label = Display_Text), 
            hjust = 0, size = 3.8, color = "black", fontface = "plain") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.8) +
  facet_grid(Label ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_y_discrete(labels = function(x) gsub("^.*\\|\\|\\|", "", x)) +
  coord_cartesian(xlim = c(axis_limit_min, axis_limit_max), clip = "off") +
  
  scale_color_manual(values = nature_colors) +
  
  labs(title = "Top 3 High-Risk Exposures by Sector and Outcome",
       subtitle = "Ranked by Relative Risk. Point estimates are shown outside the panel.",
       x = "Relative Risk (RR, 95% CI)", 
       y = "",
       color = "Industrial Sector (Department)") +
  
  theme_minimal(base_size = 13, base_family = "sans") +
  theme(
    plot.title = element_text( size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "grey40"),
    
    legend.position = "top",
    legend.title = element_text( size = 12),
    legend.text = element_text(size = 11),
    
    axis.title.x = element_text( margin = margin(t = 12)),
    axis.text.x = element_text(color = "black", size = 11),
    axis.text.y = element_text( color = "black", size = 11),
    
    strip.background = element_blank(),
    strip.text.y.left = element_text(angle = 0,  size = 12, color = "black"),
    strip.placement = "outside", 
    
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.4),
    

    panel.border = element_rect(color = "grey50", fill = NA, linewidth = 0.8),
    panel.spacing.y = unit(0.1, "lines"), 
    
    plot.margin = margin(1, 5.5, 1, 0.5, "cm")
  )

plot_width <- 8.5 
plot_height <- max(7.5, nrow(top3_per_disease) * 0.3)

save_path_png <- file.path(out_dir, "05_Top3_Targets_ForestPlot_M4_Annotated.png")
ggsave(save_path_png, plot = p_forest, width = plot_width, height = plot_height, dpi = 300)

cat(sprintf("The advanced layout with floating text outside the plot area has been generated, and the image has been saved to: %s\n", save_path_png))
