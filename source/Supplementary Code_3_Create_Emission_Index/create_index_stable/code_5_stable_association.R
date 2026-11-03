library(data.table)
library(ggplot2)
library(fixest)
library(splines)

# Stable-subset CEI association and dose-response analysis.
# This is an isolated adaptation of the original code_5 workflow.
stable_root <- "D:/emi_association/Supplementary Code_3_Create_Emission_Index/create_index_stable"
original_root <- "D:/emi_association/Supplementary Code_3_Create_Emission_Index/create_index"
input_panel <- file.path(original_root, "code_4_results/2_merged_yearly_data_new.csv")
input_grid <- file.path(stable_root, "results/tables/county_mean_emissions_for_CEI.csv")
input_scale <- file.path(stable_root, "results/tables/grid_standardization_by_fold.csv")
input_targets <- file.path(stable_root, "results/tables/Stable_priority_weights_for_forecast.csv")
output_dir <- file.path(stable_root, "results/code_5_stable_results")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!all(file.exists(c(input_panel, input_grid, input_scale, input_targets)))) {
  stop("Required stable CEI input files are missing; run run_stable_analysis.py first.")
}

panel <- fread(input_panel)[B_year >= 2017 & B_year <= 2023]
grid <- fread(input_grid)
scaler <- fread(input_scale)[fold == 0]
targets <- fread(input_targets)
diseases <- c("Blood", "Cancer", "Circulation", "Skin")
covariates <- c("C_BEDS_1000", "C_GDP_PC", "C_AGING", "C_Temp", "C_SHR_SECONDARY")

# Recreate historical stable CEIs from the selected original priority weights,
# using the same full-history grid means/SDs as the stable-index workflow.
for (disease in diseases) {
  rows <- targets[Disease_Name == disease]
  if (!nrow(rows)) stop("No stable target rows for ", disease)
  weight_by_exposure <- list()
  for (i in seq_len(nrow(rows))) {
    exps <- strsplit(rows$Exposure[i], ",\\s*(?=X_)", perl = TRUE)[[1]]
    for (exposure in exps) {
      exposure <- trimws(exposure)
      weight <- as.numeric(rows$Priority_Index[i])
      if (is.null(weight_by_exposure[[exposure]]) || weight > weight_by_exposure[[exposure]]) {
        weight_by_exposure[[exposure]] <- weight
      }
    }
  }
  cei <- rep(0, nrow(grid))
  for (exposure in names(weight_by_exposure)) {
    s <- scaler[Exposure == exposure]
    if (nrow(s) != 1 || !(exposure %in% names(grid))) stop("Missing scaler/grid exposure: ", exposure)
    z <- (grid[[exposure]] - s$mean[1]) / s$std[1]
    z[!is.finite(z)] <- 0
    cei <- cei + z * weight_by_exposure[[exposure]]
  }
  grid[, (paste0("CEI_", disease)) := cei]
}

# The source panel already contains the original CEI columns. Remove those columns
# before merging so data.table does not rename the stable CEIs to .x/.y.
old_cei_cols <- intersect(paste0("CEI_", diseases), names(panel))
if (length(old_cei_cols)) panel[, (old_cei_cols) := NULL]
panel <- merge(panel, grid[, c("B_county_id", "B_year", paste0("CEI_", diseases)), with = FALSE],
               by = c("B_county_id", "B_year"), all.x = TRUE, sort = FALSE)
all_curves <- list()
all_models <- list()

for (disease in diseases) {
  exposure <- paste0("CEI_", disease)
  outcome <- paste0("Y_", disease)
  required <- c(exposure, outcome, "C_POP", covariates, "B_county_id", "B_year")
  missing_cols <- setdiff(required, names(panel))
  if (length(missing_cols)) stop("Missing model columns for ", disease, ": ", paste(missing_cols, collapse = ", "))
  d <- panel[is.finite(get(exposure)) & is.finite(get(outcome)) &
               is.finite(C_POP) & C_POP > 0]
  d <- d[complete.cases(d[, ..covariates])]
  if (nrow(d) < 100) stop("Fewer than 100 complete county-years for ", disease)
  if (sum(d[[outcome]]) < 50) stop("Fewer than 50 hospitalizations for ", disease)
  zero_ratio <- mean(d[[exposure]] == 0)
  if (zero_ratio > 0.30) stop("CEI zero fraction exceeds 30% for ", disease)

  iqr <- IQR(d[[exposure]])
  if (!is.finite(iqr) || iqr <= 0) stop("Invalid CEI IQR for ", disease)
  d[, X_scaled := get(exposure) / iqr]
  x <- d$X_scaled
  ref <- as.numeric(quantile(x, 0.05, na.rm = TRUE))
  lo <- as.numeric(quantile(x, 0.01, na.rm = TRUE))
  hi <- as.numeric(quantile(x, 0.95, na.rm = TRUE))
  inner <- as.numeric(quantile(x, c(0.25, 0.50, 0.75), na.rm = TRUE))
  boundary <- as.numeric(quantile(x, c(0.01, 0.99), na.rm = TRUE))
  basis <- splines::ns(x, knots = inner, Boundary.knots = boundary)
  spline_cols <- paste0("spline_", seq_len(ncol(basis)))
  for (j in seq_along(spline_cols)) d[, (spline_cols[j]) := basis[, j]]
  spline_formula <- as.formula(paste0(
    outcome, " ~ ", paste(spline_cols, collapse = " + "), " + ",
    paste(covariates, collapse = " + "), " + offset(log(C_POP)) | B_county_id + B_year"
  ))
  fit <- feglm(spline_formula, family = "quasipoisson", data = d, nthreads = 4)
  coef_names <- names(coef(fit))
  if (!all(spline_cols %in% coef_names)) stop("Spline terms were dropped for ", disease)
  beta <- coef(fit)[spline_cols]
  V <- vcov(fit, cluster = ~B_county_id)[spline_cols, spline_cols, drop = FALSE]
  x_seq <- seq(lo, hi, length.out = 100)
  pred_basis <- predict(basis, newx = x_seq)
  ref_basis <- predict(basis, newx = ref)
  diff_basis <- sweep(pred_basis, 2, as.numeric(ref_basis), "-")
  log_rr <- as.vector(diff_basis %*% beta)
  se <- sqrt(pmax(rowSums((diff_basis %*% V) * diff_basis), 0))
  all_curves[[disease]] <- data.table(
    Disease = disease, Exposure_Value = x_seq, RR = exp(log_rr),
    RR_low = exp(log_rr - 1.96 * se), RR_high = exp(log_rr + 1.96 * se),
    Ref_Value = ref, IQR = iqr, Samples = nrow(d), Counties = uniqueN(d$B_county_id),
    Reference = "5th percentile", CI = "Wald 95%; county-clustered"
  )
  all_models[[disease]] <- data.table(
    Disease = disease, Outcome = outcome, Exposure = exposure, IQR = iqr,
    Samples = nobs(fit), Counties = uniqueN(d$B_county_id),
    Zero_fraction = zero_ratio, Reference_scaled = ref,
    Spline_df = length(spline_cols), Converged = fit$convergence == 0
  )
}

curve_data <- rbindlist(all_curves)
model_data <- rbindlist(all_models)
fwrite(curve_data, file.path(output_dir, "Dose_Response_CEI_IQR_Stable4_R_method.csv"))
fwrite(model_data, file.path(output_dir, "CEI_spline_model_audit_Stable4_R_method.csv"))

p <- ggplot(curve_data, aes(x = Exposure_Value, y = RR)) +
  geom_ribbon(aes(ymin = RR_low, ymax = RR_high), fill = "#52779D", alpha = 0.20) +
  geom_line(color = "#315A7D", linewidth = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "#777777", linewidth = 0.35) +
  facet_wrap(~Disease, scales = "free_x", ncol = 2) +
  labs(title = "Stable-subset CEI–hospitalization dose–response",
       subtitle = "Natural cubic spline; RR = 1 at the 5th percentile; shaded area: 95% CI",
       x = "Stable CEI (IQR-scaled)", y = "Relative risk (RR)") +
  theme_classic(base_size = 11) +
  theme(strip.background = element_rect(fill = "#F1F3F5", color = NA),
        strip.text = element_text(face = "bold"), panel.grid = element_blank())
ggsave(file.path(output_dir, "Dose_Response_CEI_IQR_Stable4_R_method.png"), p,
       width = 8.0, height = 6.0, dpi = 400)
ggsave(file.path(output_dir, "Dose_Response_CEI_IQR_Stable4_R_method.pdf"), p,
       width = 8.0, height = 6.0, device = cairo_pdf)
cat("Stable four-disease CEI dose-response outputs saved to:", output_dir, "\n")
