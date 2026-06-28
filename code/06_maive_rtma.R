#!/usr/bin/env Rscript
#
# 06_maive_rtma.R — spurious-precision and p-hacking-robust estimators.
#
# Paper: "The Returns to Education: A Comment" (on Clark and Nielsen, 2026, Kyklos)
# Author: Lee Crawfurd
#
# Added in response to a referee who noted that the funnel-based corrections all
# assume reported standard errors are honest measures of precision, and that the
# Irsova et al. (2024) menu the comment adopts includes two estimators the first
# draft omitted:
#   - MAIVE (Irsova et al. 2025, Nat. Commun.): instruments the reported SE with
#     sample size, addressing spurious precision under inverse-variance weighting.
#   - Right-truncated meta-analysis / multiple-bias meta-analysis (Mathur):
#     bounds the corrected mean under within-study p-hacking and under joint
#     across- and within-study selection.
#
# Inputs:   data/Returns_to_education.dta
# Outputs:  output/maive_rtma_results.json
#
# Notes:
#   * MAIVE uses the wild bootstrap (SE=3) with a fixed seed (123) -> deterministic.
#   * RTMA (phacking) is fit by Stan MCMC; set.seed() is called for reproducibility
#     but posterior summaries can vary at the second decimal across runs. The paper
#     reports RTMA qualitatively (a worst-case bound), not to false precision.
#   * multibiasmeta is closed-form -> exactly reproducible.

# Project-root resolution: this file lives in code/, root is one level up.
# Use the --file= argument when run via Rscript; fall back to the working dir.
this_file <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
if (length(this_file) == 1 && nzchar(this_file)) {
  project_root <- normalizePath(file.path(dirname(this_file), ".."))
} else {
  project_root <- normalizePath(".")
}
local_lib <- file.path(project_root, ".r_libs")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))

required_pkgs <- c("haven", "dplyr", "jsonlite", "MAIVE", "multibiasmeta", "phacking")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(paste("Missing required R packages:", paste(missing_pkgs, collapse = ", "),
             "\nInstall with install.packages(c(\"MAIVE\",\"multibiasmeta\",\"phacking\"))"))
}

suppressPackageStartupMessages({
  library(haven); library(dplyr); library(jsonlite)
  library(MAIVE); library(multibiasmeta); library(phacking)
})

set.seed(20260417)

dta_path <- file.path(project_root, "data", "Returns_to_education.dta")
raw <- haven::read_dta(dta_path)

# ---------------------------------------------------------------------------
# Sample 1 (MAIVE): estimate level, needs effect, SE, and sample size N.
# ---------------------------------------------------------------------------
maive_dat_full <- raw %>%
  filter(ind_est == 1,
         !is.na(effect_percentage), !is.na(se_percentage), se_percentage > 0,
         !is.na(n), n > 0)

mdat <- data.frame(
  bs       = maive_dat_full$effect_percentage,
  sebs     = maive_dat_full$se_percentage,
  Ns       = maive_dat_full$n,
  study_id = as.integer(factor(maive_dat_full$paper))
)
n_maive    <- nrow(mdat)
n_clusters <- length(unique(mdat$study_id))

# Recommended MAIVE default: PET-PEESE, unweighted, instrumented, cluster SE,
# wild bootstrap, Anderson-Rubin CI for weak instruments.
maive_run <- function(method, weight, instrument) {
  tryCatch(
    maive(dat = mdat, method = method, weight = weight, instrument = instrument,
          studylevel = 2, SE = 3, AR = 1),
    error = function(e) NULL
  )
}

# Headline: MAIVE-PET-PEESE default (unweighted, instrumented).
maive_default <- maive_run(method = 3, weight = 0, instrument = 1)

# Decomposition on the PET intercept (method = 1): isolate the effect of
# instrumenting the SE (spurious-precision fix) from the effect of dropping
# inverse-variance weights (the substantive Clark-Nielsen disagreement).
grid_beta <- function(method) {
  out <- list()
  for (w in c(0, 1)) for (ins in c(0, 1)) {
    m <- maive_run(method, w, ins)
    key <- sprintf("w%d_i%d", w, ins)
    out[[key]] <- if (is.null(m)) NA_real_ else m$beta
  }
  out
}
pet_grid       <- grid_beta(1)   # FAT-PET intercept
petpeese_grid  <- grid_beta(3)   # PET-PEESE

first_stage_F <- if (!is.null(maive_default)) maive_default[["F-test"]] else NA_real_
maive_ar_ci   <- if (!is.null(maive_default)) suppressWarnings(as.numeric(unlist(maive_default$AR_CI))) else c(NA, NA)
maive_ivw     <- maive_run(method = 3, weight = 1, instrument = 1)  # IV-weighted + instrumented

# ---------------------------------------------------------------------------
# Sample 2 (selection models): study-collapsed by inverse-variance pooling,
# matching the existing Panel B (p-uniform*, Copas, Andrews-Kasy).
# ---------------------------------------------------------------------------
coll <- raw %>%
  filter(ind_est == 1, !is.na(effect_percentage), !is.na(se_percentage), se_percentage > 0) %>%
  group_by(paper) %>%
  summarise(
    yi = sum(effect_percentage / se_percentage^2) / sum(1 / se_percentage^2),
    vi = 1 / sum(1 / se_percentage^2),
    .groups = "drop"
  ) %>%
  mutate(sei = sqrt(vi)) %>%
  filter(is.finite(yi), is.finite(vi), vi > 0)
n_studies <- nrow(coll)

# Multiple-bias meta-analysis across assumed publication-selection ratios eta.
# eta = 1 (no selection) ... eta -> large (extreme). eta = 2 corresponds to the
# Andrews-Kasy omega_hat = 0.5 already reported in the paper. Internal (within-
# study) bias set to 0 so this is a pure selection correction comparable to the
# other selection models; the worst case (every affirmative result fabricated)
# is returned separately.
mb_curve <- lapply(c(1, 2, 4, 10, 200), function(eta) {
  s <- multibias_meta(yi = coll$yi, vi = coll$vi, selection_ratio = eta,
                      bias_affirmative = 0, bias_nonaffirmative = 0,
                      favor_positive = TRUE)$stats
  s <- s[s$model == "multibias", ]
  list(eta = eta, est = s$estimate, ci_lb = s$ci_lower, ci_ub = s$ci_upper, pval = s$p_value)
})

mb_worst <- multibias_meta(yi = coll$yi, vi = coll$vi, selection_ratio = 4,
                           bias_affirmative = 0, bias_nonaffirmative = 0,
                           favor_positive = TRUE, return_worst_meta = TRUE)$stats
mb_worst_row <- mb_worst[mb_worst$model == "worst_case", ]

# Right-truncated meta-analysis (worst-case p-hacking). Stan MCMC.
rtma <- tryCatch(
  phacking_meta(yi = coll$yi, sei = coll$sei, favor_positive = TRUE, parallelize = FALSE),
  error = function(e) NULL
)
rtma_mu <- if (!is.null(rtma)) {
  s <- as.data.frame(rtma$stats)
  mu <- s[s$param == "mu", ]
  list(mode = mu$mode, median = mu$median, mean = mu$mean,
       ci_lb = mu$ci_lower, ci_ub = mu$ci_upper)
} else list(mode = NA, median = NA, mean = NA, ci_lb = NA, ci_ub = NA)

# ---------------------------------------------------------------------------
# Endogenous kink (Bom & Rachinger 2019): the one funnel-based method from the
# Irsova et al. (2024) menu not yet run. Publication selection inflates reported
# effects only where an estimate at the true mean would be insignificant
# (SE > |mu|/1.96); below that kink the effect-SE relation is flat. The kink
# a = |mu|/1.96 is endogenous, so iterate WLS (1/SE^2 weights) on the 71
# independent estimates until the intercept converges; it is the bias-corrected
# estimate. Same FE weighting and sample as the headline PET-PEESE.
# ---------------------------------------------------------------------------
ek_dat <- raw %>%
  filter(ind_est == 1, !is.na(effect_percentage), !is.na(se_percentage),
         se_percentage > 0)
ek_y  <- ek_dat$effect_percentage
ek_se <- ek_dat$se_percentage
ek_w  <- 1 / ek_se^2
ek_zcrit <- 1.96
ek_beta  <- sum(ek_w * ek_y) / sum(ek_w)        # initialise at FE weighted mean
ek_iters <- NA_integer_
for (i in seq_len(500)) {
  a    <- abs(ek_beta) / ek_zcrit
  bnew <- unname(coef(lm(ek_y ~ pmax(0, ek_se - a), weights = ek_w))[1])
  ek_iters <- i
  if (abs(bnew - ek_beta) < 1e-10) { ek_beta <- bnew; break }
  ek_beta <- bnew
}
ek_a   <- abs(ek_beta) / ek_zcrit
ek_co  <- summary(lm(ek_y ~ pmax(0, ek_se - ek_a), weights = ek_w))$coefficients

# ---------------------------------------------------------------------------
# Assemble and write.
# ---------------------------------------------------------------------------
result <- list(
  maive = list(
    n_estimates  = n_maive,
    n_clusters   = n_clusters,
    first_stage_F = first_stage_F,
    petpeese_default_unweighted_instr = if (!is.null(maive_default)) maive_default$beta else NA_real_,
    petpeese_ivweighted_instr         = if (!is.null(maive_ivw)) maive_ivw$beta else NA_real_,
    ar_ci_lb = maive_ar_ci[1], ar_ci_ub = maive_ar_ci[2],
    pet_grid      = pet_grid,       # w0_i0, w0_i1, w1_i0, w1_i1
    petpeese_grid = petpeese_grid
  ),
  multibias = list(
    n_studies = n_studies,
    curve     = mb_curve,           # est by eta = 1,2,4,10,200
    worst_case = list(est = mb_worst_row$estimate, ci_lb = mb_worst_row$ci_lower,
                      ci_ub = mb_worst_row$ci_upper, pval = mb_worst_row$p_value)
  ),
  rtma = rtma_mu,
  endogenous_kink = list(
    n            = length(ek_y),
    est          = unname(ek_co[1, 1]),
    se           = unname(ek_co[1, 2]),
    pval         = unname(ek_co[1, 4]),
    kink_se      = ek_a,
    slope        = unname(ek_co[2, 1]),
    n_above_kink = sum(ek_se > ek_a),
    iters        = ek_iters
  )
)

out_path <- file.path(project_root, "output", "maive_rtma_results.json")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat("Wrote", out_path, "\n")

# Console summary for the log.
cat(sprintf("\nMAIVE: n=%d est, %d clusters, first-stage F=%.1f\n", n_maive, n_clusters, first_stage_F))
cat(sprintf("  PET-PEESE default (unweighted, instrumented): %.2f%%\n",
            result$maive$petpeese_default_unweighted_instr))
cat(sprintf("  PET-PEESE IV-weighted, instrumented:          %.2f%%\n",
            result$maive$petpeese_ivweighted_instr))
cat(sprintf("  PET intercept grid  w0i0=%.2f w0i1=%.2f w1i0=%.2f w1i1=%.2f\n",
            pet_grid$w0_i0, pet_grid$w0_i1, pet_grid$w1_i0, pet_grid$w1_i1))
cat(sprintf("multibias (n=%d studies):\n", n_studies))
for (r in mb_curve) cat(sprintf("  eta=%-4g  %.2f%%  [%.2f, %.2f]  p=%.4f\n",
                                 r$eta, r$est, r$ci_lb, r$ci_ub, r$pval))
cat(sprintf("  worst-case: %.2f%% [%.2f, %.2f] p=%.3f\n",
            mb_worst_row$estimate, mb_worst_row$ci_lower, mb_worst_row$ci_upper, mb_worst_row$p_value))
cat(sprintf("RTMA mu: mode=%.2f median=%.2f mean=%.2f CI=[%.2f, %.2f]\n",
            rtma_mu$mode, rtma_mu$median, rtma_mu$mean, rtma_mu$ci_lb, rtma_mu$ci_ub))
