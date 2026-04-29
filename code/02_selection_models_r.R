#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript r_copas_puniform_methods.R <input_csv> <output_json> <output_plot>")
}

input_csv <- args[1]
output_json <- args[2]
output_plot <- args[3]

# Use a project-local library path first so installs don't require system write access.
# input_csv lives in data/; project root is one level up.
project_root <- normalizePath(file.path(dirname(normalizePath(input_csv)), ".."))
local_lib <- file.path(project_root, ".r_libs")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(local_lib, .libPaths()))

required_pkgs <- c("dplyr", "readr", "jsonlite", "meta", "metasens", "puniform", "weightr", "haven", "robumeta")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    paste(
      "Missing required R packages:",
      paste(missing_pkgs, collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(jsonlite)
  library(meta)
  library(metasens)
  library(puniform)
  library(weightr)
  library(haven)
  library(robumeta)
})

set.seed(20260417)

first_finite <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(NA_real_)
  }
  x_num <- suppressWarnings(as.numeric(x))
  idx <- which(is.finite(x_num))
  if (length(idx) == 0) {
    return(NA_real_)
  }
  x_num[idx[1]]
}

dat <- readr::read_csv(input_csv, show_col_types = FALSE) %>%
  transmute(
    study = as.character(study),
    effect = as.numeric(effect),
    se = as.numeric(se)
  ) %>%
  filter(is.finite(effect), is.finite(se), se > 0)

if (nrow(dat) < 3) {
  stop("Input data must contain at least 3 valid estimates.")
}

# Collapse to one estimate per study using inverse-variance (IV) pooling.
# IV pooling is correct: the within-study variance for n estimates is 1/sum(1/vi).
# mean(se^2) would inflate collapsed variances by sqrt(n)/2 on average, biasing
# non-centrality parameters for p-uniform* and weightr toward zero.
collapsed <- dat %>%
  group_by(study) %>%
  summarise(
    mean_effect = sum(effect / se^2) / sum(1 / se^2),
    mean_var    = 1 / sum(1 / se^2),
    .groups = "drop"
  ) %>%
  mutate(mean_se = sqrt(mean_var)) %>%
  filter(is.finite(mean_effect), is.finite(mean_var), mean_var > 0)

if (nrow(collapsed) < 3) {
  stop("Collapsed data must contain at least 3 studies.")
}

# Use FE-weighted mean of all input estimates to set direction (more robust
# than unweighted mean of collapsed effects).
fe_mean_wt <- sum(dat$effect / dat$se^2) / sum(1 / dat$se^2)
direction_side <- if (fe_mean_wt >= 0) "right" else "left"
copas_left <- direction_side == "right"

# p-uniform*: try bootstrap CIs first (seed already set via set.seed()).
# If bootstrap CIs are NA (common with extreme heterogeneity, I²≈96%),
# fall back to test-inversion CIs (boot=FALSE).
puni_res <- puniform::puni_star(
  yi   = collapsed$mean_effect,
  vi   = collapsed$mean_var,
  side = direction_side,
  boot = TRUE
)
# Fall back to test-inversion if bootstrap did not produce CIs
puni_ci_lb <- first_finite(puni_res$ci.lb)
puni_ci_ub <- first_finite(puni_res$ci.ub)
if (is.na(puni_ci_lb) || is.na(puni_ci_ub)) {
  puni_res_nboot <- tryCatch(
    puniform::puni_star(
      yi   = collapsed$mean_effect,
      vi   = collapsed$mean_var,
      side = direction_side,
      boot = FALSE
    ),
    error = function(e) NULL
  )
  if (!is.null(puni_res_nboot)) {
    puni_ci_lb <- first_finite(puni_res_nboot$ci.lb)
    puni_ci_ub <- first_finite(puni_res_nboot$ci.ub)
  }
}

meta_obj <- meta::metagen(
  TE = collapsed$mean_effect,
  seTE = collapsed$mean_se,
  sm = "MD",
  method.tau = "REML",
  hakn = FALSE
)

copas_res <- metasens::copas(meta_obj, left = copas_left)

png(filename = output_plot, width = 800, height = 700)
plot.copas(copas_res)
dev.off()

# Andrews-Kasy / Vevea-Hedges step-function selection model (weightr)
# Selection step at two-tailed p = 0.05: estimates how much less likely
# non-significant results are to be published relative to significant ones.
ak_est <- ak_se <- ak_ci_lb <- ak_ci_ub <- ak_pval <- ak_weight <- NA_real_

wf_res <- tryCatch(
  weightr::weightfunct(
    effect = collapsed$mean_effect,
    v      = collapsed$mean_var,
    steps  = c(0.025)   # one-tailed = two-tailed p < 0.05
  ),
  error = function(e) NULL
)

if (!is.null(wf_res)) {
  # par ordering: [tau2, mu, omega1] where mu is the bias-corrected effect
  ak_est    <- wf_res[[2]]$par[2]
  ak_se_raw <- tryCatch(
    sqrt(diag(solve(wf_res[[2]]$hessian)))[2],
    error = function(e) NA_real_
  )
  if (is.finite(ak_se_raw)) {
    ak_se    <- ak_se_raw
    ak_ci_lb <- ak_est - 1.96 * ak_se
    ak_ci_ub <- ak_est + 1.96 * ak_se
    ak_pval  <- 2 * pnorm(-abs(ak_est / ak_se))
  }
  # omega: relative publication probability for non-significant results
  ak_weight <- tryCatch(wf_res[[2]]$par[3], error = function(e) NA_real_)
}

# Cluster-robust PET/PEESE (RVE) clustered on paper/study
# Reads the full .dta to get the paper clustering variable
rve_pet_est <- rve_pet_se <- rve_pet_pval <- NA_real_
rve_peese_est <- rve_peese_se <- rve_peese_pval <- NA_real_

dta_path <- file.path(project_root, "data", "Returns_to_education.dta")
if (file.exists(dta_path)) {
  full_dat <- tryCatch({
    haven::read_dta(dta_path) %>%
      filter(ind_est == 1) %>%
      filter(!is.na(effect_percentage), !is.na(se_percentage)) %>%
      mutate(paper_id = as.integer(factor(paper)),
             eff = effect_percentage,
             stderr = se_percentage,
             stderr2 = se_percentage^2,
             var_eff = se_percentage^2)
  }, error = function(e) NULL)

  if (!is.null(full_dat) && nrow(full_dat) >= 3) {
    rve_pet_res <- tryCatch(
      robumeta::robu(eff ~ stderr, data = full_dat, studynum = paper_id,
                     var.eff.size = var_eff, small = TRUE),
      error = function(e) NULL
    )
    if (!is.null(rve_pet_res)) {
      rt <- rve_pet_res$reg_table
      int_row <- rt[grepl("Intercept", rt$labels, ignore.case = TRUE), ]
      if (nrow(int_row) == 1) {
        rve_pet_est  <- int_row$b.r
        rve_pet_se   <- int_row$SE
        rve_pet_pval <- int_row$prob
      }
    }

    rve_peese_res <- tryCatch(
      robumeta::robu(eff ~ stderr2, data = full_dat, studynum = paper_id,
                     var.eff.size = var_eff, small = TRUE),
      error = function(e) NULL
    )
    if (!is.null(rve_peese_res)) {
      rt <- rve_peese_res$reg_table
      int_row <- rt[grepl("Intercept", rt$labels, ignore.case = TRUE), ]
      if (nrow(int_row) == 1) {
        rve_peese_est  <- int_row$b.r
        rve_peese_se   <- int_row$SE
        rve_peese_pval <- int_row$prob
      }
    }
  }
}

result <- list(
  n_input = nrow(dat),
  n_collapsed = nrow(collapsed),
  direction = list(
    puniform_side = direction_side,
    copas_left = copas_left
  ),
  puniform = list(
    est    = first_finite(puni_res$est),
    se     = first_finite(puni_res$se),
    ci_lb  = if (is.finite(puni_ci_lb)) puni_ci_lb else NA_real_,
    ci_ub  = if (is.finite(puni_ci_ub)) puni_ci_ub else NA_real_,
    tau2   = first_finite(puni_res$tau2),
    pval_0 = first_finite(puni_res$pval.0),
    pval_1 = first_finite(puni_res$pval.1)
  ),
  copas = list(
    te_random = first_finite(copas_res$TE.random),
    te_adjust = first_finite(copas_res$TE.adjust),
    se_adjust = first_finite(copas_res$seTE.adjust),
    ci_lb_adjust = first_finite(copas_res$lower.adjust),
    ci_ub_adjust = first_finite(copas_res$upper.adjust),
    pval_adjust = first_finite(copas_res$pval.adjust)
  ),
  andrews_kasy = list(
    est          = ak_est,
    se           = ak_se,
    ci_lb        = ak_ci_lb,
    ci_ub        = ak_ci_ub,
    pval         = ak_pval,
    weight_nonsig = ak_weight,
    n_collapsed  = nrow(collapsed)
  ),
  rve_pet = list(
    est  = rve_pet_est,
    se   = rve_pet_se,
    pval = rve_pet_pval,
    n    = if (exists("full_dat") && !is.null(full_dat)) nrow(full_dat) else NA_integer_
  ),
  rve_peese = list(
    est  = rve_peese_est,
    se   = rve_peese_se,
    pval = rve_peese_pval
  )
)

jsonlite::write_json(result, output_json, auto_unbox = TRUE, pretty = TRUE, na = "null")
