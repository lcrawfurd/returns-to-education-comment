********************************************************************************
* 01_pub_bias_stata.do
*
* Stata implementation of the publication-bias methods for:
* "The Returns to Education: A Comment" (on Clark and Nielsen, 2026, Kyklos)
*
* Includes:
* - Egger regression
* - PET / PEESE
* - Trim-and-fill (Stata meta trimfill)
* - Caliper bunching checks around t = 1.96
*
* Comparison notes vs Clark/Nielsen do-file:
* - Their publication-bias regressions are in Returns_to_education.do
*   (publication-bias section and PET/PEESE section).
* - This script adds a side-by-side numerical comparison:
*     (a) blog-spec (full sample, weighted PET/PEESE)
*     (b) Clark/Nielsen PET/PEESE spec (SE < 10.1, weighted)
* - This script also adds formal tests not in that do-file:
*     trim-and-fill, caliper binomial tests.
********************************************************************************

version 16.0
assert c(stata_version) >= 16    // meta regress requires Stata 16+
clear all
set more off

* Detect project root. Run this script from the project root directory
* (stata-mp -b do code/blog-other-methods-stata.do) so that c(pwd) is the root.
* Fall back to the parent of c(filename)'s directory for full-path invocations.
local project_dir "`c(pwd)'"
* Validate: if data/ doesn't exist under c(pwd), try parent of c(filename) dir
capture confirm file "`project_dir'/data/Returns_to_education.dta"
if _rc {
    local this_do "`c(filename)'"
    local slash1 = strrpos("`this_do'", "/")
    local slash2 = strrpos("`this_do'", char(92))
    local slash = cond(`slash1' > `slash2', `slash1', `slash2')
    if `slash' > 0 {
        local code_dir = substr("`this_do'", 1, `slash' - 1)
        local slash1 = strrpos("`code_dir'", "/")
        local slash2 = strrpos("`code_dir'", char(92))
        local slash = cond(`slash1' > `slash2', `slash1', `slash2')
        if `slash' > 0 {
            local project_dir = substr("`code_dir'", 1, `slash' - 1)
        }
    }
}

local data_file "`project_dir'/data/Returns_to_education.dta"
local csv_out   "`project_dir'/output/stata_pub_bias_summary.csv"
local log_out   "`project_dir'/output/stata_pub_bias_summary.log"

capture log close _all
log using "`log_out'", text replace

display "Project directory: `project_dir'"
display "Loading data: `data_file'"

capture confirm file "`data_file'"
if _rc {
    display as error "Data file not found: `data_file'"
    display as error "Pass project_dir explicitly or run from the project folder."
    error 601
}

use "`data_file'", clear
keep if ind_est == 1
drop if missing(effect_percentage, se_percentage)
capture isid study_id estimate_id
if _rc {
    capture isid study_id
    if _rc {
        display as text "Note: no unique key found on study_id or study_id+estimate_id — proceeding."
    }
}

capture drop effect
capture drop se
capture drop se1
capture drop var
capture drop invvar
capture drop precision
capture drop tstat_calc
capture drop se2

gen double effect = effect_percentage
gen double se = se_percentage
gen double var = se^2
gen double invvar = 1 / var
gen double precision = 1 / se
gen double tstat_calc = effect / se
gen double se2 = se^2

* ─── Export input CSV for R selection models (Step 2 of run.sh) ─────────────
* r_copas_puniform_methods.R reads this file; deleting it breaks the pipeline.
local r_input_csv "`project_dir'/data/_r_copas_puniform_input.csv"
preserve
    keep paper effect se
    rename paper study
    export delimited using "`r_input_csv'", replace
restore
display "Wrote R input CSV: `r_input_csv'"

count
local n_ind = r(N)

display _newline(1) "============================================================"
display "Sample"
display "============================================================"
display "Independent estimates (non-missing effect/se): " `n_ind'

qui summarize effect, meanonly
local naive_mean = r(mean)

qui summarize effect [aw=invvar], meanonly
local fixed_mean = r(mean)

display "Naive mean effect (%): " %9.3f `naive_mean'
display "Fixed-effect weighted mean (%): " %9.3f `fixed_mean'

display _newline(1) "============================================================"
display "WAAP (Weighted Average of Adequately Powered studies)"
display "Ioannidis, Stanley and Doucouliagos (2017)"
display "============================================================"
* A study has 80% power to detect an effect of size theta when SE <= theta/2.802
* (two-sided alpha=0.05: z_0.025 + z_0.8 = 1.96 + 0.842 = 2.802)
local waap_thresh = abs(`fixed_mean') / 2.802
capture drop adeq_power
gen byte adeq_power = (se <= `waap_thresh')
count if adeq_power == 1
local n_waap = r(N)
display "Reference FE mean: " %9.3f `fixed_mean' "%  |  80%% power threshold (SE <= " %7.3f `waap_thresh' "%): " `n_waap' " studies"

local waap_est  = .
local waap_se   = .
local waap_pval = .
if `n_waap' >= 1 {
    qui sum effect [aw=invvar] if adeq_power == 1
    local waap_est = r(mean)
    qui sum invvar if adeq_power == 1
    local waap_se   = sqrt(1 / r(sum))
    local waap_pval = 2 * (1 - normal(abs(`waap_est' / `waap_se')))
    display "WAAP: " %9.3f `waap_est' "% (SE " %9.3f `waap_se' ", p " %9.4f `waap_pval' ", n=" `n_waap' ")"
}
else {
    local waap_est = `fixed_mean'
    display "No adequately powered studies; WAAP falls back to FE WLS"
}
drop adeq_power

display _newline(1) "============================================================"
display "Egger regression (t-stat on precision)"
display "============================================================"
qui regress tstat_calc precision, vce(robust)
local egger_intercept = _b[_cons]
local egger_intercept_se = _se[_cons]
local egger_intercept_p = 2 * ttail(e(df_r), abs(_b[_cons] / _se[_cons]))
local egger_slope = _b[precision]
local egger_slope_se = _se[precision]
local egger_slope_p = 2 * ttail(e(df_r), abs(_b[precision] / _se[precision]))

display "Intercept (bias indicator): " %9.3f `egger_intercept' ///
    " (SE " %9.3f `egger_intercept_se' ", p " %9.4f `egger_intercept_p' ")"
display "Slope (true effect):       " %9.3f `egger_slope' ///
    " (SE " %9.3f `egger_slope_se' ", p " %9.4f `egger_slope_p' ")"

display _newline(1) "============================================================"
display "PET and PEESE (inverse-variance weighted, robust SE)"
display "============================================================"
qui regress effect se [aw=invvar], vce(robust)
local pet_intercept = _b[_cons]
local pet_intercept_se = _se[_cons]
local pet_intercept_p = 2 * ttail(e(df_r), abs(_b[_cons] / _se[_cons]))
local pet_slope = _b[se]
local pet_slope_se = _se[se]
local pet_slope_p = 2 * ttail(e(df_r), abs(_b[se] / _se[se]))

display "PET intercept: " %9.3f `pet_intercept' ///
    " (SE " %9.3f `pet_intercept_se' ", p " %9.4f `pet_intercept_p' ")"
display "PET slope on SE: " %9.3f `pet_slope' ///
    " (SE " %9.3f `pet_slope_se' ", p " %9.4f `pet_slope_p' ")"

qui regress effect se2 [aw=invvar], vce(robust)
local peese_intercept = _b[_cons]
local peese_intercept_se = _se[_cons]
local peese_intercept_p = 2 * ttail(e(df_r), abs(_b[_cons] / _se[_cons]))
local peese_slope = _b[se2]
local peese_slope_se = _se[se2]
local peese_slope_p = 2 * ttail(e(df_r), abs(_b[se2] / _se[se2]))

display "PEESE intercept: " %9.3f `peese_intercept' ///
    " (SE " %9.3f `peese_intercept_se' ", p " %9.4f `peese_intercept_p' ")"
display "PEESE slope on SE^2: " %9.4f `peese_slope' ///
    " (SE " %9.4f `peese_slope_se' ", p " %9.4f `peese_slope_p' ")"

display _newline(1) "============================================================"
display "Random-effects meta-analysis (REML): heterogeneity diagnostics"
display "============================================================"

local isq = .
local tau2 = .
local hsq = .
local re_mean = .
local re_mean_se = .
local re_mean_pval = .
local re_pet_intercept = .
local re_pet_intercept_se = .
local re_pet_intercept_p = .
local re_pet_slope = .
local re_pet_slope_p = .
local re_peese_intercept = .
local re_peese_intercept_se = .
local re_peese_intercept_p = .
local re_peese_slope = .
local re_peese_slope_p = .

capture noisily meta set effect se, random(reml)
if _rc {
    display as error "meta set failed; RE estimates will be missing."
}
else {
    display _newline(1) "Random-effects summary (REML):"
    capture noisily meta summarize, random(reml)
    if _rc {
        display as error "meta summarize failed."
    }
    else {
        local re_mean    = r(theta)
        * Stata 16–17 stores SE as r(setheta); Stata 18+ may omit it.
        * Fall back: derive SE from CI bounds, then from z-statistic.
        local re_mean_se = r(setheta)
        if missing(`re_mean_se') {
            capture local re_mean_se = r(se_theta)
        }
        if missing(`re_mean_se') & !missing(r(lb)) & !missing(r(ub)) {
            local re_mean_se = (r(ub) - r(lb)) / (2 * invnormal(0.975))
        }
        if missing(`re_mean_se') & !missing(r(z)) & r(z) != 0 {
            local re_mean_se = abs(`re_mean' / r(z))
        }
        * Use r(pz) directly if available; otherwise compute from SE.
        local re_mean_pval = r(pz)
        if missing(`re_mean_pval') & !missing(`re_mean_se') & `re_mean_se' > 0 {
            local re_mean_pval = ///
                2 * (1 - normal(abs(`re_mean' / `re_mean_se')))
        }
        local isq        = r(I2)
        local tau2       = r(tau2)
        local hsq        = r(H2)
        display "RE weighted mean (%): " %9.3f `re_mean' ///
            " (SE " %9.3f `re_mean_se' ")"
        display "I-squared (%): " %9.2f `isq'
        display "tau-squared: "  %9.4f `tau2'
        display "H-squared: "    %9.3f `hsq'
    }

    display _newline(1) "Random-effects PET (REML):"
    capture noisily meta regress se, random(reml)
    if _rc {
        display as error "meta regress se (RE PET) failed."
    }
    else {
        local re_pet_intercept    = _b[_cons]
        local re_pet_intercept_se = _se[_cons]
        local re_pet_intercept_p  = 2 * (1 - normal(abs(_b[_cons] / _se[_cons])))
        local re_pet_slope        = _b[se]
        local re_pet_slope_p      = 2 * (1 - normal(abs(_b[se] / _se[se])))
        display "RE PET intercept: " %9.3f `re_pet_intercept' ///
            " (SE " %9.3f `re_pet_intercept_se' ", p " %9.4f `re_pet_intercept_p' ")"
        display "RE PET slope on SE: " %9.3f `re_pet_slope' ///
            " (p " %9.4f `re_pet_slope_p' ")"
    }

    display _newline(1) "Random-effects PEESE (REML):"
    capture noisily meta regress se2, random(reml)
    if _rc {
        display as error "meta regress se2 (RE PEESE) failed."
    }
    else {
        local re_peese_intercept    = _b[_cons]
        local re_peese_intercept_se = _se[_cons]
        local re_peese_intercept_p  = 2 * (1 - normal(abs(_b[_cons] / _se[_cons])))
        local re_peese_slope        = _b[se2]
        local re_peese_slope_p      = 2 * (1 - normal(abs(_b[se2] / _se[se2])))
        display "RE PEESE intercept: " %9.3f `re_peese_intercept' ///
            " (SE " %9.3f `re_peese_intercept_se' ", p " %9.4f `re_peese_intercept_p' ")"
        display "RE PEESE slope on SE^2: " %9.4f `re_peese_slope' ///
            " (p " %9.4f `re_peese_slope_p' ")"
    }
}

display _newline(1) "============================================================"
display "Comparison with Clark/Nielsen do-file specifications"
display "============================================================"

* Clark/Nielsen publication-bias slope regression (unweighted):
*   regress effect_percentage se_percentage if ind_est==1, r
*   regress effect_percentage se_percentage if ind_est==1 & se_percentage<10.1, r
count if se < 10.1
local n_se10 = r(N)
display "Sample size, full (blog-spec): " `n_ind'
display "Sample size, SE<10.1 (Clark/Nielsen PET filter): " `n_se10'

qui regress effect se if se < 10.1, vce(robust)
local cn_unw_slope = _b[se]
local cn_unw_slope_se = _se[se]
local cn_unw_slope_p = 2 * ttail(e(df_r), abs(_b[se] / _se[se]))

display _newline(1) "Clark/Nielsen-style unweighted slope (SE<10.1): " ///
    %9.3f `cn_unw_slope' " (SE " %9.3f `cn_unw_slope_se' ", p " %9.4f `cn_unw_slope_p' ")"
display "Blog Egger intercept (bias indicator): " %9.3f `egger_intercept' ///
    " (not directly comparable parameterization)"

* Clark/Nielsen PET/PEESE weighted specs:
*   reg effect_percentage se_percentage if ind_est==1 & se_percentage<10.1 [aw=1/(se^2)], r
*   reg effect_percentage se_percentage2 if ind_est==1 & se_percentage<10.1 [aw=1/(se^2)], r
qui regress effect se [aw=invvar] if se < 10.1, vce(robust)
local pet_cn_intercept = _b[_cons]
local pet_cn_intercept_se = _se[_cons]
local pet_cn_intercept_p = 2 * ttail(e(df_r), abs(_b[_cons] / _se[_cons]))

qui regress effect se2 [aw=invvar] if se < 10.1, vce(robust)
local peese_cn_intercept = _b[_cons]
local peese_cn_intercept_se = _se[_cons]
local peese_cn_intercept_p = 2 * ttail(e(df_r), abs(_b[_cons] / _se[_cons]))

local pet_diff = `pet_intercept' - `pet_cn_intercept'
local peese_diff = `peese_intercept' - `peese_cn_intercept'

display _newline(1) "PET intercept, blog-spec (full sample): " %9.3f `pet_intercept' ///
    " (p " %9.4f `pet_intercept_p' ")"
display "PET intercept, Clark/Nielsen spec (SE<10.1): " %9.3f `pet_cn_intercept' ///
    " (p " %9.4f `pet_cn_intercept_p' ")"
display "Difference (blog - Clark/Nielsen PET): " %9.3f `pet_diff'

display _newline(1) "PEESE intercept, blog-spec (full sample): " %9.3f `peese_intercept' ///
    " (p " %9.4f `peese_intercept_p' ")"
display "PEESE intercept, Clark/Nielsen spec (SE<10.1): " %9.3f `peese_cn_intercept' ///
    " (p " %9.4f `peese_cn_intercept_p' ")"
display "Difference (blog - Clark/Nielsen PEESE): " %9.3f `peese_diff'

display _newline(1) "Additional methods in this script not in Clark/Nielsen do-file:"
display "- meta trimfill"
display "- caliper binomial tests (bitesti)"

local tf_theta = .
local tf_se    = .

display _newline(1) "============================================================"
display "Trim-and-fill (Stata meta)"
display "============================================================"
capture noisily meta set effect se
if _rc {
    display as error "meta suite unavailable in this Stata environment; skipping trim-and-fill."
}
else {
    display "Fixed-effect summary (before trim-and-fill):"
    capture noisily meta summarize, fixed
    if _rc {
        display as error "meta summarize failed."
    }
    else {
        * Save FE estimate now; use as fallback if meta trimfill does not
        * populate r() (observed in Stata 19.5 when 0 studies are imputed).
        local tf_theta = r(theta)
        local tf_se    = r(setheta)
        if missing(`tf_se') & !missing(r(lb)) & !missing(r(ub)) {
            local tf_se = (r(ub) - r(lb)) / (2 * invnormal(0.975))
        }
        if missing(`tf_se') & !missing(r(z)) & r(z) != 0 {
            local tf_se = abs(`tf_theta' / r(z))
        }
    }

    display _newline(1) "Trim-and-fill estimate (fixed-effect):"
    capture noisily meta trimfill, fixed
    if _rc {
        display as error "meta trimfill is not available in this Stata version; skipping."
    }
    else {
        * Prefer augmented estimate from trim-and-fill if r() is populated.
        * Stata 18/19 appears not to set r(theta_a) etc. when 0 studies imputed;
        * in that case tf_theta/tf_se already hold the correct FE values above.
        local tf_theta_new = r(theta_a)
        if missing(`tf_theta_new') local tf_theta_new = r(theta)
        local tf_se_new    = r(setheta_a)
        if missing(`tf_se_new')    local tf_se_new    = r(setheta)
        if missing(`tf_se_new') & !missing(r(lb)) & !missing(r(ub)) {
            local tf_se_new = (r(ub) - r(lb)) / (2 * invnormal(0.975))
        }
        if !missing(`tf_theta_new') local tf_theta = `tf_theta_new'
        if !missing(`tf_se_new')    local tf_se    = `tf_se_new'
        display "Trim-and-fill adjusted estimate: " %9.3f `tf_theta' ///
            " (SE " %9.3f `tf_se' ")"
    }
}

display _newline(1) "============================================================"
display "Caliper bunching tests around t = 1.96"
display "============================================================"

count if tstat_calc >= 1.96 & tstat_calc < 2.46
local above_05 = r(N)
count if tstat_calc >= 1.46 & tstat_calc < 1.96
local below_05 = r(N)
local total_05 = `above_05' + `below_05'

display "Bandwidth 0.5, just above: " `above_05' "  just below: " `below_05'
if `total_05' > 0 {
    bitesti `total_05' `above_05' 0.5
}

count if tstat_calc >= 1.96 & tstat_calc < 2.96
local above_10 = r(N)
count if tstat_calc >= 0.96 & tstat_calc < 1.96
local below_10 = r(N)
local total_10 = `above_10' + `below_10'

display "Bandwidth 1.0, just above: " `above_10' "  just below: " `below_10'
if `total_10' > 0 {
    bitesti `total_10' `above_10' 0.5
}

display _newline(1) "============================================================"
display "Exporting summary table for regression-based methods"
display "============================================================"

tempfile summary
tempname posth
postfile `posth' str45 method double estimate se pvalue using "`summary'", replace

post `posth' ("Naive (unweighted) mean") (`naive_mean') (.) (.)
post `posth' ("Fixed-effect weighted mean") (`fixed_mean') (.) (.)
post `posth' ("WAAP") (`waap_est') (`waap_se') (`waap_pval')
post `posth' ("Egger intercept (bias)") (`egger_intercept') (`egger_intercept_se') (`egger_intercept_p')
post `posth' ("Egger slope (true effect)") (`egger_slope') (`egger_slope_se') (`egger_slope_p')
post `posth' ("PET intercept") (`pet_intercept') (`pet_intercept_se') (`pet_intercept_p')
post `posth' ("PEESE intercept") (`peese_intercept') (`peese_intercept_se') (`peese_intercept_p')
post `posth' ("PET intercept (Clark/Nielsen SE<10.1)") (`pet_cn_intercept') (`pet_cn_intercept_se') (`pet_cn_intercept_p')
post `posth' ("PEESE intercept (Clark/Nielsen SE<10.1)") (`peese_cn_intercept') (`peese_cn_intercept_se') (`peese_cn_intercept_p')
post `posth' ("RE weighted mean (REML)") (`re_mean') (`re_mean_se') (`re_mean_pval')
post `posth' ("RE PET intercept (REML)") (`re_pet_intercept') (`re_pet_intercept_se') (`re_pet_intercept_p')
post `posth' ("RE PEESE intercept (REML)") (`re_peese_intercept') (`re_peese_intercept_se') (`re_peese_intercept_p')
post `posth' ("I-squared (%)") (`isq') (.) (.)
post `posth' ("tau-squared") (`tau2') (.) (.)
post `posth' ("H-squared") (`hsq') (.) (.)
post `posth' ("Trim-and-fill (fixed)") (`tf_theta') (`tf_se') (.)
post `posth' ("n_waap") (`n_waap') (.) (.)

postclose `posth'

use "`summary'", clear
format estimate %9.4f
format se %9.4f
format pvalue %9.4f
export delimited using "`csv_out'", replace

display "Wrote summary CSV: `csv_out'"
log close _all
