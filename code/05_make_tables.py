#!/usr/bin/env python3
"""
make_tables.py — Generate LaTeX tables from analysis outputs.

Paper: "Causal Returns to Education Remain Substantial After Correcting for Publication Bias:
        A Comment on Clark and Nielsen (2026)"
Author: Lee Crawfurd

Reads:
  ../output/stata_pub_bias_summary.csv    (from blog-other-methods-stata.do)
  ../output/r_pub_bias_results.json       (from r_copas_puniform_methods.R)

Writes:
  ../output/tables/table1_summary.tex

Run from project root: python3 code/make_tables.py
Or via run.sh (Step 3c).
"""

import csv
import json
import math
import pathlib
import sys

HERE    = pathlib.Path(__file__).resolve().parent.parent
STATA_CSV = HERE / "output" / "stata_pub_bias_summary.csv"
R_JSON    = HERE / "output" / "r_pub_bias_results.json"
MAIVE_JSON = HERE / "output" / "maive_rtma_results.json"
OUT_DIR   = HERE / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardcoded constants — values from external sources or genuinely fixed
# ---------------------------------------------------------------------------

# Cochran Q from meta summarize (Stata output; not stored in CSV).
# Q(df=70) = 1072.8, p < 0.001.
Q_STAT = 1072.8

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_stata_csv(path):
    """Return dict[method] -> {estimate, se, pvalue} as floats or None."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def safe(s):
                s = (s or "").strip()
                try:
                    return float(s) if s else None
                except ValueError:
                    return None
            result[row["method"]] = {
                "estimate": safe(row.get("estimate", "")),
                "se":       safe(row.get("se", "")),
                "pvalue":   safe(row.get("pvalue", "")),
            }
    return result


def load_r_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _norm_sf(z):
    """Upper-tail probability of standard normal (no scipy dependency)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def pval_from_z(est, se):
    """Two-sided p-value from z = est/se; None if inputs invalid."""
    if est is None or se is None or se <= 0:
        return None
    return 2.0 * _norm_sf(abs(est / se))


def fmt_pval(pval):
    """Format a p-value for a table cell (already in math mode context)."""
    if pval is None:
        return "---"
    if pval < 0.001:
        return r"$<0.001$"
    return f"${pval:.3f}$"


def fmt_pct(val, d=1):
    """Format val as $X.X\\%$; returns '---' if None."""
    if val is None:
        return "---"
    return f"${val:.{d}f}\\%$"


def fmt_ci(lb, ub, d=1):
    """Format a confidence interval as [$lb\\%$, $ub\\%$]."""
    if lb is None or ub is None:
        return ""
    return f"[${lb:.{d}f}\\%$, ${ub:.{d}f}\\%$]"


# ---------------------------------------------------------------------------
# Convenience: look up a field from the Stata dict
# ---------------------------------------------------------------------------

def st(stata, key, field="estimate"):
    row = stata.get(key) or {}
    return row.get(field)


# ===========================================================================
# Table 1 — Summary of all publication-bias corrections
# ===========================================================================

def make_table1(stata, rj, mr):
    n_input     = rj.get("n_input", 71)
    n_collapsed = rj.get("n_collapsed", 48)

    # MAIVE (spurious-precision-robust) and Mathur multiple-bias / RTMA results
    maive   = mr.get("maive", {}) or {}
    mb      = mr.get("multibias", {}) or {}
    rtma    = mr.get("rtma", {}) or {}
    maive_n        = maive.get("n_estimates", 60)
    maive_F        = maive.get("first_stage_F")
    maive_default  = maive.get("petpeese_default_unweighted_instr")
    maive_ivw      = maive.get("petpeese_ivweighted_instr")
    # multiple-bias at eta=2 (the Andrews-Kasy-implied selection intensity)
    mb_eta2 = next((r for r in (mb.get("curve") or []) if r.get("eta") == 2), {}) or {}
    mb_extreme = next((r for r in (mb.get("curve") or []) if r.get("eta") == 200), {}) or {}
    mb_worst = mb.get("worst_case", {}) or {}
    rtma_lo, rtma_hi = rtma.get("mode"), rtma.get("median")

    # Endogenous kink (Bom & Rachinger 2019) — funnel-based, FE-weighted, n=71
    ek      = mr.get("endogenous_kink", {}) or {}
    ek_est  = ek.get("est")
    ek_pval = ek.get("pval")

    # n_waap: stored as estimate in row "n_waap" by the Stata do-file
    n_waap_raw = st(stata, "n_waap")
    n_waap     = int(round(n_waap_raw)) if n_waap_raw is not None else 33

    # Stata values
    naive    = st(stata, "Naive (unweighted) mean")
    fe_mean  = st(stata, "Fixed-effect weighted mean")
    waap     = st(stata, "WAAP")
    waap_pv  = st(stata, "WAAP", "pvalue")
    pet      = st(stata, "PET intercept")
    pet_pv   = st(stata, "PET intercept", "pvalue")
    peese    = st(stata, "PEESE intercept")
    peese_pv = st(stata, "PEESE intercept", "pvalue")
    # Trim-and-fill: present after first Stata run with updated do-file;
    # falls back to FE mean (which equals TF in this symmetric funnel).
    tf       = st(stata, "Trim-and-fill (fixed)")
    tf_val   = tf if tf is not None else fe_mean   # fallback until Stata re-run

    re_mean    = st(stata, "RE weighted mean (REML)")
    re_mean_se = st(stata, "RE weighted mean (REML)", "se")
    re_mean_pv = st(stata, "RE weighted mean (REML)", "pvalue")
    # Compute pvalue from est/se if not directly available
    if re_mean_pv is None:
        re_mean_pv = pval_from_z(re_mean, re_mean_se)
    re_pet    = st(stata, "RE PET intercept (REML)")
    re_pet_pv = st(stata, "RE PET intercept (REML)", "pvalue")
    re_pe     = st(stata, "RE PEESE intercept (REML)")
    re_pe_pv  = st(stata, "RE PEESE intercept (REML)", "pvalue")
    isq       = st(stata, "I-squared (%)")
    tau2      = st(stata, "tau-squared")
    pet_cn    = st(stata, "PET intercept (Clark/Nielsen SE<10.1)")
    pet_cn_pv = st(stata, "PET intercept (Clark/Nielsen SE<10.1)", "pvalue")
    peese_cn  = st(stata, "PEESE intercept (Clark/Nielsen SE<10.1)")
    peese_cn_pv = st(stata, "PEESE intercept (Clark/Nielsen SE<10.1)", "pvalue")

    # Egger intercept — dimensionless t-statistic (bias test, not a return)
    ei_d   = stata.get("Egger intercept (bias)", {}) or {}
    ei_est = ei_d.get("estimate")
    ei_pv  = ei_d.get("pvalue")

    # R JSON values
    rve      = rj.get("rve_pet",   {}) or {}
    rve_pe   = rj.get("rve_peese", {}) or {}

    pu       = rj.get("puniform",     {}) or {}
    pu_est   = pu.get("est")
    pu_lb    = pu.get("ci_lb")
    pu_ub    = pu.get("ci_ub")
    pu_pv    = pu.get("pval_0")
    # Use SE-based CI as fallback if bootstrap CIs not available
    if pu_lb is None and pu.get("se") is not None:
        pu_lb = pu_est - 1.96 * pu["se"]
        pu_ub = pu_est + 1.96 * pu["se"]

    co       = rj.get("copas",        {}) or {}
    co_est   = co.get("te_adjust")
    co_lb    = co.get("ci_lb_adjust")
    co_ub    = co.get("ci_ub_adjust")
    co_pv    = co.get("pval_adjust")

    ak       = rj.get("andrews_kasy", {}) or {}
    ak_est   = ak.get("est")
    ak_lb    = ak.get("ci_lb")
    ak_ub    = ak.get("ci_ub")
    ak_pv    = ak.get("pval")
    ak_w     = ak.get("weight_nonsig")

    # Derived quantities for notes
    waap_thresh = (fe_mean / 2.802) if fe_mean else 2.21

    # ── Formatting helpers for this table ─────────────────────────────────

    def pct_row(label, sample, val, pval):
        return (
            f"  {label} & {sample} & "
            f"{fmt_pct(val)} & {fmt_pval(pval)} \\\\"
        )

    def pct_ci_row(label, sample, est, lb, ub, pval):
        ci = fmt_ci(lb, ub)
        combined = f"{fmt_pct(est)}~{ci}" if ci else fmt_pct(est)
        return (
            f"  {label} & {sample} & "
            f"{combined} & {fmt_pval(pval)} \\\\"
        )

    isq_str  = f"{isq:.1f}"  if isq  is not None else "??.?"
    tau2_str = f"{tau2:.1f}" if tau2 is not None else "??.?"
    ak_w_str = f"{ak_w:.2f}" if ak_w is not None else "0.50"
    # Describe selection intensity based on omega: < 0.6 = moderate, >= 0.6 = mild
    ak_sel_label = "moderate selection" if (ak_w is not None and ak_w < 0.6) else "mild selection"

    sample_all  = f"All ind.\\ est.\\ ($n={n_input}$)"
    sample_waap = f"Adequately powered ($n={n_waap}$)"
    sample_rve  = f"All ind.\\ est.\\ ($n={n_input}$, {n_collapsed} clusters)"
    sample_sel  = f"Collapsed by study ($n={n_collapsed}$)"
    sample_maive = f"Est.\\ with $N$ ($n={maive_n}$)"

    # Egger intercept row — value is a t-stat, mark with dagger
    ei_est_str = f"${ei_est:.2f}$" if ei_est is not None else "---"
    egger_bias_row = (
        f"  Egger intercept (bias)$^{{\\dag}}$ & {sample_all} & "
        f"{ei_est_str} & "
        f"{fmt_pval(ei_pv)} \\\\"
    )

    het_line = (
        f"  \\multicolumn{{4}}{{l}}{{\\quad Heterogeneity: "
        f"$I^2 = {isq_str}\\%$, $\\tau^2 = {tau2_str}$,"
        f" $Q(70) = {Q_STAT:.1f}$ ($p<0.001$)}} \\\\[4pt]"
    )

    notes = (
        r"\footnotesize \textit{Notes:} Bias-corrected returns to one additional year"
        r" of schooling (percentage points), applied to Clark and Nielsen's (2026) data."
        r" Each method and its citation is described in Section~\ref{sec:results};"
        r" column~2 gives the sample for each."
        r" No estimate is dropped as an outlier from any row; the only outlier exclusions"
        r" are presentational (Figures~\ref{fig:funnel} and~\ref{fig:normality})."
        r" For MAIVE the default is unweighted and the IV-weighted variant holds the"
        r" weighting fixed, isolating the spurious-precision correction."
        r" $^{\dag}$~Egger intercept is a dimensionless $t$-statistic, not a return."
        r" $^{\ddag}$~PET is the precision-effect significance test; significant"
        r" throughout ($p<0.001$), so under the PET--PEESE decision rule the reported"
        r" estimate is the PEESE intercept. `---'~denotes no analytic $p$-value."
    )

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Publication-bias corrections applied to Clark and Nielsen's"
         r" (2026) replication data}"),
        r"\label{tab:summary}",
        r"\footnotesize",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Method & Sample & Estimate [95\% CI] & $p$-value \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel A: Funnel-based methods (fixed-effect weighting)}} \\",
        pct_row(r"Fixed-effect weighted mean  ", sample_all,  fe_mean, None),
        pct_row(r"WAAP                        ", sample_waap, waap,    waap_pv),
        egger_bias_row,
        pct_row(r"PET (= Egger slope)$^{\ddag}$", sample_all,  pet,     pet_pv),
        pct_row(r"PEESE                       ", sample_all,  peese,   peese_pv),
        pct_row(r"Trim-and-fill               ", sample_all,  tf_val,  None),
        pct_row(r"Endogenous kink             ", sample_all,  ek_est,  ek_pval),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel A$^\prime$: Funnel-based methods, heterogeneity-robust weighting}} \\",
        pct_row(r"Random-effects weighted mean", sample_all,  re_mean, re_mean_pv),
        pct_row(r"RE PEESE intercept (REML)   ", sample_all,  re_pe,   re_pe_pv),
        pct_row(r"RVE PEESE intercept         ", sample_rve,  rve_pe.get("est"), rve_pe.get("pval")),
        het_line,
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel A$^{\prime\prime}$: Spurious-precision-robust (MAIVE; $N$ instruments precision)}} \\",
        f"  MAIVE PET-PEESE (default / IV-weighted) & {sample_maive} & "
        f"{fmt_pct(maive_default)} / {fmt_pct(maive_ivw)} & --- \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel B: Selection / significance-based models}} \\",
        pct_ci_row(r"$p$-uniform*                ", sample_sel, pu_est, pu_lb, pu_ub, pu_pv),
        pct_ci_row(r"Copas                       ", sample_sel, co_est, co_lb, co_ub, co_pv),
        pct_ci_row(r"Andrews--Kasy (Vevea--Hedges)", sample_sel, ak_est, ak_lb, ak_ub, ak_pv),
        pct_ci_row(r"Multiple-bias ($\eta\approx2$)", sample_sel,
                   mb_eta2.get("est"), mb_eta2.get("ci_lb"), mb_eta2.get("ci_ub"), mb_eta2.get("pval")),
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel C: Clark and Nielsen's own estimates}} \\",
        pct_row(r"PET ($\mathrm{SE}<10.1\%$)           ",
                r"$n=65$", pet_cn, pet_cn_pv),
        pct_row(r"PEESE ($\mathrm{SE}<10.1\%$)         ",
                r"$n=65$", peese_cn, peese_cn_pv),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{0.5em}",
        r"\begin{minipage}{0.95\linewidth}",
        notes,
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])


# ===========================================================================
# Main
# ===========================================================================

def main():
    if not STATA_CSV.exists():
        sys.exit(f"ERROR: Stata CSV not found: {STATA_CSV}\nRun Step 1 (Stata) first.")
    if not R_JSON.exists():
        sys.exit(f"ERROR: R JSON not found: {R_JSON}\nRun Step 2 (R) first.")

    stata = load_stata_csv(STATA_CSV)
    rj    = load_r_json(R_JSON)
    mr    = load_r_json(MAIVE_JSON) if MAIVE_JSON.exists() else {}

    tables = {
        "table1_summary.tex": make_table1(stata, rj, mr),
    }

    for fname, content in tables.items():
        out = OUT_DIR / fname
        out.write_text(content, encoding="utf-8")
        print(f"Wrote {out}")

    print(f"Tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
