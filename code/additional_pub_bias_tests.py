"""
Additional publication bias tests for Clark & Nielsen (2024)
"The Returns to Education: A Meta-study"

Replicating the comprehensive battery of tests used in
Crawfurd et al. "How Much Would Reducing Lead Exposure Improve
Children's Learning Levels in the Developing World?"
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

np.random.seed(20260417)

# ============================================================
# Load data
# ============================================================
ROOT = Path(__file__).parent.parent
df = pd.read_stata(ROOT / 'data' / 'Returns_to_education.dta')

# Independent estimates only (as in Clark & Nielsen)
ind = df[df['ind_est'] == 1].copy()
ind = ind.dropna(subset=['effect_percentage', 'se_percentage'])

effect = ind['effect_percentage'].values
se = ind['se_percentage'].values
# Recompute t-stats from effect/se for consistency (some stored tstats look wrong)
tstat_raw = ind['tstat'].values
tstat = effect / se  # recompute

print("=" * 70)
print("ADDITIONAL PUBLICATION BIAS TESTS FOR CLARK & NIELSEN (2024)")
print("=" * 70)
print(f"\nN independent estimates: {len(ind)}")
print(f"Mean effect: {effect.mean():.2f}%")
print(f"Median effect: {np.median(effect):.2f}%")
print(f"Mean SE: {se.mean():.2f}%")
print(f"Negative effects: {(effect < 0).sum()} ({(effect < 0).mean()*100:.1f}%)")
print(f"Insignificant at 5%: {(np.abs(tstat) < 1.96).sum()} ({(np.abs(tstat) < 1.96).mean()*100:.1f}%)")

# ============================================================
# 1. TRIM AND FILL (Duval & Tweedie 2000)
# ============================================================
print("\n" + "=" * 70)
print("1. TRIM AND FILL (Duval & Tweedie 2000)")
print("=" * 70)

def trim_and_fill(effects, ses, max_iter=100):
    """
    Implement the Duval & Tweedie trim-and-fill estimator (L0 estimator).
    Returns adjusted mean, number of imputed studies, and imputed effects/ses.
    """
    # Inverse variance weights
    w = 1.0 / ses**2

    # Initial fixed-effect estimate
    theta_hat = np.sum(w * effects) / np.sum(w)

    for iteration in range(max_iter):
        # Rank absolute deviations from theta_hat
        deviations = effects - theta_hat
        abs_dev = np.abs(deviations)
        ranks = stats.rankdata(abs_dev)
        n = len(effects)

        # Count studies on the side with fewer (right side = positive deviations for positive bias)
        # For positive publication bias, missing studies are on the left
        right_side = deviations > 0

        # L0 estimator for number of missing studies
        # T_n = max ranks on the right side minus expected under symmetry
        right_ranks = ranks[right_side]
        S_n = np.sum(right_ranks) - np.sum(np.arange(1, sum(right_side) + 1))
        k0 = max(0, int(round((4 * S_n - n * (n + 1)) / (2 * n - 1))))

        if k0 == 0:
            break

        # Trim k0 most extreme positive studies
        sorted_idx = np.argsort(deviations)[::-1]
        keep_idx = sorted_idx[k0:]

        # Recalculate theta
        w_trim = 1.0 / ses[keep_idx]**2
        theta_new = np.sum(w_trim * effects[keep_idx]) / np.sum(w_trim)

        if abs(theta_new - theta_hat) < 1e-6:
            theta_hat = theta_new
            break
        theta_hat = theta_new

    # Now impute the missing studies by reflecting the k0 most extreme
    # positive deviations around the adjusted mean
    if k0 > 0:
        sorted_idx = np.argsort(effects - theta_hat)[::-1]
        extreme_idx = sorted_idx[:k0]
        imputed_effects = 2 * theta_hat - effects[extreme_idx]
        imputed_ses = ses[extreme_idx]

        # Combined estimate
        all_effects = np.concatenate([effects, imputed_effects])
        all_ses = np.concatenate([ses, imputed_ses])
        w_all = 1.0 / all_ses**2
        adjusted_mean = np.sum(w_all * all_effects) / np.sum(w_all)
        adjusted_se = np.sqrt(1.0 / np.sum(w_all))
    else:
        adjusted_mean = theta_hat
        adjusted_se = np.sqrt(1.0 / np.sum(w))
        imputed_effects = np.array([])
        imputed_ses = np.array([])

    return adjusted_mean, adjusted_se, k0, imputed_effects, imputed_ses

adj_mean, adj_se, k0, imp_eff, imp_ses = trim_and_fill(effect, se)
orig_w = 1.0 / se**2
orig_mean = np.sum(orig_w * effect) / np.sum(orig_w)
orig_se_mean = np.sqrt(1.0 / np.sum(orig_w))

print(f"\nOriginal fixed-effect estimate: {orig_mean:.2f}% (SE: {orig_se_mean:.2f})")
print(f"Number of imputed 'missing' studies: {k0}")
print(f"Trim-and-fill adjusted estimate: {adj_mean:.2f}% (SE: {adj_se:.2f})")
print(f"95% CI: [{adj_mean - 1.96*adj_se:.2f}, {adj_mean + 1.96*adj_se:.2f}]")
if k0 > 0:
    print(f"Imputed effect sizes: {imp_eff.round(2)}")

# ============================================================
# 2. PET-PEESE (replication + extension)
# ============================================================
print("\n" + "=" * 70)
print("2. PET-PEESE (Stanley & Doucouliagos 2014)")
print("=" * 70)

# PET: effect = b0 + b1*SE + error, weighted by 1/SE^2
X_pet = sm.add_constant(se)
wls_pet = sm.WLS(effect, X_pet, weights=1.0/se**2).fit(cov_type='HC1')
print(f"\nPET (effect ~ SE):")
print(f"  Intercept (bias-adjusted effect): {wls_pet.params[0]:.2f}% (SE: {wls_pet.bse[0]:.2f}, p={wls_pet.pvalues[0]:.4f})")
print(f"  SE coefficient (pub bias): {wls_pet.params[1]:.2f} (SE: {wls_pet.bse[1]:.2f}, p={wls_pet.pvalues[1]:.4f})")

# PEESE: effect = b0 + b1*SE^2 + error, weighted by 1/SE^2
X_peese = sm.add_constant(se**2)
wls_peese = sm.WLS(effect, X_peese, weights=1.0/se**2).fit(cov_type='HC1')
print(f"\nPEESE (effect ~ SE^2):")
print(f"  Intercept (bias-adjusted effect): {wls_peese.params[0]:.2f}% (SE: {wls_peese.bse[0]:.2f}, p={wls_peese.pvalues[0]:.4f})")
print(f"  SE^2 coefficient: {wls_peese.params[1]:.4f} (SE: {wls_peese.bse[1]:.4f}, p={wls_peese.pvalues[1]:.4f})")

# PET-PEESE decision rule: if PET intercept is significant, use PEESE
if wls_pet.pvalues[0] < 0.05:
    print(f"\n  => PET intercept is significant => Use PEESE estimate: {wls_peese.params[0]:.2f}%")
else:
    print(f"\n  => PET intercept is NOT significant => Use PET estimate: {wls_pet.params[0]:.2f}%")
    print(f"     (This suggests the bias-corrected effect may not differ from zero)")

# Excluding outliers (se > 10) as Clark & Nielsen do
mask = se < 10.1
X_pet_trim = sm.add_constant(se[mask])
wls_pet_trim = sm.WLS(effect[mask], X_pet_trim, weights=1.0/se[mask]**2).fit(cov_type='HC1')
X_peese_trim = sm.add_constant(se[mask]**2)
wls_peese_trim = sm.WLS(effect[mask], X_peese_trim, weights=1.0/se[mask]**2).fit(cov_type='HC1')
print(f"\nExcluding SE > 10 outliers (n={mask.sum()}):")
print(f"  PET intercept: {wls_pet_trim.params[0]:.2f}% (p={wls_pet_trim.pvalues[0]:.4f})")
print(f"  PEESE intercept: {wls_peese_trim.params[0]:.2f}% (p={wls_peese_trim.pvalues[0]:.4f})")

# ============================================================
# 3. p-uniform* (van Aert et al 2018) - simplified version
# ============================================================
print("\n" + "=" * 70)
print("3. p-UNIFORM* (van Aert et al 2018) - approximation")
print("=" * 70)

def p_uniform_star(effects, ses, alpha=0.05):
    """
    Simplified p-uniform* estimator.
    Under the true effect size mu, one-sided p-values of significant
    studies should be uniformly distributed.
    We find mu that makes the conditional p-values most uniform.
    """
    def objective(mu):
        # Calculate one-sided p-values under H0: effect = mu
        z_scores = (effects - mu) / ses
        p_vals = 1 - stats.norm.cdf(z_scores)

        # Only use "significant" studies (original p < alpha for positive effect)
        orig_z = effects / ses
        orig_p = 1 - stats.norm.cdf(orig_z)
        sig_mask = orig_p < alpha

        if sig_mask.sum() < 3:
            return 1e10

        # Conditional p-values for significant studies
        cond_p = p_vals[sig_mask]
        # Under true mu, these should be uniform
        # Use KS test statistic as objective
        cond_p_sorted = np.sort(cond_p)
        n_sig = len(cond_p_sorted)
        expected = np.arange(1, n_sig + 1) / (n_sig + 1)
        return np.sum((cond_p_sorted - expected)**2)

    # Grid search followed by optimization
    mu_grid = np.linspace(-10, 20, 300)
    obj_vals = [objective(mu) for mu in mu_grid]
    mu_start = mu_grid[np.argmin(obj_vals)]

    result = optimize.minimize_scalar(objective, bounds=(-15, 25), method='bounded')
    return result.x

mu_puniform = p_uniform_star(effect, se)
print(f"\np-uniform* estimate of true effect: {mu_puniform:.2f}%")
print(f"(Compare to naive mean: {effect.mean():.2f}%)")

# Also try with one-sided test for positive effects (more common in this lit)
mu_puniform_onesided = p_uniform_star(effect, se, alpha=0.10)
print(f"p-uniform* (alpha=0.10): {mu_puniform_onesided:.2f}%")

# ============================================================
# 4. COPAS SENSITIVITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("4. COPAS-STYLE SENSITIVITY ANALYSIS")
print("=" * 70)

def copas_sensitivity(effects, ses, gamma_range=np.linspace(0, 2, 20)):
    """
    Simplified Copas-style sensitivity analysis.
    Models selection probability as a function of statistical significance.
    For each gamma (strength of selection), compute the bias-corrected estimate.

    Selection model: P(selected) = Phi(gamma0 + gamma1/se)
    We approximate this by reweighting by inverse of estimated selection probability.
    """
    results = []

    for gamma1 in gamma_range:
        # Selection probability proportional to significance
        z = effects / ses
        # Higher gamma1 = more selection on significance
        if gamma1 == 0:
            weights = np.ones(len(effects))
        else:
            # Approximate selection probability
            prob_sig = stats.norm.cdf(z - 1.96)
            # Selection weight: upweight studies less likely to be selected
            selection_prob = 0.3 + 0.7 * stats.norm.cdf(gamma1 * (np.abs(z) - 1.96))
            weights = 1.0 / (selection_prob * ses**2)

        # Weighted mean
        mu_adj = np.sum(weights * effects) / np.sum(weights)
        se_adj = np.sqrt(1.0 / np.sum(1.0/ses**2))  # approximate
        results.append({'gamma1': gamma1, 'mu_adj': mu_adj, 'se_adj': se_adj})

    return pd.DataFrame(results)

copas_results = copas_sensitivity(effect, se)
print("\nSensitivity of effect to selection strength:")
print(f"{'Selection strength':>20s} {'Adjusted effect':>18s}")
print("-" * 40)
for _, row in copas_results.iloc[::3].iterrows():
    label = "None" if row['gamma1'] == 0 else f"{row['gamma1']:.2f}"
    print(f"{label:>20s} {row['mu_adj']:>14.2f}%")

print(f"\nWith no selection: {copas_results.iloc[0]['mu_adj']:.2f}%")
print(f"With strong selection: {copas_results.iloc[-1]['mu_adj']:.2f}%")

# ============================================================
# 5. Z-STATISTIC DISTRIBUTION PLOT
# ============================================================
print("\n" + "=" * 70)
print("5. Z-STATISTIC DISTRIBUTION")
print("=" * 70)

valid_t = tstat[~np.isnan(tstat)]
print(f"\nDistribution of t-statistics (n={len(valid_t)}):")
print(f"  Mean: {np.nanmean(valid_t):.2f}")
print(f"  Median: {np.nanmedian(valid_t):.2f}")
print(f"  |t| < 1.96: {(np.abs(valid_t) < 1.96).sum()} ({(np.abs(valid_t) < 1.96).mean()*100:.1f}%)")
print(f"  t > 1.96: {(valid_t > 1.96).sum()} ({(valid_t > 1.96).mean()*100:.1f}%)")
print(f"  t < 0: {(valid_t < 0).sum()} ({(valid_t < 0).mean()*100:.1f}%)")

# Caliper test around 1.96
bandwidth = 0.5
above = ((valid_t >= 1.96) & (valid_t < 1.96 + bandwidth)).sum()
below = ((valid_t >= 1.96 - bandwidth) & (valid_t < 1.96)).sum()
print(f"\n  Caliper test around 1.96 (bandwidth={bandwidth}):")
print(f"    Just above 1.96: {above}")
print(f"    Just below 1.96: {below}")
print(f"    Ratio above/below: {above/max(below,1):.2f}")
if above + below > 0:
    p_caliper = stats.binomtest(above, above + below, 0.5).pvalue
    print(f"    Binomial test p-value: {p_caliper:.4f}")

# Wider bandwidth
bandwidth2 = 1.0
above2 = ((valid_t >= 1.96) & (valid_t < 1.96 + bandwidth2)).sum()
below2 = ((valid_t >= 1.96 - bandwidth2) & (valid_t < 1.96)).sum()
print(f"\n  Caliper test around 1.96 (bandwidth={bandwidth2}):")
print(f"    Just above 1.96: {above2}")
print(f"    Just below 1.96: {below2}")
if above2 + below2 > 0:
    p_caliper2 = stats.binomtest(above2, above2 + below2, 0.5).pvalue
    print(f"    Binomial test p-value: {p_caliper2:.4f}")

# ============================================================
# 6. EGGER REGRESSION (formal)
# ============================================================
print("\n" + "=" * 70)
print("6. EGGER REGRESSION (formal test)")
print("=" * 70)

# Egger: effect/se = b0 + b1*(1/se) + error
# Equivalent to: effect = b0*se + b1 + error (precision-weighted)
# Intercept b0 != 0 indicates asymmetry/publication bias
prec = 1.0 / se
z_scores_egger = effect / se

X_egger = sm.add_constant(prec)
ols_egger = sm.OLS(z_scores_egger, X_egger).fit(cov_type='HC1')

print(f"\nEgger regression: t-stat = intercept + slope * precision")
print(f"  Intercept (bias indicator): {ols_egger.params[0]:.3f} (SE: {ols_egger.bse[0]:.3f}, p={ols_egger.pvalues[0]:.4f})")
print(f"  Slope (true effect): {ols_egger.params[1]:.3f} (SE: {ols_egger.bse[1]:.3f}, p={ols_egger.pvalues[1]:.4f})")
if ols_egger.pvalues[0] < 0.05:
    print("  => SIGNIFICANT asymmetry detected (publication bias)")
else:
    print("  => No significant asymmetry at 5% level")

# ============================================================
# 7. FIGURES
# ============================================================
print("\n" + "=" * 70)
print("7. GENERATING FIGURES")
print("=" * 70)

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

# --- Panel A: Funnel plot with significance contours ---
ax1 = fig.add_subplot(gs[0, 0])
# Significance contours
se_range = np.linspace(0.01, max(se)*1.1, 100)
for z_crit, label, ls in [(1.96, 'p=0.05', '--'), (2.58, 'p=0.01', ':')]:
    ax1.plot(orig_mean + z_crit * se_range, se_range, color='gray', ls=ls, alpha=0.5)
    ax1.plot(orig_mean - z_crit * se_range, se_range, color='gray', ls=ls, alpha=0.5, label=label if z_crit == 1.96 else None)

ax1.scatter(effect, se, s=30, alpha=0.7, zorder=5, color='steelblue')
ax1.axvline(orig_mean, color='black', ls='-', lw=1, alpha=0.5)
ax1.axvline(adj_mean, color='red', ls='--', lw=1, alpha=0.7, label=f'Trim-fill adj ({adj_mean:.1f}%)')

# Plot imputed studies
if len(imp_eff) > 0:
    ax1.scatter(imp_eff, imp_ses, s=30, alpha=0.7, zorder=5, color='red', marker='D', label=f'{k0} imputed')

ax1.set_xlabel('Effect size (%)')
ax1.set_ylabel('Standard error (%)')
ax1.set_title('A. Funnel plot with significance contours\n& trim-and-fill')
ax1.invert_yaxis()
ax1.legend(fontsize=8, loc='lower left')
ax1.axvline(0, color='gray', ls='-', lw=0.5, alpha=0.3)

# --- Panel B: Funnel plot (standard orientation) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(effect, 1.0/se, s=30, alpha=0.7, color='steelblue')
ax2.axvline(orig_mean, color='black', ls='-', lw=1, alpha=0.5, label=f'FE mean ({orig_mean:.1f}%)')
ax2.axvline(0, color='gray', ls='--', lw=0.5, alpha=0.3)
ax2.set_xlabel('Effect size (%)')
ax2.set_ylabel('Precision (1/SE)')
ax2.set_title('B. Funnel plot (precision)')
ax2.legend(fontsize=8)

# --- Panel C: Z-statistic distribution ---
ax3 = fig.add_subplot(gs[1, 0])
valid_t_plot = valid_t[(valid_t > -5) & (valid_t < 10)]
bins = np.arange(-5, 10.5, 0.5)
ax3.hist(valid_t_plot, bins=bins, color='steelblue', alpha=0.7, edgecolor='white')
ax3.axvline(1.96, color='red', ls='--', lw=2, label='t = 1.96 (p=0.05)')
ax3.axvline(2.58, color='orange', ls='--', lw=1.5, label='t = 2.58 (p=0.01)')
ax3.axvline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
ax3.set_xlabel('t-statistic')
ax3.set_ylabel('Frequency')
ax3.set_title('C. Distribution of t-statistics')
ax3.legend(fontsize=8)

# --- Panel D: PET-PEESE ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(se, effect, s=30, alpha=0.7, color='steelblue')
se_plot = np.linspace(0, max(se)*1.05, 100)
# PET line
ax4.plot(se_plot, wls_pet.params[0] + wls_pet.params[1]*se_plot,
         color='red', ls='--', lw=2, label=f'PET: {wls_pet.params[0]:.1f}%')
# PEESE curve
ax4.plot(se_plot, wls_peese.params[0] + wls_peese.params[1]*se_plot**2,
         color='darkgreen', ls='-', lw=2, label=f'PEESE: {wls_peese.params[0]:.1f}%')
ax4.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
ax4.set_xlabel('Standard error (%)')
ax4.set_ylabel('Effect size (%)')
ax4.set_title('D. PET-PEESE')
ax4.legend(fontsize=8)

# --- Panel E: Copas sensitivity ---
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(copas_results['gamma1'], copas_results['mu_adj'], 'b-', lw=2)
ax5.axhline(0, color='gray', ls='--', lw=0.5)
ax5.axhline(effect.mean(), color='red', ls=':', alpha=0.5, label=f'Naive mean ({effect.mean():.1f}%)')
ax5.set_xlabel('Selection strength (gamma)')
ax5.set_ylabel('Adjusted effect (%)')
ax5.set_title('E. Copas-style sensitivity')
ax5.legend(fontsize=8)

# --- Panel F: Distribution of effects with normal fit ---
ax6 = fig.add_subplot(gs[2, 1])
eff_plot = effect[effect < 40]  # exclude extreme outlier
bins_eff = np.arange(-15, 35, 2.5)
n_hist, bins_hist, _ = ax6.hist(eff_plot, bins=bins_eff, color='steelblue', alpha=0.7,
                                  edgecolor='white', density=True, label='Observed')
# Fit normal
mu_fit, sigma_fit = stats.norm.fit(eff_plot)
x_norm = np.linspace(-15, 35, 200)
ax6.plot(x_norm, stats.norm.pdf(x_norm, mu_fit, sigma_fit), 'r-', lw=2,
         label=f'Normal fit (mu={mu_fit:.1f}, sigma={sigma_fit:.1f})')
# Normal centered at 0
ax6.plot(x_norm, stats.norm.pdf(x_norm, 0, sigma_fit), 'g--', lw=1.5,
         label=f'Normal at 0 (sigma={sigma_fit:.1f})')
ax6.axvline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
ax6.set_xlabel('Effect size (%)')
ax6.set_ylabel('Density')
ax6.set_title("F. Distribution of effects")
ax6.legend(fontsize=8)

plt.suptitle('Additional Publication Bias Tests: Clark & Nielsen (2024)',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(ROOT / 'output' / 'publication_bias_tests.png', dpi=150, bbox_inches='tight')
print("Saved figure: publication_bias_tests.png")

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL PUBLICATION BIAS ESTIMATES")
print("=" * 70)

summary = pd.DataFrame({
    'Method': [
        'Naive (unweighted) mean',
        'Fixed-effect (inverse-variance weighted) mean',
        'Egger regression slope (true effect)',
        'PET (Stanley & Doucouliagos)',
        'PET (excl. SE > 10)',
        'PEESE (Stanley & Doucouliagos)',
        'PEESE (excl. SE > 10)',
        'Trim-and-fill (Duval & Tweedie)',
        'p-uniform* (van Aert et al)',
    ],
    'Estimate (%)': [
        effect.mean(),
        orig_mean,
        ols_egger.params[1],  # slope = true effect in Egger
        wls_pet.params[0],
        wls_pet_trim.params[0],
        wls_peese.params[0],
        wls_peese_trim.params[0],
        adj_mean,
        mu_puniform,
    ],
    'SE': [
        np.nan,
        orig_se_mean,
        ols_egger.bse[1],
        wls_pet.bse[0],
        wls_pet_trim.bse[0],
        wls_peese.bse[0],
        wls_peese_trim.bse[0],
        adj_se,
        np.nan,
    ]
})

summary['p-value'] = [
    np.nan,
    np.nan,
    ols_egger.pvalues[1],
    wls_pet.pvalues[0],
    wls_pet_trim.pvalues[0],
    wls_peese.pvalues[0],
    wls_peese_trim.pvalues[0],
    np.nan,
    np.nan,
]

print(summary.to_string(index=False, float_format='%.3f'))

# Save summary table
summary.to_csv(ROOT / 'output' / 'publication_bias_summary.csv', index=False)
print(f"\nSaved summary table: publication_bias_summary.csv")

# ============================================================
# 9. ADDITIONAL: Shapiro-Wilk test of normality of effects
# ============================================================
print("\n" + "=" * 70)
print("9. ADDITIONAL DIAGNOSTICS")
print("=" * 70)

# Test normality
sw_stat, sw_p = stats.shapiro(effect)
print(f"\nShapiro-Wilk normality test on effect sizes:")
print(f"  W = {sw_stat:.4f}, p = {sw_p:.4f}")
if sw_p < 0.05:
    print("  => Effect sizes are NOT normally distributed (reject normality)")
else:
    print("  => Cannot reject normality")

# Skewness test
skew = stats.skew(effect)
skew_z, skew_p = stats.skewtest(effect)
print(f"\nSkewness: {skew:.3f} (z={skew_z:.2f}, p={skew_p:.4f})")
if skew > 0 and skew_p < 0.05:
    print("  => Significant RIGHT skew - consistent with missing negative results")

# Kolmogorov-Smirnov test against normal centered at 0
ks_stat, ks_p = stats.kstest((effect - 0) / effect.std(), 'norm')
print(f"\nKS test (effects vs normal centered at 0):")
print(f"  D = {ks_stat:.4f}, p = {ks_p:.4f}")

# K-S test against normal centered at fitted mean
ks_stat2, ks_p2 = stats.kstest((effect - mu_fit) / sigma_fit, 'norm')
print(f"KS test (effects vs fitted normal at {mu_fit:.1f}):")
print(f"  D = {ks_stat2:.4f}, p = {ks_p2:.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
