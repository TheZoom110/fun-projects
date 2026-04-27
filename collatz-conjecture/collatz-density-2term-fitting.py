# ── REPLACE the models dict and fitting section with this ─────────────────

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re

def parse_output(text):
    """
    Extracts (N, density) from terminal output lines like:
      N=  10,000,000  |  same-T: 4,964,705  | ...
    Works regardless of spacing or comma formatting.
    """
    pairs = []
    pattern = r'([\d]+),([\d]+)'
    for line in text.splitlines():
        m = re.search(pattern, line)
        if m:
            n       = int(m.group(1))
            same_t  = int(m.group(2))
            # N here is the upper end of the pair, pairs checked = N - start
            # start = 2, so pairs checked = N - 2
            pairs_checked = n - 2
            if pairs_checked > 0:
                density = same_t / pairs_checked
                pairs.append((n, density))
    return pairs

# Load data
with open('collatz-10pow8-report.csv', 'r') as f:
    raw_large = f.read()

# Fine-resolution small-N data — paste your low-N output here as a string
# or load from a second file if you saved it
with open('collatz-10pow6-finer-report.csv', 'r') as f:
    raw_small = f.read()

all_data = sorted(set(parse_output(raw_large) + parse_output(raw_small)),
                  key=lambda x: x[0])

Ns_all   = np.array([d[0] for d in all_data])
dens_all = np.array([d[1] for d in all_data])
logN_all = np.log10(Ns_all)

print(f"Total data points: {len(all_data)}")
print(f"N range: {Ns_all.min():,} to {Ns_all.max():,}")
print(f"Density range: {dens_all.min():.6f} to {dens_all.max():.6f}")

# ── SPLIT INTO TWO REGIMES ─────────────────────────────────────────────────
# Boundary chosen where the curvature visibly changes
BOUNDARY = 100_000

mask_small = Ns_all <  BOUNDARY
mask_large = Ns_all >= BOUNDARY

Ns_small,   dens_small   = Ns_all[mask_small],   dens_all[mask_small]
Ns_large,   dens_large   = Ns_all[mask_large],   dens_all[mask_large]
logN_small, logN_large   = logN_all[mask_small],  logN_all[mask_large]

print(f"\nSmall-N regime (N < {BOUNDARY:,}): {mask_small.sum()} points")
print(f"Large-N regime (N ≥ {BOUNDARY:,}): {mask_large.sum()} points")

# ── MODEL DEFINITIONS ──────────────────────────────────────────────────────

def exp_approach_1(logN, a, b, r):
    """Single-term: d = a - b*r^log10(N)"""
    return a - b * np.power(r, logN)

def exp_approach_2(logN, a, b1, r1, b2, r2):
    """Two-term: d = a - b1*r1^log10(N) - b2*r2^log10(N)"""
    return a - b1 * np.power(r1, logN) - b2 * np.power(r2, logN)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def aic(y_true, y_pred, k):
    n   = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k

# ── FIT 1: LARGE-N ONLY (asymptotic regime) ───────────────────────────────
print("\n" + "="*65)
print("FIT A: Large-N asymptotic regime (N ≥ 100,000)")
print("="*65)

popt_large, pcov_large = curve_fit(
    exp_approach_1, logN_large, dens_large,
    p0=[0.60, 0.34, 0.845], maxfev=10000)
a_L, b_L, r_L = popt_large
perr_L = np.sqrt(np.diag(pcov_large))
pred_large = exp_approach_1(logN_large, *popt_large)
alpha_L = -np.log10(r_L)

print(f"  a  = {a_L:.6f} ± {perr_L[0]:.6f}   ← d* (limiting density)")
print(f"  b  = {b_L:.6f} ± {perr_L[1]:.6f}")
print(f"  r  = {r_L:.6f} ± {perr_L[2]:.6f}")
print(f"  α  = {alpha_L:.6f}")
print(f"  R² = {r_squared(dens_large, pred_large):.6f}")
print(f"  AIC= {aic(dens_large, pred_large, 3):.2f}")

# Validate on 10^9 (should be in large data)
n9 = 1_000_000_000
pred_9 = exp_approach_1(np.log10(n9), *popt_large)
print(f"\n  Prediction at 10^9: {pred_9:.6f}  (actual: 0.526051)")
print(f"  Error: {pred_9 - 0.526051:+.6f}")

# ── FIT 2: TWO-TERM MODEL ON ALL DATA ────────────────────────────────────
print("\n" + "="*65)
print("FIT B: Two-term model (all N)")
print("="*65)

# Initial guess: large-N params + fast-decaying small-N correction
# ERROR: p0_2 = [a_L, b_L, r_L, 0.5, 0.2]
p0_2 = [
    np.clip(a_L, 0.55, 0.70),
    np.clip(b_L, 0.00, 1.00),
    np.clip(r_L, 0.80, 0.99),
    0.5,   # b2 — already a literal, fine
    0.20   # r2 — already a literal, fine
]
bounds_2 = ([0.55, 0.0, 0.80, 0.0, 0.01],
            [0.70, 1.0, 0.99, 2.0, 0.60])

popt_2, pcov_2 = curve_fit(
    exp_approach_2, logN_all, dens_all,
    p0=p0_2, bounds=bounds_2, maxfev=20000)
a2, b1, r1, b2, r2 = popt_2
perr_2 = np.sqrt(np.diag(pcov_2))
pred_2_all = exp_approach_2(logN_all, *popt_2)
alpha1 = -np.log10(r1)
alpha2 = -np.log10(r2)

print(f"  a  = {a2:.6f} ± {perr_2[0]:.6f}   ← d* (limiting density)")
print(f"  b1 = {b1:.6f} ± {perr_2[1]:.6f}   ← large-N amplitude")
print(f"  α1 = {alpha1:.6f}              ← large-N exponent")
print(f"  b2 = {b2:.6f} ± {perr_2[3]:.6f}   ← small-N amplitude")
print(f"  α2 = {alpha2:.6f}              ← small-N exponent")
print(f"  R² = {r_squared(dens_all, pred_2_all):.6f}")
print(f"  AIC= {aic(dens_all, pred_2_all, 5):.2f}")
print(f"\n  d(N) ≈ {a2:.5f}"
      f" - {b1:.5f}·N^(-{alpha1:.5f})"
      f" - {b2:.5f}·N^(-{alpha2:.5f})")

pred_9_2 = exp_approach_2(np.log10(n9), *popt_2)
print(f"\n  Prediction at 10^9: {pred_9_2:.6f}  (actual: 0.526051)")
print(f"  Error: {pred_9_2 - 0.526051:+.6f}")

# ── RESIDUALS ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Residual analysis")
print("="*65)

for label, y_true, y_pred, logN_use in [
    ("Fit A large-N only", dens_large, pred_large, logN_large),
    ("Fit B two-term all", dens_all,   pred_2_all, logN_all)
]:
    resid = y_true - y_pred
    signs = np.sign(resid)
    runs  = 1 + np.sum(signs[1:] != signs[:-1])
    print(f"\n  {label}:")
    print(f"    RMSE      : {np.sqrt(np.mean(resid**2)):.8f}")
    print(f"    Max resid : {resid.max():+.6f}")
    print(f"    Min resid : {resid.min():+.6f}")
    print(f"    Sign runs : {runs}  (random ≈ {len(resid)//2})")

# ── EXTRAPOLATION TABLE ───────────────────────────────────────────────────
extrap = [1e8, 1e9, 1e10, 1e12, 1e15, 1e20]
print("\n" + "="*65)
print(f"{'N':<8}", end='')
for en in extrap:
    print(f"  10^{int(np.log10(en))}", end='')
print()
print("-"*65)
for label, fn, params in [
    ("Fit A", exp_approach_1, popt_large),
    ("Fit B", exp_approach_2, popt_2)
]:
    print(f"{label:<8}", end='')
    for en in extrap:
        v = fn(np.log10(en), *params)
        flag = "!" if v > 1.0 else " "
        print(f"  {flag}{v:.4f}", end='')
    print()

# ── PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Left: both fits over full data range
ax = axes[0]
ax.scatter(logN_all, dens_all, s=3, color='black', alpha=0.4,
           label='Data', zorder=3)
lx = np.linspace(logN_all.min(), logN_all.max(), 1000)
ax.plot(lx, exp_approach_1(lx, *popt_large),
        color='blue', lw=2, label='Fit A: large-N only')
ax.plot(lx, exp_approach_2(lx, *popt_2),
        color='orange', lw=2, label='Fit B: two-term all')
ax.axvline(np.log10(BOUNDARY), color='grey', lw=1, ls='--',
           label=f'Boundary N={BOUNDARY:,}')
ax.set_xlabel('$\\log_{{10}} N$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Model fits', fontsize=13)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Middle: residuals
ax = axes[1]
resid_A = dens_large - pred_large
resid_B = dens_all   - pred_2_all
ax.scatter(logN_large, resid_A, s=3, color='blue',
           alpha=0.5, label='Fit A residuals')
ax.scatter(logN_all,   resid_B, s=3, color='orange',
           alpha=0.5, label='Fit B residuals')
ax.axhline(0, color='black', lw=1)
ax.axvline(np.log10(BOUNDARY), color='grey', lw=1, ls='--')
ax.set_xlabel('$\\log_{{10}} N$', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Residuals', fontsize=13)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Right: extrapolation
ax = axes[2]
ax.scatter(logN_all, dens_all, s=3, color='black', alpha=0.3,
           label='Data', zorder=3)
ext_lx = np.linspace(2, 20, 1000)
ax.plot(ext_lx, exp_approach_1(ext_lx, *popt_large),
        color='blue', lw=2, label=f'Fit A  d*={a_L:.4f}')
ax.plot(ext_lx, exp_approach_2(ext_lx, *popt_2),
        color='orange', lw=2, label=f'Fit B  d*={a2:.4f}')
ax.axhline(1.0, color='black', lw=1, ls='--', label='Ceiling = 1')
ax.axhline(a_L, color='blue',   lw=1, ls=':')
ax.axhline(a2,  color='orange', lw=1, ls=':')
ax.set_xlabel('$\\log_{{10}} N$', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Extrapolation to $10^{20}$', fontsize=13)
ax.set_ylim(0.28, 0.75)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('collatz-10pow8-finer-2term-fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to collatz_twoterm_fit.png")