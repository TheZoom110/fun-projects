# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 20:48:15 2026

@author: souka
"""
### Update Line 39 & 171

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── 1. PARSE ────────────────────────────────────────────────────────────────

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


with open('collatz-E3-E8-finer-report-merged.csv', 'r') as f:
    raw = f.read()

data   = parse_output(raw)
Ns     = np.array([d[0] for d in data])
dens   = np.array([d[1] for d in data])
logN   = np.log10(Ns)

print(f"Parsed {len(data)} data points")
print(f"N range: {Ns.min():,} to {Ns.max():,}")
print(f"Density range: {dens.min():.6f} to {dens.max():.6f}")

# ── 2. MODEL DEFINITIONS ────────────────────────────────────────────────────

models = {}

# Model 1: Linear-log  d = a + b*log10(N)
def m_linlog(logN, a, b):
    return a + b * logN
models['Linear-log'] = (m_linlog, [0.3, 0.016], r'$a + b\log N$')

# Model 2: Log-log  d = a + b*log10(log10(N))
def m_loglog(logN, a, b):
    return a + b * np.log10(logN)
models['Log-log'] = (m_loglog, [0.4, 0.1], r'$a + b\log\log N$')

# Model 3: Power-log  d = a + b*log10(N)^c
def m_powlog(logN, a, b, c):
    return a + b * np.power(logN, c)
models['Power-log'] = (m_powlog, [0.3, 0.01, 1.2], r'$a + b(\log N)^c$')

# Model 4: Rational-log  d = a*log10(N) / (b + log10(N))  → saturates at a
def m_ratlog(logN, a, b):
    return a * logN / (b + logN)
models['Rational-log'] = (m_ratlog, [0.8, 5.0], r'$\frac{a\log N}{b+\log N}$')

# Model 5: Exponential-approach  d = a - b*r^log10(N)  → saturates at a
def m_expapp(logN, a, b, r):
    return a - b * np.power(r, logN)
models['Exp-approach'] = (m_expapp, [0.65, 0.5, 0.85], r'$a - b \cdot r^{\log N}$')

# ── 3. FIT AND SCORE ────────────────────────────────────────────────────────

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def aic(y_true, y_pred, k):
    """Akaike Information Criterion — lower is better."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k

results = {}
print("\n" + "="*70)
print(f"{'Model':<18} {'R²':>8} {'RMSE':>12} {'AIC':>10} {'Params'}")
print("="*70)

for name, (fn, p0, _) in models.items():
    try:
        popt, _ = curve_fit(fn, logN, dens, p0=p0, maxfev=10000)
        pred    = fn(logN, *popt)
        r2      = r_squared(dens, pred)
        rmse    = np.sqrt(np.mean((dens - pred)**2))
        aic_val = aic(dens, pred, len(popt))
        results[name] = (popt, pred, r2, rmse, aic_val)
        param_str = ', '.join([f'{p:.5f}' for p in popt])
        print(f"{name:<18} {r2:>8.6f} {rmse:>12.8f} {aic_val:>10.2f}  [{param_str}]")
    except Exception as e:
        print(f"{name:<18} FAILED: {e}")

best = min(results, key=lambda k: results[k][4])
print(f"\nBest model by AIC: {best}")

# ── 4. EXTRAPOLATION ────────────────────────────────────────────────────────

extrap_Ns = [1e8, 1e9, 1e10, 1e12, 1e15, 1e20]
print("\n" + "="*70)
print(f"{'Model':<18}", end='')
for en in extrap_Ns:
    print(f"  10^{int(np.log10(en))}", end='')
print()
print("="*70)

for name, (popt, _, _, _, _) in results.items():
    fn = models[name][0]
    print(f"{name:<18}", end='')
    for en in extrap_Ns:
        val = fn(np.log10(en), *popt)
        marker = " !" if val > 1.0 else "  "
        print(f"{marker}{val:>6.3f}", end='')
    print()

print("\n(!) = density exceeds 1.0, model breaks down there")

# ── 5. PLOT ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

# Left: fit over data range
ax = axes[0]
ax.scatter(logN, dens, s=4, color='black', alpha=0.5, label='Data', zorder=5)
for (name, (popt, pred, r2, _, _)), col in zip(results.items(), colors):
    fn = models[name][0]
    ax.plot(logN, pred, color=col, lw=1.5,
            label=f'{name} (R²={r2:.5f})')
ax.set_xlabel('$\\log_{10} N$', fontsize=12)
ax.set_ylabel('Density of same-T pairs', fontsize=12)
ax.set_title('Model fits over data range', fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: extrapolation to 10^20
ax = axes[1]
ext_logN = np.linspace(logN.min(), 20, 500)
ax.scatter(logN, dens, s=4, color='black', alpha=0.5, label='Data', zorder=5)
ax.axhline(1.0, color='black', lw=1, ls='--', label='Density = 1 (ceiling)')
ax.axhline(0.5, color='grey', lw=0.8, ls=':', label='Density = 0.5')
for (name, (popt, _, _, _, _)), col in zip(results.items(), colors):
    fn = models[name][0]
    ext_dens = fn(ext_logN, *popt)
    ax.plot(ext_logN, ext_dens, color=col, lw=1.5, label=name)
ax.set_xlabel('$\\log_{10} N$', fontsize=12)
ax.set_ylabel('Density of same-T pairs', fontsize=12)
ax.set_title('Extrapolation to $10^{20}$', fontsize=13)
ax.set_ylim(0.4, 1.1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('collatz-E3-E8-finer-curve-fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to collatz_density_fit.png")

# ── 6. RESIDUAL ANALYSIS ────────────────────────────────────────────────────

print("\n" + "="*70)
print("Residual analysis — checking for systematic trend in best model")
print("="*70)
best_pred = results[best][1]
residuals = dens - best_pred
print(f"Best model: {best}")
print(f"Max residual  : {residuals.max():+.8f}")
print(f"Min residual  : {residuals.min():+.8f}")
print(f"Mean residual : {residuals.mean():+.8f}")
print(f"Std residual  : {residuals.std():.8f}")

# Check if residuals are randomly signed or systematically trending
signs = np.sign(residuals)
runs  = 1 + np.sum(signs[1:] != signs[:-1])
print(f"Sign runs     : {runs} (high = random, low = systematic trend)")