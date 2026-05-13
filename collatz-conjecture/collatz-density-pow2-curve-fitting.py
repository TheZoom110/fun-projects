import numpy as np
from scipy.optimize import curve_fit

# ── DATA ───────────────────────────────────────────────────────────────────
raw = [
    ( 4,           2), ( 5,           7), ( 6,          18),
    ( 7,          40), ( 8,          86), ( 9,         180),
    (10,         373), (11,         774), (12,       1_624),
    (13,       3_381), (14,       6_935), (15,      14_224),
    (16,      29_177), (17,      59_570), (18,     121_388),
    (19,     246_903), (20,     500_898), (21,   1_014_412),
    (22,   2_053_126), (23,   4_153_081), (24,   8_393_046),
    (25,  16_951_586), (26,  34_216_182), (27,  69_029_924),
    (28, 139_204_551), (29, 280_583_387), (30, 565_306_123),
]

ks     = np.array([r[0] for r in raw])
same_T = np.array([r[1] for r in raw], dtype=float)
Ns     = 2.0 ** ks
pairs  = Ns - 2
dens   = same_T / pairs
logN   = np.log10(Ns)   # = k * log10(2)

# Large-k only: asymptotic regime where curvature is low
# k >= 16 chosen from where increment ratios stabilise
K_BOUNDARY = 16
mask_large = ks >= K_BOUNDARY
logN_large = logN[mask_large]
dens_large = dens[mask_large]

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def aic(y_true, y_pred, k_params):
    n   = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k_params

def one_term(logN, a, b, r):
    return a - b * np.power(r, logN)

def two_term(logN, a, b1, r1, b2, r2):
    return a - b1 * np.power(r1, logN) - b2 * np.power(r2, logN)

# ── FIT A: ONE-TERM, LARGE-K ONLY (k >= 16) ───────────────────────────────
print("=" * 65)
print(f"FIT A: One-term, large-k only (k ≥ {K_BOUNDARY}, {mask_large.sum()} points)")
print("=" * 65)

pA, cA = curve_fit(one_term, logN_large, dens_large,
                   p0=[0.60, 0.34, 0.845], maxfev=10000)
predA  = one_term(logN_large, *pA)
eA     = np.sqrt(np.diag(cA))
aA     = pA[0]
alphaA = -np.log10(pA[2])

print(f"  a  = {aA:.8f} ± {eA[0]:.8f}   ← d*")
print(f"  b  = {pA[1]:.8f} ± {eA[1]:.8f}")
print(f"  r  = {pA[2]:.8f} ± {eA[2]:.8f}")
print(f"  α  = {alphaA:.8f}")
print(f"  R² = {r_squared(dens_large, predA):.8f}")
print(f"  AIC= {aic(dens_large, predA, 3):.4f}")
print(f"\n  d(N) ≈ {aA:.6f} - {pA[1]:.6f} · N^(-{alphaA:.6f})")

# ── FIT B: TWO-TERM, ALL 27 POINTS ────────────────────────────────────────
print("\n" + "=" * 65)
print("FIT B: Two-term, all 27 points")
print("=" * 65)

pB, cB = curve_fit(two_term, logN, dens,
                   p0=[aA, pA[1], pA[2], 0.5, 0.3],
                   bounds=([0.55, 0.0, 0.50, 0.0, 0.01],
                           [0.75, 2.0, 0.99, 5.0, 0.70]),
                   maxfev=30000)
predB  = two_term(logN, *pB)
eB     = np.sqrt(np.diag(cB))
aB     = pB[0]
alpha1 = -np.log10(pB[2])
alpha2 = -np.log10(pB[4])

print(f"  a  = {aB:.8f} ± {eB[0]:.8f}   ← d*")
print(f"  b1 = {pB[1]:.8f} ± {eB[1]:.8f}   ← slow-decay amplitude")
print(f"  α1 = {alpha1:.8f}              ← slow-decay exponent")
print(f"  b2 = {pB[3]:.8f} ± {eB[3]:.8f}   ← fast-decay amplitude")
print(f"  α2 = {alpha2:.8f}              ← fast-decay exponent")
print(f"  R² = {r_squared(dens, predB):.8f}")
print(f"  AIC= {aic(dens, predB, 5):.4f}")
print(f"\n  d(N) ≈ {aB:.6f}"
      f" - {pB[1]:.6f}·N^(-{alpha1:.6f})"
      f" - {pB[3]:.6f}·N^(-{alpha2:.6f})")

# ── RESIDUAL ANALYSIS ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Residual analysis")
print("=" * 65)
for label, pred, y_true in [
    ("Fit A (large-k only)", predA, dens_large),
    ("Fit B (two-term all)", predB, dens)
]:
    resid = y_true - pred
    signs = np.sign(resid)
    runs  = 1 + np.sum(signs[1:] != signs[:-1])
    print(f"\n  {label}:")
    print(f"    RMSE      : {np.sqrt(np.mean(resid**2)):.8f}")
    print(f"    Max resid : {resid.max():+.8f}")
    print(f"    Min resid : {resid.min():+.8f}")
    print(f"    Sign runs : {runs}  (random ≈ {len(resid)//2})")

# ── PER-POINT TABLE ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Per-point comparison (all 27 points)")
print("=" * 65)
predA_all = one_term(logN, *pA)
print(f"  {'k':>4}  {'N':>14}  {'actual':>10}  "
      f"{'Fit A':>10}  {'Fit B':>10}  "
      f"{'A err%':>9}  {'B err%':>9}")
print("  " + "-"*75)
for i, (k, _) in enumerate(raw):
    N   = 2**k
    act = dens[i]
    vA  = predA_all[i]
    vB  = predB[i]
    eAp = 100*(act-vA)/act
    eBp = 100*(act-vB)/act
    flag = " ←" if k < K_BOUNDARY else ""
    print(f"  {k:>4}  {N:>14,}  {act:>10.6f}  "
          f"{vA:>10.6f}  {vB:>10.6f}  "
          f"{eAp:>+9.4f}  {eBp:>+9.4f}{flag}")

# ── CONSTANT PROXIMITY CHECK ───────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"Proximity to known constants  (d* from Fit A = {aA:.8f})")
print("=" * 65)
candidates = {
    "3/5"                : 3/5,
    "log(2)"             : np.log(2),
    "1/log2(3)"          : 1/np.log2(3),
    "log2(3)-1"          : np.log2(3)-1,
    "1-log10(3)"         : 1-np.log10(3),
    "1-1/e"              : 1-1/np.e,
    "1/(1+log10(2))"     : 1/(1+np.log10(2)),
    "log10(2)+log10(3)"  : np.log10(2)+np.log10(3),
    "sqrt(2)-1"          : np.sqrt(2)-1,
    "2-sqrt(2)"          : 2-np.sqrt(2),
    "pi/2-1"             : np.pi/2-1,
    "1/pi+1/3"           : 1/np.pi+1/3,
    "1-log(2)/2"         : 1-np.log(2)/2,
    "2*log10(3)-1/2"     : 2*np.log10(3)-0.5,
    "log10(4)"           : np.log10(4),
    "1/sqrt(e)"          : 1/np.sqrt(np.e),
}
print(f"  {'Expression':<22}  {'Value':>12}  {'Δ':>12}  {'|Δ|/d*%':>10}")
print("  " + "-"*62)
for name, val in sorted(candidates.items(), key=lambda x: abs(x[1]-aA)):
    delta = aA - val
    print(f"  {name:<22}  {val:>12.8f}  {delta:>+12.8f}  {100*abs(delta)/aA:>10.6f}%")

# ── EXTRAPOLATION ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Extrapolation")
print("=" * 65)
print(f"  {'k (2^k)':>12}  {'Fit A':>10}  {'Fit B':>10}")
print("  " + "-"*38)
for kk in [31, 32, 40, 50, 64, 100, 200]:
    logN_e = kk * np.log10(2)
    vA = one_term(logN_e, *pA)
    vB = two_term(logN_e, *pB)
    print(f"  2^{kk:<10}  {vA:>10.6f}  {vB:>10.6f}")