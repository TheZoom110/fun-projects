# Cell 1: Core Collatz Engine
# Computes (s, r) for each N without storing full trajectory.
# Uses Python's native arbitrary-precision integers — no overflow ever.

def collatz_sr(n):
    """
    Returns (s, r) where:
        s = number of odd steps  (3n+1)
        r = number of even steps (n/2)
    for the Collatz trajectory of n, terminating at 1.
    Returns None if n < 1.
    """
    if n < 1:
        return None
    s, r = 0, 0
    while n != 1:
        if n & 1:        # n is odd
            n = 3 * n + 1
            s += 1
        else:            # n is even
            n >>= 1
            r += 1
    return s, r


def collatz_sr_batch(start, end):
    """
    Computes (s, r) for all N in [start, end] inclusive.
    Returns a list of (s, r) tuples indexed from start.
    """
    return [collatz_sr(n) for n in range(start, end + 1)]


def get_print_interval(n):
    """
    Returns the print interval appropriate for the current N:
      every 100    for N <       1,000
      every 1,000  for N <      10,000
      every 10,000 for N <     100,000
      every 100,000 for N <  1,000,000
      every 500,000 for N >= 1,000,000
    """
    if   n <     1_000: return       100
    elif n <    10_000: return     1_000
    elif n <   100_000: return    10_000
    elif n < 1_000_000: return   100_000
    else:               return   500_000


# Quick sanity check
test_cases = [(3, (2, 5)), (5, (1, 4)), (7, (5, 11)), (27, (41, 70))]
print("Sanity checks (N: expected (s,r) → got (s,r)):")
all_pass = True
for n, expected in test_cases:
    got = collatz_sr(n)
    status = "✅" if got == expected else "❌"
    print(f"  N={n}: expected {expected} → got {got} {status}")
    if got != expected:
        all_pass = False
if all_pass:
    print("\nAll sanity checks passed. Engine is correct.")

# Cell 2: Three-Statement Verifier
# Checks all three statements over a range, reports counterexamples.

def verify_statements(start, end):
    """
    For all consecutive pairs (N, N+1) in [start, end-1], checks:

    Statement 1: T(N) = T(N+1)  →  s(N) = s(N+1)  [and r auto-follows]
    Statement 2: s(N) = s(N+1)  →  r(N) = r(N+1)
    Statement 3: r(N) = r(N+1)  →  s(N) = s(N+1)

    Returns a dict with counts and all counterexamples found.
    """
    counts = {
        "pairs_checked":      0,
        "same_T":             0,   # pairs where T(N) = T(N+1)
        "same_s":             0,   # pairs where s(N) = s(N+1)
        "same_r":             0,   # pairs where r(N) = r(N+1)
        "all_three_equal":    0,   # pairs where T, s, r all equal
        "s1_violations":      0,   # Statement 1 violated
        "s2_violations":      0,   # Statement 2 violated
        "s3_violations":      0,   # Statement 3 violated
    }
    counterexamples = {"s1": [], "s2": [], "s3": []}
    MAX_STORE = 10  # store at most this many counterexamples each

    # We compute one step ahead to avoid recomputing
    prev_s, prev_r = collatz_sr(start)

    for n in range(start, end):
        curr_s, curr_r = collatz_sr(n + 1)

        T_prev = prev_s + prev_r
        T_curr = curr_s + curr_r

        same_T = (T_prev == T_curr)
        same_s = (prev_s == curr_s)
        same_r = (prev_r == curr_r)

        counts["pairs_checked"] += 1
        if same_T: counts["same_T"] += 1
        if same_s: counts["same_s"] += 1
        if same_r: counts["same_r"] += 1
        if same_T and same_s and same_r: counts["all_three_equal"] += 1

        # Statement 1: same_T → same_s (and same_r)
        if same_T and not same_s:
            counts["s1_violations"] += 1
            if len(counterexamples["s1"]) < MAX_STORE:
                counterexamples["s1"].append({
                    "N": n,
                    "T_N": T_prev, "s_N": prev_s, "r_N": prev_r,
                    "T_N1": T_curr, "s_N1": curr_s, "r_N1": curr_r
                })

        # Statement 2: same_s → same_r
        if same_s and not same_r:
            counts["s2_violations"] += 1
            if len(counterexamples["s2"]) < MAX_STORE:
                counterexamples["s2"].append({
                    "N": n,
                    "s_N": prev_s, "r_N": prev_r,
                    "s_N1": curr_s, "r_N1": curr_r
                })

        # Statement 3: same_r → same_s
        if same_r and not same_s:
            counts["s3_violations"] += 1
            if len(counterexamples["s3"]) < MAX_STORE:
                counterexamples["s3"].append({
                    "N": n,
                    "s_N": prev_s, "r_N": prev_r,
                    "s_N1": curr_s, "r_N1": curr_r
                })

        prev_s, prev_r = curr_s, curr_r

        if (n + 1) % get_print_interval(n + 1) == 0:
            n_checked = counts["pairs_checked"]
            print(f"  N={n+1:>12,}  |  "
                  f"same-T: {counts['same_T']:>8,}  |  "
                  f"S1 violations: {counts['s1_violations']}  |  "
                  f"S2 violations: {counts['s2_violations']}  |  "
                  f"S3 violations: {counts['s3_violations']}")

    return counts, counterexamples


def print_report(counts, counterexamples, start, end):
    print("\n" + "="*65)
    print(f"  VERIFICATION REPORT: N = {start:,} to {end:,}")
    print("="*65)
    p = counts["pairs_checked"]
    print(f"\n  Consecutive pairs checked : {p:>12,}")
    print(f"  Pairs with T(N)=T(N+1)   : {counts['same_T']:>12,}  "
          f"({100*counts['same_T']/p:.3f}%)")
    print(f"  Pairs with s(N)=s(N+1)   : {counts['same_s']:>12,}  "
          f"({100*counts['same_s']/p:.3f}%)")
    print(f"  Pairs with r(N)=r(N+1)   : {counts['same_r']:>12,}  "
          f"({100*counts['same_r']/p:.3f}%)")
    print(f"  Pairs with all three equal: {counts['all_three_equal']:>12,}  "
          f"({100*counts['all_three_equal']/p:.3f}%)")

    for label, key in [("Statement 1 (same-T → same-s)", "s1"),
                        ("Statement 2 (same-s → same-r)", "s2"),
                        ("Statement 3 (same-r → same-s)", "s3")]:
        v = counts[f"{key}_violations"]
        status = "✅ NO VIOLATIONS" if v == 0 else f"❌ {v} VIOLATIONS FOUND"
        print(f"\n  {label}:")
        print(f"    {status}")
        if counterexamples[key]:
            print(f"    First counterexample: {counterexamples[key][0]}")

    # Density trend within the range
    print(f"\n  Density of same-T pairs  : "
          f"{counts['same_T']/p:.6f}  (~1 in {p//max(counts['same_T'],1):.1f})")
    print("="*65)

# Cell 3: Run Verification
# Adjust START and END to control the range.
# On Colab free tier: ~10^7 takes ~2 min, ~10^8 takes ~20 min,
# ~10^9 takes ~3-4 hours. Use GPU runtime for no speedup (CPU-bound).
# For >10^8, consider running in batches (see Cell 4).

import time

START = 2
END   = 1_000_000    # Change to 100_000_000 for 10^8, etc.

print(f"Verifying N = {START:,} to {END:,}")
print(f"Progress (adaptive intervals: 100/1K/10K/100K/500K by scale):\n")

t0 = time.time()
counts, counterexamples = verify_statements(START, END)
elapsed = time.time() - t0

print_report(counts, counterexamples, START, END)
print(f"\n  Total time: {elapsed:.1f}s  ({(END-START)/elapsed:,.0f} N/sec)")