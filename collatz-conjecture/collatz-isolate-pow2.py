"""
Counts isolated numbers (not part of any same-T streak) at each power of 2.
A number N is isolated iff T(N-1) != T(N) and T(N+1) != T(N).
"""

import time
import multiprocessing
print(multiprocessing.cpu_count())
MAX_K = 15    # reports at 2^2, 2^3, ..., 2^MAX_K

def collatz_T(n):
    t = 0
    while n != 1:
        n = 3 * n + 1 if n & 1 else n >> 1
        t += 1
    return t

def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0

END      = 2 ** MAX_K
isolated = 0

print(f"Counting isolated numbers from N=2 to 2^{MAX_K} = {END:,}\n")
print(f"  {'k':>4}  {'N=2^k':>15}  {'isolated':>13}  "
      f"{'total N':>13}  {'isolated%':>11}")
print(f"  {'-'*4}  {'-'*15}  {'-'*13}  {'-'*13}  {'-'*11}")

t0 = time.time()

# Initialise sliding window at n=2
T_prev = collatz_T(1)
T_curr = collatz_T(2)

for n in range(2, END + 1):
    T_next = collatz_T(n + 1) if n < END else None

    # N is isolated if both neighbours have different T
    if T_prev != T_curr and (T_next is None or T_curr != T_next):
        isolated += 1

    if is_power_of_2(n):
        k       = n.bit_length() - 1
        total_N = n - 1          # integers in [2, n]
        print(f"  {k:>4}  {n:>15,}  {isolated:>13,}  "
              f"{total_N:>13,}  {100*isolated/total_N:>10.3f}%")

    elif (n & 2_097_151) == 0:
        print(f"  ... {n:>15,}", flush=True)

    T_prev = T_curr
    T_curr = T_next if T_next is not None else T_curr

print(f"\n  Total time: {time.time()-t0:.1f}s")