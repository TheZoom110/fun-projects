"""
Counts isolated numbers (not part of any same-T streak) at each power of 2.
Uses multiprocessing — workers are fully independent, no inter-process data.

Strategy:
  1. Split [2, 2^MAX_K] into NCORES equal chunks.
  2. Each worker counts isolated numbers in its chunk, treating its first and
     last elements as if their external neighbours have a different T (boundary
     assumption). This may misclassify at most 2 elements per chunk.
  3. After all workers finish, one fixup pass recomputes T at each chunk
     boundary (4 values per boundary) and corrects any misclassification.
"""

import time
import multiprocessing as mp

MAX_K  = 30
NCORES = 4

# ── CORE ──────────────────────────────────────────────────────────────────

def collatz_T(n):
    t = 0
    while n != 1:
        n = 3 * n + 1 if n & 1 else n >> 1
        t += 1
    return t

def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0

# ── WORKER ────────────────────────────────────────────────────────────────

def process_chunk(args):
    """
    Count isolated numbers in [chunk_start, chunk_end].

    Boundary assumption: T of the element just before chunk_start and just
    after chunk_end are treated as "different" from the chunk's boundary
    elements. This is corrected by the main process after all workers finish.

    Returns:
        isolated   : count of isolated numbers in this chunk
        checkpoints: list of (k, isolated_up_to_2^k_within_chunk)
        T_boundary : dict with T values at the first 2 and last 2 elements
                     — needed for fixup
    """
    chunk_start, chunk_end = args

    isolated    = 0
    checkpoints = []
    T_boundary  = {}   # positions → T(position)

    T_prev = None      # Left boundary unknown → treated as "different"
    T_curr = collatz_T(chunk_start)

    for n in range(chunk_start, chunk_end + 1):
        T_next = collatz_T(n + 1) if n < chunk_end else None

        # Treat unknown boundary T as "different"
        left_diff  = (T_prev is None) or (T_prev != T_curr)
        right_diff = (T_next is None) or (T_curr != T_next)

        if left_diff and right_diff:
            isolated += 1

        # Record boundary T values (first 2 and last 2 elements)
        if n <= chunk_start + 1 or n >= chunk_end - 1:
            T_boundary[n] = T_curr

        if is_power_of_2(n):
            checkpoints.append((n.bit_length() - 1, isolated))

        elif (n & 2_097_151) == 0:
            print(f"  [worker {chunk_start}-{chunk_end}] ... {n:>15,}",
                  flush=True)

        T_prev = T_curr
        T_curr = T_next

    return isolated, checkpoints, T_boundary

# ── FIXUP ─────────────────────────────────────────────────────────────────

def compute_adjustments(chunks, results):
    """
    For each internal boundary between chunk i (ending at b) and
    chunk i+1 (starting at b+1), compute corrections to isolated counts.

    Each worker assumed:
      - Its first element had a left  neighbour with a different T  (+0 or -1)
      - Its last  element had a right neighbour with a different T  (+0 or -1)

    We compute actual T values at the boundary and correct any errors.

    Returns:
        adj_start[i]: adjustment to chunk i's count for its FIRST element
        adj_end[i]  : adjustment to chunk i's count for its LAST  element
    """
    n_chunks    = len(chunks)
    adj_start   = [0] * n_chunks
    adj_end     = [0] * n_chunks

    for i in range(n_chunks - 1):
        b    = chunks[i][1]       # last element of chunk i
        b1   = chunks[i + 1][0]  # first element of chunk i+1 (= b + 1)

        T_bnd = results[i][2]      # T_boundary dict for chunk i
        T_bn1 = results[i + 1][2]  # T_boundary dict for chunk i+1

        T_bm1 = T_bnd.get(b - 1)   # T(b-1) — second-to-last of chunk i
        T_b   = T_bnd.get(b)        # T(b)   — last of chunk i
        T_b1  = T_bn1.get(b1)       # T(b+1) — first of chunk i+1
        T_b2  = T_bn1.get(b1 + 1)   # T(b+2) — second of chunk i+1

        # Was b correctly classified?
        # Worker i assumed T(b+1) != T(b). Actual right diff = T(b) != T(b+1).
        # Worker i counted b as isolated if T(b-1) != T(b) [left diff was real].
        # Correction needed if T(b) == T(b+1) and T(b-1) != T(b):
        #   b was counted as isolated but should not be → adj_end[i] = -1
        if T_b == T_b1 and (T_bm1 is None or T_bm1 != T_b):
            adj_end[i] = -1

        # Was b+1 correctly classified?
        # Worker i+1 assumed T(b) != T(b+1). Actual left diff = T(b) != T(b+1).
        # Worker counted b+1 as isolated if T(b+2) != T(b+1).
        # Correction needed if T(b) == T(b+1) and T(b+2) != T(b+1):
        #   b+1 was counted as isolated but should not be → adj_start[i+1] = -1
        if T_b == T_b1 and (T_b2 is None or T_b1 != T_b2):
            adj_start[i + 1] = -1

    return adj_start, adj_end

# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    END = 2 ** MAX_K

    print(f"Counting isolated numbers from N=2 to 2^{MAX_K} = {END:,}")
    print(f"Using {NCORES} cores\n")
    print(f"  {'k':>4}  {'N=2^k':>15}  {'isolated':>13}  "
          f"{'total N':>13}  {'isolated%':>11}")
    print(f"  {'-'*4}  {'-'*15}  {'-'*13}  {'-'*13}  {'-'*11}")

    # ── SPLIT INTO EQUAL CHUNKS ───────────────────────────────────────────
    total     = END - 2 + 1
    base      = total // NCORES
    remainder = total % NCORES

    chunks = []
    start  = 2
    for i in range(NCORES):
        size = base + (1 if i < remainder else 0)
        end  = start + size - 1
        chunks.append((start, end))
        start = end + 1

    t0 = time.time()

    # ── RUN WORKERS IN PARALLEL ───────────────────────────────────────────
    with mp.Pool(processes=NCORES) as pool:
        results = list(pool.imap(process_chunk, chunks))
    # results[i] = (isolated, checkpoints, T_boundary)

    # ── BOUNDARY FIXUP ────────────────────────────────────────────────────
    adj_start, adj_end = compute_adjustments(chunks, results)

    # ── ACCUMULATE RESULTS ────────────────────────────────────────────────
    # Corrected total per chunk
    corrected = [results[i][0] + adj_start[i] + adj_end[i]
                 for i in range(NCORES)]

    # Prefix sums: prefix[i] = sum of corrected totals for chunks 0..i-1
    prefix = [0] * NCORES
    for i in range(1, NCORES):
        prefix[i] = prefix[i - 1] + corrected[i - 1]

    # Identify which chunk each power of 2 lives in
    chunk_ranges = chunks

    all_k = sorted({k for _, checkpoints, _ in results for k, _ in checkpoints})

    for k in all_k:
        power = 2 ** k
        for i, (cs, ce) in enumerate(chunk_ranges):
            if cs <= power <= ce:
                # partial count within chunk i up to 2^k
                partial = next(c for kk, c in results[i][1] if kk == k)
                # apply start adjustment (affects all elements in chunk i)
                partial += adj_start[i]
                # apply end adjustment only if 2^k is the last element of chunk i
                if power == ce:
                    partial += adj_end[i]
                total_isolated = prefix[i] + partial
                total_N        = power - 1   # integers in [2, 2^k]
                print(f"  {k:>4}  {power:>15,}  {total_isolated:>13,}  "
                      f"{total_N:>13,}  {100*total_isolated/total_N:>10.3f}%")
                break

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s  "
          f"({(END - 2) / elapsed:,.0f} N/sec)")

if __name__ == "__main__":
    main()