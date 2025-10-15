import numpy as np
from math import gcd
import galois
from collections import defaultdict


def solve_gf2(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ax = b over GF(2) using Gaussian elimination.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        Solution vector or None if no solution exists
    """
    A = A.astype(int)
    b = b.astype(int)
    m, n = A.shape

    # Augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        # Find pivot
        pivot_found = False
        for row in range(pivot_row, m):
            if Ab[row, col] == 1:
                # Swap rows
                if row != pivot_row:
                    Ab[[pivot_row, row]] = Ab[[row, pivot_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_cols.append(col)

        # Eliminate
        for row in range(m):
            if row != pivot_row and Ab[row, col] == 1:
                Ab[row] = (Ab[row] + Ab[pivot_row]) % 2

        pivot_row += 1

    # Check for inconsistency
    for row in range(pivot_row, m):
        if Ab[row, -1] == 1:
            raise ValueError(f"No solution exists\n{A}\n{b}")

    # Back substitution - find particular solution
    x = np.zeros(n, dtype=int)

    for i in range(len(pivot_cols) - 1, -1, -1):
        col = pivot_cols[i]
        # Find the row with pivot in this column
        row = i

        # Calculate value for this variable
        val = Ab[row, -1]
        for j in range(col + 1, n):
            val = (val + Ab[row, j] * x[j]) % 2
        x[col] = val

    return x


def solve_modular_linear_additive(x: int, z: int, d: int) -> int:
    """
    Find smallest non-negative integer n such that (x + z*n) % d == 0.
    Returns None if no solution exists
    """
    x, z, d = int(x), int(z), int(d)
    g = gcd(z, d)
    if x % g != 0:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    # Reduce the equation modulo d // g
    z_ = z // g
    d_ = d // g
    x_ = (-x // g) % d_

    # compute modular inverse
    try:
        z_inv = pow(z_, -1, d_)
    except ValueError:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    n = (x_ * z_inv) % d_
    return n


def solve_modular_linear_system(B, v):
    """
    Solve x @ B = v over GF(p) using row-reduction (RREF)
    """
    GF = type(B)
    m, n = B.shape
    # Transpose to solve B^T x^T = v^T
    A = GF(np.hstack([B.T, v.reshape(-1, 1)]))
    R = A.row_reduce()

    num_vars = B.shape[0]
    x = GF.Zeros(num_vars)

    # Identify pivot columns and back-substitute
    pivot_cols = []
    for row in R:
        nz = np.nonzero(row[:-1])[0]
        if len(nz) > 0:
            pivot_cols.append(nz[0])

    for i, col in enumerate(pivot_cols):
        x[col] = R[i, -1]

    # Verify solution
    if not np.array_equal(x @ B, v):
        raise Exception("Failed to solve linear system")
    return x


def mod_inv(M: np.ndarray, p: int):
    GFP = galois.GF(p)
    M = GFP(M)
    return np.linalg.inv(M)


def gf_rref(A: np.ndarray, GF: type) -> tuple[np.ndarray, list[int]]:
    A = GF(A)
    m, n = A.shape
    R = A.copy()
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if R[rr, c] != GF(0):
                piv = rr
                break
        if piv is None:
            continue
        if piv != r:
            R[[r, piv], :] = R[[piv, r], :]
        inv = GF(1) / R[r, c]
        R[r, :] *= inv
        for rr in range(m):
            if rr != r and R[rr, c] != GF(0):
                R[rr, :] -= R[rr, c] * R[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return R, pivots


def gf_solve(A: np.ndarray, b: np.ndarray, GF: type) -> np.ndarray:
    """
    Solve A x = b over GF(p). Returns one particular solution or raises ValueError if inconsistent.
    """
    A = GF(A)
    b = GF(b).reshape(-1, 1)
    m, n = A.shape
    M = np.hstack((A, b))
    R, _ = gf_rref(M, GF)
    # consistency check
    for i in range(m):
        if np.all(R[i, :n] == GF(0)) and R[i, n] != GF(0):
            raise ValueError("Inconsistent linear system over GF(p).")
    # particular solution (free vars = 0)
    x = GF.Zeros(n)
    row = 0
    for c in range(n):
        if row < m and R[row, c] == GF(1) and np.all(R[row, :c] == GF(0)):
            s = GF(0)
            for j in range(c + 1, n):
                s += R[row, j] * x[j]
            x[c] = R[row, n] - s
            row += 1
    return np.asarray(x).reshape(n, 1)



def get_linear_dependencies(vectors: np.ndarray,
                            p: int | list[int] | np.ndarray
                            ) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    """
    Find linear dependencies among rows of `vectors` over finite fields.

    Parameters
    ----------
    vectors : np.ndarray
        Matrix of shape (m, n). Each row is a vector.
    p : int | list[int] | np.ndarray
        - If int: all columns are over GF(p).
        - If list/array of length n: p[j] is the prime for column j.
        - If list/array of length n//2: per-qudit primes; each value is duplicated for X/Z columns.
        - If list/array of length m: p[i] is the prime field per row (legacy mode).

    Returns
    -------
    pivot_indices : list[int]
        Indices of linearly independent rows.
    dependencies : dict[int, list[tuple[int, int]]]
        Mapping: row index -> list of (pivot_index, coefficient) such that
        row = sum(coeff * pivot).
    """
    m, n = vectors.shape  # m = rows (vectors), n = columns

    # Normalize p into per-column primes when possible.
    p_cols: list[int]
    mode: str
    if isinstance(p, int):
        p_cols = [int(p)] * n
        mode = "per-column"
    elif isinstance(p, (list, np.ndarray)):
        lp = len(p)
        if lp == n:
            p_cols = [int(x) for x in p]
            mode = "per-column"
        elif lp == n // 2 and n % 2 == 0:
            # Per-qudit list provided; duplicate for X/Z halves
            p_cols = [int(x) for x in np.repeat(p, 2)]
            mode = "per-column"
        elif lp == m:
            # Legacy behavior: per-row primes
            ps = [int(x) for x in p]
            mode = "per-row"
        else:
            raise AssertionError(
                f"Length of p must be either rows={m}, cols={n}, or cols/2={n//2} (for qudits). Got {lp}"
            )
    else:
        raise TypeError(f"p must be int or list/np.ndarray of ints, it is {type(p)}")

    pivot_indices: list[int] = []
    dependencies: dict[int, list[tuple[int, int]]] = {}

    if mode == "per-row":
        # Group rows by prime and process each group independently
        prime_groups = defaultdict(list)
        for i, prime in enumerate(ps):
            prime_groups[prime].append(i)

        for prime, indices in prime_groups.items():
            if len(indices) == 1:
                pivot_indices.append(indices[0])
                continue

            GF = galois.GF(prime)
            V = GF(vectors[indices, :])

            # Identify group pivots incrementally
            group_pivots = []
            seen = GF.Zeros((0, V.shape[1]))
            for idx_in_group, i in enumerate(indices):
                candidate = V[idx_in_group]
                A = GF(np.vstack([seen, candidate]))
                R = A.row_reduce()
                rank = sum(1 for row in R if np.any(row != 0))
                if rank > seen.shape[0]:
                    group_pivots.append(i)
                    seen = A
            pivot_indices.extend(group_pivots)

            # Dependencies within the group
            if len(group_pivots) > 0:
                B = V[[indices.index(j) for j in group_pivots], :]
                for idx_in_group, i in enumerate(indices):
                    if i in group_pivots:
                        continue
                    x = solve_modular_linear_system(B, V[idx_in_group])
                    deps = [(group_pivots[j], int(x[j])) for j in range(len(x)) if x[j] != 0]
                    dependencies[i] = deps

        return pivot_indices, dependencies

    # Per-column primes path
    # Build prime -> column indices map
    prime_cols: dict[int, list[int]] = defaultdict(list)
    for j, prime in enumerate(p_cols):
        prime_cols[int(prime)].append(j)

    unique_primes = list(prime_cols.keys())

    # Track seen rows per-prime on the respective column subsets
    seen_per_prime: dict[int, np.ndarray] = {}
    for q in unique_primes:
        GFq = galois.GF(q)
        seen_per_prime[q] = GFq.Zeros((0, len(prime_cols[q])))

    # Select pivots: a row is independent if it increases rank for any prime on its column subset
    for i in range(m):
        increases_any = False
        updated_seen = {}
        for q in unique_primes:
            cols = prime_cols[q]
            if len(cols) == 0:
                continue
            GFq = galois.GF(q)
            candidate = GFq(vectors[i, cols])
            A_prev = seen_per_prime[q]
            A = GFq(np.vstack([A_prev, candidate]))
            R = A.row_reduce()
            rank_new = sum(1 for row in R if np.any(row != 0))
            if rank_new > A_prev.shape[0]:
                increases_any = True
                updated_seen[q] = A
            else:
                updated_seen[q] = A_prev

        if increases_any:
            pivot_indices.append(i)
            for q in unique_primes:
                seen_per_prime[q] = updated_seen[q]
        else:
            # Mark as dependent; coefficients are computed later as best-effort
            continue

    # Compute dependencies best-effort:
    # If all columns share the same prime, compute exact coefficients as before.
    if len(unique_primes) == 1:
        q = unique_primes[0]
        GFq = galois.GF(q)
        cols = prime_cols[q]
        Vq = GFq(vectors[:, cols])
        Bq = Vq[pivot_indices, :]
        for i in range(m):
            if i in pivot_indices:
                continue
            x = solve_modular_linear_system(Bq, Vq[i])
            deps = [(pivot_indices[j], int(x[j])) for j in range(len(x)) if x[j] != 0]
            dependencies[i] = deps
    else:
        # Mixed primes: return independent set; dependency coefficients are non-unique across fields.
        dependencies = {}

    return pivot_indices, dependencies

