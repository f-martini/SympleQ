import numpy as np
from math import gcd
import galois
from collections import defaultdict


def solve_linear_system_over_gf(A: np.ndarray, b: np.ndarray, GF: type | int) -> np.ndarray:
    """
    Solve the system A @ x = b over a finite field.
    Returns one particular solution with free variables set to zero.
    Raises ValueError if the system is inconsistent.
    """
    if isinstance(GF, int):
        GF = galois.GF(GF)

    A_gf = GF(A)
    b_gf = GF(b).reshape(-1, 1)

    if A_gf.ndim != 2:
        raise ValueError("Coefficient matrix must be 2-dimensional.")
    if A_gf.shape[0] != b_gf.shape[0]:
        raise ValueError(
            f"Incompatible shapes for A ({A_gf.shape}) and b ({b_gf.shape})."
        )

    m, n = A_gf.shape
    augmented = np.hstack((A_gf, b_gf))
    R = augmented.copy()

    pivot_rows: list[tuple[int, int]] = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if R[r, col] != GF(0):
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            R[[row, pivot], :] = R[[pivot, row], :]

        pivot_val = R[row, col]
        R[row, :] /= pivot_val

        for r in range(m):
            if r != row and R[r, col] != GF(0):
                R[r, :] -= R[r, col] * R[row, :]

        pivot_rows.append((row, col))
        row += 1
        if row == m:
            break

    for r in range(m):
        if all(R[r, c] == GF(0) for c in range(n)) and R[r, n] != GF(0):
            raise ValueError("Inconsistent linear system over GF(p).")

    x = GF.Zeros(n)
    for row_idx, pivot_col in reversed(pivot_rows):
        value = R[row_idx, n]
        for j in range(pivot_col + 1, n):
            if R[row_idx, j] != GF(0):
                value -= R[row_idx, j] * x[j]
        x[pivot_col] = value

    return x.copy()


def solve_gf2(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Solve the linear system Ax = b over GF(2) using Gaussian elimination.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        Solution vector or None if no solution exists
    """
    GF2 = galois.GF(2)
    try:
        solution = solve_linear_system_over_gf(
            np.asarray(A, dtype=int) % 2, np.asarray(b, dtype=int) % 2, GF2
        )
    except ValueError:
        return None

    return solution.view(np.ndarray).astype(int, copy=False)


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
    solution = solve_linear_system_over_gf(B.T, v, GF)
    if not np.array_equal(solution @ B, v):
        raise ValueError("Failed to solve linear system over GF(p).")
    return solution


def gf_solve(A: np.ndarray, b: np.ndarray, GF: type) -> np.ndarray:
    """
    Solve A x = b over GF(p). Returns one particular solution or raises ValueError if inconsistent.
    """
    solution = solve_linear_system_over_gf(A, b, GF)
    return solution.reshape(-1, 1)


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
