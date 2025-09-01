import numpy as np
from math import gcd


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
    Solve x @ B = v over GF(p) for x (coefficients of pivot rows).
    B: shape (r, m)
    v: shape (m,)
    Returns: x (length r) or None if no solution
    """
    GF = type(B)
    r, m = B.shape

    # Solve: x @ B = v  <=>  B^T @ x^T = v^T
    A = B.T          # shape (m x r)
    b = v.reshape(-1, 1)  # shape (m x 1)

    Ab = GF(np.hstack((A, b)))  # shape (m x (r+1))
    R = Ab.row_reduce()

    # Check for inconsistency: row of [0 ... 0 | nonzero]
    for row in R:
        if np.all(row[:-1] == 0) and row[-1] != 0:
            raise Exception("Inconsistent system")  # inconsistent system

    # Back-substitution
    num_vars = A.shape[1]
    x = GF.Zeros(num_vars)
    pivot_rows = 0
    for i in range(R.shape[0]):
        row = R[i]
        nz = np.nonzero(row[:-1])[0]
        if len(nz) == 0:
            continue  # skip zero row
        pivot_col = nz[0]
        x[pivot_col] = row[-1]
        pivot_rows += 1

    if np.array_equal(x @ B, v):
        return x
    else:
        raise Exception("Failed to solve linear system")
