"""
This file takes care of phase correction for symplectic automorphisms.

It uses H\\Omega P = \\Delta \\phi

Finds P for some target phase changes \\Delta \\phi, given the tableau H and symplectic form \\Omega.
"""

import numpy as np
import galois
from quaos.core.circuits.utils import symplectic_form


# --- RREF / solve / nullspace over GF(p) ---

def gf_rref(A, GF):
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


def gf_rank(A, GF):
    _, piv = gf_rref(A, GF)
    return len(piv)


def gf_nullspace(A, GF):
    A = GF(A)
    m, n = A.shape
    R, piv = gf_rref(A, GF)
    piv = set(piv)
    free = [j for j in range(n) if j not in piv]
    if not free:
        return []
    basis = []
    for f in free:
        x = GF.Zeros(n)
        x[f] = GF(1)
        row = 0
        for c in range(n):
            if c in piv:
                s = GF(0)
                for j in range(c + 1, n):
                    if R[row, j] != GF(0) and x[j] != GF(0):
                        s += R[row, j] * x[j]
                x[c] = -s
                row += 1
                if row == m:
                    break
        basis.append(x.reshape(n, 1))
    return basis


def gf_solve_particular(A, b, GF):
    """
    Solve A x = b over GF(p). Return one particular solution or raise ValueError if inconsistent.
    """
    A = GF(A)
    b = GF(b).reshape(-1, 1)
    m, n = A.shape
    M = np.hstack((A, b))
    R, piv = gf_rref(M, GF)
    # Check consistency
    for i in range(m):
        if np.all(R[i, :n] == GF(0)) and R[i, n] != GF(0):
            raise ValueError("No solution: system A x = b is inconsistent over GF(p).")
    # Reconstruct one solution with all free vars set to 0
    x = GF.Zeros(n)
    row = 0
    for c in range(n):
        if row < m and R[row, c] == GF(1) and np.all(R[row, :c] == GF(0)):
            s = GF(0)
            for j in range(c + 1, n):
                s += R[row, j] * x[j]
            x[c] = R[row, n] - s
            row += 1
    return x.reshape(n, 1)

# --- Optional: greedy sparsifier to reduce Hamming weight of P ---


def hamming_weight(x, GF):
    return int(np.count_nonzero(x != GF(0)))


def sparsify_solution(x0, N_basis, p, GF, passes=1):
    """
    Given particular solution x0 and nullspace basis vectors N_i,
    greedily pick coefficients \\alpha_i ∈ GF(p) to reduce Hamming weight of x.
    """
    if not N_basis:
        return x0
    x = x0.copy()
    for _ in range(passes):
        improved = False
        for N in N_basis:
            best_alpha = GF(0)
            best_w = hamming_weight(x, GF)
            for a_int in range(p):
                a = GF(a_int)
                x_try = x + a * N
                w = hamming_weight(x_try, GF)
                if w < best_w:
                    best_w = w
                    best_alpha = a
            if best_alpha != GF(0):
                x = x + best_alpha * N
                improved = True
        if not improved:
            break
    return x


def pauli_phase_correction(pauli_sum_tableau, delta_phi, p, minimize=False, passes=1):
    """
    Solve H Ω P = Δφ over GF(p).
    Returns:
        P             : (2n x 1) GF vector (one solution)
        is_unique     : True iff solution is unique
        nullity       : dim Null(H Ω)
    Args:
        minimize : if True, greedily reduce Hamming weight using the nullspace
        passes   : number of greedy passes (if minimize=True)
    """
    GF = galois.GF(p)
    pauli_sum_tableau = GF(pauli_sum_tableau)
    Omega = GF(symplectic_form(pauli_sum_tableau.shape[1] // 2))
    A = pauli_sum_tableau @ Omega
    b = GF(delta_phi).reshape(-1, 1)
    # Particular solution (or raises if inconsistent)
    P0 = gf_solve_particular(A, b, GF)
    # Nullspace and uniqueness
    N = gf_nullspace(A, GF)  # list of (2n x 1)
    nullity = len(N)
    is_unique = (nullity == 0)
    if minimize and nullity > 0:
        P = sparsify_solution(P0, N, p, GF, passes=passes)
    else:
        P = P0
    # Sanity check
    assert np.all((A @ P) == b), "Internal: solution check failed over GF(p)."
    return P, is_unique, nullity


if __name__ == "__main__":
    p = 5
    GF = galois.GF(p)

    # Dimensions: H is m x 2n, Omega is 2n x 2n, delta_phi is m x 1
    # Example (replace with your real H, Omega, dphi):
    m, n = 4, 3
    H = GF.Random((m, 2 * n))
    delta_phi = GF.Random(m)

    P, is_unique, nullity = pauli_phase_correction(H, delta_phi, p, minimize=True, passes=2)
    print("P:", P.reshape(-1))
    print("unique?", is_unique, "nullity:", nullity)
