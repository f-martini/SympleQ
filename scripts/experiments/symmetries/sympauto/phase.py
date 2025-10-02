from __future__ import annotations
from typing import Optional
import numpy as np
from algebra import gauss_solve_mod2, nullspace_mod2, in_image_mod2
from utils import i_pow, almost_zero

def qF_parity_vector_via_U(tableau: np.ndarray, F: np.ndarray) -> np.ndarray:
    N, two_n = tableau.shape
    n = two_n // 2
    U = np.zeros((two_n, two_n), dtype=int)
    U[n:, :n] = 1
    U_F = (F % 4) @ (U % 4) @ (F.T % 4) % 4
    d = np.diag(U_F) % 4
    Ppart = (2 * np.triu(U_F) - np.diag(d)) % 4
    q2 = np.zeros(N, dtype=int)
    for i in range(N):
        a = tableau[i, :] % 4
        p1 = int(d @ a) % 4
        p2 = int(a @ ((Ppart @ a) % 4)) % 4
        q2[i] = (-p1 + p2) & 1
    return q2

def build_phase_rhs_via_U(tableau: np.ndarray, phases: Optional[np.ndarray], pi: np.ndarray, F: np.ndarray, p: int) -> np.ndarray:
    N, two_n = tableau.shape
    n = two_n // 2
    mod = 2 * int(p)
    phi = np.zeros(N, dtype=int) if phases is None else (np.asarray(phases, dtype=int) % p)
    U = np.zeros((two_n, two_n), dtype=int)
    U[n:, :n] = 1
    U_F = (F % mod) @ (U % mod) @ (F.T % mod) % mod
    d = np.diag(U_F) % mod
    Ppart = (2 * np.triu(U_F) - np.diag(d)) % mod
    r = np.zeros(N, dtype=int)
    for i in range(N):
        a = (tableau[i, :] % mod).astype(int)
        p1 = int(d @ a) % mod
        p2 = int(a @ ((Ppart @ a) % mod)) % mod
        qF = (-p1 + p2) % mod
        r[i] = ( (2 * ((phi[pi[i]] - phi[i]) % p)) - qF ) % mod
    return r

def gauss_solve_mod4_unit(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    A = (M.astype(int) % 4).copy()
    y = (b.astype(int) % 4).copy()
    m, n = A.shape
    piv_cols, piv_rows = [], []
    r = 0
    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if (A[i, c] & 1) == 1:
                pivot = i; break
        if pivot == -1:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            y[r], y[pivot] = y[pivot], y[r]
        inv = 1 if (A[r, c] % 4) == 1 else 3
        A[r, :] = (A[r, :] * inv) % 4
        y[r] = (y[r] * inv) % 4
        piv_cols.append(c); piv_rows.append(r)
        for i in range(m):
            if i == r: continue
            f = A[i, c] % 4
            if f:
                A[i, :] = (A[i, :] - f * A[r, :]) % 4
                y[i] = (y[i] - f * y[r]) % 4
        r += 1
        if r == m: break
    for i in range(m):
        if np.all(A[i, :] % 4 == 0) and (y[i] % 4) != 0:
            return None
    x = np.zeros(n, dtype=int)
    for k in reversed(range(len(piv_cols))):
        i = piv_rows[k]; c = piv_cols[k]
        s = 0
        for j in range(n):
            if j != c and A[i, j] % 4:
                s = (s + A[i, j] * x[j]) % 4
        x[c] = (y[i] - s) % 4
    return x

def solve_h_qubits_mod4_with_lift(M: np.ndarray, b: np.ndarray, diag: dict | None = None) -> np.ndarray | None:
    # 1) direct Z4
    h4 = gauss_solve_mod4_unit(M, b)
    if h4 is not None and np.array_equal((M @ h4) % 4, b % 4):
        return h4 % 4
    if diag is not None: diag["phase_mod4_infeasible"] += 1

    # 2) 2-adic lift + nullspace attempts
    M2, b2 = (M % 2), (b % 2)
    h0 = gauss_solve_mod2(M2, b2)
    if h0 is None:
        if diag is not None: diag["phase_mod2_infeasible"] += 1
        return None

    resid = (b - (M @ h0) % 4) % 4
    if np.any(resid & 1):
        if diag is not None: diag["phase_lift_parity_fail"] += 1
        return None

    rhs2 = ((resid // 2) % 2).astype(int)
    delta = gauss_solve_mod2(M2, rhs2)
    if delta is not None:
        h = (h0 + 2 * delta) % 4
        if np.array_equal((M @ h) % 4, b % 4):
            return h

    ns = nullspace_mod2(M2)
    for mask in range(1, 1 << min(len(ns), 10)):  # modest breadth
        h0_try = h0.copy()
        for bit, v in enumerate(ns[:10]):
            if (mask >> bit) & 1:
                h0_try ^= v
        resid = (b - (M @ h0_try) % 4) % 4
        if np.any(resid & 1):
            continue
        rhs2 = ((resid // 2) % 2).astype(int)
        delta = gauss_solve_mod2(M2, rhs2)
        if delta is None:
            continue
        h = (h0_try + 2 * delta) % 4
        if np.array_equal((M @ h) % 4, b % 4):
            return h

    if diag is not None: diag["phase_lift_delta_fail"] += 1
    return None

def verify_weights_gate_global_phase(
    tableau: np.ndarray,
    coeffs_raw: np.ndarray | None,
    pi: np.ndarray,
    F: np.ndarray,
    h: np.ndarray,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    clamp_clifford: bool = True,  # snap global to {1,i,-1,-i} if close
) -> bool:
    if coeffs_raw is None:
        return True
    N, two_n = tableau.shape
    n = two_n // 2
    U = np.zeros((two_n, two_n), dtype=int)
    U[n:, :n] = 1
    U_F = (F % 4) @ (U % 4) @ (F.T % 4) % 4
    d = np.diag(U_F) % 4
    Ppart = (2 * np.triu(U_F) - np.diag(d)) % 4

    g_ref: complex | None = None
    for i in range(N):
        a = (tableau[i, :] % 4).astype(int)
        p1 = int(d @ a) % 4
        p2 = int(a @ ((Ppart @ a) % 4)) % 4
        qF = (-p1 + p2) % 4
        acq = (int(h @ a) + qF) % 4
        lhs = complex(coeffs_raw[pi[i]])
        rhs = i_pow(acq) * complex(coeffs_raw[i])

        if almost_zero(lhs) and almost_zero(rhs):
            continue
        if almost_zero(lhs) != almost_zero(rhs):
            return False

        gi = lhs / rhs
        if g_ref is None:
            if clamp_clifford:
                candidates = [1+0j, 0+1j, -1+0j, 0-1j]
                best = min(candidates, key=lambda z: abs(gi - z))
                if abs(gi - best) <= 1e-8:
                    g_ref = best
                else:
                    g_ref = gi
            else:
                g_ref = gi
        else:
            if abs(lhs - g_ref * rhs) > max(atol, rtol * max(abs(lhs), abs(g_ref * rhs))):
                return False
    return True

if __name__ == "__main__":
    # smoke: trivial
    T = np.array([[1,0,1,0],[0,1,0,1]])
    F = np.eye(4, dtype=int)
    h = np.zeros(4, dtype=int)
    c = np.array([1+0j, 1+0j])
    ok = verify_weights_gate_global_phase(T, c, np.array([0,1]), F, h)
    assert ok
    print("[phase] ok")
