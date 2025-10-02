from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from quaos.utils import get_linear_dependencies
from numba import njit
from quaos.core.circuits.target import map_pauli_sum_to_target_tableau as _map_sum_to_target
from collections import defaultdict

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# =============================================================================
# Small utilities
# =============================================================================
def _in_image_mod2(M: np.ndarray, b: np.ndarray) -> bool:
    """Return True iff b is in the column space of M over GF(2)."""
    x = _gauss_solve_mod2(M % 2, b % 2)
    return x is not None

def _qF_parity_vector_via_U(tableau: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Return qF(a_i) mod 2 for all rows i, using the SAME U-form as your gate.
    For qubits this equals r mod 2 since 2*(phi_pi-phi) is even.
    """
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
        q2[i] = (-p1 + p2) & 1  # mod 2
    return q2

def _perm_index_to_perm_map(pi: np.ndarray) -> Dict[int, int]:
    return {i: int(pi[i]) for i in range(pi.size)}

def _is_identity_perm_idx(pi: np.ndarray) -> bool:
    return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))

# =============================================================================
# Coefficient discretization (stable binning for floats/complex)
# =============================================================================

def _discretize_coeffs(
    coeffs: Optional[np.ndarray],
    rel: float = 1e-8,
    abs_tol: float = 1e-12,
) -> Optional[np.ndarray]:
    """
    Map scalar/complex coefficients to stable integer colour IDs with
    relative + absolute tolerance. Used to enforce coefficient-colour equality.
    """
    if coeffs is None:
        return None
    c = np.asarray(coeffs)

    def q(x: np.ndarray) -> np.ndarray:
        step = np.maximum(np.abs(x) * rel, abs_tol)
        return np.rint(x / step).astype(np.int64)

    if np.iscomplexobj(c):
        qr = q(c.real)
        qi = q(c.imag)
        return (qr.astype(np.int64) << 32) ^ (qi.astype(np.int64) & ((1 << 32) - 1))
    else:
        return q(c)

# =============================================================================
# WL-1 base partition on edge-coloured complete graph S (ordering only)
# =============================================================================

def _wl_colors_from_S(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray] = None,
    max_rounds: int = 10,
) -> np.ndarray:
    """
    1-WL color refinement on the complete edge-colored graph with edge color S[i,j] in GF(p).
    Seed key: (coeff[i], histogram of S[i,*] mod p).
    We'll use the resulting colors for *ordering only* (not a hard constraint).
    """
    n = S_mod.shape[0]
    hist = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts = np.bincount(S_mod[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette: Dict[Tuple, int] = {}
    color = np.empty(n, dtype=np.int64)
    for i in range(n):
        coeff_key = None if coeffs is None else (coeffs[i].item() if hasattr(coeffs[i], "item") else coeffs[i])
        seed_key = (coeff_key, tuple(hist[i]))
        color[i] = palette.setdefault(seed_key, len(palette))

    for _ in range(max_rounds):
        new_keys = []
        for i in range(n):
            d: Dict[Tuple[int, int], int] = {}
            row = S_mod[i]
            for j in range(n):
                key = (int(color[j]), int(row[j]))
                d[key] = d.get(key, 0) + 1
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2: Dict[Tuple, int] = {}
        new_color = np.empty(n, dtype=np.int64)
        changed = False
        for i, key in enumerate(new_keys):
            c = palette2.setdefault(key, len(palette2))
            new_color[i] = c
            changed |= (c != color[i])
        color = new_color
        if not changed:
            break
    return color

def _color_classes(color: np.ndarray) -> Dict[int, List[int]]:
    classes: Dict[int, List[int]] = {}
    for i, c in enumerate(color):
        classes.setdefault(int(c), []).append(i)
    for c in classes:
        classes[c].sort()
    return classes

# =============================================================================
# Local S-consistency (Numba)
# =============================================================================

@njit(cache=True, fastmath=True)
def _consistent_all(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Enforce S-consistency of (i -> y) against all already-mapped vertices.
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True

@njit(cache=True, fastmath=True)
def _consistent_vs_basis_only(S_mod: np.ndarray, phi: np.ndarray, mapped_basis_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Enforce S-consistency of (i -> y) against already-mapped *basis* vertices only.
    """
    for t in range(mapped_basis_idx.size):
        j = mapped_basis_idx[t]
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True

# =============================================================================
# Global S check
# =============================================================================

def _check_symplectic_invariance_mod(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)

# Phase consistency checks

def _almost_equal_complex(a: complex, b: complex, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

def _i_pow(k: int) -> complex:
    k &= 3
    return (1+0j, 0+1j, -1+0j, 0-1j)[k]

def _verify_weights_gate_exact(tableau, coeffs_raw, pi, F, h, *, atol=1e-9, rtol=1e-9, qubits_global=True) -> bool:
    if coeffs_raw is None:
        return True

    N, two_n = tableau.shape
    n = two_n // 2
    mod4 = 4

    # Build U_F as in the gate
    U = np.zeros((two_n, two_n), dtype=int)
    U[n:, :n] = 1
    U_F = (F % mod4) @ (U % mod4) @ (F.T % mod4) % mod4
    d = np.diag(U_F) % mod4
    Ppart = (2 * np.triu(U_F) - np.diag(d)) % mod4

    def ipow(k: int) -> complex:
        k &= 3
        return (1+0j, 0+1j, -1+0j, 0-1j)[k]

    # Find a reference global phase g from the first nonzero row
    g_ref = None

    for i in range(N):
        a = (tableau[i, :] % mod4).astype(int)
        p1 = int(d @ a) % mod4
        p2 = int(a @ ((Ppart @ a) % mod4)) % mod4
        qF = (-p1 + p2) % mod4
        acq = (int(h @ a) + qF) % mod4

        lhs = complex(coeffs_raw[pi[i]])
        rhs = ipow(acq) * complex(coeffs_raw[i])

        if abs(rhs) <= max(atol, rtol * abs(lhs)) and abs(lhs) <= max(atol, rtol * abs(rhs)):
            # both effectively zero → OK
            continue
        if abs(rhs) <= max(atol, rtol * abs(lhs)) or abs(lhs) <= max(atol, rtol * abs(rhs)):
            # one zero, the other nonzero → fail
            return False

        gi = lhs / rhs  # candidate global phase for this row

        if g_ref is None:
            if qubits_global:
                # snap to nearest in {1,i,-1,-i}
                candidates = [1+0j, 0+1j, -1+0j, 0-1j]
                g_ref = min(candidates, key=lambda z: abs(gi - z))
                if abs(gi - g_ref) > 1e-8:
                    # if it’s not very close to a Clifford phase, still accept as long as consistent across rows
                    g_ref = gi
            else:
                g_ref = gi
        else:
            if abs(lhs - g_ref * rhs) > max(atol, rtol * max(abs(lhs), abs(g_ref * rhs))):
                return False

    return True


# =============================================================================
# Generator matrix from (independents, dependencies)
# =============================================================================

def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int], np.ndarray]:
    """
    Build G in systematic form (k × N), columns ordered by `labels` (0..N-1).
    Returns (G, basis_order, basis_mask[idx]=True if labels[idx] in basis).
    """
    GF = galois.GF(p)
    basis_order = sorted(independent)
    k, n = len(basis_order), len(labels)
    label_to_col = {lab: j for j, lab in enumerate(labels)}
    basis_index = {b: i for i, b in enumerate(basis_order)}

    G_int = np.zeros((k, n), dtype=int)
    for b in basis_order:
        G_int[basis_index[b], label_to_col[b]] = 1
    for d, pairs in dependencies.items():
        j = label_to_col[d]
        acc: Dict[int, int] = {}
        for b, m in pairs:
            acc[b] = (acc.get(b, 0) + int(m)) % p
        for b, m in acc.items():
            G_int[basis_index[b], j] = m % p
    G = GF(G_int)

    basis_mask = np.zeros(n, dtype=bool)
    for b in basis_order:
        basis_mask[label_to_col[b]] = True
    return G, basis_order, basis_mask

def _check_code_automorphism(
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],
    pi: np.ndarray,
    p: int,
) -> bool:
    """
    Code/matroid test: ∃U ∈ GL(k,p) with U G[:,pi] = G (field solve).
    """
    GF = galois.GF(p)
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    B_cols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[B_cols]
    C = G[:, PBcols]  # k × k FieldArray
    try:
        U = np.linalg.solve(C, GF.Identity(C.shape[0]))
    except np.linalg.LinAlgError:
        return False
    return np.array_equal(U @ G[:, pi], G)

# =============================================================================
# Modular linear algebra (GF(p), plus CRT/Hensel for mod 2p solves)
# =============================================================================


def _inv_mod_prime(a: int, p: int) -> int:
    p = int(p); a = int(a) % p
    if a == 0:
        raise ZeroDivisionError("no inverse")
    return pow(a, p - 2, p)

def _gauss_solve_mod_prime(M_int: np.ndarray, b_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    """
    Solve M x = b (mod p) with row-reduced echelon-like elimination; returns x or None.
    M_int: (m×n) int, b_int: (m,)
    """
    m, n = M_int.shape
    M = (M_int % p).astype(int).copy()
    b = (b_int % p).astype(int).copy()
    row = 0
    pivots: List[Tuple[int, int]] = []
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
            b[row], b[pivot] = b[pivot], b[row]
        inv = _inv_mod_prime(int(M[row, col]), int(p))
        M[row, col] = 1
        if col + 1 < n:
            M[row, col + 1 :] = (M[row, col + 1 :] * inv) % p
        b[row] = (b[row] * inv) % p
        for r in range(row + 1, m):
            if M[r, col] != 0:
                f = M[r, col] % p
                M[r, col] = 0
                if col + 1 < n:
                    M[r, col + 1 :] = (M[r, col + 1 :] - f * M[row, col + 1 :]) % p
                b[r] = (b[r] - f * b[row]) % p
        pivots.append((row, col))
        row += 1
        if row == m:
            break
    # consistency
    for r in range(row, m):
        if np.all(M[r, :] % p == 0) and (b[r] % p != 0):
            return None
    x = np.zeros(n, dtype=int)
    for r, c in reversed(pivots):
        s = 0 if c + 1 >= n else int(np.dot(M[r, c + 1 :] % p, x[c + 1 :] % p) % p)
        x[c] = (b[r] - s) % p
    return x % p

def _gauss_inverse_mod_prime(M_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    p = int(p)
    n = int(M_int.shape[0])
    assert M_int.shape == (n, n)
    A = (M_int % p).astype(int).copy()
    I = np.eye(n, dtype=int)
    A_aug = np.concatenate([A, I], axis=1)
    row = 0
    for col in range(n):
        pivot = -1
        for r in range(row, n):
            if A_aug[r, col] % p != 0:
                pivot = r
                break
        if pivot == -1:
            return None
        if pivot != row:
            A_aug[[row, pivot]] = A_aug[[pivot, row]]
        inv = _inv_mod_prime(int(A_aug[row, col]), p)
        A_aug[row, col:] = (A_aug[row, col:] * inv) % p
        for r in range(n):
            if r == row:
                continue
            f = A_aug[r, col] % p
            if f != 0:
                A_aug[r, col:] = (A_aug[r, col:] - f * A_aug[row, col:]) % p
        row += 1
        if row == n:
            break
    if not np.array_equal(A_aug[:, :n] % p, np.eye(n, dtype=int)):
        return None
    return A_aug[:, n:] % p

# =============================================================================
# Phase helpers (canonical: tableau is N×2n; F acts on the right; π permutes rows)
# =============================================================================

def _build_P_phase(n_qud: int, p: int) -> np.ndarray:
    """
    Standard symplectic form for phase calculations, modulo 2p.
    """
    P_upper = np.hstack([np.zeros((n_qud, n_qud), dtype=int), np.eye(n_qud, dtype=int)])
    P_lower = np.hstack([(-np.eye(n_qud, dtype=int)) % (2 * p), np.zeros((n_qud, n_qud), dtype=int)])
    return (np.vstack([P_upper, P_lower]) % (2 * p)).astype(int)

def _left_inverse_GF_tableau(tableau: np.ndarray, p: int) -> Optional[np.ndarray]:
    """
    Given canonical tableau (N × 2n), return L_left (2n × N) s.t. L_left @ tableau = I_{2n} (mod p),
    or None if rank(tableau) < 2n.
    """
    N, two_n = tableau.shape
    insane_convention_tableau = tableau.T % p  # 2n × N
    L_left = np.zeros((two_n, N), dtype=int)
    # For each unit vector e_j in dimension 2n, solve (tableau^T) x = e_j
    for j in range(two_n):
        e = np.zeros(two_n, dtype=int); e[j] = 1
        x = _gauss_solve_mod_prime(insane_convention_tableau, e, p)
        if x is None:
            return None
        L_left[j, :] = x % p
    return L_left % p

def _build_F_right_rowperm(
    tableau: np.ndarray,
    L_left: Optional[np.ndarray],
    pi: np.ndarray,
    p: int
) -> Optional[np.ndarray]:
    """
    Build F (2n × 2n) so that (tableau @ F) == (Π @ tableau) (mod p), where Π permutes rows by pi.
    For p=2, use the transvection-based construction (successive symplectic transvections) and RETURN.
    For p!=2, fall back to the left-inverse method (and verify).
    """
    p = int(p)
    N, two_n = tableau.shape
    Pi = np.eye(N, dtype=int)[pi, :]  # N×N row permutation

    if p == 2:
        # --- Use your trusted transvection construction (already imported as _map_sum_to_target) ---
        A_src = (tableau % 2).astype(int)        # N × 2n
        A_tgt = (Pi @ A_src) % 2                 # N × 2n
        F = _map_sum_to_target(A_src, A_tgt) % 2

        # Verify mapping and symplecticity
        if not np.array_equal((tableau @ F) % 2, (Pi @ tableau) % 2):
            return None

        P2 = _build_P_phase(two_n // 2, 1) % 2
        if not np.array_equal((F.T @ P2 @ F) % 2, P2):
            return None

        return F  # IMPORTANT: do not fall through to the left-inverse path

    # --- General p path: left-inverse (requires full row rank) ---
    if L_left is None:
        return None
    F = (L_left @ (Pi @ (tableau % p))) % p  # 2n×2n

    # Verify mapping and symplecticity
    if not np.array_equal((tableau % p) @ F % p, (Pi @ (tableau % p)) % p):
        return None

    Pp = _build_P_phase(two_n // 2, p) % p
    if not np.array_equal((F.T @ Pp @ F) % p, Pp):
        return None

    return F % p

def _solve_h_qubits_mod4_with_lift(M: np.ndarray, b: np.ndarray, diag: dict | None = None) -> np.ndarray | None:
    """
    Solve M h = b (mod 4). Try a direct Z4 elimination (unit pivots) first;
    fallback to 2-adic lift from mod 2 with a small nullspace multi-start.
    Returns h (length 2n) or None.
    """
    # 1) direct mod-4
    h4 = _gauss_solve_mod4_unit(M, b)  # your existing routine
    if h4 is not None and np.array_equal((M @ h4) % 4, b % 4):
        return h4 % 4
    if diag is not None: diag["phase_mod4_infeasible"] += 1

    # 2) 2-adic lift
    M2, b2 = (M % 2), (b % 2)
    h0 = _gauss_solve_mod2(M2, b2)     # your existing GF(2) solver
    if h0 is None:
        if diag is not None: diag["phase_mod2_infeasible"] += 1
        return None
    resid = (b - (M @ h0) % 4) % 4
    if np.any(resid & 1):
        if diag is not None: diag["phase_lift_parity_fail"] += 1
        return None
    rhs2 = ((resid // 2) % 2).astype(int)
    delta = _gauss_solve_mod2(M2, rhs2)
    if delta is not None:
        h = (h0 + 2 * delta) % 4
        if np.array_equal((M @ h) % 4, b % 4):
            return h

    # 3) small nullspace multi-start to find a liftable particular solution
    ns = _nullspace_mod2(M2)  # your existing nullspace builder
    from itertools import product
    L = len(ns)
    for mask in product([0, 1], repeat=min(L, 5)):
        h0_try = h0.copy()
        for bit, v in zip(mask, ns):
            if bit: h0_try ^= v
        resid = (b - (M @ h0_try) % 4) % 4
        if np.any(resid & 1):
            continue
        rhs2 = ((resid // 2) % 2).astype(int)
        delta = _gauss_solve_mod2(M2, rhs2)
        if delta is None:
            continue
        h = (h0_try + 2 * delta) % 4
        if np.array_equal((M @ h) % 4, b % 4):
            return h
    if diag is not None: diag["phase_lift_delta_fail"] += 1
    return None


def _build_phase_rhs_via_U(tableau: np.ndarray, phases: Optional[np.ndarray], pi: np.ndarray, F: np.ndarray, p: int) -> np.ndarray:
    """
    r_i = 2*(phi_{pi(i)} - phi_i) - Q_F(a_i)  (mod 2p),
    where Q_F(a) = -diag(U_F)·a + a^T (2*triu(U_F)-diag(diag(U_F))) a,
    and U_F = F U F^T with U = [[0,0],[I,0]] in your convention.
    """
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


def _gauss_solve_mod2(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Solve M x = b over GF(2).
    Returns one solution x (0/1 ints) or None if inconsistent.
    Free variables are set to 0.
    """
    A = (M.astype(np.uint8) & 1).copy()
    y = (b.astype(np.uint8) & 1).copy()
    m, n = A.shape

    piv_cols = []
    piv_rows = []
    r = 0
    for c in range(n):
        # find pivot
        pivot = -1
        for i in range(r, m):
            if A[i, c]:
                pivot = i
                break
        if pivot == -1:
            continue
        # swap into row r
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            y[r], y[pivot] = y[pivot], y[r]
        piv_cols.append(c)
        piv_rows.append(r)
        # eliminate in all other rows
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
                y[i] ^= y[r]
        r += 1
        if r == m:
            break

    # inconsistency: 0 = 1
    for i in range(m):
        if not A[i].any() and y[i]:
            return None

    # back-substitution (free vars = 0)
    x = np.zeros(n, dtype=np.uint8)
    for k in reversed(range(len(piv_cols))):
        i = piv_rows[k]
        c = piv_cols[k]
        s = 0
        row = A[i]
        # sum row[j]*x[j] for j != c
        # (uint8 xor is sum mod 2)
        for j in range(n):
            if j != c and row[j]:
                s ^= x[j]
        x[c] = y[i] ^ s
    return x.astype(int)


def _nullspace_mod2(M: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of GF(2) nullspace basis vectors v with M v = 0 (mod 2).
    """
    A = (M % 2).astype(np.uint8)
    m, n = A.shape
    piv_col = [-1]*m
    r = 0
    col_used = [False]*n
    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if A[i, c]:
                pivot = i; break
        if pivot == -1: continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        piv_col[r] = c
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        col_used[c] = True
        r += 1
        if r == m: break
    free_cols = [j for j in range(n) if not col_used[j]]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # back-substitute to fill pivot columns
        for i in reversed(range(r)):
            c = piv_col[i]
            s = 0
            row = A[i]
            for j in range(n):
                if j != c and row[j]:
                    s ^= v[j]
            v[c] = s
        basis.append(v.astype(int))
    return basis  # list of length = nullity


def _gauss_solve_mod4_unit(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Solve M x = b (mod 4), preferring unit pivots (1 or 3). Works well since your M is 0/1.
    Returns x mod 4 or None if inconsistent.
    """
    A = (M.astype(int) % 4).copy()
    y = (b.astype(int) % 4).copy()
    m, n = A.shape

    piv_cols, piv_rows = [], []
    r = 0
    for c in range(n):
        # find a row with an odd entry (1 or 3) in column c from row r down
        pivot = -1
        for i in range(r, m):
            if (A[i, c] & 1) == 1:  # odd => invertible in Z4
                pivot = i
                break
        if pivot == -1:
            continue

        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            y[r], y[pivot] = y[pivot], y[r]

        # normalize pivot to 1
        inv = 1 if (A[r, c] % 4) == 1 else 3  # 3 is inverse of 3 mod 4
        A[r, :] = (A[r, :] * inv) % 4
        y[r] = (y[r] * inv) % 4

        piv_cols.append(c)
        piv_rows.append(r)

        # eliminate in all other rows
        for i in range(m):
            if i == r:
                continue
            factor = A[i, c] % 4
            if factor:
                A[i, :] = (A[i, :] - factor * A[r, :]) % 4
                y[i] = (y[i] - factor * y[r]) % 4

        r += 1
        if r == m:
            break

    # inconsistency: 0 = nonzero
    for i in range(m):
        if np.all(A[i, :] % 4 == 0) and (y[i] % 4) != 0:
            return None

    # back-substitute with free vars = 0
    x = np.zeros(n, dtype=int)
    for k in reversed(range(len(piv_cols))):
        i = piv_rows[k]
        c = piv_cols[k]
        # A[i,c] == 1 now
        s = 0
        for j in range(n):
            if j != c and A[i, j] % 4:
                s = (s + A[i, j] * x[j]) % 4
        x[c] = (y[i] - s) % 4
    return x
def _solve_tableau_h_eq_r_mod_2p(tableau: np.ndarray, r: np.ndarray, p: int, diag: dict | None = None) -> np.ndarray | None:
    p = int(p)
    mod = 2 * p
    M = (tableau % mod).astype(int)
    b = (np.asarray(r, dtype=int) % mod)
    n = M.shape[1]

    if p == 2:
        h4 = _gauss_solve_mod4_unit(M, b)
        if h4 is not None:
            h4 = np.asarray(h4, dtype=int).reshape(-1)
            if h4.size == n and np.array_equal((M @ h4) % 4, b % 4):
                return h4 % 4
        if diag is not None: diag["phase_mod4_infeasible"] += 1

        M2, b2 = (M % 2), (b % 2)
        h0 = _gauss_solve_mod2(M2, b2)
        if h0 is None:
            if diag is not None: diag["phase_mod2_infeasible"] += 1
            return None
        h0 = np.asarray(h0, dtype=int).reshape(-1)
        if h0.size != n:
            if diag is not None: diag["phase_bad_h0_shape"] += 1
            return None

        resid = (b - (M @ h0) % 4) % 4
        if np.any(resid & 1):
            if diag is not None: diag["phase_lift_parity_fail"] += 1
            return None
        rhs2 = ((resid // 2) % 2).astype(int)
        delta = _gauss_solve_mod2(M2, rhs2)
        if delta is not None:
            delta = np.asarray(delta, dtype=int).reshape(-1)
            if delta.size != n:
                if diag is not None: diag["phase_bad_delta_shape"] += 1
                return None
            h = (h0 + 2 * delta) % 4
            if np.array_equal((M @ h) % 4, b % 4):
                return h

        # try nullspace alternatives
        ns = _nullspace_mod2(M2)
        ns = [np.asarray(v, dtype=int).reshape(-1) for v in ns]
        for v in ns:
            if v.size != n:
                if diag is not None: diag["phase_bad_ns_vec_shape"] += 1
                return None
        from itertools import product
        L = len(ns)
        max_masks = 1 << min(L, 5)
        for mask in product([0,1], repeat=min(L, 5)):
            h0_try = h0.copy()
            for bit, v in zip(mask, ns):
                if bit: h0_try ^= v
            resid = (b - (M @ h0_try) % 4) % 4
            if np.any(resid & 1): 
                continue
            rhs2 = ((resid // 2) % 2).astype(int)
            delta = _gauss_solve_mod2(M2, rhs2)
            if delta is None:
                continue
            delta = np.asarray(delta, dtype=int).reshape(-1)
            if delta.size != n: 
                continue
            h = (h0_try + 2 * delta) % 4
            if np.array_equal((M @ h) % 4, b % 4):
                return h
        if diag is not None: diag["phase_lift_delta_fail"] += 1
        return None

    # p odd (CRT)
    h2 = _gauss_solve_mod2(M % 2, b % 2)
    if h2 is None:
        if diag is not None: diag["phase_mod2_infeasible"] += 1
        return None
    hp = _gauss_solve_mod_prime(M % p, b % p, p)
    if hp is None:
        if diag is not None: diag["phase_modp_infeasible"] += 1
        return None
    h2 = np.asarray(h2, dtype=int).reshape(-1)
    hp = np.asarray(hp, dtype=int).reshape(-1)
    if h2.size != n or hp.size != n:
        if diag is not None: diag["phase_bad_crt_shape"] += 1
        return None

    inv2_modp = pow(2, p - 2, p)
    h = (h2 + 2 * ((hp - (h2 % p)) * inv2_modp % p)) % (2 * p)
    h = np.asarray(h, dtype=int).reshape(-1)
    if h.size != n:
        if diag is not None: diag["phase_bad_h_shape_return"] += 1
        return None
    return h


# =============================================================================
# Dependent placement using signatures (UG vs G) + basis-only S checks
# =============================================================================

def _place_dependents_with_U(
    *,
    U: galois.FieldArray,
    G: galois.FieldArray,
    basis_mask: np.ndarray,
    coeffs: Optional[np.ndarray],      # WL colorized coeffs (for candidate filtering)
    S_mod: np.ndarray,
    phi: np.ndarray,
    used: np.ndarray,
    basis_order_idx: List[int],
    p: int,
    tableau: Optional[np.ndarray],
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_left: Optional[np.ndarray],
    rank_full: bool,
    return_phase: bool,
    diag: dict,
    coeffs_raw: Optional[np.ndarray],   # <-- ADD THIS (raw complex)
) -> Optional[Dict[str, object]]:
    """
    After basis is mapped, determine U = C^{-1}. Use signatures of UG vs G to
    finish dependents with minimal search; check S and code globally at the leaf,
    and (optionally) solve for phases (right-acting F, row permutation).
    """
    N = G.shape[1]
    G_nd = G.view(np.ndarray) % p
    UG = U @ G
    UG_nd = UG.view(np.ndarray) % p

    def sig_col(nd: np.ndarray, j: int) -> Tuple[int, ...]:
        return tuple(int(v) for v in nd[:, j])

    sig_G = [sig_col(G_nd, j) for j in range(N)]
    sig_UG = [sig_col(UG_nd, j) for j in range(N)]

    from collections import defaultdict
    free_by_sig: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for y in range(N):
        if not used[y]:
            free_by_sig[sig_UG[y]].append(y)

    dependents = [i for i in range(N) if not basis_mask[i]]
    dep_candidates: Dict[int, List[int]] = {}
    for i in dependents:
        lst = free_by_sig.get(sig_G[i], [])
        if coeffs is not None:
            lst = [y for y in lst if coeffs[i] == coeffs[y]]
        if not lst:
            diag['dep_cand_empty'] += 1
            return None
        dep_candidates[i] = lst

    dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))
    mapped_basis = np.asarray([b for b in basis_order_idx if phi[b] >= 0], dtype=np.int64)

    def dfs_dep(t: int) -> Optional[Dict[str, object]]:
        if t >= len(dep_order):
            pi = phi.copy()
            if _is_identity_perm_idx(pi):
                diag['is_identity'] += 1
                return None
            if not _check_symplectic_invariance_mod(S_mod, pi):
                diag['S_fail'] += 1
                return None
            if not _check_code_automorphism(G, [int(b) for b in basis_order_idx], list(range(N)), pi, p):
                diag['code_fail'] += 1
                return None

            # Build F (transvections for p=2; left-inverse fallback otherwise)
            # Build F (transvections for p=2; left-inverse fallback otherwise)
            F_int = _build_F_right_rowperm(tableau, L_left, pi, p)
            if F_int is None:
                diag['F_fail'] += 1
                return None

            # 1) Build RHS with the SAME U-form your gate uses (matches acquired_phase exactly)
            r = _build_phase_rhs_via_U(tableau, phases, pi, F_int, p=2)  # N-vector mod 4

            # 2) Parity must match image of tableau mod 2
            r2 = (r & 1)
            if not _in_image_mod2(tableau, r2):
                diag["phase_parity_image_fail"] += 1
                return None

            # 3) Double-check our qF parity computation matches r parity (should always hold)
            q2 = _qF_parity_vector_via_U(tableau, F_int)
            if not np.array_equal(r2, q2):
                diag["qF_parity_mismatch"] += 1
                return None

            # 2) Solve tableau @ h == r (mod 2p) using robust solver (no SymPy/SciPy)
            # Try direct Z4 elimination first
            h = _solve_h_qubits_mod4_with_lift(tableau, r, diag)

            if h is None:
                diag["phase_fail"] += 1
                return None  # <-- bail before any reshape/matmul

            h = np.asarray(h, dtype=int).reshape(-1)
            if h.ndim != 1 or h.size != tableau.shape[1]:
                diag["phase_bad_h_shape"] += 1
                return None

            r = np.asarray(r, dtype=int).reshape(-1)
            if r.ndim != 1 or r.size != tableau.shape[0]:
                diag["phase_bad_r_shape"] += 1
                return None

            # verify the linear system exactly
            res = (tableau @ h - r) % (2 * p)
            if np.any(res):
                diag["phase_linear_residual_nonzero"] += 1
                diag["phase_linear_bad_rows"] = int(np.count_nonzero(res))
                return None

            if h is None:
                diag["phase_mod4_infeasible"] += 1
            else:
                # Verify quickly
                if np.array_equal((tableau @ h) % 4, r % 4):
                    pass
                else:
                    h = None

            if h is None:
                # Fallback: 2-adic lift with a few alternative mod-2 particular solutions
                M2 = tableau % 2
                b2 = r % 2
                h0 = _gauss_solve_mod2(M2, b2)
                if h0 is None:
                    diag["phase_mod2_infeasible"] += 1
                    return None

                resid = (r - (tableau @ h0) % 4) % 4
                if np.any(resid & 1):
                    diag["phase_lift_parity_fail"] += 1
                    return None

                rhs2 = ((resid // 2) % 2).astype(int)
                delta = _gauss_solve_mod2(M2, rhs2)
                if delta is None:
                    # Try a few alternative particular solutions: h0' = h0 + sum t_i * ns_i
                    ns = _nullspace_mod2(M2)
                    tried = 0
                    found = None
                    # small bounded search; tweak K if you want
                    K = min(32, max(1, 1 << min(len(ns), 5)))
                    # try all if tiny, else random subset
                    from itertools import product
                    if len(ns) <= 5:
                        it = product([0,1], repeat=len(ns))
                    else:
                        import random
                        it = ([random.getrandbits(1) for _ in range(len(ns))] for _ in range(K))
                    for coeffs_bits in it:
                        tried += 1
                        h0_try = h0.copy()
                        for bit, v in zip(coeffs_bits, ns):
                            if bit: h0_try = (h0_try ^ v)  # XOR mod 2
                        resid = (r - (tableau @ h0_try) % 4) % 4
                        if np.any(resid & 1):
                            continue
                        rhs2 = ((resid // 2) % 2).astype(int)
                        delta = _gauss_solve_mod2(M2, rhs2)
                        if delta is not None:
                            found = (h0_try + 2 * delta) % 4
                            break
                    if found is None:
                        diag["phase_lift_delta_fail"] += 1
                        return None
                    h = found
                else:
                    h = (h0 + 2 * delta) % 4


            if h is None:
                diag['phase_fail'] += 1
                return None

            # Final exact check against your gate’s convention (needs RAW complex weights!)
            if not _verify_weights_gate_exact(tableau, coeffs_raw, pi, F_int, h):
                diag['weights_fail'] += 1
                return None

            rec: Dict[str, object] = {"perm": _perm_index_to_perm_map(pi)}
            rec["h"] = h
            rec["F"] = F_int
            return rec

        i = dep_order[t]
        for y in dep_candidates[i]:
            if used[y]:
                continue
            if not _consistent_vs_basis_only(S_mod, phi, mapped_basis, i, y):
                continue
            phi[i] = y
            used[y] = True
            sol = dfs_dep(t + 1)
            if sol is not None:
                return sol
            phi[i] = -1
            used[y] = False
        return None

    return dfs_dep(0)

# =============================================================================
# Basis-first search (WL ordering only) + dependent placement by signatures
# =============================================================================

def _basis_first_search(
    *,
    S_mod: np.ndarray,
    coeffs: Optional[np.ndarray],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],  # ordering only
    G: galois.FieldArray,
    basis_order_idx: List[int],
    basis_mask: np.ndarray,
    p: int,
    k_wanted: int,
    # Phase context (optional):
    tableau: Optional[np.ndarray],  # N × 2n
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_left: Optional[np.ndarray],   # 2n × N
    rank_full: bool,
    return_phase: bool,
    diag: dict,
    coeffs_raw: np.ndarray,
) -> List[Dict[str, object]]:
    """
    Basis-first mapping. WL colors are used as a heuristic *ordering only*.
    Coefficient colours are hard constraints. After basis is fixed, compute U=C^{-1}
    (GF(p)) and place dependents via signature buckets; at leaf verify global S, code,
    and (optionally) phases using right-acting F and row permutation.
    """

    N = S_mod.shape[0]
    results: List[Dict[str, object]] = []
    k = len(basis_order_idx)

    # Order basis variables by increasing WL-class size (MRV-ish)
    basis_idx_seq = sorted(basis_order_idx, key=lambda i: len(base_classes[int(base_colors[i])]))

    phi = -np.ones(N, dtype=int)
    used = np.zeros(N, dtype=bool)
    GF = galois.GF(p)

    def dfs_basis(t: int) -> bool:
        if len(results) >= k_wanted:
            return True
        if t >= k:
            Bcols = np.array(basis_order_idx, dtype=int)
            PBcols = phi[Bcols]
            C = G[:, PBcols]
            C_int = (C.view(np.ndarray) % p)
            U_int = _gauss_inverse_mod_prime(C_int, p)
            if U_int is None:
                return False
            U = GF(U_int)
            sol = _place_dependents_with_U(
                                            U=U, G=G, basis_mask=basis_mask, coeffs=coeffs,
                                            S_mod=S_mod, phi=phi, used=used, basis_order_idx=basis_order_idx, p=p,
                                            tableau=tableau, P_phase=P_phase, phases=phases,
                                            L_left=L_left, rank_full=rank_full, return_phase=return_phase,
                                            diag=diag,
                                            coeffs_raw=coeffs_raw,  # <-- pass raw complex weights here
)
            if sol is not None:
                results.append(sol)
                return True
            return False

        i = basis_idx_seq[t]
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # WL ordering only; coefficient colours are hard constraints
        bi = int(base_colors[i])
        pool = base_classes[bi]
        candidates = [y for y in pool if not used[y] and (coeffs is None or coeffs[i]==coeffs[y])]

        candidates.sort(key=lambda y: (base_colors[y], y))

        for y in candidates:
            if not _consistent_all(S_mod, phi, mapped_idx, i, y):
                continue
            phi[i] = y
            used[y] = True
            if dfs_basis(t + 1):
                return True
            phi[i] = -1
            used[y] = False
        return False

    dfs_basis(0)
    return results[:k_wanted]


def find_k_automorphisms_symplectic(
    H,
    *,
    k: int = 1,
    basis_first: str = "any",          # "any" or "off" (use algebraic fallback only)
    return_phase: bool = True,
) -> List[Dict[str, object]]:
    """
    Returns up to k automorphisms π (row permutations) with optional phase vector h and
    (if rank-full) a right-acting F (2n×2n) such that tableau @ F == Π @ tableau (mod p).

    Conventions:
      - tableau := H.tableau() is N × 2n (rows=Paulis, cols=qudit axes).
      - F acts from the right; π permutes rows.
      - get_linear_dependencies(tableau, p) is called with the canonical N×2n shape.
    """
    # ----- Extract data from H -----
    tableau = np.asarray(H.tableau(), dtype=int)               # N × 2n (canonical; rows=terms)
    S = np.asarray(H.symplectic_product_matrix(), dtype=int)   # N × N
    coeffs_raw = None if getattr(H, "weights", None) is None else np.asarray(H.weights)
    phases_raw = None if getattr(H, "phases", None) is None else np.asarray(H.phases, dtype=int)
    dims = list(H.dimensions)

    # ----- Field & shape checks -----
    if len(set(dims)) != 1:
        raise ValueError("All local dimensions must be equal (prime p).")
    p = int(dims[0])
    n_qud = len(dims)
    two_n = 2 * n_qud

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square; got {S.shape}.")
    N = int(S.shape[0])
    if tableau.shape != (N, two_n):
        raise ValueError(f"Expected tableau shape (N,2n)=({N},{two_n}); got {tableau.shape}.")

    independent, dependencies = get_linear_dependencies(tableau, p)

    labels = list(range(N))
    G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, labels, p)
    basis_order_idx = [int(i) for i in basis_order]

    # ----- WL (ordering only) + coefficient colours (hard constraint) -----
    S_mod = (S % p).astype(np.int64, copy=False)
    coeff_cols = _discretize_coeffs(coeffs_raw)
    base_colors = _wl_colors_from_S(S_mod, p, coeffs=coeff_cols, max_rounds=10)
    base_classes = _color_classes(base_colors)  # used for ordering heuristics only

    # ----- Phase context (optional) -----
    # We can always attempt phase feasibility; if H.phases is None we treat it as zeros.
    phases_vec = np.zeros(N, dtype=int) if phases_raw is None else (phases_raw % p)
    P_phase = _build_P_phase(n_qud, p)

    # Build a left-inverse L_left so that L_left @ tableau = I_{2n}, if rank-full.
    L_left = _left_inverse_GF_tableau(tableau, p)  # 2n × N or None
    rank_full = L_left is not None

    # ----- Primary fast path: basis-first search -----
    results: List[Dict[str, object]] = []
    diag = defaultdict(int)  # <-- add this

    # primary fast path
    if basis_first == "any":
        results = _basis_first_search(
            S_mod=S_mod,
            coeffs=coeff_cols,
            base_colors=base_colors,
            base_classes=base_classes,
            G=G,
            basis_order_idx=basis_order_idx,
            basis_mask=basis_mask,
            p=p,
            k_wanted=k,
            tableau=tableau,
            P_phase=P_phase,
            phases=phases_vec,
            L_left=L_left,
            rank_full=rank_full,
            return_phase=return_phase,
            diag=diag,                # <-- pass diag down
            coeffs_raw=coeffs_raw
        )
        if results:
            print("[diag] " + ", ".join(f"{k}={v}" for k,v in diag.items()))
            return results

    diag["dep_dfs_dead"] += 1
    print("[diag] " + ", ".join(f"{k}={v}" for k,v in diag.items()))
    return []


if __name__ == "__main__":
    from quaos.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from quaos.core.circuits import SWAP, Gate, SUM, Hadamard, PHASE, Circuit

    sym = SWAP(0, 1, 2)
    n_qudits = 3
    n_paulis_pre_sym = 6
    H = random_gate_symmetric_hamiltonian(sym, n_qudits, n_paulis_pre_sym, scrambled=False)

    scramble_circuit = Circuit([2] * n_qudits)
    scramble_circuit.add_gate(SUM(1, 2, 2))
    scramble_circuit.add_gate(Hadamard(0, 2))
    scramble_circuit.add_gate(PHASE(0, 2))
    H = scramble_circuit.act(H)

    independent, dependencies = get_linear_dependencies(H.tableau(), H.dimensions)

    S = H.symplectic_product_matrix()
    coeffs = H.weights

    # assert H.standard_form() == sym.act(H).standard_form()

    print('independent = ', independent)
    print('dependencies = ', dependencies)

    print('tableau = ', H.tableau())
    print('S = ', S)
    print('coeffs = ', coeffs)

    perms = find_k_automorphisms_symplectic(H)

    print('permutations = ', perms)

    sym_gate = Gate('symmetry', [i for i in range(H.n_qudits())], perms[0]['F'].T, 2, perms[0]['h'])

    assert np.all(sym_gate.act(H).standard_form().tableau() == H.standard_form(
    ).tableau()), f'{sym_gate.act(H).standard_form().tableau()}\n{H.standard_form().tableau()}'
    print('Tableau is ok!')
    assert sym_gate.act(H).standard_form() == H.standard_form(
    ), f'{sym_gate.act(H).standard_form().__str__()}\n{H.standard_form().__str__()}'

    print('Passed!')
