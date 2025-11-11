from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from sympleq.utils import get_linear_dependencies

# Optional: if you want to JIT tiny kernels, you can import numba and
# swap in numba-accelerated versions of consistency checks. The pure-numpy
# versions below are correct and reasonably fast for medium N.
# from numba import njit

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _perm_index_to_perm_map(pi: np.ndarray) -> Dict[int, int]:
    return {i: int(pi[i]) for i in range(pi.size)}


def _is_identity_perm_idx(pi: np.ndarray) -> bool:
    return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))


# ---------------------------------------------------------------------------
# Robust coefficient discretization (relative + absolute tolerance)
# ---------------------------------------------------------------------------

def _discretize_coeffs(
    coeffs: Optional[np.ndarray],
    rel: float = 1e-8,
    abs_tol: float = 1e-12,
) -> Optional[np.ndarray]:
    """
    Map float/complex coefficients to stable integer colour IDs using
    per-entry quantization with relative + absolute tolerance:
      q(x) = round( x / (abs(x)*rel + abs_tol) )
    Two values x,y that satisfy |x-y| <= |x|*rel + abs_tol
    will map to the same integer bin (and similarly for complex by parts).
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
        # stable 64-bit packing of two signed 64-bit ints
        packed = (qr.astype(np.int64) << 32) ^ (qi.astype(np.int64) & ((1 << 32) - 1))
        return packed
    else:
        return q(c)


# ---------------------------------------------------------------------------
# WL-1 base partition on edge-coloured complete graph S
# ---------------------------------------------------------------------------

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
    This is a safe isomorphism invariant; we use it only as a *base* partition.
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


# ---------------------------------------------------------------------------
# Local S-consistency checks
# ---------------------------------------------------------------------------

def _consistent_all(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """Enforce S-consistency of (i -> y) against *all* already-mapped vertices."""
    for j in mapped_idx:
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True


def _consistent_vs_basis_only(S_mod: np.ndarray, phi: np.ndarray, mapped_basis_idx: np.ndarray, i: int, y: int) -> bool:
    """Enforce S-consistency of (i -> y) against mapped *basis* vertices only."""
    for j in mapped_basis_idx:
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True


# ---------------------------------------------------------------------------
# Global checks
# ---------------------------------------------------------------------------

def _check_symplectic_invariance_mod(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)


def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int], np.ndarray]:
    """
    Build G in systematic form (k x N), columns ordered by `labels` (0..N-1).
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
    """Code/matroid test: ∃U in GL(k,p) with U G[:,pi] = G (field solve)."""
    GF = galois.GF(p)
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    B_cols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[B_cols]
    C = G[:, PBcols]  # k x k FieldArray
    try:
        U = np.linalg.solve(C, GF.Identity(C.shape[0]))
    except np.linalg.LinAlgError:
        return False
    return np.array_equal(U @ G[:, pi], G)


# ---------------------------------------------------------------------------
# Modular linear algebra helpers (GF(p), mod 2p via CRT/Hensel)
# ---------------------------------------------------------------------------

def _inv_mod_prime(a: int, p: int) -> int:
    p = int(p)
    a = int(a) % p
    if a == 0:
        raise ZeroDivisionError("no inverse")
    return pow(a, p - 2, p)


def _gauss_solve_mod_prime(M_int: np.ndarray, b_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    m, n = M_int.shape
    M = (M_int % p).astype(int).copy()
    b = (b_int % p).astype(int).copy()
    row = 0
    pivots: List[Tuple[int, int]] = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
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
        pivot = None
        for r in range(row, n):
            if A_aug[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
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


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _normalize_tableau(A_raw: np.ndarray, two_n: int) -> np.ndarray:
    if A_raw.shape[0] == two_n:
        return A_raw.astype(int, copy=False)
    if A_raw.shape[1] == two_n:
        return A_raw.T.astype(int, copy=False)
    raise ValueError(f"Tableau shape {A_raw.shape} not compatible with 2n={two_n}.")


def _build_P_phase(n_qud: int, p: int) -> np.ndarray:
    P_upper = np.hstack([np.zeros((n_qud, n_qud), dtype=int), np.eye(n_qud, dtype=int)])
    P_lower = np.hstack([(-np.eye(n_qud, dtype=int)) % (2 * p), np.zeros((n_qud, n_qud), dtype=int)])
    return (np.vstack([P_upper, P_lower]) % (2 * p)).astype(int)


def _right_inverse_GF(A: np.ndarray, p: int) -> Optional[np.ndarray]:
    A = A.T
    rows, N = A.shape  # rows = 2n
    L = np.zeros((N, rows), dtype=int)
    for i in range(rows):
        e = np.zeros(rows, dtype=int)
        e[i] = 1
        x = _gauss_solve_mod_prime(A % p, e, p)
        if x is None:
            return None
        L[:, i] = x % p
    return L % p


def _build_F_from_perm_with_L(A_c: np.ndarray, L_right: np.ndarray, pi: np.ndarray, p: int) -> np.ndarray:
    # Using A_T = A_c^T (2n × N) and P_pi to reorder columns:
    A_T = A_c.T % p
    N = A_T.shape[1]
    Ppi = np.eye(N, dtype=int)[:, pi]       # N×N permutation
    F = (A_T @ Ppi) % p
    F = (F @ L_right) % p                   # 2n×2n
    return F

def _build_phase_rhs(A_c: np.ndarray, phases: Optional[np.ndarray], pi: np.ndarray,
                     Delta_mod2: np.ndarray, p: int) -> np.ndarray:
    mod2 = 2 * int(p)
    N = A_c.shape[0]
    phi = np.zeros(N, dtype=int) if phases is None else (np.asarray(phases, dtype=int) % p)
    r = np.zeros(N, dtype=int)
    for i in range(N):
        a = (A_c[i, :] % mod2).astype(int)           # row vector length 2n
        quad = int((a @ ((Delta_mod2 @ a) % mod2)) % mod2)
        dphi = int((2 * ((phi[pi[i]] - phi[i]) % p)) % mod2)
        r[i] = (dphi - quad) % mod2
    return r

def _solve_Mh_eq_r_mod_2p(M: np.ndarray, r: np.ndarray, p: int) -> Optional[np.ndarray]:
    # M is N×(2n); solve M h ≡ r (mod 2p) using same CRT/Hensel logic as before
    if p % 2 == 1:
        h_p = _gauss_solve_mod_prime(M % p, r % p, p)
        if h_p is None: return None
        h_2 = _gauss_solve_mod_prime(M % 2, r % 2, 2)
        if h_2 is None: return None
        inv2 = _inv_mod_prime(2, p)
        h = np.zeros_like(h_p, dtype=int)
        for i in range(h.size):
            k = (int(h_p[i]) - int(h_2[i])) % p
            u = (k * inv2) % p
            h[i] = (int(h_2[i]) + 2 * int(u)) % (2 * p)
        return h % (2*p)
    else:
        h2 = _gauss_solve_mod_prime(M % 2, r % 2, 2)
        if h2 is None: return None
        e = (r % 4 - (M % 4) @ (h2 % 4)) % 4
        if np.any(e % 2 != 0): return None
        y = _gauss_solve_mod_prime(M % 2, (e // 2) % 2, 2)
        if y is None: return None
        return (h2 + 2 * y) % 4


def _solve_phase_system_variants(
    A: np.ndarray,          # N×2n
    phases: Optional[np.ndarray],
    pi: np.ndarray,
    P_phase: np.ndarray,
    F_int: Optional[np.ndarray],  # 2n×2n mod p
    p: int,
    rank_full: bool,
) -> Optional[np.ndarray]:
    mod2 = 2 * int(p)

    def try_rhs(r_vec: np.ndarray) -> Optional[np.ndarray]:
        h = _solve_Mh_eq_r_mod_2p(A, r_vec, p)
        if h is None: return None
        lhs = (A @ (h % mod2)) % mod2
        return h % mod2 if np.array_equal(lhs, r_vec % mod2) else None

    # Prefer using Delta from a concrete F if available
    if F_int is not None:
        Delta = (F_int.T @ (P_phase % mod2) @ F_int - (P_phase % mod2)) % mod2
        for Delta_try in (Delta, (-Delta) % mod2):   # handle sign convention differences
            r = _build_phase_rhs(A, phases, pi, Delta_try, p)
            h = try_rhs(r)
            if h is not None:
                return h

    # Underdetermined case: no F — fall back to r = 2(φπ - φ)
    if not rank_full:
        phi = np.zeros(A.shape[0], dtype=int) if phases is None else (np.asarray(phases, dtype=int) % p)
        r = (2 * ((phi[pi] - phi) % p)) % mod2
        h = try_rhs(r)
        if h is not None:
            return h

    return None


# ---------------------------------------------------------------------------
# Dependent placement using signature buckets + DFS (basis-only S checks)
# ---------------------------------------------------------------------------

def _place_dependents_with_U(
    *,
    U: galois.FieldArray,
    G: galois.FieldArray,
    basis_mask: np.ndarray,
    coeffs: Optional[np.ndarray],
    S_mod: np.ndarray,
    phi: np.ndarray,
    used: np.ndarray,
    basis_order_idx: List[int],  # indices (0..N-1)
    p: int,
    # Phase context (optional):
    tableau: Optional[np.ndarray],
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_right: Optional[np.ndarray],
    rank_full: bool,
    return_phase: bool,
) -> Optional[Dict[str, object]]:
    """
    Build candidate lists per dependent via column signatures of UG vs G,
    then DFS with *basis-only* local S consistency. On success, verify global S,
    code, and (optionally) phases, and return the solution dict.
    """
    N = G.shape[1]
    GF = galois.GF(p)

    G_nd = G.view(np.ndarray) % p
    UG = U @ G
    UG_nd = UG.view(np.ndarray) % p

    def sig_col(mat_nd: np.ndarray, j: int) -> Tuple[int, ...]:
        return tuple(int(v) for v in mat_nd[:, j])

    # Precompute signatures
    sig_G = [sig_col(G_nd, j) for j in range(N)]
    sig_UG = [sig_col(UG_nd, j) for j in range(N)]

    # Free targets by signature
    from collections import defaultdict
    free_by_sig: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for y in range(N):
        if not used[y]:
            free_by_sig[sig_UG[y]].append(y)

    # Build per-dependent candidate lists
    dependents = [i for i in range(N) if not basis_mask[i]]

    dep_candidates: Dict[int, List[int]] = {}
    for i in dependents:
        lst = free_by_sig.get(sig_G[i], [])
        if coeffs is not None:
            lst = [y for y in lst if coeffs[i] == coeffs[y]]
        if not lst:
            return None
        dep_candidates[i] = lst

    # Order dependents by MRV (fewest candidates first)
    dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))

    # Basis-mapped set for basis-only S check
    mapped_basis = np.asarray([b for b in basis_order_idx if phi[b] >= 0], dtype=np.int64)

    def dfs_dep(t: int) -> Optional[Dict[str, object]]:
        if t >= len(dep_order):
            pi = phi.copy()
            # Global S-invariance and code check
            if _is_identity_perm_idx(pi):
                return None
            if not _check_symplectic_invariance_mod(S_mod, pi):
                return None
            if not _check_code_automorphism(G, [int(b) for b in basis_order_idx], list(range(N)), pi, p):
                return None

            rec: Dict[str, object] = {"perm": _perm_index_to_perm_map(pi)}
            # Optional phase feasibility
            if tableau is not None and P_phase is not None:
                F_int = _build_F_from_perm_with_L(
                    tableau, L_right, pi, p) if rank_full and L_right is not None else None
                h = _solve_phase_system_variants(
                    A=tableau, phases=phases, pi=pi,
                    P_phase=P_phase, F_int=F_int, p=p, rank_full=rank_full,
                )
                if h is None:
                    return None
                if return_phase:
                    rec["h"] = h
                    if F_int is not None:
                        rec["F"] = F_int
            return rec

        i = dep_order[t]
        for y in dep_candidates[i]:
            if used[y]:
                continue
            if not _consistent_vs_basis_only(S_mod, phi, mapped_basis, i, y):
                continue
            # place and recurse
            phi[i] = y
            used[y] = True
            sol = dfs_dep(t + 1)
            if sol is not None:
                return sol
            # backtrack
            phi[i] = -1
            used[y] = False
        return None

    return dfs_dep(0)


# ---------------------------------------------------------------------------
# Basis-first search (single fast path). Dependents handled by DFS per signature.
# ---------------------------------------------------------------------------
def _basis_first_search(
    *,
    S_mod: np.ndarray,
    coeffs: Optional[np.ndarray],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],  # kept for ordering; we won't constrain to these sets
    G: galois.FieldArray,
    basis_order_idx: List[int],  # indices (0..N-1)
    basis_mask: np.ndarray,
    p: int,
    k_wanted: int,
    # Phase context (optional):
    tableau: Optional[np.ndarray],
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_right: Optional[np.ndarray],
    rank_full: bool,
    return_phase: bool,
) -> List[Dict[str, object]]:
    """
    Basis-first mapping with coefficient-colour constraint. WL classes are used
    only to order candidates (heuristic), not as a feasibility constraint.
    At the basis leaf, compute U=C^{-1} over GF(p) and place dependents via
    signature buckets (basis-only S checks). Verify global S, code, and (optionally) phases.
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
            # Basis fixed -> U = C^{-1}
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
                L_right=L_right, rank_full=rank_full, return_phase=return_phase,
            )
            if sol is not None:
                results.append(sol)
                return True
            return False

        i = basis_idx_seq[t]
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # --- WL is ordering only; coefficient colours are a hard constraint
        if coeffs is None:
            candidates = [y for y in range(N) if not used[y]]
        else:
            candidates = [y for y in range(N) if (not used[y]) and (coeffs[i] == coeffs[y])]

        # Order by WL colour (heuristic) then index for determinism
        candidates.sort(key=lambda y: (base_colors[y], y))

        # Local S vs already-mapped vertices (currently all basis so far)
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

# ---------------------------------------------------------------------------
# Algebraic safety-net (no local S checks during construction). Gated by N.
# ---------------------------------------------------------------------------

def _algebraic_fallback_search(
    *,
    S_mod: np.ndarray,
    G: galois.FieldArray,
    basis_order_idx: List[int],
    basis_mask: np.ndarray,
    coeffs: Optional[np.ndarray],
    p: int,
    k_wanted: int,
    # Phase context (optional):
    tableau: Optional[np.ndarray],
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_right: Optional[np.ndarray],
    rank_full: bool,
    return_phase: bool,
) -> List[Dict[str, object]]:
    """
    Final safety-net for small N: no local S pruning while constructing π.
    Basis mapped by coefficient colours only (then require C invertible). Dependents
    mapped by exact UG/G column signatures and colours. At leaf: enforce global S,
    code, and (optionally) phases. Enumerates with MRV but avoids factorial blowups
    in practice due to strong signature constraints.
    """
    N = G.shape[1]
    results: List[Dict[str, object]] = []

    G_nd = G.view(np.ndarray) % p
    sig_G = [tuple(int(v) for v in G_nd[:, j]) for j in range(N)]

    phi = -np.ones(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    def place_basis(t: int) -> bool:
        if len(results) >= k_wanted:
            return True
        if t >= len(basis_order_idx):
            # build U
            Bcols = np.array(basis_order_idx, dtype=int)
            PBcols = phi[Bcols]
            C = G[:, PBcols]
            C_int = (C.view(np.ndarray) % p)
            U_int = _gauss_inverse_mod_prime(C_int, p)
            if U_int is None:
                return False
            U = galois.GF(p)(U_int)
            return place_dependents(U)

        i = basis_order_idx[t]
        # simple candidates: any unused with same coeff colour (if any)
        cand = [y for y in range(N) if not used[y] and (coeffs is None or coeffs[i] == coeffs[y])]
        for y in cand:
            phi[i] = y
            used[y] = True
            if place_basis(t + 1):
                return True
            phi[i] = -1
            used[y] = False
        return False

    def place_dependents(U: galois.FieldArray) -> bool:
        if len(results) >= k_wanted:
            return True
        UG = U @ G
        UG_nd = UG.view(np.ndarray) % p
        sig_UG = [tuple(int(v) for v in UG_nd[:, j]) for j in range(N)]

        from collections import defaultdict
        free_by_sig: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for y in range(N):
            if not used[y]:
                free_by_sig[sig_UG[y]].append(y)

        dep_idx = [i for i in range(N) if not basis_mask[i]]
        dep_idx.sort(key=lambda i: len(free_by_sig.get(sig_G[i], [])))

        def dfs_dep(t: int) -> bool:
            if len(results) >= k_wanted:
                return True
            if t >= len(dep_idx):
                pi = phi.copy()
                if _is_identity_perm_idx(pi):
                    return False
                if not _check_symplectic_invariance_mod(S_mod, pi):
                    return False
                if not _check_code_automorphism(G, [int(b) for b in basis_order_idx], list(range(N)), pi, p):
                    return False
                rec: Dict[str, object] = {"perm": _perm_index_to_perm_map(pi)}
                if tableau is not None and P_phase is not None:
                    F_int = _build_F_from_perm_with_L(
                        tableau, L_right, pi, p) if rank_full and L_right is not None else None
                    h = _solve_phase_system_variants(
                        A=tableau, phases=phases, pi=pi,
                        P_phase=P_phase, F_int=F_int, p=p, rank_full=rank_full,
                    )
                    if h is None:
                        return False
                    if return_phase:
                        rec["h"] = h
                        if F_int is not None:
                            rec["F"] = F_int
                results.append(rec)
                return True

            i = dep_idx[t]
            cand = [y for y in free_by_sig.get(sig_G[i], []) if (coeffs is None or coeffs[i] == coeffs[y])]
            for y in cand:
                if used[y]:
                    continue
                phi[i] = y
                used[y] = True
                if dfs_dep(t + 1):
                    return True
                phi[i] = -1
                used[y] = False
            return False

        return dfs_dep(0)

    place_basis(0)
    return results[:k_wanted]


# ---------------------------------------------------------------------------
# Public API: accept a Pauli-sum H (as in your environment)
# ---------------------------------------------------------------------------
def find_k_automorphisms_symplectic(
    H,
    *,
    k: int = 1,
    basis_first: str = "any",
    dynamic_refine_every: int = 0,
    p2_bitset: str = "auto",
    enforce_base_on_dependents: bool = False,
    return_phase: bool = True,
    safety_net_max_N: int = 48,
) -> List[Dict[str, object]]:
    # 0) Extract from H
    A_raw = np.asarray(H.tableau, dtype=int)          # <-- canonical: (N, 2n) (rows=terms, cols=qudits)
    S = np.asarray(H.symplectic_product_matrix(), dtype=int)  # (N, N)
    coeffs_raw = None if getattr(H, "weights", None) is None else np.asarray(H.weights)
    phases_raw = None if getattr(H, "phases", None) is None else np.asarray(H.phases, dtype=int)
    dims = list(H.dimensions)

    # Field / sizes
    if len(set(dims)) != 1:
        raise ValueError("All local dimensions must be equal (prime p).")
    p = int(dims[0])
    n_qud = len(dims)
    two_n = 2 * n_qud

    # Authoritative N from S, and assert canonical tableau shape
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square; got {S.shape}.")
    N = int(S.shape[0])
    if A_raw.shape != (N, two_n):
        raise ValueError(f"Expected tableau shape (N,2n)=({N},{two_n}); got {A_raw.shape}.")

    # 1) Build presentation (G) **without transposing** (your util expects N×2n)
    independent, dependencies = get_linear_dependencies(A_raw, p)

    labels = list(range(N))
    G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, labels, p)
    basis_order_idx = [int(i) for i in basis_order]

    # 2) WL (ordering only) on S with coefficient colours
    S_mod = (S % p).astype(np.int64, copy=False)
    coeff_cols = _discretize_coeffs(coeffs_raw)
    base_colors = _wl_colors_from_S(S_mod, p, coeffs=coeff_cols, max_rounds=10)
    base_classes = _color_classes(base_colors)

    # 3) Phase context (optional): if you need (2n×N), transpose **here only**
    phase_ctx = {}
    if phases_raw is not None:
        # Convert to (2n × N) only for the phase system / F build
        A_cols = A_raw.T  # (2n, N) — local use only; dependencies already done from A_raw
        P_phase = _build_P_phase(n_qud, p)
        L_right = _right_inverse_GF(A_cols, p)  # None if rank-deficient
        phase_ctx = dict(
            tableau=A_cols,             # (2n × N) for phase routines
            P_phase=P_phase,
            phases=(phases_raw % p),
            L_right=L_right,
            rank_full=(L_right is not None),
            return_phase=bool(return_phase),
        )
    else:
        # No phases: leave phase_ctx empty; no transpose needed anywhere
        pass

    # 4) Primary fast path
    if basis_first == "any":
        sols = _basis_first_search(
            S_mod=S_mod,
            coeffs=coeff_cols,
            base_colors=base_colors,
            base_classes=base_classes,  # used for ordering only in your latest version
            G=G,
            basis_order_idx=basis_order_idx,
            basis_mask=basis_mask,
            p=p,
            k_wanted=k,
            **phase_ctx,
        )
        if sols:
            return sols

    # 5) Algebraic fallback (small N)
    if N <= int(safety_net_max_N):
        sols = _algebraic_fallback_search(
            S_mod=S_mod,
            G=G,
            basis_order_idx=basis_order_idx,
            basis_mask=basis_mask,
            coeffs=coeff_cols,
            p=p,
            k_wanted=k,
            **phase_ctx,
        )
        if sols:
            return sols

    return []


if __name__ == "__main__":
    from sympleq.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from sympleq.core.circuits import SWAP, Gate

    sym = SWAP(0, 1, 2)
    n_qudits = 3
    n_paulis_pre_sym = 6
    H = random_gate_symmetric_hamiltonian(sym, n_qudits, n_paulis_pre_sym, scrambled=True)

    independent, dependencies = get_linear_dependencies(H.tableau, H.dimensions)

    S = H.symplectic_product_matrix()
    coeffs = H.weights

    # assert H.standard_form() == sym.act(H).standard_form()

    print('independent = ', independent)
    print('dependencies = ', dependencies)

    print('tableau = ', H.tableau)
    print('S = ', S)
    print('coeffs = ', coeffs)

    perms = find_k_automorphisms_symplectic(H, safety_net_max_N=99999)

    print('permutations = ', perms)

    sym_gate = Gate('symmetry', [i for i in range(H.n_qudits())], perms[0]['F'], 2, perms[0]['h'])

    assert sym_gate.act(H).standard_form() == H.standard_form(
    ), f'{sym_gate.act(H).standard_form().__str__()}\n{H.standard_form().__str__()}'

    print('Passed!')
