from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from quaos.utils import get_linear_dependencies
from numba import njit
from quaos.core.circuits.target import map_pauli_sum_to_target_tableau as _map_sum_to_target
Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# =============================================================================
# Small utilities
# =============================================================================

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

def _almost_equal_complex(a: complex, b: complex, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

def _verify_weights_with_h(
    tableau: np.ndarray,   # N × 2n
    coeffs: Optional[np.ndarray],  # length N (complex)
    pi: np.ndarray,         # length N row permutation
    h: np.ndarray,          # length 2n, modulo 2p
    p: int
) -> bool:
    if coeffs is None:
        return True  # nothing to check
    mod2 = 2 * int(p)
    N = tableau.shape[0]
    # phase exponent per row: t_i = (a_i · h) mod 2p
    t = (tableau @ (h % mod2)) % mod2

    if p == 2:
        for i in range(N):
            if t[i] & 1:
                # print(f"[weights] odd t[{i}]={t[i]} -> unexpected ±i from h-only")
                return False
            factor = 1.0 if (t[i] % 4 == 0) else -1.0
            lhs = coeffs[i]
            rhs = factor * coeffs[pi[i]]
            if not _almost_equal_complex(lhs, rhs):
                print(f"[weights] mismatch at row {i}: coeff[i]={lhs}, factor={factor}, coeff[pi[i]]={coeffs[pi[i]]}")
                return False
        return True
    else:
        # For general prime p, you’d map the exponent t_i (mod 2p) to a p-th root of unity.
        # Placeholder: accept for now (or implement your desired mapping).
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


def _build_phase_rhs_tableau(tableau: np.ndarray, phases: Optional[np.ndarray], pi: np.ndarray,
                             Delta_mod2: np.ndarray, p: int) -> np.ndarray:
    """
    Build r (length N) where r_i = 2(φ_{π(i)} - φ_i) - a_i Δ a_i^T  (mod 2p),
    with a_i the *row* i of tableau (N × 2n).
    """
    mod2 = 2 * int(p)
    N = tableau.shape[0]
    phi = np.zeros(N, dtype=int) if phases is None else (np.asarray(phases, dtype=int) % p)
    r = np.zeros(N, dtype=int)
    for i in range(N):
        a = (tableau[i, :] % mod2).astype(int)
        quad = int(a @ ((Delta_mod2 @ a) % mod2)) % mod2
        dphi = int((2 * ((phi[pi[i]] - phi[i]) % p)) % mod2)
        r[i] = (dphi - quad) % mod2
    return r

def _solve_tableau_h_eq_r_mod_2p(tableau: np.ndarray, r: np.ndarray, p: int) -> Optional[np.ndarray]:
    """
    Solve (tableau) h ≡ r  (mod 2p), where tableau is N × 2n, r is length N.
    Returns h (length 2n) modulo 2p, or None if inconsistent.
    """
    M = tableau  # N × 2n
    if p % 2 == 1:
        h_p = _gauss_solve_mod_prime(M, r, p)
        if h_p is None:
            return None
        h_2 = _gauss_solve_mod_prime(M, r, 2)
        if h_2 is None:
            return None
        inv2 = _inv_mod_prime(2, p)
        h = np.zeros_like(h_p, dtype=int)
        for i in range(h.size):
            k = (int(h_p[i]) - int(h_2[i])) % p
            u = (k * inv2) % p
            h[i] = (int(h_2[i]) + 2 * int(u)) % (2 * p)
        return h % (2 * p)
    else:
        # p == 2 -> lift mod 2 to mod 4
        h2 = _gauss_solve_mod_prime(M, r, 2)
        if h2 is None:
            return None
        e = (r % 4 - (M % 4) @ (h2 % 4)) % 4
        if np.any(e % 2 != 0):
            return None
        y = _gauss_solve_mod_prime(M, (e // 2) % 2, 2)
        if y is None:
            return None
        return (h2 + 2 * y) % 4  # 2p == 4

def _solve_phase_system_variants_tableau(
    tableau: np.ndarray,          # N × 2n
    phases: Optional[np.ndarray],
    pi: np.ndarray,
    P_phase: np.ndarray,
    F_int: Optional[np.ndarray],  # 2n × 2n mod p
    p: int,
    rank_full: bool,
) -> Optional[np.ndarray]:
    """
    Try to solve (tableau) h ≡ r (mod 2p). Prefer Δ from a concrete F when available.
    If rank is not full (no F), fall back to r = 2(φπ - φ).
    """
    mod2 = 2 * int(p)

    def try_rhs(r_vec: np.ndarray) -> Optional[np.ndarray]:
        h = _solve_tableau_h_eq_r_mod_2p(tableau, r_vec, p)
        if h is None:
            return None
        lhs = (tableau @ (h % mod2)) % mod2
        return h % mod2 if np.array_equal(lhs % mod2, r_vec % mod2) else None

    if F_int is not None:
        Delta = (F_int.T @ (P_phase % mod2) @ F_int - (P_phase % mod2)) % mod2
        for D in (Delta, (-Delta) % mod2):  # handle sign conventions
            r = _build_phase_rhs_tableau(tableau, phases, pi, D, p)
            h = try_rhs(r)
            if h is not None:
                return h

    if not rank_full:
        N = tableau.shape[0]
        phi = np.zeros(N, dtype=int) if phases is None else (np.asarray(phases, dtype=int) % p)
        r = (2 * ((phi[pi] - phi) % p)) % mod2
        h = try_rhs(r)
        if h is not None:
            return h

    return None

# =============================================================================
# Dependent placement using signatures (UG vs G) + basis-only S checks
# =============================================================================

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
    tableau: Optional[np.ndarray],   # N × 2n
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_left: Optional[np.ndarray],    # 2n × N
    rank_full: bool,
    return_phase: bool,
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
            return None
        dep_candidates[i] = lst

    dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))
    mapped_basis = np.asarray([b for b in basis_order_idx if phi[b] >= 0], dtype=np.int64)

    def dfs_dep(t: int) -> Optional[Dict[str, object]]:
        if t >= len(dep_order):
            pi = phi.copy()
            if _is_identity_perm_idx(pi):
                return None
            if not _check_symplectic_invariance_mod(S_mod, pi):
                return None
            if not _check_code_automorphism(G, [int(b) for b in basis_order_idx], list(range(N)), pi, p):
                return None

            rec: Dict[str, object] = {"perm": _perm_index_to_perm_map(pi)}

            # Build/verify F BEFORE phase solve.
            F_int = _build_F_right_rowperm(tableau, L_left, pi, p)  # returns None if fails mapping/symplecticity
            if F_int is None:
                return None

            # Phase feasibility with the CORRECT F
            h = _solve_phase_system_variants_tableau(
                tableau=tableau, phases=phases, pi=pi,
                P_phase=_build_P_phase(tableau.shape[1] // 2, p),  # or reuse cached P_phase
                F_int=F_int, p=p, rank_full=(L_left is not None),
            )
            if h is None:
                return None
            
            if not _verify_weights_with_h(tableau, getattr(coeffs, "orig", coeffs), pi, h, p):
                return None  # reject this π: it doesn’t preserve weights once rephasing by h is applied

            # Pack result
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
            )
            if sol is not None:
                results.append(sol)
                return True
            return False

        i = basis_idx_seq[t]
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # WL ordering only; coefficient colours are hard constraints
        if coeffs is None:
            candidates = [y for y in range(N) if not used[y]]
        else:
            candidates = [y for y in range(N) if (not used[y]) and (coeffs[i] == coeffs[y])]

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

# =============================================================================
# Algebraic fallback (small N) – no local S during construction
# =============================================================================

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
    tableau: Optional[np.ndarray],  # N × 2n
    P_phase: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_left: Optional[np.ndarray],   # 2n × N
    rank_full: bool,
    return_phase: bool,
) -> List[Dict[str, object]]:
    """
    Small-N safety net: construct π with coefficient-colour constraints only (no local S),
    then verify global S, code, and phases at the leaf. Dependents placed via UG vs G signatures.
    """
    N = G.shape[1]
    results: List[Dict[str, object]] = []

    phi = -np.ones(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    def place_basis(t: int) -> bool:
        if len(results) >= k_wanted:
            return True
        if t >= len(basis_order_idx):
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
        G_nd = G.view(np.ndarray) % p
        UG = U @ G
        UG_nd = UG.view(np.ndarray) % p

        sig_G = [tuple(int(v) for v in G_nd[:, j]) for j in range(N)]
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
                    F_int = None
                    if rank_full and L_left is not None:
                        F_int = _build_F_right_rowperm(tableau, L_left, pi, p)
                        if F_int is None:
                            return False
                    h = _solve_phase_system_variants_tableau(
                        tableau=tableau, phases=phases, pi=pi,
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

def find_k_automorphisms_symplectic(
    H,
    *,
    k: int = 1,
    basis_first: str = "any",          # "any" or "off" (use algebraic fallback only)
    safety_net_max_N: int = 64,        # enable algebraic fallback for small N
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
    if basis_first == "any":
        results = _basis_first_search(
            S_mod=S_mod,
            coeffs=coeff_cols,
            base_colors=base_colors,
            base_classes=base_classes,  # ordering only
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
        )
        if results:
            return results

    print('Needs safety net')
    # ----- Algebraic fallback (for small N) -----
    if N <= int(safety_net_max_N):
        results = _algebraic_fallback_search(
            S_mod=S_mod,
            G=G,
            basis_order_idx=basis_order_idx,
            basis_mask=basis_mask,
            coeffs=coeff_cols,
            p=p,
            k_wanted=k,
            tableau=tableau,
            P_phase=P_phase,
            phases=phases_vec,
            L_left=L_left,
            rank_full=rank_full,
            return_phase=return_phase,
        )
        if results:
            return results

    # No solutions found
    return []


if __name__ == "__main__":
    from quaos.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from quaos.core.circuits import SWAP, Gate

    sym = SWAP(0, 1, 2)
    n_qudits = 3
    n_paulis_pre_sym = 6
    H = random_gate_symmetric_hamiltonian(sym, n_qudits, n_paulis_pre_sym, scrambled=False)

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

    sym_gate = Gate('symmetry', [i for i in range(H.n_qudits())], perms[0]['F'], 2, perms[0]['h'])

    assert sym_gate.act(H).standard_form() == H.standard_form(
    ), f'{sym_gate.act(H).standard_form().__str__()}\n{H.standard_form().__str__()}'

    print('Passed!')
