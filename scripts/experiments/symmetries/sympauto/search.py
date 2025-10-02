from __future__ import annotations
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
import galois

from utils import perm_index_to_map, is_identity_perm_idx, discretize_coeffs
from algebra import in_image_mod2
from wl import wl1_colors, color_classes
from code import build_generator_matrix, compute_U_from_basis, check_code_automorphism
from symplectic import build_F_right_rowperm, left_inverse_tableau, check_S_invariance
from phase import (
    build_phase_rhs_via_U, qF_parity_vector_via_U,
    solve_h_qubits_mod4_with_lift, verify_weights_gate_global_phase
)

# ------------ Local S-consistency (NumPy fast path; drop-in for numba if desired) -----------
def consistent_all(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]: return False
        if S_mod[j, i] != S_mod[yj, y]: return False
    return True

def consistent_vs_basis(S_mod: np.ndarray, phi: np.ndarray, mapped_basis_idx: np.ndarray, i: int, y: int) -> bool:
    for t in range(mapped_basis_idx.size):
        j = mapped_basis_idx[t]
        yj = phi[j]
        if yj < 0:
            continue
        if S_mod[i, j] != S_mod[y, yj]: return False
        if S_mod[j, i] != S_mod[yj, y]: return False
    return True

# --------------------------- Dependent placement via U-signatures ---------------------------

def _place_dependents_with_U(
    *,
    U: galois.FieldArray,
    G: galois.FieldArray,
    basis_mask: np.ndarray,
    coeffs: Optional[np.ndarray],
    S_mod: np.ndarray,
    phi: np.ndarray,
    used: np.ndarray,
    basis_order_idx: List[int],
    p: int,
    # Phase context:
    tableau: Optional[np.ndarray],
    phases: Optional[np.ndarray],
    L_left: Optional[np.ndarray],
    rank_full: bool,
    diag: dict,
    coeffs_raw: Optional[np.ndarray],
    transvection_builder,  # callable or None
) -> Optional[Dict[str, object]]:

    N = G.shape[1]
    G_nd = G.view(np.ndarray) % p
    UG_nd = (U @ G).view(np.ndarray) % p

    def sig_col(nd: np.ndarray, j: int) -> tuple[int,...]:
        return tuple(int(v) for v in nd[:, j])

    sig_G  = [sig_col(G_nd, j) for j in range(N)]
    sig_UG = [sig_col(UG_nd, j) for j in range(N)]

    free_by_sig: Dict[tuple[int,...], List[int]] = defaultdict(list)
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
            diag["dep_cand_empty"] += 1
            return None
        dep_candidates[i] = lst

    dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))
    mapped_basis = np.asarray([b for b in basis_order_idx if phi[b] >= 0], dtype=np.int64)

    def dfs_dep(t: int) -> Optional[Dict[str, object]]:
        if t >= len(dep_order):
            pi = phi.copy()
            if is_identity_perm_idx(pi):
                diag['is_identity'] += 1
                return None
            if not check_S_invariance(S_mod, pi):
                diag['S_fail'] += 1
                return None
            if not check_code_automorphism(G, pi, [int(b) for b in basis_order_idx], list(range(N)), p):
                diag['code_fail'] += 1
                return None

            # Build F (p=2 transvections; else left-inverse fallback)
            F_int = build_F_right_rowperm(tableau, pi, p=2, transvection_builder=transvection_builder, L_left=L_left)
            if F_int is None:
                diag['F_fail'] += 1
                return None

            # Phase RHS and parity sanity
            r = build_phase_rhs_via_U(tableau, phases, pi, F_int, p=2)  # mod 4
            r2 = (r & 1)
            if not in_image_mod2(tableau, r2):
                diag["phase_parity_image_fail"] += 1
                return None
            q2 = qF_parity_vector_via_U(tableau, F_int)
            if not np.array_equal(r2, q2):
                diag["qF_parity_mismatch"] += 1
                return None

            # Solve for h (robust)
            h = solve_h_qubits_mod4_with_lift(tableau, r, diag)
            if h is None:
                diag["phase_fail"] += 1
                return None

            # Final global-phase tolerant verification
            if not verify_weights_gate_global_phase(tableau, coeffs_raw, pi, F_int, h):
                diag['weights_fail'] += 1
                return None

            return {"perm": perm_index_to_map(pi), "h": h, "F": F_int}

        i = dep_order[t]
        for y in dep_candidates[i]:
            if used[y]: continue
            if not consistent_vs_basis(S_mod, phi, mapped_basis, i, y):
                continue
            phi[i] = y; used[y] = True
            sol = dfs_dep(t + 1)
            if sol is not None:
                return sol
            phi[i] = -1; used[y] = False
        return None

    return dfs_dep(0)

# ----------------------------------------- Main API -----------------------------------------

def find_k_automorphisms_symplectic(
    H,
    *,
    k: int = 1,
    coeff_hard: bool = True,          # set False to soften coeff filter
    wl_order_only: bool = True,       # WL(1) only for ordering; no hard WL constraint
    transvection_builder=None,        # inject your tested mapper for p=2
) -> List[Dict[str, object]]:

    # Extract
    tableau = np.asarray(H.tableau(), dtype=int)               # N × 2n
    S = np.asarray(H.symplectic_product_matrix(), dtype=int)   # N × N
    coeffs_raw = None if getattr(H, "weights", None) is None else np.asarray(H.weights)
    phases_raw = None if getattr(H, "phases", None) is None else np.asarray(H.phases, dtype=int)
    dims = list(H.dimensions)

    # Field / shape
    if len(set(dims)) != 1:
        raise ValueError("All local dimensions must be equal (prime p).")
    p = int(dims[0]); assert p == 2, "This implementation targets qubits first (p=2)."
    n_qud = len(dims); two_n = 2 * n_qud
    N = int(S.shape[0])
    if tableau.shape != (N, two_n):
        raise ValueError(f"Expected tableau shape (N,2n)=({N},{two_n}); got {tableau.shape}.")

    # Dependencies → G
    from quaos.utils import get_linear_dependencies  # existing
    independent, dependencies = get_linear_dependencies(tableau, p)
    labels = list(range(N))
    G, basis_order, basis_mask = build_generator_matrix(independent, dependencies, labels, p)
    basis_order_idx = [int(i) for i in basis_order]

    # WL ordering + coefficient colours
    S_mod = (S % p).astype(np.int64, copy=False)
    coeff_cols = discretize_coeffs(coeffs_raw) if coeff_hard else None
    base_colors = wl_order_only and wl1_colors(S_mod, p, coeffs=coeff_cols, max_rounds=10) or np.zeros(N, dtype=int)
    base_classes = color_classes(base_colors)

    phases_vec = np.zeros(N, dtype=int) if phases_raw is None else (phases_raw % p)
    L_left = left_inverse_tableau(tableau, p)  # may be None; F builder will handle

    results: List[Dict[str, object]] = []
    diag = defaultdict(int)

    # ---- Basis mapping DFS (MRV via WL class size) with coeff hard filter ----
    k_needed = max(1, int(k))
    phi = -np.ones(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    # order basis by smallest WL class (MRV-ish)
    basis_idx_seq = sorted(basis_order_idx, key=lambda i: len(base_classes[int(base_colors[i])]))

    def dfs_basis(t: int) -> bool:
        if len(results) >= k_needed:
            return True
        if t >= len(basis_idx_seq):
            U = compute_U_from_basis(G, phi, basis_order_idx, p)
            if U is None:
                return False
            sol = _place_dependents_with_U(
                U=U, G=G, basis_mask=basis_mask, coeffs=coeff_cols,
                S_mod=S_mod, phi=phi, used=used, basis_order_idx=basis_order_idx, p=p,
                tableau=tableau, phases=phases_vec, L_left=L_left, rank_full=(L_left is not None),
                diag=diag, coeffs_raw=coeffs_raw, transvection_builder=transvection_builder
            )
            if sol is not None:
                results.append(sol)
                return True
            return False

        i = basis_idx_seq[t]
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
        pool = list(range(N)) if not wl_order_only else color_classes(base_colors)[int(base_colors[i])]
        # coefficient filter (hard or soft)
        candidates = [y for y in pool if not used[y] and (coeff_cols is None or coeff_cols[i] == coeff_cols[y])]
        # tie-breaker for stability
        candidates.sort()
        for y in candidates:
            # local S-consistency against already mapped
            if not consistent_all(S_mod, phi, mapped_idx, i, y):
                continue
            phi[i] = y; used[y] = True
            if dfs_basis(t + 1):
                return True
            phi[i] = -1; used[y] = False
        return False

    dfs_basis(0)

    # ------------------- Fallback: soften coefficient filter if empty -------------------
    if not results and coeff_hard:
        # try again with coeff_hard = False once (robustness over pruning)
        return find_k_automorphisms_symplectic(H, k=k, coeff_hard=False, wl_order_only=wl_order_only, transvection_builder=transvection_builder)

    # ------------------- Diagnostics (optional print here) -------------------
    if not results:
        # minimal diag print to match your style
        print("[diag] " + ", ".join(f"{k}={v}" for k, v in diag.items()))

    return results[:k_needed]

if __name__ == "__main__":
    # Minimal synthetic test: build a tableau, apply a row perm π, see if we recover it.
    class DummyH:
        def __init__(self, T, S, weights, phases, p):
            self._T = T; self._S = S; self.weights = weights; self.phases = phases; self.dimensions = [p]*(T.shape[1]//2)
        def tableau(self): return self._T
        def symplectic_product_matrix(self): return self._S
        def n_qudits(self): return self._T.shape[1]//2
        def standard_form(self): return self  # for external tests

    # Toy data
    T = np.array([[1,0,1,0,0,1],
                  [1,0,1,0,0,0],
                  [1,0,0,0,0,0],
                  [1,1,0,1,0,1],
                  [0,1,0,0,1,0],
                  [0,1,1,0,1,0],
                  [1,0,1,1,1,0],
                  [1,0,1,1,0,0],
                  [0,0,1,0,1,0]], dtype=int)
    # simple S from rows (mod 2)
    def S_from_rows(A):
        n = A.shape[1]//2
        X = A[:, :n] % 2; Z = A[:, n:] % 2
        return (X @ Z.T + Z @ X.T) % 2
    S = S_from_rows(T)
    w = np.array([1+0j, -1+1j, 0+1j, 1+0j, -1+1j, 0+1j, 2+0j, 1+0j, 1+0j])
    H = DummyH(T, S, w, None, 2)

    res = find_k_automorphisms_symplectic(H, k=1, transvection_builder=None)  # left-inverse path if full rank
    print("[search] results:", res)
