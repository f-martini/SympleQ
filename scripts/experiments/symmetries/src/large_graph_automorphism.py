
# fast_many_dependents.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois

# Optional JIT
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    NUMBA_OK = False

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]


# =============================================================================
# Basic helpers
# =============================================================================

def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))

def _perm_index_to_perm_map(labels: List[int], pi: np.ndarray) -> Dict[int, int]:
    return {labels[j]: labels[int(pi[j])] for j in range(len(labels))}

def _is_identity_perm_idx(pi: np.ndarray) -> bool:
    return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))


# =============================================================================
# Build generator matrix G over GF(p) from (independent, dependencies)
# =============================================================================

def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int], np.ndarray]:
    """
    Columns ordered by `labels`. Independent columns form I_k; dependent column d has
    entries from `dependencies[d]`, with multiplicities reduced mod p.
    """
    GF = galois.GF(p)
    basis_order = sorted(independent)
    k, n = len(basis_order), len(labels)
    label_to_col = {lab: j for j, lab in enumerate(labels)}
    basis_index = {b: i for i, b in enumerate(basis_order)}

    # sanity: deps reference only basis
    dep_keys = set(dependencies.keys())
    inter = dep_keys & set(basis_order)
    if inter:
        raise ValueError(f"Labels cannot be both independent and dependent: {sorted(inter)}")
    for d, pairs in dependencies.items():
        for b, _ in pairs:
            if b not in basis_order:
                raise ValueError(f"Dependency {d} references non-basis label {b}")

    G_int = np.zeros((k, n), dtype=int)
    # basis = identity
    for b in basis_order:
        G_int[basis_index[b], label_to_col[b]] = 1
    # dependents
    for d, pairs in dependencies.items():
        j = label_to_col[d]
        acc = {}
        for b, m in pairs:
            acc[b] = (acc.get(b, 0) + int(m)) % p
        for b, m in acc.items():
            if m % p != 0:
                G_int[basis_index[b], j] = m % p

    G = GF(G_int)

    basis_mask = np.zeros(n, dtype=bool)
    for b in basis_order:
        basis_mask[label_to_col[b]] = True
    return G, basis_order, basis_mask


# =============================================================================
# WL-1 base partition (safe) on S, seeded with coeffs and optional G-invariants
# =============================================================================

def _wl_colors_from_S(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray] = None,
    col_invariants: Optional[np.ndarray] = None,  # shape (n, t) ints; optional extras
    max_rounds: int = 10,
) -> np.ndarray:
    """
    1-WL color refinement on the edge-colored complete graph defined by S_mod (values in 0..p-1).
    Seed key: (coeff[i], col_invariants[i,*], row-histogram of S[i,*]).
    """
    n = S_mod.shape[0]
    hist = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts = np.bincount(S_mod[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette = {}
    color = np.empty(n, dtype=np.int64)
    for i in range(n):
        coeff_key = None if coeffs is None else (coeffs[i].item() if hasattr(coeffs[i], "item") else coeffs[i])
        inv_key = () if col_invariants is None else tuple(int(x) for x in np.atleast_1d(col_invariants[i]))
        seed_key = (coeff_key, inv_key, tuple(hist[i]))
        color[i] = palette.setdefault(seed_key, len(palette))

    for _ in range(max_rounds):
        new_keys = []
        for i in range(n):
            d = {}
            row = S_mod[i]
            for j in range(n):
                key = (int(color[j]), int(row[j]))
                d[key] = d.get(key, 0) + 1
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2 = {}
        new_color = np.empty(n, dtype=np.int64)
        changed = False
        for i, key in enumerate(new_keys):
            c = palette2.setdefault(key, len(palette2))
            new_color[i] = c
            if c != color[i]:
                changed = True
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
# Symplectic consistency checks (incremental)
# =============================================================================

@njit(cache=True, fastmath=True)
def _consistent_numba(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    General p: check S[i, j] == S[y, phi[j]] and S[j, i] == S[phi[j], y] for mapped j.
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True

def _build_bitrows_binary(S_mod: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    For p=2 only. Pack each row's 0/1 into 64-bit chunks.
    Returns (bits[n, C], chunks=C).
    """
    n = S_mod.shape[0]
    C = (n + 63) // 64
    bits = np.zeros((n, C), dtype=np.uint64)
    # vectorized packing
    for i in range(n):
        for j in range(n):
            if S_mod[i, j] & 1:
                bits[i, j >> 6] |= (1 << (j & 63))
    return bits, C

@njit(cache=True, fastmath=True)
def _consistent_bitset(bits: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    p=2 bitset version. For each mapped j, compare bit S[i,j] to S[y,phi[j]] and S[j,i] to S[phi[j],y].
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        # read bit S[i,j] vs S[y,yj]
        bi = (bits[i, j >> 6] >> (j & 63)) & 1
        by = (bits[y, yj >> 6] >> (yj & 63)) & 1
        if bi != by:
            return False
        # symmetric counterpart
        bji = (bits[j, i >> 6] >> (i & 63)) & 1
        byy = (bits[yj, y >> 6] >> (y & 63)) & 1
        if bji != byy:
            return False
    return True


# =============================================================================
# Leaf checks
# =============================================================================

def _check_symplectic_invariance_mod(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)

def _check_code_automorphism(
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],
    pi: np.ndarray
) -> bool:
    """
    Check ∃ U with U G P = G. Implemented via U = (G[:, P(B)])^{-1} and equality test.
    """
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    Bcols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[Bcols]
    C = G[:, PBcols]
    try:
        U = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return False
    Gp = G[:, pi]
    return np.array_equal(U @ Gp, G)


# =============================================================================
# Basis-first (complete) with fast dependent placement via UG signatures
# =============================================================================

def find_k_automorphisms_many_dependents(
    independent: List[int],
    dependencies: DepPairs,
    *,
    S: np.ndarray,
    p: int = 2,
    k: int = 1,
    S_labels: Optional[List[int]] = None,
    require_nontrivial: bool = True,
    coeffs: Optional[np.ndarray] = None,
    coeff_labels: Optional[List[int]] = None,
    # NEW toggles
    restrict_basis_by_WL: bool = True,
    enforce_base_on_dependents: bool = False,
    try_unrestricted_if_none: bool = True,
) -> List[Dict[int, int]]:
    """
    Basis-first (complete) solver optimized for n >> k.
    - WL base partition is computed from S (+coeffs) ONLY (no G-derived features).
    - If restrict_basis_by_WL=True, basis images must lie in same WL class (safe).
    - If enforce_base_on_dependents=True, dependents must also respect WL classes (off by default).
    - If no solution is found and try_unrestricted_if_none=True, rerun with both WL restrictions OFF.
    """
    # --- shared setup (wrapped so we can reuse for fallback) ------------------
    def _run_core(restrict_basis: bool, restrict_deps: bool) -> List[Dict[int,int]]:
        pres_labels = _labels_union(independent, dependencies)
        n = len(pres_labels)

        # Align S
        if S_labels is not None:
            lab_to_pos = {lab: i for i, lab in enumerate(S_labels)}
            idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
            S_aligned = S[np.ix_(idx, idx)]
        else:
            if S.shape != (n, n):
                raise ValueError("S shape does not match number of labels; supply S_labels.")
            S_aligned = S
        S_mod = np.mod(S_aligned, p).astype(np.int64, copy=False)

        # Align coeffs
        coeffs_aligned = None
        if coeffs is not None:
            coeffs_arr = np.asarray(coeffs)
            if coeff_labels is not None:
                lab_to_pos = {lab: i for i, lab in enumerate(coeff_labels)}
                idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
                coeffs_aligned = coeffs_arr[idx]
            else:
                if coeffs_arr.shape[0] != n:
                    raise ValueError("coeffs length does not match number of labels; or provide coeff_labels.")
                coeffs_aligned = coeffs_arr

        # Build G & basis mask
        G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, pres_labels, p)
        lab_to_idx = {lab: i for i, lab in enumerate(pres_labels)}
        basis_idx = [lab_to_idx[b] for b in basis_order]
        k_rank = len(basis_idx)

        # WL base partition (SAFE: from S + coeffs only)
        base_colors = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned, col_invariants=None, max_rounds=10)
        base_classes = _color_classes(base_colors)

        # S-consistency kernels
        if p == 2:
            bits, _ = _build_bitrows_binary(S_mod)
            def consistent(phi, mapped, i, y):
                return _consistent_bitset(bits, phi, mapped, int(i), int(y))
        else:
            def consistent(phi, mapped, i, y):
                return _consistent_numba(S_mod, phi, mapped, int(i), int(y))

        # Order basis by most constrained (WL class size) if we are restricting;
        # otherwise keep the natural (given) order.
        if restrict_basis:
            basis_idx_sorted = sorted(basis_idx, key=lambda i: len(base_classes[int(base_colors[i])]))
        else:
            basis_idx_sorted = list(basis_idx)

        # State
        phi = -np.ones(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        results: List[Dict[int,int]] = []

        # helpers for dependent stage
        def col_key_field(col: galois.FieldArray) -> Tuple[int, ...]:
            return tuple(int(v) for v in col)

        def build_target_map(UG: galois.FieldArray, available: np.ndarray) -> Dict[Tuple[int, ...], List[int]]:
            m: Dict[Tuple[int, ...], List[int]] = {}
            for y in range(n):
                if not available[y]:
                    continue
                key = col_key_field(UG[:, y])
                m.setdefault(key, []).append(y)
            return m

        def place_dependents_fast_iterative(U: galois.FieldArray) -> bool:
            """
            Iterative (non-recursive) dependent placement.
            After the basis images are fixed and U = (G[:, P(B)])^{-1} is known, we form UG
            and, for each dependent i, restrict candidates to y with UG[:,y] == G[:,i].
            We then perform an explicit-stack DFS over all dependents ordered by candidate
            list size, enforcing symplectic consistency and (optional) coefficient/WL filters.
            """
            # -- Precompute UG and target signature map
            UG = U @ G  # k x n
            def col_key_field(col: galois.FieldArray) -> Tuple[int, ...]:
                return tuple(int(v) for v in col)
            # All currently available targets (unused only)
            available = ~used
            target_map: Dict[Tuple[int, ...], List[int]] = {}
            for y in range(n):
                if not available[y]:
                    continue
                key = col_key_field(UG[:, y])
                target_map.setdefault(key, []).append(y)

            # Build dependents and their raw candidate lists from UG signatures
            dependents = [i for i in range(n) if not basis_mask[i]]
            dep_sig = {i: col_key_field(G[:, i]) for i in dependents}

            # Early pigeonhole check and candidate base lists
            cand0: Dict[int, List[int]] = {}
            for i in dependents:
                cand = target_map.get(dep_sig[i], [])
                if len(cand) == 0:
                    return False  # no target matches UG-signature -> dead
                cand0[i] = cand

            # Order variables (dependents) by initial candidate-count (MRV)
            var_order = sorted(dependents, key=lambda i: len(cand0[i]))

            # Explicit DFS state
            t = 0                                   # depth / index into var_order
            choice_ptr = [0] * len(var_order)       # per-level pointer into filtered candidates
            filtered_cands: List[List[int]] = [None] * len(var_order)  # caches per level

            while True:
                if t == len(var_order):
                    # All dependents placed; check and record
                    pi = phi.copy()
                    if require_nontrivial and _is_identity_perm_idx(pi):
                        # Backtrack
                        t -= 1
                        if t < 0:
                            return False
                        # undo last assignment below
                    else:
                        if _check_symplectic_invariance_mod(S_mod, pi) and _check_code_automorphism(G, basis_order, pres_labels, pi):
                            results.append(_perm_index_to_perm_map(pres_labels, pi))
                            return True
                        # if it fails, backtrack

                if t < 0:
                    return False  # exhausted

                i = var_order[t]
                bi = int(base_colors[i])

                # Build / refresh filtered candidates at this level
                if filtered_cands[t] is None:
                    base_list = cand0[i]
                    # Filter by global 'used', coeffs, (optional) WL class on dependents
                    cand_list = []
                    for y in base_list:
                        if used[y]:
                            continue
                        if coeffs_aligned is not None and coeffs_aligned[i] != coeffs_aligned[y]:
                            continue
                        if enforce_base_on_dependents and int(base_colors[y]) != bi:
                            continue
                        cand_list.append(y)
                    filtered_cands[t] = cand_list
                    choice_ptr[t] = 0

                # Advance to next feasible candidate y that passes local S-consistency
                placed = False
                cand_list = filtered_cands[t]
                while choice_ptr[t] < len(cand_list):
                    y = cand_list[choice_ptr[t]]
                    choice_ptr[t] += 1
                    # Local symplectic consistency vs already mapped vertices
                    mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
                    if not consistent(phi, mapped_idx, i, y):
                        continue
                    # Commit assignment
                    phi[i] = y
                    used[y] = True
                    # Descend
                    t += 1
                    if t < len(var_order):
                        filtered_cands[t] = None  # ensure rebuild at next level
                    placed = True
                    break

                if placed:
                    continue  # go deeper

                # No candidate worked at this level -> backtrack
                filtered_cands[t] = None
                choice_ptr[t] = 0
                t -= 1
                if t < 0:
                    return False
                # Undo previous level's assignment
                prev_i = var_order[t]
                y_prev = phi[prev_i]
                if y_prev >= 0:
                    used[y_prev] = False
                    phi[prev_i] = -1
                # Loop continues to try the next candidate at the previous level


        def dfs_basis(t: int) -> bool:
            if len(results) >= k:
                return True
            if t >= k_rank:
                PBcols = phi[basis_idx]
                C = G[:, PBcols]
                try:
                    U = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    return False
                return place_dependents_fast_iterative(U)

            i = basis_idx_sorted[t]
            mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
            # candidate images for basis i
            if restrict_basis:
                pool = base_classes[int(base_colors[i])]
            else:
                pool = list(range(n))
            cand = [y for y in pool if not used[y]]
            if coeffs_aligned is not None:
                cand = [y for y in cand if coeffs_aligned[i] == coeffs_aligned[y]]

            for y in cand:
                if not consistent(phi, mapped_idx, i, y):
                    continue
                phi[i] = y
                used[y] = True
                if dfs_basis(t + 1):
                    return True
                phi[i] = -1
                used[y] = False
            return False

        dfs_basis(0)
        return results[:k]

    # --- first pass: your fast settings --------------------------------------
    sols = _run_core(
        restrict_basis=restrict_basis_by_WL,
        restrict_deps=enforce_base_on_dependents,
    )
    if sols or not try_unrestricted_if_none:
        return sols

    # --- fallback pass: GUARANTEED complete (no WL restrictions at all) ------
    # This cannot prune any valid automorphism; it's slower only when needed.
    return _run_core(restrict_basis=False, restrict_deps=False)


if __name__ == "__main__":
    from quaos.models.chemistry import water_molecule
    from quaos.utils import get_linear_dependencies
    from quaos.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from quaos.core.circuits import SWAP

    H = water_molecule()
    # sym = SWAP(0, 1, 2)
    # H = random_gate_symmetric_hamiltonian(sym, 10, 20, scrambled=True)

    independent, dependencies = get_linear_dependencies(H.tableau(), H.dimensions)

    S = H.symplectic_product_matrix()

    perms = find_k_automorphisms_many_dependents(independent, dependencies, S=S, p=2, k=2, require_nontrivial=True,
                                                 coeffs=H.weights)
    print(perms)
