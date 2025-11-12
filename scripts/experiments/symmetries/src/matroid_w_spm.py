from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from numba import njit

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# =============================================================================
# Utilities
# =============================================================================


def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))


def _perm_index_to_perm_map(labels: List[int], pi: np.ndarray) -> Dict[int, int]:
    return {labels[j]: labels[int(pi[j])] for j in range(len(labels))}

# =============================================================================
# Generator matrix over GF(p) (vectorized; no mixed types)
# =============================================================================


def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int], np.ndarray]:
    """
    Build G in systematic form (k x n), columns ordered by `labels`.
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
        for b, m in pairs:
            G_int[basis_index[b], j] += int(m)
    G_int %= p
    G = GF(G_int)

    basis_mask = np.zeros(n, dtype=bool)
    for b in basis_order:
        basis_mask[label_to_col[b]] = True
    return G, basis_order, basis_mask

# =============================================================================
# WL-1 base partition (safe). You can extend the seed with extra invariants.
# =============================================================================


def _wl_colors_from_S(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray] = None,
    col_invariants: Optional[np.ndarray] = None,  # shape (n, t) ints; optional extras for seeding
    max_rounds: int = 10
) -> np.ndarray:
    """
    1-WL color refinement on the complete edge-colored graph with edge color S[i,j] in GF(p).
    Seed key: (coeff[i], col_invariants[i,*], row-histogram-of-S[i,*]).
    This is a safe isomorphism invariant; we use it as a *base* partition (hard constraint).
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
        # count pairs (neighbor_color, edge_value)
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
# Incremental S-consistency (p-agnostic and p=2 bitset variant)
# =============================================================================


@njit(cache=True, fastmath=True)
def _consistent_numba(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Require S[i, j] == S[y, phi[j]] and S[j, i] == S[phi[j], y] for all mapped j.
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True


# ---- Optional p=2 bitset kernel --------------------------------------------

def _build_bitrows_binary(S_mod: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    For p=2 only. Pack each row's 0/1 into chunks of 64 bits.
    Returns (bits[n, C], chunks=C). Column j lives at chunk=j>>6, bit=(j & 63).
    """
    n = S_mod.shape[0]
    C = (n + 63) // 64
    bits = np.zeros((n, C), dtype=np.uint64)
    for i in range(n):
        for j in range(n):
            if S_mod[i, j] & 1:
                bits[i, j >> 6] |= (1 << (j & 63))
    return bits, C


@njit(cache=True, fastmath=True)
def _consistent_bitset(bits: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Same logic as _consistent_numba but reading single bits from packed rows.
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        # read bit S[i,j]
        bi = (bits[i, j >> 6] >> (j & 63)) & 1
        # read bit S[y,yj]
        by = (bits[y, yj >> 6] >> (yj & 63)) & 1
        if bi != by:
            return False
        # read bit S[j,i] vs S[yj,y]
        bji = (bits[j, i >> 6] >> (i & 63)) & 1
        byy = (bits[yj, y >> 6] >> (y & 63)) & 1
        if bji != byy:
            return False
    return True


# =============================================================================
# Full checks at a leaf
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
    Linear-code test over GF(p): ∃ U with U G P = G ?
    Let C = G[:, P(B)]; if invertible, U = C^{-1} and check U G P == G.
    """
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    B_cols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[B_cols]
    C = G[:, PBcols]
    try:
        U = np.linalg.inv(C)  # works on galois.FieldArray
    except np.linalg.LinAlgError:
        return False
    Gp = G[:, pi]
    return np.array_equal(U @ Gp, G)


# =============================================================================
# Basis-first (complete) solver: basis_first="any"
# =============================================================================

def _basis_first_any(
    S_mod: np.ndarray,
    *,
    coeffs: Optional[np.ndarray],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],
    G: galois.FieldArray,
    basis_order: List[int],
    basis_mask: np.ndarray,
    labels: List[int],
    k_wanted: int,
    require_nontrivial: bool,
    p2_bitset: bool,
    enforce_base_on_dependents: bool = False,   # <-- NEW: relaxed by default
) -> List[Dict[int, int]]:
    """
    Complete search that branches only on basis labels (images can be any labels).
    After basis images fixed and C invertible, compute U=C^{-1} and place dependents
    by requiring U @ G[:, y] == G[:, i], plus symplectic consistency and coeff equality.
    If enforce_base_on_dependents is True, y must also lie in the *base* WL class of i.
    """
    n = S_mod.shape[0]
    results: List[Dict[int, int]] = []

    # Order basis variables by increasing base-class size (MRV-ish)
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    basis_idx = [lab_to_idx[b] for b in basis_order]
    basis_idx = sorted(basis_idx, key=lambda i: len(base_classes[int(base_colors[i])]))

    # Prepared consistency kernel
    if p2_bitset:
        bits, _ = _build_bitrows_binary(S_mod)

        def consistent(phi, mapped, i, y):
            return _consistent_bitset(bits, phi, mapped, int(i), int(y))
    else:
        def consistent(phi, mapped, i, y):
            return _consistent_numba(S_mod, phi, mapped, int(i), int(y))

    # State for basis placement
    phi = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)

    def place_dependents_with_U(U: galois.FieldArray) -> bool:
        """
        Try to complete mapping for dependents only.
        Use signature equality U @ G[:, y] == G[:, i] as a hard filter,
        plus S-consistency and coefficient equality.
        """
        ncols = G.shape[1]
        UG = U @ G  # k x n

        # precompute exact column tuples (int) for equality check
        sig_y = [tuple(int(v) for v in UG[:, y]) for y in range(ncols)]
        sig_i = [tuple(int(v) for v in G[:, i]) for i in range(ncols)]

        dependents = [i for i in range(n) if not basis_mask[i]]
        # candidate lists per dependent
        dep_candidates: Dict[int, List[int]] = {}
        for i in dependents:
            # pool: either all unused labels, or only the base class
            if enforce_base_on_dependents:
                pool = base_classes[int(base_colors[i])]
            else:
                pool = range(n)
            lst = []
            for y in pool:
                if used[y]:
                    continue
                if coeffs is not None and coeffs[i] != coeffs[y]:
                    continue
                if sig_y[y] != sig_i[i]:
                    continue
                lst.append(y)
            dep_candidates[i] = lst

        # order by fewest candidates
        dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))

        def dfs_dep(t: int) -> bool:
            if t >= len(dep_order):
                pi = phi.copy()
                if require_nontrivial and np.all(pi == np.arange(n, dtype=pi.dtype)):
                    return False
                if not _check_symplectic_invariance_mod(S_mod, pi):
                    return False
                if not _check_code_automorphism(G, basis_order, labels, pi):
                    return False
                results.append(_perm_index_to_perm_map(labels, pi))
                return True

            i = dep_order[t]
            mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

            for y in dep_candidates[i]:
                if used[y]:
                    continue
                if not consistent(phi, mapped_idx, i, y):
                    continue
                # place
                phi[i] = y
                used[y] = True
                if dfs_dep(t + 1):
                    return True
                # undo
                phi[i] = -1
                used[y] = False
            return False

        return dfs_dep(0)

    # DFS over basis images
    def dfs_basis(t: int) -> bool:
        if len(results) >= k_wanted:
            return True
        if t >= len(basis_idx):
            # All basis mapped: compute U and finish dependents
            PBcols = phi[basis_idx]
            C = G[:, PBcols]
            try:
                U = np.linalg.inv(C)
                U = G.__class__(U)  # Convert to galois.FieldArray
            except np.linalg.LinAlgError:
                return False
            return place_dependents_with_U(U)

        i = basis_idx[t]
        bi = int(base_colors[i])
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # Feasible candidates for this basis variable (base partition + coeffs)
        candidates = [y for y in base_classes[bi] if not used[y]]
        if coeffs is not None:
            candidates = [y for y in candidates if coeffs[i] == coeffs[y]]

        for y in candidates:
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
    return results[:k_wanted]

# =============================================================================
# Fallback full DFS (feasible by base partition; complete). Kept simple/serial.
# =============================================================================


def _full_dfs_complete(
    S_mod: np.ndarray,
    *,
    coeffs: Optional[np.ndarray],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],
    G: galois.FieldArray,
    basis_order: List[int],
    basis_mask: np.ndarray,
    labels: List[int],
    k_wanted: int,
    require_nontrivial: bool,
    p2_bitset: bool,
    dynamic_refine_every: int = 0,
) -> List[Dict[int, int]]:
    """
    Complete interleaved DFS that maps all labels with feasibility constrained ONLY by base partition.
    Dynamic WL (if enabled) is used only for ordering every `dynamic_refine_every` steps.
    """
    n = S_mod.shape[0]
    results: List[Dict[int, int]] = []

    # Prepared consistency kernel
    if p2_bitset:
        bits, _ = _build_bitrows_binary(S_mod)

        def consistent(phi, mapped, i, y):
            return _consistent_bitset(bits, phi, mapped, int(i), int(y))
    else:
        def consistent(phi, mapped, i, y):
            return _consistent_numba(S_mod, phi, mapped, int(i), int(y))

    # Static domain order from base classes (largest first)
    base_order = sorted(base_classes.keys(), key=lambda c: -len(base_classes[c]))
    domain_order = [i for c in base_order for i in base_classes[c]]

    phi = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)

    steps = 0
    cur_colors = base_colors.copy()

    def select_next() -> int:
        # MRV measured against base feasibility
        best_i, best_rem = -1, 10**9
        for i in domain_order:
            if phi[i] >= 0:
                continue
            bi = int(base_colors[i])
            rem = sum((not used[y] and (coeffs is None or coeffs[i] == coeffs[y])) for y in base_classes[bi])
            if rem < best_rem:
                best_i, best_rem = i, rem
                if rem <= 1:
                    break
        return best_i

    def at_leaf(pi: np.ndarray) -> bool:
        if require_nontrivial and np.all(pi == np.arange(n, dtype=pi.dtype)):
            return False
        if not _check_symplectic_invariance_mod(S_mod, pi):
            return False
        if not _check_code_automorphism(G, basis_order, labels, pi):
            return False
        return True

    def maybe_dynamic_refine():
        nonlocal cur_colors
        if dynamic_refine_every <= 0:
            return
        # very light 1-WL just to order (we do not change feasibility!)
        cur_colors = _wl_colors_from_S(S_mod, int(2), coeffs=coeffs, col_invariants=None, max_rounds=1)

    def dfs() -> bool:
        nonlocal steps
        if len(results) >= k_wanted:
            return True
        if np.all(phi >= 0):
            pi = phi.copy()
            if at_leaf(pi):
                results.append(_perm_index_to_perm_map(labels, pi))
                return True
            return False

        if dynamic_refine_every and (steps % dynamic_refine_every == 0):
            maybe_dynamic_refine()
        steps += 1

        i = select_next()
        bi = int(base_colors[i])
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # Order candidates by current colors (ordering heuristic only)
        candidate = [y for y in base_classes[bi] if not used[y]]
        if coeffs is not None:
            candidate = [y for y in candidate if coeffs[i] == coeffs[y]]
        candidate.sort(key=lambda y: cur_colors[y])

        for y in candidate:
            if not consistent(phi, mapped_idx, i, y):
                continue
            phi[i] = y
            used[y] = True
            if dfs():
                return True
            phi[i] = -1
            used[y] = False
        return False

    dfs()
    return results[:k_wanted]

# =============================================================================
# Public API with toggles
# =============================================================================


def find_k_automorphisms_symplectic(
    independent: List[int],
    dependencies: DepPairs,
    S: np.ndarray,
    p: int,
    k: int = 1,
    S_labels: Optional[List[int]] = None,
    require_nontrivial: bool = True,
    # Strategy
    basis_first: str = "any",          # "off" | "any" (complete) | "basis_only" (heuristic)
    fallback_full_if_empty: bool = True,   # <-- NEW
    dynamic_refine_every: int = 0,
    coeffs: Optional[np.ndarray] = None,
    coeff_labels: Optional[List[int]] = None,
    extra_column_invariants: str = "none",
    p2_bitset: str = "auto",
    enforce_base_on_dependents: bool = False,
) -> List[Dict[int, int]]:
    """
    Return up to k automorphisms preserving S and the vector set. See flags above.
    """
    pres_labels = _labels_union(independent, dependencies)
    n = len(pres_labels)

    # Align S
    if S_labels is not None:
        lab_to_pos = {lab: i for i, lab in enumerate(S_labels)}
        idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
        S_aligned = S[np.ix_(idx, idx)]
    else:
        if S.shape != (n, n):
            raise ValueError("S shape does not match the number of labels; supply S_labels.")
        S_aligned = S
    S_mod = np.mod(S_aligned, p).astype(np.int64, copy=False)

    # Align coeffs
    coeffs_aligned = None
    if coeffs is not None:
        coeffs = np.asarray(coeffs)
        if coeff_labels is not None:
            lab_to_pos = {lab: i for i, lab in enumerate(coeff_labels)}
            idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
            coeffs_aligned = coeffs[idx]
        else:
            if coeffs.shape[0] != n:
                raise ValueError("coeffs length does not match number of labels; supply coeff_labels.")
            coeffs_aligned = coeffs

    # Extra invariants from G? Be careful: these are *not* invariants under left GL(k,p)!
    if extra_column_invariants != "none":
        # Strong advice: leave this as "none" unless you know left GL(k,p) preserves your chosen invariants.
        G_for_inv, _, _ = _build_generator_matrix(independent, dependencies, pres_labels, p)
        if extra_column_invariants == "hist":
            inv = np.zeros((n, min(p, 16)), dtype=np.int64)
            for j in range(n):
                col = np.array([int(x) for x in G_for_inv[:, j]])
                cnt = np.bincount(col, minlength=p)
                inv[j, :min(p, 16)] = cnt[:min(p, 16)]
        else:
            raise ValueError("extra_column_invariants must be 'none', 'support', or 'hist'.")

    # Base (safe) partition from S (+coeffs only)
    base_colors = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned, col_invariants=None, max_rounds=10)
    base_classes = _color_classes(base_colors)

    # Build G & basis mask
    G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, pres_labels, p)

    # p=2 bitset?
    use_bitset = (p == 2 and (p2_bitset is True or (p2_bitset == "auto" and n <= 256)))

    if basis_first in ("any", "basis_only"):
        # If "basis_only", shrink feasible targets for basis to basis indices
        if basis_first == "basis_only":
            basis_set_idx = set(np.where(basis_mask)[0])
            base_classes_for_basis = {
                c: [y for y in ys if y in basis_set_idx] for c, ys in base_classes.items()
            }
        else:
            base_classes_for_basis = base_classes

        sols = _basis_first_any(
            S_mod,
            coeffs=coeffs_aligned,
            base_colors=base_colors,
            base_classes=base_classes_for_basis,  # used for basis; dependents controlled by flag
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            k_wanted=k, require_nontrivial=require_nontrivial,
            p2_bitset=use_bitset,
            enforce_base_on_dependents=bool(enforce_base_on_dependents),
        )
        if sols or not fallback_full_if_empty:
            return sols
        # Safety net: fall back once to full DFS (complete)
        return _full_dfs_complete(
            S_mod,
            coeffs=coeffs_aligned,
            base_colors=base_colors,
            base_classes=base_classes,
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            k_wanted=k, require_nontrivial=require_nontrivial,
            p2_bitset=use_bitset,
            dynamic_refine_every=int(dynamic_refine_every),
        )

    if basis_first == "off":
        return _full_dfs_complete(
            S_mod,
            coeffs=coeffs_aligned,
            base_colors=base_colors,
            base_classes=base_classes,
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            k_wanted=k, require_nontrivial=require_nontrivial,
            p2_bitset=use_bitset,
            dynamic_refine_every=int(dynamic_refine_every),
        )

    raise ValueError("basis_first must be 'off', 'any', or 'basis_only'.")


if __name__ == "__main__":
    from sympleq.utils import get_linear_dependencies
    from sympleq.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from sympleq.core.circuits import SWAP

    # sym = SWAP(0, 1, 2)
    # H = random_gate_symmetric_hamiltonian(sym, 2, 4, scrambled=True)

    # independent, dependencies = get_linear_dependencies(H.tableau, H.dimensions)

    # S = H.symplectic_product_matrix()
    # coeffs = H.weights

    independent = [0, 1, 2, 3]
    dependencies = {4: [(1, 1), (3, 1)], 5: [(0, 1), (2, 1)], 6: [(0, 1), (1, 1)]}
    S = np.array([[0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 1, 1, 1, 0]])
    coeffs = np.array([0.+1.j,  0.+1.j, -1.+0.j,  1.+0.j, -1.+0.j,  1.+0.j,  2.+0.j])
    print('independent = ', independent)
    print('dependencies = ', dependencies)

    print('S = ', S)
    print('coeffs = ', coeffs)

    perms = find_k_automorphisms_symplectic(independent, dependencies,
                                            S=S, p=2, k=1,
                                            coeffs=coeffs)

    print('permutations = ', perms)  # either [{'perm':..., 'h':...}] when return_phase=True, or [perm_dict]
