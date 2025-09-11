# symplectic_aut_fast.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from numba import njit
from joblib import Parallel, delayed

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# ============================================================
# Small helpers
# ============================================================
def _is_identity_perm_map(perm_map: Dict[int, int]) -> bool:
    return all(k == v for k, v in perm_map.items())

def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))

def _perm_index_to_perm_map(labels: List[int], pi: np.ndarray) -> Dict[int, int]:
    return {labels[j]: labels[int(pi[j])] for j in range(len(labels))}

# ============================================================
# Generator matrix over GF(p) (vectorized; no mixed types)
# ============================================================
def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int]]:
    """
    Build G in systematic form (k x n), columns ordered by `labels`.
    Independent columns form I_k; dependent columns are coefficient vectors in that basis.
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
    return G, basis_order

# ============================================================
# WL-1 colors from S (seed with coefficients)
# ============================================================
def _wl_colors_from_S(S: np.ndarray, p: int, coeffs: Optional[np.ndarray] = None, max_rounds: int = 10) -> np.ndarray:
    """
    1-WL color refinement on the complete edge-colored graph with edge color S[i,j] mod p.
    Seed color = (coeff[i], row-histogram). Uses Python dict hashing; fast enough in practice.
    """
    S = np.mod(S, p)
    n = S.shape[0]

    # Seed: (coeff, row histogram)
    hist = np.zeros((n, p), dtype=int)
    for i in range(n):
        counts = np.bincount(S[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette = {}
    color = np.empty(n, dtype=int)
    for i in range(n):
        coeff_key = None if coeffs is None else (coeffs[i].item() if hasattr(coeffs[i], "item") else coeffs[i])
        seed_key = (coeff_key, tuple(hist[i].tolist()))
        color[i] = palette.setdefault(seed_key, len(palette))

    # Refinement
    for _ in range(max_rounds):
        new_keys = []
        for i in range(n):
            d = {}
            row = S[i]
            # count pairs (neighbor_color, edge_value)
            for j in range(n):
                key = (int(color[j]), int(row[j]))
                d[key] = d.get(key, 0) + 1
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2 = {}
        new_color = np.empty(n, dtype=int)
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

# ============================================================
# Fast checks (Numba-jitted where it matters)
# ============================================================
def _check_symplectic_invariance(S: np.ndarray, pi: np.ndarray, p: int) -> bool:
    """
    Check P^T S P == S (mod p) using index permutation pi.
    Pure NumPy path (fast: one remap + equality).
    """
    S_mod = np.mod(S, p)
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)

@njit(cache=True, fastmath=True)
def _consistent_numba(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Numba-jitted incremental consistency:
    require S[i, j] == S[y, phi[j]] and S[j, i] == S[phi[j], y] for all mapped j.
    - S_mod: (n,n) int array mod p
    - phi: shape (n,), filled with -1 for unmapped, else target index
    - mapped_idx: shape (m,), the indices j with phi[j] >= 0
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True

def _check_code_automorphism(G: galois.FieldArray, basis_order: List[int], labels: List[int], pi: np.ndarray, p: int) -> bool:
    """
    Linear-code test over GF(p): ∃ U with U G P = G ?
    Let C = G[:, P(B)]; if invertible, U = C^{-1} and check U G P == G.
    """
    # columns of basis after permutation
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

# ============================================================
# Worker DFS (used by both serial and parallel entry points)
# ============================================================
def _dfs_search(
    S_mod: np.ndarray,
    colors: np.ndarray,
    classes: Dict[int, List[int]],
    *,
    coeffs: Optional[np.ndarray],
    k_limit: int,
    forbid_identity: bool,
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],
    # Optional initial assignment (for parallel fan-out)
    phi_init: Optional[np.ndarray] = None,
    used_target_init: Optional[np.ndarray] = None,
    used_by_class_init: Optional[Dict[int, set]] = None,
) -> List[Dict[int, int]]:
    """
    Serial DFS that can start from a partially assigned mapping (phi_init).
    Returns up to k_limit permutations as label->label maps.
    """
    n = S_mod.shape[0]
    # State arrays
    phi = -np.ones(n, dtype=np.int64) if phi_init is None else phi_init.copy()
    used_target = np.zeros(n, dtype=bool) if used_target_init is None else used_target_init.copy()
    used_by_class = {c: set() for c in classes} if used_by_class_init is None else {c: set(v) for c, v in used_by_class_init.items()}

    # Domain order: largest classes first
    class_order = sorted(classes.keys(), key=lambda c: -len(classes[c]))
    domain_order = [i for c in class_order for i in classes[c]]

    results: List[Dict[int, int]] = []

    def select_next_index() -> int:
        best_i, best_rem = -1, 10**9
        for i in domain_order:
            if phi[i] >= 0:
                continue
            c = int(colors[i])
            rem = len(classes[c]) - len(used_by_class[c])
            if rem < best_rem:
                best_i, best_rem = i, rem
                if rem <= 1:
                    break
        return best_i

    def at_leaf(pi: np.ndarray) -> bool:
        if forbid_identity and np.all(pi == np.arange(n, dtype=pi.dtype)):
            return False
        if not _check_symplectic_invariance(S_mod, pi, p=int(1)):  # S_mod already mod p == integers
            # (p argument isn't used by this function; pass dummy)
            return False
        return _check_code_automorphism(G, basis_order, labels, pi, p=int(1))

    def dfs() -> bool:
        # stop early if enough results
        if len(results) >= k_limit:
            return True
        # complete?
        if np.all(phi >= 0):
            pi = phi.copy()
            if at_leaf(pi):
                results.append(_perm_index_to_perm_map(labels, pi))
                return len(results) >= k_limit
            return False

        i = select_next_index()
        ci = int(colors[i])
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        for y in classes[ci]:
            if used_target[y]:
                continue
            if coeffs is not None and coeffs[i] != coeffs[y]:
                continue
            if not _consistent_numba(S_mod, phi, mapped_idx, int(i), int(y)):
                continue

            # place
            phi[i] = y
            used_target[y] = True
            used_by_class[ci].add(y)

            if dfs():
                if len(results) >= k_limit:
                    return True

            # undo
            phi[i] = -1
            used_target[y] = False
            used_by_class[ci].remove(y)

        return False

    dfs()
    return results

# ============================================================
# Top-level complete search with optional joblib parallel fan-out
# ============================================================
def find_k_automorphisms_symplectic(
    independent: List[int],
    dependencies: DepPairs,
    *,
    S: np.ndarray,
    p: int,
    k: int = 1,
    S_labels: Optional[List[int]] = None,
    forbid_identity: bool = True,
    coeffs: Optional[np.ndarray] = None,
    coeff_labels: Optional[List[int]] = None,
    # Parallelization knobs
    n_jobs: int = 1,
    parallel_min_branch: int = 8,
) -> List[Dict[int, int]]:
    """
    Return up to k non-trivial automorphisms (label->label maps) that preserve:
      (A) the symplectic product matrix S (P^T S P = S mod p), and
      (B) the set of vectors given by (independent, dependencies) over GF(p).
    Also enforce that labels only map to equal coefficients when `coeffs` is provided.

    Speedups:
      - Numba: accelerated incremental consistency check.
      - joblib: first branching level parallelized when candidate count >= parallel_min_branch.
    """
    # Establish working label order
    pres_labels = _labels_union(independent, dependencies)
    n = len(pres_labels)

    # Align S to pres_labels (if necessary)
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

    # WL colors (seeded with coefficients)
    colors = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned)
    classes = _color_classes(colors)

    # Build generator matrix once (GF(p))
    G, basis_order = _build_generator_matrix(independent, dependencies, pres_labels, p)

    # Prepare initial branching: choose first variable by MRV
    class_order = sorted(classes.keys(), key=lambda c: -len(classes[c]))
    domain_order = [i for c in class_order for i in classes[c]]

    def select_next_index(phi: np.ndarray, used_by_class: Dict[int, set]) -> int:
        best_i, best_rem = -1, 10**9
        for i in domain_order:
            if phi[i] >= 0:
                continue
            c = int(colors[i])
            rem = len(classes[c]) - len(used_by_class[c])
            if rem < best_rem:
                best_i, best_rem = i, rem
                if rem <= 1:
                    break
        return best_i

    # If n_jobs == 1, just run serial DFS
    if n_jobs == 1:
        return _dfs_search(
            S_mod, colors, classes,
            coeffs=coeffs_aligned, k_limit=k, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, labels=pres_labels,
        )

    # Otherwise parallelize the first branching level
    phi0 = -np.ones(n, dtype=np.int64)
    used_target0 = np.zeros(n, dtype=bool)
    used_by_class0 = {c: set() for c in classes}
    i0 = select_next_index(phi0, used_by_class0)
    ci0 = int(colors[i0])

    # Candidate targets in same color/coeff class consistent with empty mapping
    cand_y = []
    for y in classes[ci0]:
        if coeffs_aligned is not None and coeffs_aligned[i0] != coeffs_aligned[y]:
            continue
        # With empty mapping, consistency is always true; keep y
        cand_y.append(y)

    if len(cand_y) < parallel_min_branch:
        # Not enough fan-out; run serial
        return _dfs_search(
            S_mod, colors, classes,
            coeffs=coeffs_aligned, k_limit=k, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, labels=pres_labels,
        )

    # Distribute a small quota per worker to avoid over-solving
    tasks = len(cand_y)
    per_worker = max(1, (k + tasks - 1) // tasks)

    def _worker(y0: int) -> List[Dict[int, int]]:
        # Start from phi[i0] = y0
        phi_w = phi0.copy(); phi_w[i0] = y0
        used_t_w = used_target0.copy(); used_t_w[y0] = True
        used_c_w = {c: set(v) for c, v in used_by_class0.items()}
        used_c_w[ci0].add(y0)

        return _dfs_search(
            S_mod, colors, classes,
            coeffs=coeffs_aligned, k_limit=per_worker, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, labels=pres_labels,
            phi_init=phi_w, used_target_init=used_t_w, used_by_class_init=used_c_w,
        )

    # Run workers; stop when we have k
    results: List[Dict[int, int]] = []
    for chunk in Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_worker)(y) for y in cand_y):
        results.extend(chunk)
        if len(results) >= k:
            break
    return results[:k]

# ============================================================
# Example usage (remove or adapt in your codebase)
# ============================================================
if __name__ == "__main__":
    p = 2
    independent = [1, 3, 4]
    dependencies = {
        2: [(1, 1), (3, 1)],
        5: [(1, 1), (4, 1)],
    }
    # Labels are [1,2,3,4,5]
    S = np.array([
        [0,1,0,0,1],
        [1,0,1,0,1],
        [0,1,0,0,0],
        [0,0,0,0,1],
        [1,1,0,1,0],
    ], dtype=np.int64)

    # Optional coefficients: same length as labels; equal coefficients must map to equal
    coeffs = np.array([0, 0, 1, 1, 0], dtype=int)  # example

    perms = find_k_automorphisms_symplectic(
        independent, dependencies,
        S=S, p=p, k=3, S_labels=[1,2,3,4,5],
        forbid_identity=True,
        coeffs=coeffs, coeff_labels=[1,2,3,4,5],
        n_jobs=4,                  # parallel fan-out
        parallel_min_branch=2,     # start parallelism when >= 2 first-branch candidates
    )
    for i, pm in enumerate(perms, 1):
        print(f"[{i}] {pm}")
