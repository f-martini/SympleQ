# symplectic_aut.py
import numpy as np
import networkx as nx  # used for API consistency; WL implemented below for speed/control
import galois
from sympy.combinatorics.permutations import Permutation
from typing import Dict, List, Tuple, Optional

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]


# ============================================================
# Utilities
# ============================================================
def _is_identity_perm_map(perm_map: Dict[int, int]) -> bool:
    return all(k == v for k, v in perm_map.items())


def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))


def _build_generator_matrix(
    independent: List[int],
    dependencies: DepPairs,
    labels: List[int],
    p: int,
) -> Tuple[galois.FieldArray, List[int], Dict[int, int]]:
    """
    Build generator matrix G over GF(p) in systematic form:
      - columns ordered by 'labels' (sorted),
      - independent columns = I_k,
      - dependent columns = integer accumulation mod p, then cast to GF(p) in one go.
    """
    GF = galois.GF(p)

    basis_order = sorted(independent)
    k = len(basis_order)
    n = len(labels)
    label_to_col = {lab: j for j, lab in enumerate(labels)}
    basis_index = {b: i for i, b in enumerate(basis_order)}

    # Start as INT array (fast), fill, then cast once to GF(p)
    G_int = np.zeros((k, n), dtype=int)

    # independent columns = identity
    for b in basis_order:
        i = basis_index[b]
        j = label_to_col[b]
        G_int[i, j] = 1  # integer; we mod p later and cast to GF

    # dependent columns: accumulate integers then mod p
    for d, pairs in dependencies.items():
        j = label_to_col[d]
        for b, m in pairs:
            i = basis_index[b]
            G_int[i, j] += int(m)  # still integer accumulation

    G_int %= p
    G = GF(G_int)  # one cast; entire matrix becomes a FieldArray over GF(p)
    return G, basis_order, label_to_col

def _perm_map_to_index_perm(labels: List[int], perm_map: Dict[int, int]) -> np.ndarray:
    """
    Convert mapping {label -> label} into a column permutation array 'pi' such that
    (G @ P) == G[:, pi], i.e., pi[j] = index of permuted label for column j.
    """
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    pi = np.empty(len(labels), dtype=int)
    for j, lab in enumerate(labels):
        pi[j] = lab_to_idx[perm_map[lab]]
    return pi


def _perm_index_to_perm_map(labels: List[int], pi: np.ndarray) -> Dict[int, int]:
    """
    Convert index permutation array 'pi' into mapping {label -> label}.
    'pi[j]' is the new column index of original column j.
    Equivalently, label_j -> labels[pi[j]].
    """
    return {labels[j]: labels[pi[j]] for j in range(len(labels))}


def _check_symplectic_invariance(S: np.ndarray, pi: np.ndarray, p: int) -> bool:
    """
    Check P^T S P == S (mod p) using index permutation 'pi' (columns/rows both permuted).
    """
    # Numba: JIT this index remapping & comparison for large n.
    S_mod = np.mod(S, p)
    S_perm = S_mod[np.ix_(pi, pi)]
    return np.array_equal(S_perm, S_mod)


def _check_code_automorphism(
    G,                    # galois.FieldArray, shape (k, n)
    basis_order,          # List[int]
    labels,               # List[int]
    pi: np.ndarray,       # index permutation, shape (n,)
    p: int,
) -> bool:
    """
    Linear-code check over GF(p): ∃ U ∈ GL(k,p) such that U G P = G ?
    Since G is systematic, let C = G[:, P(B)], U := C^{-1}, check U G P == G.
    """
    # columns of basis after permutation
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    Bcols = [lab_to_idx[b] for b in basis_order]
    PBcols = pi[Bcols]

    C = G[:, PBcols]
    try:
        # Use galois’ modular inverse (works on FieldArray)
        U = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return False

    Gp = G[:, pi]
    left = U @ Gp
    return np.array_equal(left, G)


# ============================================================
# WL (1-dimensional) color refinement on the edge-colored complete graph from S
# ============================================================
def _wl_colors_from_S(S: np.ndarray, p: int, max_rounds: int = 10) -> np.ndarray:
    """
    Compute stable WL-1 colors from the symplectic product matrix S (mod p).
    Returns an array 'color' of shape (n,) with small consecutive integers.
    """
    # Numba: Speed up the histogram computations inside the loop.
    S = np.mod(S, p)
    n = S.shape[0]
    # Seed colors: row histogram over GF(p)
    # hist[i] = tuple(counts of values 0..p-1 in row i)
    hist = np.zeros((n, p), dtype=int)
    for i in range(n):
        counts = np.bincount(S[i], minlength=p)
        hist[i, :p] = counts[:p]
    # map hist rows to initial color ids
    # (use dict of tuples -> color id)
    palette = {}
    color = np.empty(n, dtype=int)
    for i in range(n):
        key = tuple(hist[i].tolist())
        color[i] = palette.setdefault(key, len(palette))

    for _ in range(max_rounds):
        # neighbor histogram keyed by (neighbor_color, edge_value)
        # Build composite keys per node; we’ll hash as tuple of pairs
        new_keys = []
        for i in range(n):
            # count pairs (color[j], S[i,j])
            # fast path: accumulate in dict
            # (for very large n, consider counting by blocking in NumPy)
            d = {}
            row = S[i]
            for j in range(n):
                key = (color[j], int(row[j]))
                d[key] = d.get(key, 0) + 1
            items = tuple(sorted(d.items()))
            new_keys.append((int(color[i]), items))

        # compress to new color ids
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
# COMPLETE backtracking guided by WL colors + incremental S-consistency
# ============================================================
def _find_k_permutations_complete(
    independent: List[int],
    dependencies: DepPairs,
    S: np.ndarray,      # assumed aligned to 'labels' order
    labels: List[int],
    *,
    p: int,
    k: int,
    forbid_identity: bool = True,
) -> List[Dict[int, int]]:
    """
    Complete search (won't miss) for permutations pi of indices:
      - incremental enforcement of P^T S P == S (row/col consistency),
      - final check: linear-code automorphism U G P = G over GF(p).
    Returns up to k non-trivial permutations as label->label maps.
    """
    n = len(labels)
    S_mod = np.mod(S, p)
    # Build G once
    G, basis_order, _ = _build_generator_matrix(independent, dependencies, labels, p)

    # WL colors from S
    colors = _wl_colors_from_S(S_mod, p)
    classes = _color_classes(colors)

    # Per-class used-target bookkeeping
    used_by_class = {c: set() for c in classes}

    # Partial mapping: phi[i] = y (index -> index), -1 if unmapped
    phi = -np.ones(n, dtype=int)
    used_target = np.zeros(n, dtype=bool)

    # Domain order: largest classes first (reduces branching)
    class_order = sorted(classes.keys(), key=lambda c: -len(classes[c]))
    domain_order = [i for c in class_order for i in classes[c]]

    results: List[Dict[int, int]] = []

    # Precompute quick structures for consistency
    # (Optionally Numba JIT this routine for large n)
    def consistent(i: int, y: int) -> bool:
        # Check with already mapped j -> phi[j]:
        # S[i, j] == S[y, phi[j]]  and  S[j, i] == S[phi[j], y]
        row_i = S_mod[i]
        col_i = S_mod[:, i]
        row_y = S_mod[y]
        col_y = S_mod[:, y]
        for j in np.nonzero(phi >= 0)[0]:
            yj = phi[j]
            if row_i[j] != row_y[yj]:
                return False
            if col_i[j] != col_y[yj]:
                return False
        return True

    def select_next_index() -> int:
        # MRV: smallest remaining candidates in its class
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

    def at_leaf_check_and_store() -> bool:
        # full permutation in index form
        pi = phi.copy()
        if forbid_identity and np.all(pi == np.arange(n, dtype=int)):
            return False
        # (A) symplectic invariance (should hold by construction, but keep)
        if not _check_symplectic_invariance(S_mod, pi, p):
            return False
        # (B) linear-code automorphism check over GF(p)
        if not _check_code_automorphism(G, basis_order, labels, pi, p):
            return False
        # Accept
        perm_map = _perm_index_to_perm_map(labels, pi)
        results.append(perm_map)
        return True

    # DFS backtracking
    def dfs() -> bool:
        if len(results) >= k:
            return True
        # if complete
        if np.all(phi >= 0):
            if at_leaf_check_and_store():
                if len(results) >= k:
                    return True
            return False

        i = select_next_index()
        ci = int(colors[i])
        # iterate candidate targets within same color class
        for y in classes[ci]:
            if used_target[y]:
                continue
            if y in used_by_class[ci]:
                # already used within the class
                continue
            if not consistent(i, y):
                continue

            # place
            phi[i] = y
            used_target[y] = True
            used_by_class[ci].add(y)

            # recurse
            if dfs():
                if len(results) >= k:
                    return True

            # undo
            phi[i] = -1
            used_target[y] = False
            used_by_class[ci].remove(y)

        return False

    # Joblib: For large instances, parallelize the first branching layer:
    # e.g., spawn a task per candidate y for the first i.
    # This sequential version is simpler and avoids overhead on small/medium n.
    dfs()
    return results


# ============================================================
# Public API
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
) -> List[Dict[int, int]]:
    """
    Find up to k non-trivial automorphisms (label->label maps) that:
      1) preserve the symplectic product matrix S: P^T S P = S (mod p),
      2) preserve the vector multiset represented by (independent, dependencies) over GF(p).

    Parameters
    ----------
    independent : list[int]
        Basis labels (independent set).
    dependencies : dict[int, list[tuple[int,int]]]
        Mapping: dependent label -> list of (basis_label, multiplicity). Multiplicities are integers; arithmetic is mod p.
    S : np.ndarray
        Symplectic product matrix. If 'S_labels' is given, rows/cols are in that order; otherwise
        rows/cols are assumed to be ordered by 'sorted(independent ∪ dependencies.keys())'.
    p : int
        Prime field characteristic.
    k : int
        Number of non-trivial automorphisms to return.
    S_labels : Optional[list[int]]
        If provided, the label order corresponding to rows/cols of S.
    forbid_identity : bool
        If True (default), the identity permutation is never returned.

    Returns
    -------
    List[Dict[int,int]]
        Up to k label->label mappings (each a full permutation).

    Notes
    -----
    - Completeness: The WL-guided backtracking over S (with incremental consistency) enumerates
      exactly the permutations preserving S; each is then filtered by the GF(p) linear-code check.
      Therefore, if a symplectic automorphism that also preserves the vector set exists, it will be found.
    - Performance tips:
        * This is fast when WL colors split labels well. If a class is very large (high symmetry),
          consider adding a time cap and/or seeding colors with extra cheap invariants if available.
        * See comments marked "Numba" and "Joblib" for straightforward speedups.
    """
    # Establish our working label order
    pres_labels = _labels_union(independent, dependencies)
    if S_labels is not None:
        # Reorder S to match 'pres_labels'
        lab_to_pos = {lab: i for i, lab in enumerate(S_labels)}
        idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
        S_aligned = S[np.ix_(idx, idx)]
    else:
        # Assume S already matches sorted label order
        S_aligned = S
        # sanity: shape must agree
        n = len(pres_labels)
        if S_aligned.shape != (n, n):
            raise ValueError("S shape does not match the number of labels; provide S_labels to disambiguate.")

    # Run complete search
    sols = _find_k_permutations_complete(
        independent, dependencies, S_aligned, pres_labels, p=p, k=k, forbid_identity=forbid_identity
    )
    return sols


# ============================================================
# Convenience: pretty-print & SymPy conversion
# ============================================================
def as_sympy_permutation(perm_map: Dict[int, int], labels: Optional[List[int]] = None) -> Permutation:
    """
    Convert {label->label} to a SymPy Permutation acting on the set of labels.
    If 'labels' is None, it uses sorted(keys) as support.
    """
    if labels is None:
        labels = sorted(perm_map.keys())
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    images = [lab_to_idx[perm_map[lab]] for lab in labels]
    return Permutation(images)


if __name__ == "__main__":

    p = 2
    independent = [1, 3, 4]
    dependencies = {2: [(1, 1), (3, 1)], 5: [(1, 1), (4, 1)]}

    # Example S (rows/cols in order [1,2,3,4,5]); replace with your real symplectic product matrix
    S = np.array([
        [0,1,0,0,1],
        [1,0,1,0,1],
        [0,1,0,0,0],
        [0,0,0,0,1],
        [1,1,0,1,0],
    ], dtype=int)

    perms = find_k_automorphisms_symplectic(
        independent, dependencies, S=S, p=p, k=3, S_labels=[1,2,3,4,5], forbid_identity=True
    )
    for i, pm in enumerate(perms, 1):
        print(f"[{i}] {pm}")
        print("  SymPy cycles:", as_sympy_permutation(pm).cyclic_form)
