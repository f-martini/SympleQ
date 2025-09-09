# symplectic_aut_ir_fast.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from numba import njit
from joblib import Parallel, delayed

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# ------------------------------- small utils -------------------------------
def _is_identity_perm_map(perm_map: Dict[int, int]) -> bool:
    return all(k == v for k, v in perm_map.items())

def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))

def _perm_index_to_perm_map(labels: List[int], pi: np.ndarray) -> Dict[int, int]:
    return {labels[j]: labels[int(pi[j])] for j in range(len(labels))}

# ----------------------- generator matrix over GF(p) -----------------------
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

# -------------------- WL-1 colors from S (with tags/coeffs) ----------------
def _wl_colors_from_S(
    S: np.ndarray, p: int,
    *, coeffs: Optional[np.ndarray] = None,
    tags: Optional[np.ndarray] = None,
    max_rounds: int = 10
) -> np.ndarray:
    """
    1-WL color refinement on the complete edge-colored graph with edge color S[i,j] mod p.
    Seed key: (coeff[i], tag[i], row-histogram-of-S[i,*]).
    'tags' is an int array (len n) with -1 for "none" and unique ints for individualized nodes.
    """
    S = np.mod(S, p)
    n = S.shape[0]

    # Seed by (coeff, tag, row histogram)
    hist = np.zeros((n, p), dtype=int)
    for i in range(n):
        counts = np.bincount(S[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette = {}
    color = np.empty(n, dtype=int)
    for i in range(n):
        coeff_key = None if coeffs is None else (coeffs[i].item() if hasattr(coeffs[i], "item") else coeffs[i])
        tag_key = None if tags is None else int(tags[i])
        seed_key = (coeff_key, tag_key, tuple(hist[i].tolist()))
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

# ------------------------ fast checks (numba-assisted) ---------------------
def _check_symplectic_invariance_mod(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    """Check P^T S P == S given S_mod already reduced mod p."""
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)

@njit(cache=True, fastmath=True)
def _consistent_numba(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Incremental consistency:
      require S[i, j] == S[y, phi[j]] and S[j, i] == S[phi[j], y] for all mapped j.
    """
    for t in range(mapped_idx.size):
        j = mapped_idx[t]
        yj = phi[j]
        if S_mod[i, j] != S_mod[y, yj]:
            return False
        if S_mod[j, i] != S_mod[yj, y]:
            return False
    return True

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
    Bcols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[Bcols]
    C = G[:, PBcols]
    try:
        U = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return False
    Gp = G[:, pi]
    return np.array_equal(U @ Gp, G)

# --------------------------- DFS worker with IR ----------------------------
def _dfs_search(
    S_mod: np.ndarray,
    p: int,
    base_colors: np.ndarray,                     # <-- hard feasibility partition
    base_classes: Dict[int, List[int]],          # <-- hard feasibility partition
    *,
    coeffs: Optional[np.ndarray],
    k_limit: int,
    forbid_identity: bool,
    G: galois.FieldArray,
    basis_order: List[int],
    basis_mask: np.ndarray,
    labels: List[int],
    wl_rounds_per_step: int = 1,                 # dynamic WL used only for ordering
    # optional partial state for parallel fan-out
    phi_init: Optional[np.ndarray] = None,
    used_target_init: Optional[np.ndarray] = None,
    used_by_class_init: Optional[Dict[int, set]] = None,
) -> List[Dict[int, int]]:
    """
    COMPLETE DFS over permutations that preserve S (feasibility constrained only by base WL classes):
      - Feasibility: candidates y for i come from the *base* class base_classes[base_colors[i]].
      - Ordering: we still compute a *dynamic* WL palette each step (individualize–refine) to
        choose the next variable (MRV) and to order candidate y's, but we do NOT drop feasible y
        just because dynamic WL split them.
      - Early code pruning: once the entire basis is mapped, compute U = C^{-1} and for any
        dependent i→y require U @ G[:, y] == G[:, i] before recursing deeper.
    """
    n = S_mod.shape[0]

    # State
    phi = -np.ones(n, dtype=np.int64) if phi_init is None else phi_init.copy()
    used_target = np.zeros(n, dtype=bool) if used_target_init is None else used_target_init.copy()

    # Track used targets per *base* class (fast MRV and symmetry breaking if desired)
    if used_by_class_init is None:
        used_by_class = {c: set() for c in base_classes}
    else:
        used_by_class = {c: set(v) for c, v in used_by_class_init.items()}
        for c in base_classes:
            used_by_class.setdefault(c, set())

    # Deterministic domain order based on *base* classes (largest first)
    base_order = sorted(base_classes.keys(), key=lambda c: -len(base_classes[c]))
    domain_order = [i for c in base_order for i in base_classes[c]]

    results: List[Dict[int, int]] = []

    # Precompute basis metadata
    basis_indices = np.where(basis_mask)[0]
    k = len(basis_indices)

    # ---- dynamic WL palette (for ordering only)
    def dynamic_palette() -> Tuple[np.ndarray, Dict[int, List[int]]]:
        # Individualize currently mapped domain nodes and their images
        tags = np.full(n, -1, dtype=int)
        mapped_dom = np.where(phi >= 0)[0]
        for t, i in enumerate(mapped_dom):
            tags[i] = 2 * t
            tags[phi[i]] = 2 * t + 1
        # Run a small number of WL rounds; do NOT use these classes for feasibility
        cur_colors = _wl_colors_from_S(S_mod, p, coeffs=coeffs, tags=tags, max_rounds=max(1, wl_rounds_per_step))
        cur_classes = _color_classes(cur_colors)
        return cur_colors, cur_classes

    # ---- choose next variable: MRV measured against *base* candidates
    def select_next_index(cur_colors: np.ndarray) -> int:
        best_i, best_rem = -1, 10**9
        for i in domain_order:
            if phi[i] >= 0:
                continue
            bi = int(base_colors[i])
            # remaining feasible targets in base class not yet used
            rem = len(base_classes[bi]) - len(used_by_class[bi])
            if rem < best_rem:
                best_i, best_rem = i, rem
                if rem <= 1:
                    break
        return best_i

    # ---- leaf acceptance
    def at_leaf(pi: np.ndarray) -> bool:
        if forbid_identity and np.all(pi == np.arange(n, dtype=pi.dtype)):
            return False
        if not _check_symplectic_invariance_mod(S_mod, pi):
            return False
        return _check_code_automorphism(G, basis_order, labels, pi)

    # ---- DFS (cur_colors only for ordering; feasibility via base_* only)
    def dfs(cur_colors: np.ndarray) -> bool:
        if len(results) >= k_limit:
            return True
        if np.all(phi >= 0):
            pi = phi.copy()
            if at_leaf(pi):
                results.append(_perm_index_to_perm_map(labels, pi))
                return len(results) >= k_limit
            return False

        i = select_next_index(cur_colors)
        bi = int(base_colors[i])
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)

        # Order candidates y by dynamic color (heuristic), but iterate over the *base* class
        base_cands = [y for y in base_classes[bi] if not used_target[y]]
        # Optional coefficient hard constraint
        if coeffs is not None:
            base_cands = [y for y in base_cands if coeffs[i] == coeffs[y]]

        # Tie-break by dynamic colors (stable heuristic)
        y_order = sorted(base_cands, key=lambda y: cur_colors[y])

        # Early U setup flag: compute once when all basis mapped
        U: Optional[galois.FieldArray] = None
        have_U = False

        for y in y_order:
            # Incremental S-consistency
            if not _consistent_numba(S_mod, phi, mapped_idx, int(i), int(y)):
                continue

            # Compute U the moment the full basis is mapped (first time only)
            if not have_U:
                mapped_basis = [j for j in basis_indices if phi[j] >= 0]
                # If i is a basis index, include tentative y
                if basis_mask[i]:
                    mapped_basis = mapped_basis + [i]
                if len(mapped_basis) == k:
                    # Build PBcols per tentative assignment
                    PBcols = np.array([phi[j] if j != i else y for j in basis_indices], dtype=int)
                    C = G[:, PBcols]
                    try:
                        U = np.linalg.inv(C)
                        have_U = True
                    except np.linalg.LinAlgError:
                        # This assignment cannot lead to a code automorphism
                        continue

            # If U exists and i is dependent, enforce column equality now
            if have_U and not basis_mask[i]:
                if not np.array_equal(U @ G[:, y], G[:, i]):
                    continue

            # place i -> y
            phi[i] = y
            used_target[y] = True
            used_by_class[bi].add(y)

            # Dynamic palette only for ordering deeper
            next_colors, _ = dynamic_palette()

            # Recurse
            done = dfs(next_colors)
            if done:
                return True

            # Undo
            phi[i] = -1
            used_target[y] = False
            used_by_class[bi].remove(y)

        return False

    # Start with a small dynamic palette for ordering
    cur_colors, _ = dynamic_palette()
    dfs(cur_colors)
    return results


# ---------------- top-level with adaptive two-level fan-out ----------------
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
    max_initial_tasks: int = 256,
    two_level_fanout: bool = True,
    wl_rounds_per_step: int = 1,
) -> List[Dict[int, int]]:
    """
    Return up to k non-trivial automorphisms preserving:
      (A) S: P^T S P = S (mod p), and
      (B) the vector set (independent, dependencies) over GF(p).
    Feasibility is constrained ONLY by the *base* WL partition from S (+coeffs).
    Dynamic WL is used strictly for ordering (MRV / value order), not as a hard constraint.
    """
    # Working labels
    pres_labels = _labels_union(independent, dependencies)
    n = len(pres_labels)

    # Align S to pres_labels and reduce mod p
    if S_labels is not None:
        lab_to_pos = {lab: i for i, lab in enumerate(S_labels)}
        idx = np.array([lab_to_pos[lab] for lab in pres_labels], dtype=int)
        S_aligned = S[np.ix_(idx, idx)]
    else:
        if S.shape != (n, n):
            raise ValueError("S shape does not match the number of labels; supply S_labels.")
        S_aligned = S
    S_mod = np.mod(S_aligned, p)
    if S_mod.dtype != np.int64:
        S_mod = S_mod.astype(np.int64, copy=False)

    # Align coeffs to pres_labels
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

    # --- Base WL partition: ONLY hard feasibility constraint
    base_colors  = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned, tags=None, max_rounds=10)
    base_classes = _color_classes(base_colors)

    # Build G and basis mask once
    G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, pres_labels, p)

    # Serial path
    if n_jobs == 1:
        return _dfs_search(
            S_mod, p,
            base_colors, base_classes,
            coeffs=coeffs_aligned, k_limit=k, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            wl_rounds_per_step=wl_rounds_per_step,
        )

    # Parallel fan-out: prepare a dynamic palette for ordering only
    colors_init = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned, tags=None, max_rounds=1)
    classes_init = _color_classes(colors_init)

    # Root partial state
    phi0 = -np.ones(n, dtype=np.int64)
    used_target0 = np.zeros(n, dtype=bool)
    used_by_class0 = {c: set() for c in base_classes}   # track per *base* class

    # Helper: MRV selection using dynamic palette but counting *base* feasibility
    base_order = sorted(base_classes.keys(), key=lambda c: -len(base_classes[c]))
    domain_order = [i for c in base_order for i in base_classes[c]]
    def select_next_index(colors: np.ndarray) -> int:
        best_i, best_rem = -1, 10**9
        for i in domain_order:
            if phi0[i] >= 0:
                continue
            bi = int(base_colors[i])
            rem = len(base_classes[bi]) - len(used_by_class0[bi])
            if rem < best_rem:
                best_i, best_rem = i, rem
                if rem <= 1:
                    break
        return best_i

    # First variable (ordering only)
    i0 = select_next_index(colors_init)
    bi0 = int(base_colors[i0])

    # First-level candidates from the *base* class (feasibility), with coeffs if any
    cand_y0 = [y for y in base_classes[bi0]
               if (coeffs_aligned is None or coeffs_aligned[i0] == coeffs_aligned[y])]

    # If branching is small, run serial to avoid overhead
    if len(cand_y0) < parallel_min_branch:
        return _dfs_search(
            S_mod, p,
            base_colors, base_classes,
            coeffs=coeffs_aligned, k_limit=k, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            wl_rounds_per_step=wl_rounds_per_step,
        )

    # Build initial tasks (1 or 2 levels)
    tasks: List[Tuple[np.ndarray, np.ndarray, Dict[int, set]]] = []

    if two_level_fanout:
        for y0 in cand_y0:
            # place i0 -> y0
            phi1 = phi0.copy(); phi1[i0] = y0
            used_t1 = used_target0.copy(); used_t1[y0] = True
            used_c1 = {c: set(v) for c, v in used_by_class0.items()}
            used_c1[bi0].add(y0)

            # Dynamic palette after one placement (ordering only)
            tags = np.full(n, -1, dtype=int); tags[i0] = 0; tags[y0] = 1
            colors1 = _wl_colors_from_S(S_mod, p, coeffs=coeffs_aligned, tags=tags, max_rounds=wl_rounds_per_step)

            # Choose second variable (ordering only)
            # We can reuse select_next_index but with the new colors; it only reads base usage.
            # Temporarily set phi0 so MRV skips mapped:
            old_phi0_i0 = phi0[i0]; phi0[i0] = y0
            i1 = select_next_index(colors1)
            phi0[i0] = old_phi0_i0

            if i1 == -1:
                tasks.append((phi1, used_t1, used_c1))
                if len(tasks) >= max_initial_tasks:
                    break
                continue

            bi1 = int(base_colors[i1])
            cands_y1 = [y1 for y1 in base_classes[bi1]
                        if (not used_t1[y1]) and
                           (coeffs_aligned is None or coeffs_aligned[i1] == coeffs_aligned[y1])]

            for y1 in cands_y1:
                phi2 = phi1.copy(); phi2[i1] = y1
                used_t2 = used_t1.copy(); used_t2[y1] = True
                used_c2 = {c: set(v) for c, v in used_c1.items()}
                used_c2[bi1].add(y1)
                tasks.append((phi2, used_t2, used_c2))
                if len(tasks) >= max_initial_tasks:
                    break
            if len(tasks) >= max_initial_tasks:
                break
    else:
        for y0 in cand_y0:
            phi1 = phi0.copy(); phi1[i0] = y0
            used_t1 = used_target0.copy(); used_t1[y0] = True
            used_c1 = {c: set(v) for c, v in used_by_class0.items()}
            used_c1[bi0].add(y0)
            tasks.append((phi1, used_t1, used_c1))
            if len(tasks) >= max_initial_tasks:
                break

    if not tasks:
        return []

    # Distribute a small quota per worker (complete up to k, not “all”)
    per_worker = max(1, (k + len(tasks) - 1) // len(tasks))

    def _worker(state_tuple) -> List[Dict[int, int]]:
        phi_w, used_t_w, used_c_w = state_tuple
        return _dfs_search(
            S_mod, p,
            base_colors, base_classes,
            coeffs=coeffs_aligned, k_limit=per_worker, forbid_identity=forbid_identity,
            G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
            wl_rounds_per_step=wl_rounds_per_step,
            phi_init=phi_w, used_target_init=used_t_w, used_by_class_init=used_c_w,
        )

    results: List[Dict[int, int]] = []
    for chunk in Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_worker)(t) for t in tasks):
        results.extend(chunk)
        if len(results) >= k:
            break
    return results[:k]


# ------------------------------ example usage ------------------------------
if __name__ == "__main__":
    p = 2
    independent = [1, 3, 4]
    dependencies = {
        2: [(1, 1), (3, 1)],
        5: [(1, 1), (4, 1)],
    }
    S = np.array([
        [0,1,0,0,1],
        [1,0,1,0,1],
        [0,1,0,0,0],
        [0,0,0,0,1],
        [1,1,0,1,0],
    ], dtype=np.int64)
    coeffs = np.array([0, 0, 1, 1, 0], dtype=int)  # optional; or None

    perms = find_k_automorphisms_symplectic(
        independent, dependencies,
        S=S, p=p, k=3, S_labels=[1,2,3,4,5],
        forbid_identity=True,
        coeffs=coeffs, coeff_labels=[1,2,3,4,5],
        n_jobs=4, parallel_min_branch=2, max_initial_tasks=128,
        two_level_fanout=True, wl_rounds_per_step=1,
    )
    for i, pm in enumerate(perms, 1):
        print(f"[{i}] {pm}")
