from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from graph_colour_utils import (dynamic_refine_colors_IR, init_candidates_from_base, ac3_filter_domains,
                                selective_wl2_split)
from collections import Counter
from quaos.core.paulis import PauliSum
from quaos.core.circuits.target import find_map_to_target_pauli_sum
from quaos.core.circuits import Gate, Circuit
from phase_correction import pauli_phase_correction
# ------------------------------------------------------------
# Small GF(p) linear algebra helpers (deterministic, exact)
# ------------------------------------------------------------

def gf_rank(A: galois.FieldArray) -> int:
    """Row-reduce over GF(p) and return rank. Deterministic pivoting (left-to-right)."""
    GF = type(A)
    p = GF.characteristic
    M = A.copy()
    m, n = M.shape
    r = 0
    for c in range(n):
        # find pivot at/below row r
        piv = None
        for i in range(r, m):
            if M[i, c] != 0:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        # normalize pivot row
        inv = pow(int(M[r, c]), -1, p)
        M[r, :] = (M[r, :] * inv)
        # eliminate other rows
        for i in range(m):
            if i == r:
                continue
            if M[i, c] != 0:
                fac = int(M[i, c])
                M[i, :] = (M[i, :] - fac * M[r, :])
        r += 1
        if r == m:
            break
    return r


def gf_inverse(C: galois.FieldArray) -> galois.FieldArray:
    """Compute C^{-1} over GF(p) via augmented RREF. Raises LinAlgError if singular."""
    GF = type(C)
    p = GF.characteristic
    k = C.shape[0]
    assert C.shape == (k, k)
    A = C.copy()
    Id = GF.Identity(k)
    # Augment [A | I]
    Aug = np.concatenate([A, Id], axis=1)
    m, n2 = Aug.shape
    n = k  # left width

    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if Aug[i, c] != 0:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            Aug[[r, piv]] = Aug[[piv, r]]
        inv = pow(int(Aug[r, c]), -1, p)
        Aug[r, :] = Aug[r, :] * inv
        for i in range(m):
            if i == r:
                continue
            if Aug[i, c] != 0:
                fac = int(Aug[i, c])
                Aug[i, :] = Aug[i, :] - fac * Aug[r, :]
        r += 1
    # Check left block is identity
    if not np.array_equal(Aug[:, :k], GF.Identity(k)):
        raise np.linalg.LinAlgError("Matrix is singular over GF(p)")
    return Aug[:, k:]


# ------------------------------------------------------------
# S-consistency check (simple, readable version)
# ------------------------------------------------------------

def _is_identity_perm(pi: np.ndarray) -> bool:
    return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))


def s_consistent(S_mod: np.ndarray, phi: np.ndarray, mapped_idx: np.ndarray, i: int, y: int) -> bool:
    """
    Local symplectic consistency for a new placement i -> y:
      For all already mapped j (with image phi[j]), we require
        S[i,j] == S[y,phi[j]]  and  S[j,i] == S[phi[j],y].
    """
    row_i = S_mod[i]
    for j in mapped_idx:
        yj = phi[j]
        if S_mod[y, yj] != row_i[j]:  # forward
            return False
        if S_mod[yj, y] != S_mod[j, i]:  # backward (symmetry for directed colors)
            return False
    return True


def _check_symplectic_invariance_mod(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)


def _check_code_automorphism(
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],
    pi: np.ndarray,
    p: int = 2
) -> bool:
    """
    Linear-code test over GF(p): ∃ U with U G P = G ?
    Uses a *field* solve (no numpy int RHS) to avoid mixed-type pitfalls.
    """
    GF = galois.GF(p)

    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    B_cols = np.array([lab_to_idx[b] for b in basis_order], dtype=int)
    PBcols = pi[B_cols]
    C = G[:, PBcols]                         # k x k over GF(p)

    try:
        U = np.linalg.solve(C, GF.Identity(C.shape[0]))  # solve in GF(p)
    except np.linalg.LinAlgError:
        return False

    Gp = G[:, pi]
    return np.array_equal(U @ Gp, G)



# ------------------------------------------------------------
# Signature helper for columns over GF(p)
# ------------------------------------------------------------

def col_signature(col: galois.FieldArray) -> Tuple[int, ...]:
    """Hashable signature of a GF(p) column (used to bucket candidates)."""
    return tuple(int(x) for x in col)


# ------------------------------------------------------------
# basis-first search
# ------------------------------------------------------------

def basis_first_search(
    S_mod: np.ndarray,                          # (n, n) symplectic product matrix mod p
    G: galois.FieldArray,                       # (k, n) generator matrix [I | D] over GF(p)
    basis_order: List[int],                     # labels (indices into columns of G/S_mod) for k independent elements
    base_colors: np.ndarray,                    # 1-WL color per row (hard feasibility classes)
    base_classes: Dict[int, List[int]],         # color -> sorted list of indices
    labels: List[int],                          # presentation labels, same order as columns of G / rows of S_mod
    *,
    coeffs: Optional[np.ndarray] = None,        # optional: preserve weights (exact equality)
    enforce_base_on_dependents: bool = False,   # if True, also constrain dependents to WL classes
    k_wanted: int = 1,                          # number of permutations to return
    p: int = 2
) -> List[Dict[int, int]]:
    """
    Improved 'basis-first' solver:
      1) Map the k basis elements using WL+S-consistency, with *incremental independence checks*.
      2) Once all k placed, compute U = C^{-1} (exact GF(p)).
      3) Use UG signatures to bucket dependents; DFS them with WL+S-consistency (cheap).
      4) Collect up to k_wanted permutations.

    Returns: list of {label_src -> label_dst} dicts.
    """
    n = S_mod.shape[0]

    # Map from label -> position index
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    basis_idx = [lab_to_idx[b] for b in sorted(basis_order)]

    # Order basis variables: smallest WL class first (MRV)
    basis_idx.sort(key=lambda i: len(base_classes[int(base_colors[i])]))

    # Search state
    phi = -np.ones(n, dtype=int)   # image index per vertex (row)
    used = np.zeros(n, dtype=bool)  # which targets are taken
    results: List[Dict[int, int]] = []

    # -------- Incremental independence check helpers --------
    # We keep the currently selected columns C_t (k x t) and verify rank(C_t) == t.
    def independent_with_new(y_list: List[int]) -> bool:
        if not y_list:
            return True
        C = G[:, np.array(y_list, dtype=int)]
        return (gf_rank(C) == C.shape[1])

    # -------- Dependents finisher (after U known) --------
    def finish_dependents_with_U(U: galois.FieldArray) -> None:
        """
        Use UG to bucket candidate images for each dependent; DFS them with S-consistency.
        """
        nonlocal results
        if len(results) >= k_wanted:
            return

        UG = U @ G                        # k x n (FieldArray)
        need_sig = {i: col_signature(G[:, i]) for i in range(n)}

        # Pre-bucket available targets by signature
        available = ~used
        bucket: Dict[Tuple[int, ...], List[int]] = {}
        for y in range(n):
            if not available[y]:
                continue
            s = col_signature(UG[:, y])
            bucket.setdefault(s, []).append(y)

        # Dependent indices (those not in the mapped basis set)
        basis_set = set(basis_idx)
        dependents = [i for i in range(n) if i not in basis_set]

        # Build candidate lists per dependent
        dep_candidates: Dict[int, List[int]] = {}
        for i in dependents:
            cand = bucket.get(need_sig[i], [])
            if not cand:
                return  # impossible under this U
            # Optional: keep dependents within their WL base class
            if enforce_base_on_dependents:
                bi = int(base_colors[i])
                cand = [y for y in cand if y in base_classes[bi]]
                if not cand:
                    return
            # Optional: match coefficients (if you enforce weight-preserving autos)
            if coeffs is not None:
                cand = [y for y in cand if coeffs[y] == coeffs[i]]
                if not cand:
                    return
            dep_candidates[i] = cand

        # MRV on dependents
        dep_order = sorted(dependents, key=lambda i: len(dep_candidates[i]))

        # DFS dependents
        def dfs_dep(t: int) -> None:
            nonlocal results
            if len(results) >= k_wanted:
                return
            if t == len(dep_order):
                pi = phi.copy()
                if _is_identity_perm(pi):
                    return
                # Global checks (S and code) are expected upstream; keep S check here for safety
                if not np.array_equal(S_mod[np.ix_(pi, pi)], S_mod):
                    return
                # Success: emit mapping as {label -> label}
                results.append({labels[j]: labels[int(pi[j])] for j in range(n)})
                return

            i = dep_order[t]
            mapped_idx = np.where(phi >= 0)[0]

            for y in dep_candidates[i]:
                if used[y]:
                    continue
                if not s_consistent(S_mod, phi, mapped_idx, i, y):
                    continue
                # place
                phi[i] = y
                used[y] = True
                dfs_dep(t + 1)
                if len(results) >= k_wanted:
                    return
                # undo
                phi[i] = -1
                used[y] = False

        dfs_dep(0)

    # -------- DFS basis with incremental independence --------
    selected_targets: List[int] = []  # images for the basis columns, in the DFS order

    def dfs_basis(t: int) -> None:
        nonlocal results, selected_targets
        if len(results) >= k_wanted:
            return

        if t == len(basis_idx):
            # All basis images chosen; build C and invert to get U
            PBcols = np.array(selected_targets, dtype=int)       # images of the basis columns
            C = G[:, PBcols]                                     # k x k over GF(p)
            try:
                U = gf_inverse(C)                                # exact over GF(p)
            except np.linalg.LinAlgError:
                return
            finish_dependents_with_U(U)
            return

        i = basis_idx[t]
        bi = int(base_colors[i])
        mapped_idx = np.where(phi >= 0)[0]

        # Candidates = same WL class; filter by coeffs if requested
        candidates = [y for y in base_classes[bi] if not used[y]]
        if coeffs is not None:
            candidates = [y for y in candidates if coeffs[y] == coeffs[i]]

        # Try each candidate with S-consistency + incremental independence
        for y in candidates:
            if not s_consistent(S_mod, phi, mapped_idx, i, y):
                continue

            # Tentatively add y to the selected basis images and check independence
            trial = selected_targets + [y]
            if not independent_with_new(trial):
                continue

            # place
            phi[i] = y
            used[y] = True
            selected_targets.append(y)

            dfs_basis(t + 1)
            if len(results) >= k_wanted:
                return

            # undo
            selected_targets.pop()
            used[y] = False
            phi[i] = -1

    dfs_basis(0)
    return results[:k_wanted]


# def smallest_wl_block_first_search(
#     S_mod: np.ndarray,
#     p: int,
#     *,
#     # Feasibility partition (from WL-1 on S, optionally with invariants)
#     base_colors: np.ndarray,
#     base_classes: Dict[int, List[int]],

#     # Matroid/code data for the final leaf checks
#     G: galois.FieldArray,
#     basis_order: List[int],
#     labels: List[int],

#     # Optional invariants for IR seeding (e.g., parallel IDs, circuit counts)
#     col_invariants: Optional[np.ndarray] = None,

#     # Weights: only include if you truly want weight-preserving automorphisms
#     coeffs: Optional[np.ndarray] = None,

#     # How many automorphisms to collect
#     k_wanted: int = 1,

#     # === A) IR dynamic refinement (ordering/pruning) ===
#     use_ir_ordering: bool = True,
#     ir_rounds: int = 2,              # 1–3 is plenty
#     ir_every_steps: int = 0,         # 0 => only when a big domain exists; else recompute every X steps

#     # === B) Selective WL-2 pre-split (feasibility refinement) ===
#     use_selective_wl2: bool = True,
#     wl2_size_threshold: int = 64,
#     wl2_rounds: int = 1,

#     # === C) AC-3 domain filtering (global consistency) ===
#     use_ac3: bool = True,
#     ac3_symmetric: bool = True,

#     # Local forward checking (kept simple)
#     use_forward_check: bool = True,
# ) -> List[Dict[int, int]]:
#     """
#     Interleaved WL-guided DFS (contrast with `basis_first_search` which fixes a basis first).

#     Strategy:
#       • Optionally (B) refine feasibility by splitting the fattest WL class via selective WL-2.
#       • Build candidate domains from WL classes (and coeffs if requested).
#       • Run (C) AC-3 to reach global arc-consistency before search.
#       • DFS: at each step pick the variable in the smallest WL block (MRV), try candidates
#         ordered by (A) IR dynamic WL seeded by the current partial mapping.

#     Returns up to `k_wanted` automorphisms as {label_src -> label_dst} dicts.
#     """
#     n = S_mod.shape[0]
#     assert S_mod.shape == (n, n)

#     # ---------- B) optional pre-split of fattest WL class ----------
#     if use_selective_wl2:
#         split = selective_wl2_split(S_mod, p, base_colors,
#                                     size_threshold=wl2_size_threshold,
#                                     rounds=wl2_rounds)
#         if split is not None:
#             base_colors, base_classes = split  # refine feasibility safely

#     # ---------- Build initial domains from WL classes (+ coeffs if any) ----------
#     D = init_candidates_from_base(n, base_classes, base_colors, coeffs)

#     # ---------- C) global arc-consistency before search ----------
#     if use_ac3:
#         ok = ac3_filter_domains(S_mod, p, D, symmetric=ac3_symmetric)
#         if not ok:
#             return []

#     # ---------- DFS state ----------
#     phi = -np.ones(n, dtype=int)      # image per source vertex; -1 => unmapped
#     used = np.zeros(n, dtype=bool)     # used targets
#     results: List[Dict[int, int]] = []
#     steps = 0

#     # MRV variable selection (smallest current domain; tie-break by base color)
#     def select_next_var() -> int:
#         best_i, best_sz, best_col = -1, 10**9, 10**9
#         for i in range(n):
#             if phi[i] >= 0:
#                 continue
#             dom = [y for y in D[i] if not used[y]]
#             sz = len(dom)
#             if sz < best_sz or (sz == best_sz and base_colors[i] < best_col):
#                 best_i, best_sz, best_col = i, sz, base_colors[i]
#         return best_i

#     # A) dynamic IR-based WL colours for smarter ordering (and optional pruning if desired)
#     def maybe_ir_colors() -> Optional[np.ndarray]:
#         nonlocal steps
#         if not use_ir_ordering:
#             return None
#         if ir_every_steps > 0 and (steps % ir_every_steps != 0):
#             return None
#         if ir_every_steps == 0:
#             # only recompute when a large domain exists (heuristic)
#             max_dom = max((len([y for y in D[i] if not used[y]])
#                            for i in range(n) if phi[i] < 0), default=0)
#             if max_dom <= 16:
#                 return None
#         return dynamic_refine_colors_IR(
#             S_mod, p, base_colors, phi, coeffs, col_invariants, rounds=ir_rounds
#         )

#     # Local forward checking: shrink neighbors’ domains after placing i -> y
#     def forward_check(i: int, y: int, backup: Dict[int, List[int]]) -> bool:
#         if not use_forward_check:
#             return True
#         for j in range(n):
#             if j == i or phi[j] >= 0:
#                 continue
#             old = D[j]
#             # Keep z that maintain local S-consistency with the new placement
#             new = [z for z in old if (not used[z]) and
#                    S_mod[i, j] == S_mod[y, z] and
#                    S_mod[j, i] == S_mod[z, y]]
#             if len(new) != len(old):
#                 backup[j] = old
#                 D[j] = new
#                 if not new:
#                     return False
#         return True

#     def dfs() -> bool:
#         nonlocal steps
#         if len(results) >= k_wanted:
#             return True

#         # Done?
#         if np.all(phi >= 0):
#             pi = phi.copy()
#             if _is_identity_perm(pi):
#                 return False  # reject identity
#             # Global checks (same as your leaf stage)
#             if not _check_symplectic_invariance_mod(S_mod, pi):
#                 return False
#             if not _check_code_automorphism(G, basis_order, labels, pi, p):
#                 return False
#             results.append({labels[j]: labels[int(pi[j])] for j in range(n)})
#             return len(results) >= k_wanted

#         steps += 1

#         i = select_next_var()
#         if i < 0:
#             return False

#         # Candidate list = domain minus used
#         candidates = [y for y in D[i] if not used[y]]
#         if not candidates:
#             return False

#         # A) order by IR colours if available (else base colors)
#         ir_cols = maybe_ir_colors()
#         if ir_cols is not None:
#             candidates.sort(key=lambda y: (ir_cols[y], y))
#         else:
#             candidates.sort(key=lambda y: (base_colors[y], y))

#         mapped_idx = np.where(phi >= 0)[0]

#         for y in candidates:
#             # Local S-consistency against already mapped vertices
#             if not s_consistent(S_mod, phi, mapped_idx, i, y):
#                 continue

#             # Place
#             phi[i] = y
#             used[y] = True

#             # Domain edits snapshot
#             backup: Dict[int, List[int]] = {}
#             if not forward_check(i, y, backup):
#                 # undo
#                 for j, old in backup.items():
#                     D[j] = old
#                 used[y] = False
#                 phi[i] = -1
#                 continue

#             # Light AC-3 sweep if domains changed
#             if use_ac3 and backup:
#                 if not ac3_filter_domains(S_mod, p, D, symmetric=ac3_symmetric):
#                     for j, old in backup.items():
#                         D[j] = old
#                     used[y] = False
#                     phi[i] = -1
#                     continue

#             # Recurse
#             done = dfs()
#             if done and len(results) >= k_wanted:
#                 return True

#             # Undo placement + domain edits
#             for j, old in backup.items():
#                 D[j] = old
#             used[y] = False
#             phi[i] = -1

#         return False

#     dfs()
#     return results[:k_wanted]


def pick_strategy(
    *,
    k: int,                     # number of independent elements (rank)
    n: int,                     # number of rows / terms
    basis_idx: List[int],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],
    coeffs: Optional[np.ndarray] = None,
    # heuristics
    max_basis_product: int = 50000,
    max_largest_basis_cell: int = 64,
) -> str:
    """
    Return 'basis_first' or 'wl_block_first' based on cheap diagnostics.

    Heuristics:
      • avg_basis_domain = average size of WL class for basis vars (filtered by coeffs if provided)
      • basis_product ≈ k * avg_basis_domain  (rough proxy for basis search width)
      • largest_basis_cell = max WL class size among basis vars
    """
    # avg basis domain
    sizes = []
    for i in basis_idx:
        cls = base_classes[int(base_colors[i])]
        if coeffs is None:
            sizes.append(len(cls))
        else:
            sizes.append(sum(1 for y in cls if coeffs[y] == coeffs[i]))
    avg_basis_domain = (sum(sizes) / len(sizes)) if sizes else 0
    basis_product = int(k * max(1, avg_basis_domain))
    largest_basis_cell = max(sizes) if sizes else 0

    # Decision
    if k <= n // 2 and basis_product <= max_basis_product and largest_basis_cell <= max_largest_basis_cell:
        return 'basis_first'
    return 'wl_block_first'


# def smallest_wl_block_first_search(
#     S_mod: np.ndarray,
#     p: int,
#     *,
#     base_colors: np.ndarray,
#     base_classes: Dict[int, List[int]],
#     G: galois.FieldArray,
#     basis_order: List[int],
#     labels: List[int],
#     coeffs: Optional[np.ndarray] = None,
#     k_wanted: int = 1,
# ) -> List[Dict[int, int]]:
#     """
#     Compatibility version of the original _full_dfs_complete:
#       - Feasibility constrained ONLY by base WL classes (+ optional coeff equality)
#       - No AC-3, no forward-check, no IR, no WL-2.
#       - Variable selection = MRV using base feasibility.
#       - Candidate ordering = base colors.
#       - Global checks at leaf: nontrivial, S-invariance, code automorphism.
#     """
#     n = S_mod.shape[0]
#     results: List[Dict[int, int]] = []

#     # State
#     phi  = -np.ones(n, dtype=np.int64)
#     used = np.zeros(n, dtype=bool)

#     def select_next() -> int:
#         # MRV measured against base feasibility (unmapped targets in same WL class and same coeff if any)
#         best_i, best_rem = -1, 10**9
#         for i in range(n):
#             if phi[i] >= 0:
#                 continue
#             bi = int(base_colors[i])
#             rem = 0
#             if coeffs is None:
#                 rem = sum((not used[y]) for y in base_classes[bi])
#             else:
#                 rem = sum((not used[y] and coeffs[i] == coeffs[y]) for y in base_classes[bi])
#             if rem < best_rem:
#                 best_i, best_rem = i, rem
#                 if rem <= 1:
#                     break
#         return best_i

#     def at_leaf(pi: np.ndarray) -> bool:
#         # Non-trivial
#         if np.array_equal(pi, np.arange(n, dtype=pi.dtype)):
#             return False
#         # S invariance
#         if not np.array_equal(S_mod[np.ix_(pi, pi)], S_mod):
#             return False
#         # Code/matroid invariance
#         if not _check_code_automorphism(G, basis_order, labels, pi, p):
#             return False
#         return True

#     def s_consistent_local(i: int, y: int) -> bool:
#         mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
#         row_i = S_mod[i]
#         for j in mapped_idx:
#             yj = phi[j]
#             if S_mod[y, yj] != row_i[j]:            # forward
#                 return False
#             if S_mod[yj, y] != S_mod[j, i]:         # backward
#                 return False
#         return True

#     def dfs() -> bool:
#         if len(results) >= k_wanted:
#             return True
#         if np.all(phi >= 0):
#             pi = phi.copy()
#             if at_leaf(pi):
#                 results.append({labels[j]: labels[int(pi[j])] for j in range(n)})
#                 return len(results) >= k_wanted
#             return False

#         i = select_next()
#         bi = int(base_colors[i])

#         # Candidates: within base WL class (and coeff equal if requested), not used
#         if coeffs is None:
#             cand = [y for y in base_classes[bi] if not used[y]]
#         else:
#             cand = [y for y in base_classes[bi] if (not used[y] and coeffs[i] == coeffs[y])]

#         # Order by current base color only (matches original)
#         cand.sort(key=lambda y: base_colors[y])

#         for y in cand:
#             if not s_consistent_local(i, y):
#                 continue
#             phi[i]  = y
#             used[y] = True
#             if dfs():
#                 if len(results) >= k_wanted:
#                     return True
#             phi[i]  = -1
#             used[y] = False
#         return False

#     dfs()
#     return results[:k_wanted]


def basis_first_search(
    S_mod: np.ndarray,
    p: int,
    *,
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],
    coeffs: Optional[np.ndarray] = None,
    enforce_base_on_dependents: bool = False,
    k_wanted: int = 1,
) -> List[Dict[int, int]]:
    GF = galois.GF(p)
    n = S_mod.shape[0]
    results: List[Dict[int, int]] = []

    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    basis_idx  = [lab_to_idx[b] for b in sorted(basis_order)]
    basis_idx.sort(key=lambda i: len(base_classes[int(base_colors[i])]))

    phi  = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)

    def s_consistent_local(i: int, y: int) -> bool:
        mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
        row_i = S_mod[i]
        for j in mapped_idx:
            yj = phi[j]
            if S_mod[y, yj] != row_i[j]:
                return False
            if S_mod[yj, y] != S_mod[j, i]:
                return False
        return True

    def col_sig_int(col: galois.FieldArray) -> Tuple[int, ...]:
        return tuple(int(v) for v in col)

    def finish_dependents(U: galois.FieldArray) -> None:
        if len(results) >= k_wanted:
            return
        UG = U @ G

        # ---- ORIGINAL MULTISET PRECHECK (strict) ----
        need = Counter(col_sig_int(G[:, j]) for j in range(n) if j not in basis_idx)
        have = Counter(col_sig_int(UG[:, j]) for j in range(n) if not used[j])
        if need != have:
            return

        # Pre-bucket by UG signature (available targets only)
        bucket: Dict[Tuple[int, ...], List[int]] = {}
        for y in range(n):
            if not used[y]:
                s = col_sig_int(UG[:, y])
                bucket.setdefault(s, []).append(y)

        # Dependents = non-basis indices
        dependents = [i for i in range(n) if i not in set(basis_idx)]

        # Candidate list per dependent, identical filters
        dep_cands: Dict[int, List[int]] = {}
        for i in dependents:
            s = col_sig_int(G[:, i])
            lst = bucket.get(s, [])
            if not lst:
                return
            if enforce_base_on_dependents:
                bi = int(base_colors[i])
                lst = [y for y in lst if y in base_classes[bi]]
                if not lst:
                    return
            if coeffs is not None:
                lst = [y for y in lst if coeffs[y] == coeffs[i]]
                if not lst:
                    return
            dep_cands[i] = lst

        dep_order = sorted(dependents, key=lambda i: len(dep_cands[i]))

        def dfs_dep(t: int) -> None:
            if len(results) >= k_wanted:
                return
            if t == len(dep_order):
                pi = phi.copy()
                if np.array_equal(pi, np.arange(n, dtype=pi.dtype)):
                    return
                if not np.array_equal(S_mod[np.ix_(pi, pi)], S_mod):
                    return
                if not _check_code_automorphism(G, basis_order, labels, pi, p):
                    return
                results.append({labels[j]: labels[int(pi[j])] for j in range(n)})
                return

            i = dep_order[t]
            mapped_idx = np.where(phi >= 0)[0].astype(np.int64)
            for y in dep_cands[i]:
                if used[y]:
                    continue
                if not s_consistent_local(i, y):
                    continue
                phi[i]  = y
                used[y] = True
                dfs_dep(t + 1)
                if len(results) >= k_wanted:
                    return
                phi[i]  = -1
                used[y] = False

        dfs_dep(0)

    def dfs_basis(t: int) -> None:
        if len(results) >= k_wanted:
            return
        if t == len(basis_idx):
            PBcols = np.array([phi[i] for i in basis_idx], dtype=int)
            C = G[:, PBcols]
            try:
                U = gf_inverse(C)  # exact GF(p) inverse
            except np.linalg.LinAlgError:
                return
            finish_dependents(U)
            return

        i = basis_idx[t]
        bi = int(base_colors[i])
        if coeffs is None:
            cand = [y for y in base_classes[bi] if not used[y]]
        else:
            cand = [y for y in base_classes[bi] if (not used[y] and coeffs[i] == coeffs[y])]

        for y in cand:
            if not s_consistent_local(i, y):
                continue
            phi[i]  = y
            used[y] = True
            dfs_basis(t + 1)
            if len(results) >= k_wanted:
                return
            phi[i] = -1
            used[y] = False

    dfs_basis(0)
    return results[:k_wanted]


def smallest_wl_block_first_search(
    pauli_sum: PauliSum,
    independent_labels: List[int],
    dependent_labels: List[int],
    S_mod: np.ndarray,
    p: int,
    *,
    # Feasibility partition (WL-1 on S; whatever you already computed)
    base_colors: np.ndarray,
    base_classes: Dict[int, List[int]],

    # Matroid/code data for final checks
    G: galois.FieldArray,
    basis_order: List[int],
    labels: List[int],

    # Optional invariants for IR seeding (e.g., parallel IDs, circuit counts)
    col_invariants: Optional[np.ndarray] = None,

    # Hard constraint: equal-weight mapping (pass discrete IDs if needed)
    coeffs: Optional[np.ndarray] = None,

    # Result count
    k_wanted: int = 1,

    # ---------- NEW FEATURES (all OFF by default for safety) ----------
    # A) Partial-mapping-aware WL for ordering only (no feasibility change)
    enable_ir_ordering: bool = False,
    ir_rounds: int = 2,
    ir_every_steps: int = 0,    # 0 => recompute only when a large domain exists

    # B) Selective WL-2 pre-split of largest WL cell (safe feasibility refinement)
    enable_selective_wl2: bool = False,
    wl2_size_threshold: int = 64,
    wl2_rounds: int = 1,

    # C) Global arc-consistency on candidate sets (feasibility refinement)
    enable_ac3: bool = False,
    ac3_symmetric: bool = True,

    # Local forward checking (feasibility refinement)
    enable_forward_check: bool = False,
) -> List[Dict[int, int]]:
    """
    Interleaved WL-guided DFS that preserves the behavior of the original solver
    when all feature flags are left at their defaults.

    Enable features gradually:
      - enable_ir_ordering=True     (ordering only; never prunes)
      - enable_selective_wl2=True   (safe WL split before search)
      - enable_forward_check=True   (local pruning)
      - enable_ac3=True             (global pruning)
    """
    n = S_mod.shape[0]
    assert S_mod.shape == (n, n)

    # ---------- (B) Selective WL-2 pre-split (optional, safe) ----------
    if enable_selective_wl2:
        split = selective_wl2_split(S_mod, p, base_colors,
                                    size_threshold=wl2_size_threshold,
                                    rounds=wl2_rounds)
        if split is not None:
            base_colors, base_classes = split  # feasibility refined safely

    # ---------- Candidate domains from WL classes (+ coeffs if hard constraint) ----------
    D = init_candidates_from_base(n, base_classes, base_colors, coeffs)

    # ---------- (C) Global AC-3 before search (optional) ----------
    if enable_ac3:
        if not ac3_filter_domains(S_mod, p, D, symmetric=ac3_symmetric):
            return []

    # ---------- DFS state ----------
    phi = -np.ones(n, dtype=int)   # image per source vertex; -1 => unmapped
    used = np.zeros(n, dtype=bool)
    results: List[Dict[int, int]] = []
    steps = 0

    def _is_identity_perm(pi: np.ndarray) -> bool:
        return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))

    def s_consistent_local(i: int, y: int) -> bool:
        mapped_idx = np.where(phi >= 0)[0]
        row_i = S_mod[i]
        for j in mapped_idx:
            yj = phi[j]
            if S_mod[y, yj] != row_i[j]:
                return False
            if S_mod[yj, y] != S_mod[j, i]:
                return False
        return True

    # MRV selection: smallest current domain; tie-break by base color
    def select_next_var() -> int:
        best_i, best_sz, best_col = -1, 10**9, 10**9
        for i in range(n):
            if phi[i] >= 0:
                continue
            dom = [y for y in D[i] if not used[y]]
            sz = len(dom)
            if sz < best_sz or (sz == best_sz and base_colors[i] < best_col):
                best_i, best_sz, best_col = i, sz, base_colors[i]
        return best_i

    # (A) IR-based colors for ordering only (never prunes)
    def maybe_ir_colors() -> Optional[np.ndarray]:
        nonlocal steps
        if not enable_ir_ordering:
            return None
        if ir_every_steps > 0 and (steps % ir_every_steps != 0):
            return None
        if ir_every_steps == 0:
            # Only recompute when a large domain exists (speed heuristic)
            max_dom = max((len([y for y in D[i] if not used[y]])
                          for i in range(n) if phi[i] < 0), default=0)
            if max_dom <= 16:
                return None
        return dynamic_refine_colors_IR(S_mod, p, base_colors, phi, coeffs, col_invariants, rounds=ir_rounds)

    # Local forward checking (optional)
    def forward_check(i: int, y: int, backup: Dict[int, List[int]]) -> bool:
        if not enable_forward_check:
            return True
        for j in range(n):
            if j == i or phi[j] >= 0:
                continue
            old = D[j]
            # Keep z that maintain local S-consistency with i->y
            new = [z for z in old if (not used[z]) and
                   S_mod[i, j] == S_mod[y, z] and
                   S_mod[j, i] == S_mod[z, y]]
            if len(new) != len(old):
                backup[j] = old
                D[j] = new
                if not new:
                    return False
        return True

    def leaf_checks(pi: np.ndarray) -> bool:
        # cheap checks first to catch easy misses
        if _is_identity_perm(pi):
            return False
        if not np.array_equal(S_mod[np.ix_(pi, pi)], S_mod):
            return False

        H_i = pauli_sum.copy()[independent_labels]
        H_t = pauli_sum.copy()[pi[independent_labels]]

        # The part below can for sure be optimized...

        F, h, _, _ = find_map_to_target_pauli_sum(H_i, H_t)
        SG = Gate('Symmetry', [i for i in range(pauli_sum.n_qudits())], F.T, 2, h)
        if np.all(pauli_sum.standard_form().tableau() != SG.act(pauli_sum).standard_form().tableau()):
            print("VERY BAD FAIL")
            return False
        H_out = SG.act(pauli_sum)
        H_out.weight_to_phase()
        copy2 = pauli_sum.copy()
        copy2.weight_to_phase()
        phases_out = H_out.phases
        phases_tgt = copy2[pi].phases
        if not np.all(copy2.weights == H_out.weights):
            print('Weights changed - bad')
            return False
        delta_phases = (phases_tgt - phases_out)
        pauli = pauli_phase_correction(H_out.tableau(), delta_phases, p)
        if pauli is None:
            pauli = pauli_phase_correction(H_out.tableau(), - delta_phases, p)
            if pauli is None:
                return False
        C = Circuit(pauli_sum.dimensions, [SG, pauli])
        final = C.act(pauli_sum.copy())
        if final.standard_form() != pauli_sum.standard_form():
            print('Failed after thinking phases were fine - bad')
            return False
        if not _check_code_automorphism(G, basis_order, labels, pi, p):
            print('HOW?')
            return False
        return True

    def dfs() -> bool:
        nonlocal steps
        if len(results) >= k_wanted:
            return True

        if np.all(phi >= 0):
            pi = phi.copy()
            if not leaf_checks(pi):
                return False
            results.append({labels[j]: labels[int(pi[j])] for j in range(n)})
            return len(results) >= k_wanted

        steps += 1
        i = select_next_var()
        if i < 0:
            return False

        # Candidates = current domain minus used
        candidates = [y for y in D[i] if not used[y]]
        if not candidates:
            return False

        # Order candidates: IR colors if available, else base colors; prefer moves (y != i)
        ir_cols = maybe_ir_colors()
        if ir_cols is not None:
            candidates.sort(key=lambda y: (y == i, ir_cols[y], y))
        else:
            candidates.sort(key=lambda y: (y == i, base_colors[y], y))

        for y in candidates:
            if not s_consistent_local(i, y):
                continue
            # place
            phi[i] = y
            used[y] = True

            # domain edits snapshot
            backup: Dict[int, List[int]] = {}
            if not forward_check(i, y, backup):
                for j, old in backup.items():
                    D[j] = old
                used[y] = False
                phi[i] = -1
                continue

            # (C) small AC-3 sweep only if domains changed
            if enable_ac3 and backup:
                if not ac3_filter_domains(S_mod, p, D, symmetric=ac3_symmetric):
                    for j, old in backup.items():
                        D[j] = old
                    used[y] = False
                    phi[i] = -1
                    continue

            # Recurse
            done = dfs()
            if done and len(results) >= k_wanted:
                return True

            # Undo placement + domain edits
            for j, old in backup.items():
                D[j] = old
            used[y] = False
            phi[i] = -1

        return False

    dfs()
    return results[:k_wanted]
