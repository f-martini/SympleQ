from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import galois
from numba import njit
from quaos.core.circuits.target import find_map_to_target_pauli_sum
from phase_correction import pauli_phase_correction, _solve_mod_linear_system
from quaos.core.paulis import PauliSum
from quaos.core.circuits import Gate, Circuit
Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]

# =============================================================================
# Utilities
# =============================================================================


def _labels_union(independent: List[int], dependencies: DepPairs) -> List[int]:
    return sorted(set(independent) | set(dependencies.keys()))


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


def _build_base_partition(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray],
    col_invariants: Optional[np.ndarray],
    max_rounds: int = 10,
    color_mode: str = "wl",      # "wl" | "coeffs_only" | "none"
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Build the base colors & classes:
      - "wl":          WL-1 on S (optionally seeded with coeffs/invariants)
      - "coeffs_only": colors = coefficient IDs only (strict weight-preservation, no WL)
      - "none":        everyone in one color (true brute-force; only weights + S-consistency prune)
    """
    n = S_mod.shape[0]

    if color_mode == "none":
        base_colors = np.zeros(n, dtype=np.int64)

    elif color_mode == "coeffs_only":
        if coeffs is None:
            base_colors = np.zeros(n, dtype=np.int64)
        else:
            # compress coeffs to stable int IDs
            _, inv = np.unique(np.asarray(coeffs), return_inverse=True)
            base_colors = inv.astype(np.int64, copy=False)

    elif color_mode == "wl":
        base_colors = _wl_colors_from_S(
            S_mod, p, coeffs=coeffs, col_invariants=col_invariants, max_rounds=max_rounds
        )
    else:
        raise ValueError("color_mode must be 'wl', 'coeffs_only', or 'none'.")

    return base_colors, _color_classes(base_colors)


def _perm_dict_to_index(labels: List[int], perm: Dict[int,int]) -> np.ndarray:
    """Convert {label->label} dict to index permutation over `labels`."""
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    pi = np.arange(len(labels), dtype=int)
    for src_lab, dst_lab in perm.items():
        i = lab_to_idx[src_lab]; j = lab_to_idx[dst_lab]
        pi[i] = j
    return pi


def _debug_check_coloring_allows_perm(
    *,
    labels: List[int],
    base_colors: np.ndarray,
    coeffs_aligned: Optional[np.ndarray],
    debug_perm_dict: Dict[int,int],
    header: str = "DEBUG"
) -> None:
    """
    Print a clear diagnostic if the current coloring (and weight constraint) forbids `debug_perm_dict`.
    No exception is raised; this is a debug aide.
    """
    pi = _perm_dict_to_index(labels, debug_perm_dict)
    bad_color_idx = np.where(base_colors != base_colors[pi])[0]
    if bad_color_idx.size:
        examples = [(labels[i], labels[pi[i]], int(base_colors[i]), int(base_colors[pi[i]]))
                    for i in bad_color_idx[:8]]
        print(f"{header}: known permutation is BLOCKED by base colors at {bad_color_idx.size} positions.")
        print(f"{header}: examples (src, dst, color[src], color[dst]): {examples}")
        print(f"{header}: try color_mode='coeffs_only' or color_mode='none'")
    else:
        print(f"{header}: base colors ALLOW the known permutation.")

    if coeffs_aligned is not None:
        bad_coeff_idx = np.where(coeffs_aligned != coeffs_aligned[pi])[0]
        if bad_coeff_idx.size:
            ex2 = [(labels[i], labels[pi[i]], coeffs_aligned[i], coeffs_aligned[pi[i]])
                   for i in bad_coeff_idx[:8]]
            print(f"{header}: known permutation VIOLATES weight-preservation at {bad_coeff_idx.size} positions.")
            print(f"{header}: examples (src, dst, w[src], w[dst]): {ex2}")
        else:
            print(f"{header}: weights also allow the known permutation.")


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
# Fallback full DFS (feasible by base partition; complete). Kept simple/serial.
# =============================================================================


def _full_dfs_complete(
    pauli_sum: PauliSum,
    independent_labels: List[int],
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
    p2_bitset: bool,
    dynamic_refine_every: int = 0,
    F_known_debug: Optional[np.ndarray] = None,
    debug_known_permutation: Optional[Dict[int,int]] = None,
) -> List[Circuit]:
    """
    Complete interleaved DFS that maps all labels with feasibility constrained ONLY by base partition.
    Dynamic WL (if enabled) is used only for ordering every `dynamic_refine_every` steps.
    """
    n = S_mod.shape[0]
    results = []

    p = int(pauli_sum.lcm)

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

    def at_leaf(pi: np.ndarray) -> Circuit | None:
        # ---- structural checks ----
        if np.array_equal(pi, np.arange(pauli_sum.n_paulis(), dtype=pi.dtype)):
            return None
        if not np.array_equal(S_mod[np.ix_(pi, pi)], S_mod):
            return None
        if not _check_code_automorphism(G, basis_order, labels, pi):
            return None

        # ---- 1) find F, h0 from the permutation
        H_basis_src = pauli_sum.copy()[independent_labels]
        H_basis_tgt = pauli_sum.copy()[pi[independent_labels]]
        F, h0, _, _ = find_map_to_target_pauli_sum(H_basis_src, H_basis_tgt)
        if F_known_debug is not None and np.array_equal(F.T, F_known_debug):
            print('DEBUG: found known F failure means it cant correct h')

        nq   = pauli_sum.n_qudits()
        twoN = 2 * nq

        # ---- 2) apply that lift ON THE FULL HAMILTONIAN ----
        SG_F = Gate('Symmetry', list(range(nq)), F.T, pauli_sum.dimensions, np.asarray(h0, dtype=int))
        H_full_in = pauli_sum.copy()
        H_full_tg = pauli_sum.copy()[pi]
        H_full_F = SG_F.act(H_full_in.copy())

        # ---- 3) build residual phases Δ on the FULL SET (mod 2p) ----
        Htg = H_full_tg.copy(); Htg.weight_to_phase()
        HF  = H_full_F.copy();  HF.weight_to_phase()
        delta = (Htg.phases - HF.phases) % (2 * int(pauli_sum.lcm))

        # Optional qubit lift tweak: if any residual is odd, try canonical h0 from F
        if p == 2 and np.any(delta % 2 != 0):
            n  = nq
            F2 = F % 2
            A, B = F2[:n, :n], F2[:n, n:]
            C, D = F2[n:, :n], F2[n:, n:]
            hx0 = np.diag((A @ B.T) % 2) % 2
            hz0 = np.diag((C @ D.T) % 2) % 2
            h0_alt = np.concatenate([hx0, hz0]).astype(int)

            SG_F_alt  = Gate('Symmetry', list(range(nq)), F.T, pauli_sum.dimensions, h0_alt)
            H_full_Fa = SG_F_alt.act(H_full_in.copy())
            HFa       = H_full_Fa.copy(); HFa.weight_to_phase()
            delta_alt = (Htg.phases - HFa.phases) % 4
            if (delta_alt % 2).sum() < (delta % 2).sum():
                h0, SG_F, H_full_F, HF, delta = h0_alt, SG_F_alt, H_full_Fa, HFa, delta_alt

        # ---- 4) solve for h on the FULL post-F tableau ----
        T_in_full = H_full_in.tableau().astype(int, copy=False)
        h_lin = solve_phase_vector_h_from_residual(T_in_full, delta, pauli_sum.dimensions, verbose=True)
        if h_lin is None:
            return None

        # ---- 5) compose final symmetry and verify against H (perm-insensitive via standard_form) ----
        two_lcm = 2 * int(pauli_sum.lcm)
        h0_mod = np.asarray(h0, dtype=int) % two_lcm
        h_lin_mod = np.asarray(h_lin, dtype=int) % two_lcm
        h_tot = (h0_mod + h_lin_mod) % two_lcm
        SG    = Gate('Symmetry', list(range(nq)), F.T, pauli_sum.dimensions, h_tot)

        H_out_cf = SG.act(pauli_sum.copy()).standard_form()
        H_ref_cf = pauli_sum.standard_form()

        # ensure both are in phase mode consistently
        H_out_cf.weight_to_phase(); H_ref_cf.weight_to_phase()

        if not np.array_equal(H_out_cf.tableau(), H_ref_cf.tableau()):
            return None
        if not np.all((H_ref_cf.phases - H_out_cf.phases) % two_lcm == 0):
            return None

        return Circuit(pauli_sum.dimensions, [SG])

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
            leaf = at_leaf(pi)
            if leaf is not None:
                results.append(leaf)
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
    pauli_sum: PauliSum,
    independent: List[int],
    dependencies: DepPairs,
    S: np.ndarray,
    p: int,
    k: int = 1,
    S_labels: Optional[List[int]] = None,
    # Strategy
    dynamic_refine_every: int = 0,
    coeffs: Optional[np.ndarray] = None,
    coeff_labels: Optional[List[int]] = None,
    extra_column_invariants: str = "none",
    p2_bitset: str = "auto",
    F_known_debug: Optional[np.ndarray] = None,
    debug_known_permutation: Optional[Dict[int, int]] = None,
    color_mode: str = "wl",   # "wl" | "coeffs_only" | "none"
    max_wl_rounds: int = 10,
) -> List[Circuit]:
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

     # Build G as before
    G, basis_order, basis_mask = _build_generator_matrix(independent, dependencies, pres_labels, p)

    # SAFE: no extra invariants unless you *really* mean it
    col_invariants = None
    if extra_column_invariants != "none":
        G_for_inv, _, _ = _build_generator_matrix(independent, dependencies, pres_labels, p)
        if extra_column_invariants == "hist":
            inv = np.zeros((n, min(p, 16)), dtype=np.int64)
            for j in range(n):
                col = np.array([int(x) for x in G_for_inv[:, j]])
                cnt = np.bincount(col, minlength=p)
                inv[j, :min(p, 16)] = cnt[:min(p, 16)]
            col_invariants = inv
        else:
            raise ValueError("extra_column_invariants must be 'none' or 'hist'.")

    # ---------- NEW: choose the base partition via color_mode ----------
    base_colors, base_classes = _build_base_partition(
        S_mod, p,
        coeffs=coeffs_aligned,               # OK to seed WL with coeff IDs if you want
        col_invariants=col_invariants if color_mode == "wl" else None,
        max_rounds=max_wl_rounds,
        color_mode=color_mode,
    )

        # --- : quick debug feasibility check of known permutation under current coloring ---
    # if debug_known_permutation is not None:
    #     _debug_check_coloring_allows_perm(
    #         labels=pres_labels,
    #         base_colors=base_colors,
    #         coeffs_aligned=coeffs_aligned,   # weight preservation check too
    #         debug_perm_dict=debug_known_permutation,
    #         header="COLOR-CHECK"
    #     )

    # p=2 bitset?
    use_bitset = (p == 2 and (p2_bitset is True or (p2_bitset == "auto" and n <= 256)))

    return _full_dfs_complete(
        pauli_sum,
        independent,
        S_mod,
        coeffs=coeffs_aligned,
        base_colors=base_colors,
        base_classes=base_classes,
        G=G, basis_order=basis_order, basis_mask=basis_mask, labels=pres_labels,
        k_wanted=k,
        p2_bitset=use_bitset,
        dynamic_refine_every=int(dynamic_refine_every),
        F_known_debug=F_known_debug,
        debug_known_permutation=debug_known_permutation,
    )


def _gf_rank(A_int: np.ndarray, p: int) -> int:
    """Rank over GF(p) via simple row-echelon pivot count."""
    GF = galois.GF(p)
    A = GF(A_int % p).copy()
    N, M = A.shape
    r = 0  # next pivot row
    for c in range(M):
        # find pivot in column c, below (and including) row r
        pivot = None
        for i in range(r, N):
            if A[i, c] != GF(0):
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            tmp = A[r, :].copy()
            A[r, :], A[pivot, :] = A[pivot, :], tmp
        inv = GF(1) / A[r, c]
        A[r, :] = A[r, :] * inv
        for i in range(N):
            if i == r:
                continue
            if A[i, c] != GF(0):
                A[i, :] = A[i, :] - A[i, c] * A[r, :]
        r += 1
        if r == N:
            break
    return r

def _gf_solve_one_solution(A_int: np.ndarray, b_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    """Gauss–Jordan elimination over GF(p). Return one solution (free vars=0) or None if inconsistent."""
    GF = galois.GF(p)
    A = GF(A_int % p)
    b = GF(b_int % p)
    N, M = A.shape
    Ab = np.concatenate([A, b.reshape(-1, 1)], axis=1)

    row = 0
    pivots = []
    for col in range(M):
        pivot = None
        for r in range(row, N):
            if Ab[r, col] != GF(0):
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            tmp = Ab[row, :].copy()
            Ab[row, :], Ab[pivot, :] = Ab[pivot, :], tmp
        inv = GF(1) / Ab[row, col]
        Ab[row, :] = Ab[row, :] * inv
        for r in range(N):
            if r == row:
                continue
            if Ab[r, col] != GF(0):
                Ab[r, :] = Ab[r, :] - Ab[r, col] * Ab[row, :]
        pivots.append(col)
        row += 1
        if row == N:
            break

    # inconsistency: [0 ... 0 | nonzero]
    for r in range(N):
        if np.all(Ab[r, :M] == GF(0)) and Ab[r, M] != GF(0):
            return None

    x = GF.Zeros(M)
    for r, c in enumerate(pivots):
        if r < N:
            x[c] = Ab[r, M]
    return np.array([int(v) for v in x], dtype=int)

# ---------- main solver with diagnostics ----------

def solve_phase_vector_h_from_residual(
    tableau_in: np.ndarray,   # (N, 2n) ints; rows of input Paulis
    delta_2L: np.ndarray,     # (N,) ints mod 2L; desired phase corrections
    dimensions: np.ndarray | List[int],
    *,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Solve tableau_in @ h ≡ delta_2L (mod 2L), where h is the additional phase vector.

    Returns h or None if inconsistent.
    """
    A = np.asarray(tableau_in, dtype=int)
    b = np.asarray(delta_2L, dtype=int)
    dims = np.asarray(dimensions, dtype=int)
    if A.ndim != 2:
        raise ValueError("tableau_in must be a 2D array")
    N, M = A.shape
    if M != 2 * dims.size:
        raise ValueError("tableau_in columns must equal 2 * number of qudits")

    L = int(np.lcm.reduce(dims))
    modulus = 2 * L

    sol = _solve_mod_linear_system(A % modulus, b % modulus, modulus)
    if sol is not None:
        if verbose:
            print("[phase] direct solve mod", modulus, "succeeded")
        return sol % modulus

    if verbose:
        print("[phase] direct mod", modulus, "solve failed")

    # Fallbacks when the general solver is unavailable (e.g. missing SymPy)
    if dims.size and np.all(dims == dims[0]):
        p = int(dims[0])
        if p == 2:
            if np.any(b % 2 != 0):
                if verbose:
                    print("[phase] qubit fallback: residual has odd entries, cannot fix")
                return None
            A2 = A % 2
            b2 = ((b // 2) % 2).astype(int)
            h2 = _gf_solve_one_solution(A2, b2, 2)
            if h2 is not None:
                return (2 * h2) % modulus
        else:
            A_p = A % p
            b_p = b % p
            h_p = _gf_solve_one_solution(A_p, b_p, p)
            if h_p is not None:
                return h_p % modulus

    return None

def _row_basis_indices(A_int: np.ndarray, p: int, want_cols: int) -> np.ndarray:
    """Select indices of rows giving up to `want_cols` independent equations over GF(p)."""
    GF = galois.GF(p)
    A = GF(A_int % p).copy()
    N, M = A.shape
    used = np.zeros(N, dtype=bool)
    basis = []
    col = 0
    for _ in range(N):
        if col >= M:
            break
        pick = None
        for r in range(N):
            if used[r]:
                continue
            if A[r, col] != GF(0):
                pick = r; break
        if pick is None:
            col += 1
            continue
        basis.append(pick)
        used[pick] = True
        inv = GF(1) / A[pick, col]
        A[pick, :] = A[pick, :] * inv
        for r in range(N):
            if r == pick or used[r]:
                continue
            if A[r, col] != GF(0):
                A[r, :] = A[r, :] - A[r, col] * A[pick, :]
        col += 1
        if len(basis) >= want_cols:
            break
    return np.array(basis, dtype=int)

def known_permutation(pauli_sum: PauliSum, symmetry: Gate):
    pi = {}
    for i in range(pauli_sum.n_paulis()):
        pauli_out = symmetry.act(pauli_sum[i])
        index = -1
        for j in range(pauli_sum.n_paulis()):
            if pauli_out == pauli_sum[j]:
                index = j
                break
        if index == -1:
            return None
        pi[i] = index

    return pi


if __name__ == "__main__":
    from quaos.core.finite_field_solvers import get_linear_dependencies
    from quaos.models.random_hamiltonian import random_gate_symmetric_hamiltonian

    from quaos.core.circuits import SWAP, SUM, PHASE, Hadamard, Circuit

    failed = 0
    for _ in range(3):
        sym = SWAP(0, 1, 2)
        # symC = Circuit.from_random(2, 10, [2, 2])
        # print(symC)
        # sym = symC.composite_gate()
        H = random_gate_symmetric_hamiltonian(sym, 50, 120, scrambled=False)
        C = Circuit.from_random(H.n_qudits(), 100, H.dimensions).composite_gate()
        # C = Circuit(H.dimensions, [Hadamard(i, 2) for i in range(H.n_qudits())])
        H = C.act(H)
        H.weight_to_phase()
        scrambled_sym = Circuit(H.dimensions, [C.inv(), sym, C]).composite_gate()
        assert H.standard_form() == scrambled_sym.act(H).standard_form(
        ), f"\n{H.standard_form().__str__()}\n{sym.act(H).standard_form().__str__()}"

        known_perm = known_permutation(H, scrambled_sym)
        print(scrambled_sym.phase_vector)
        # print('known_perm = ', known_perm)

        independent, dependencies = get_linear_dependencies(H.tableau(), 2)
        known_F = scrambled_sym.symplectic
        circ = find_k_automorphisms_symplectic(H, independent, dependencies,
                                               S=H.symplectic_product_matrix(), p=2, k=1,
                                               coeffs=H.weights,                  # hard weight constraint
                                               dynamic_refine_every=0,            # keep ordering fixed
                                               F_known_debug=known_F,
                                               debug_known_permutation=known_perm,)

        print(len(circ))
        if len(circ) == 0:
            failed += 1
        else:
            for c in circ:
                print(np.all(c.composite_gate().symplectic == known_F) and np.all(
                    c.composite_gate().phase_vector == scrambled_sym.phase_vector))
                H_s = H.standard_form()
                H_out = c.act(H).standard_form()
                H_s.weight_to_phase()
                H_out.weight_to_phase()
                print(np.all(H_s.tableau() == H_out.tableau()))
                print(np.all(H_s.phases == H_out.phases))
                print(np.all(H_s.weights == H_out.weights))
                # print(H_s.weights)
                # print(H_out.weights)
                # print(c.composite_gate().symplectic)

                if c.act(H).standard_form() != H.standard_form():
                    failed += 1

    print('Failed = ', failed)
