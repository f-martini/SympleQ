import numpy as np
from typing import Tuple, List
from .modular_helpers import (mod_p, rank_mod, _solve_linear, nullspace_mod, rref_mod, omega_matrix, inv_mod_mat,
                              inv_mod_scalar, matmul_mod, is_symplectic)
from .minimal_block_size import rcf_prepass
# =========================
# GF(p) linear algebra utils
# =========================


def independent_columns(B: np.ndarray, p: int) -> np.ndarray:
    """Return a column-subset of B with independent columns over GF(p) (single RREF pass)."""
    if B.size == 0:
        return B
    _, piv = rref_mod(mod_p(B, p), p)
    piv = [pc for pc in piv if pc < B.shape[1]]
    return B[:, piv] if piv else np.zeros((B.shape[0], 0), dtype=np.int64)


# =========================
# Mode graph helpers
# =========================

def _mode_graph_from_S(S: np.ndarray, p: int) -> List[List[int]]:
    """
    Build adjacency for 'modes' (pairs). S is [U_all | V_all].
    Connect i--j if any entry in the 4x4 cross-block between modes i and j is nonzero mod p.
    """
    n2 = S.shape[0]
    assert n2 % 2 == 0
    k = n2 // 2  # number of modes
    adj = [[] for _ in range(k)]
    M = mod_p(S, p)
    for i in range(k):
        rows_i = [i, k + i]
        for j in range(i + 1, k):
            cols_j = [j, k + j]
            block_ij = M[np.ix_(rows_i, cols_j)]
            block_ji = M[np.ix_([j, k + j], [i, k + i])]
            if np.any(block_ij % p != 0) or np.any(block_ji % p != 0):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _components(adj: List[List[int]]) -> List[List[int]]:
    """Connected components from adjacency list."""
    k = len(adj)
    seen = [False] * k
    comps = []
    for s in range(k):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _mode_permutation_for_blocks(S: np.ndarray, p: int) -> np.ndarray:
    """
    Permute modes so that:
      (1) Nontrivial components with >= 2 modes first, sorted by size asc (tie: rank desc),
      (2) then nontrivial singletons (rank desc),
      (3) then trivial components (size asc).
    """
    n2 = S.shape[0]
    k = n2 // 2
    adj = _mode_graph_from_S(S, p)
    comps = _components(adj)

    M = mod_p(S - np.eye(n2, dtype=np.int64), p)

    def comp_rank(comp_modes: list[int]) -> int:
        idx = comp_modes + [c + k for c in comp_modes]
        return rank_mod(M[np.ix_(idx, idx)], p)

    info = []
    for comp in comps:
        r = comp_rank(comp)
        info.append({"modes": comp, "n_modes": len(comp), "rank": r})

    non_triv_ge2 = [c for c in info if c["rank"] > 0 and c["n_modes"] >= 2]
    non_triv_1 = [c for c in info if c["rank"] > 0 and c["n_modes"] == 1]
    triv = [c for c in info if c["rank"] == 0]

    non_triv_ge2.sort(key=lambda c: (c["n_modes"], -c["rank"]))
    non_triv_1.sort(key=lambda c: -c["rank"])
    triv.sort(key=lambda c: c["n_modes"])

    order_modes: list[int] = []
    for bucket in (non_triv_ge2, non_triv_1, triv):
        for c in bucket:
            order_modes.extend(c["modes"])

    P = np.zeros((k, k), dtype=np.int64)
    for new_pos, old_mode in enumerate(order_modes):
        P[new_pos, old_mode] = 1
    return P


def _apply_mode_permutation_to_ST(S: np.ndarray, T: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permute modes with the same P in U and V: Π = diag(P, P).
    Preserves Omega; produces S' = Π^{-1} S Π, T' = T Π.
    """
    n2 = S.shape[0]
    k = n2 // 2
    P = _mode_permutation_for_blocks(S, p)
    Pi = np.block([[P, np.zeros((k, k), dtype=np.int64)],
                  [np.zeros((k, k), dtype=np.int64), P]])
    Pi_inv = Pi.T
    S2 = mod_p(Pi_inv @ S @ Pi, p)
    T2 = mod_p(T @ Pi, p)
    return S2, T2

# =========================
# Symplectic structures
# =========================


# Safe scalar from 1x1 array (no deprecation warnings)
def _scalar(x: np.ndarray) -> int:
    return int(np.asarray(x, dtype=np.int64).reshape(()))


def _split_uv(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given canonical [U | V], return (U, V)."""
    k2 = T.shape[1]
    assert k2 % 2 == 0
    k = k2 // 2
    return T[:, :k], T[:, k:]


# =========================
# Canonical symplectic basis from a span
# =========================

def symplectic_basis_from_span(B: np.ndarray, p: int) -> np.ndarray:
    """
    Input: B (n2 x s) spans an even-dimensional non-degenerate subspace.
    Output: T (n2 x 2k) canonical, T^T Omega T = Omega_k, ordered [U | V].
    """
    n2 = B.shape[0]
    Omega = omega_matrix(n2 // 2, p)

    # Independent spanning columns S (single RREF)
    S = independent_columns(B, p)
    dS = S.shape[1]
    if dS % 2 != 0:
        raise ValueError("Subspace dimension is odd; cannot form symplectic basis")

    def as_col(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.int64)
        return v.reshape(-1, 1)

    def pair(a: np.ndarray, b: np.ndarray) -> int:
        return _scalar((a.T @ Omega @ b) % p)

    def row_from_vec(vec: np.ndarray) -> np.ndarray:
        vec = as_col(vec)
        r = mod_p(S.T @ Omega @ vec, p).reshape(-1)
        return r

    def row_from_a(a: np.ndarray) -> np.ndarray:
        a = as_col(a)
        r = mod_p(a.T @ Omega @ S, p).reshape(-1)
        return r

    U: List[np.ndarray] = []
    V: List[np.ndarray] = []
    used = np.zeros(dS, dtype=bool)

    while (len(U) + len(V)) < dS:
        # pick new a independent of current span(U|V)
        M = np.column_stack(U + V) if (U or V) else np.zeros((n2, 0), dtype=np.int64)
        a = None
        for j in range(dS):
            if used[j]:
                continue
            candidate = S[:, j: j + 1]
            if rank_mod(np.concatenate([M, candidate], axis=1), p) > rank_mod(M, p):
                a = candidate
                used[j] = True
                break
        if a is None:
            # fallback: random combo (rare)
            rng = np.random.default_rng(1337)
            for _ in range(128):
                coeffs = rng.integers(0, p, size=(dS, 1), dtype=np.int64)
                candidate = mod_p(S @ coeffs, p)
                if rank_mod(np.concatenate([M, candidate], axis=1), p) > rank_mod(M, p):
                    a = candidate
                    break
            if a is None:
                raise RuntimeError("Failed to extend independent vector in span(B)")

        # omega-orthogonalize a against existing pairs: a ← a - <a,v>u + <a,u>v
        for u, v in zip(U, V):
            av = pair(a, v)
            au = pair(a, u)
            if av:
                a = mod_p(a - av * u, p)
            if au:
                a = mod_p(a + au * v, p)

        # Solve for b in span(S): <b,u>=0, <b,v>=0, and <a,b>=1
        rows = []
        rhs = []
        for u, v in zip(U, V):
            rows.append(row_from_vec(u))
            rhs.append(0)
            rows.append(row_from_vec(v))
            rhs.append(0)
        rows.append(row_from_a(a))
        rhs.append(1)

        A = mod_p(np.vstack(rows), p)
        b_vec = np.array(rhs, dtype=np.int64).reshape(-1, 1)
        b = _solve_linear(A, b_vec, p)
        b = mod_p(S @ b, p)

        ab = pair(a, b)
        if ab % p == 0:
            raise RuntimeError("Zero pairing for constructed (a,b)")
        if ab % p != 1 % p:
            b = mod_p(b * inv_mod_scalar(ab, p), p)

        U.append(a)
        V.append(b)

    T = np.column_stack([u for u in U] + [v for v in V])  # [U | V]
    G = mod_p(T.T @ Omega @ T, p)
    if rank_mod(G, p) != T.shape[1]:
        raise RuntimeError("Degenerate Gram in symplectic_basis_from_span")
    return T

# =========================
# Minimal symplectic block via Krylov + partner
# =========================


def krylov_closure(F: np.ndarray, v: np.ndarray, p: int, cap: int | None = None) -> np.ndarray:
    """Krylov span K = span{v, Fv, ..., F^{k-1}v} until rank stops increasing."""
    n2 = F.shape[0]
    cap_local: int = n2 if cap is None else int(cap)
    V = np.zeros((n2, 0), dtype=np.int64)
    w = v.reshape(-1, 1)
    r0 = 0
    for _ in range(cap_local):
        can = np.concatenate([V, w], axis=1)
        r = rank_mod(can, p)
        if r > r0:
            V = can
            r0 = r
            w = matmul_mod(F, w, p)
        else:
            break
    return V


def build_partner_for_krylov(F: np.ndarray, K: np.ndarray, p: int) -> np.ndarray:
    """
    Given K = span{v, Fv, ..., F^{k-1}v}, find z with constraints:
      <v, F^b z> = δ_{b, k-1} for b=0..k-1.
    Efficient vector recurrence for rows r_b = (F^b)^T Omega v.
    """
    n2 = F.shape[0]
    Omega = omega_matrix(n2 // 2, p)
    k = K.shape[1]
    v0 = K[:, 0:1]  # column

    # rb_0 = Omega v0, rb_{b+1} = F^T rb_b
    rb = mod_p(Omega @ v0, p)
    rows = [rb.T.reshape(-1)]
    for _ in range(1, k):
        rb = mod_p(F.T @ rb, p)
        rows.append(rb.T.reshape(-1))
    A = np.stack(rows, axis=0)           # (k, n2)
    b_vec = np.zeros((k, 1), dtype=np.int64)
    b_vec[-1, 0] = 1                       # δ_{b,k-1}
    z = _solve_linear(A, b_vec, p)         # (n2,1)
    return z


def _restricted_action(F: np.ndarray, T_blk: np.ndarray, p: int) -> np.ndarray:
    """Return S_blk = T_blk^{-1} F T_blk using J^{-1} T^T Omega."""
    n2 = F.shape[0]
    Omega = omega_matrix(n2 // 2, p)
    J = mod_p(T_blk.T @ Omega @ T_blk, p)  # should be Omega_k
    J_inv = inv_mod_mat(J, p)
    left_inv = matmul_mod(J_inv, matmul_mod(T_blk.T, Omega, p), p)
    return matmul_mod(left_inv, matmul_mod(F, T_blk, p), p)


def _minimal_block_from_seeds(
    F: np.ndarray,
    p: int,
    seeds: List[np.ndarray],
    min_block_size: int = 0
) -> np.ndarray | None:
    """
    Unified worker: tries seeds, returns best T_blk or None.
    Score: smallest size (>= min_block_size if >0), tie-break by rank(S_blk - I) descending.
    """
    n2 = F.shape[0]
    n = n2 // 2
    Omega = omega_matrix(n, p)

    best: tuple[int, int, np.ndarray] | None = None  # (size, r_score, T_blk)

    for v in seeds:
        if v.size == 0 or np.all(v % p == 0):
            continue
        K = krylov_closure(F, v, p)
        if K.shape[1] == 0:
            continue
        try:
            z = build_partner_for_krylov(F, K, p)
        except RuntimeError:
            continue
        Kz = krylov_closure(F, z, p)
        W = np.concatenate([K, Kz], axis=1)

        B = independent_columns(W, p)
        if B.shape[1] % 2 != 0:
            continue
        if min_block_size and B.shape[1] < min_block_size:
            continue

        G = mod_p(B.T @ Omega @ B, p)
        if rank_mod(G, p) != B.shape[1]:
            continue

        try:
            T_blk = symplectic_basis_from_span(B, p)  # [U|V]
        except Exception:
            continue

        S_blk = _restricted_action(F, T_blk, p)
        r_score = rank_mod(mod_p(S_blk - np.eye(S_blk.shape[0], dtype=np.int64), p), p)
        if r_score == 0:
            continue  # reject trivial block

        size = T_blk.shape[1]
        if (best is None or size < best[0] or (size == best[0] and r_score > best[1])):
            best = (size, r_score, T_blk)
            if min_block_size and size == min_block_size and r_score > 0:
                break
            if not min_block_size and size == 2:
                break

    return None if best is None else best[2]


def minimal_symplectic_block_in_complement(
    F: np.ndarray, p: int, N: np.ndarray, trials: int = 64
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Find the smallest-dimension invariant non-degenerate symplectic block W for F,
    **constrained to the subspace span(N)** (columns of N).
    Returns (T_blk, S_blk) in ambient coordinates, or None if none found.
    """
    rng = np.random.default_rng(2025)

    seeds: List[np.ndarray] = [N[:, i: i + 1] for i in range(N.shape[1])]
    for _ in range(trials):
        coeffs = rng.integers(0, p, size=(N.shape[1], 1), dtype=np.int64)
        seeds.append(mod_p(N @ coeffs, p))

    T_blk = _minimal_block_from_seeds(F, p, seeds, min_block_size=0)
    if T_blk is None:
        return None
    S_blk = _restricted_action(F, T_blk, p)
    return T_blk, S_blk


def minimal_symplectic_block(F: np.ndarray, p: int, trials: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a smallest-dimension invariant, non-degenerate symplectic block W for F.
    Returns (T_blk, S_blk).
    """
    n2 = F.shape[0]
    rng = np.random.default_rng(2025)

    seeds: List[np.ndarray] = [np.eye(n2, dtype=np.int64)[:, i: i + 1] for i in range(n2)]
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    T_blk = _minimal_block_from_seeds(F, p, seeds, min_block_size=0)
    if T_blk is None:
        raise RuntimeError("Failed to find a non-degenerate invariant symplectic block")
    S_blk = _restricted_action(F, T_blk, p)
    return T_blk, S_blk


# =========================
# Local symplectic completion and global peel
# =========================

def complete_symplectic_local(T_blk: np.ndarray, p: int) -> np.ndarray:
    """
    Given T_blk (n2_sub x k2) a canonical symplectic basis of a non-degenerate subspace W,
    complete to a full symplectic basis of the trailing subspace:
       T_local = [U_W | U_perp | V_W | V_perp], with T_local^T Omega_sub T_local = Omega_sum.
    """
    n2_sub = T_blk.shape[0]
    Omega_sub = omega_matrix(n2_sub // 2, p)

    # Omega-orthogonal complement: W_perp = {x | T_blk^T Omega x = 0}
    A = mod_p(T_blk.T @ Omega_sub, p)           # k2 x n2_sub
    N = nullspace_mod(A, p)               # n2_sub x d_perp
    d_perp = N.shape[1]
    if d_perp % 2 != 0:
        raise RuntimeError("Complement has odd dimension (cannot symplectically complete)")

    if d_perp > 0:
        T_perp = symplectic_basis_from_span(N, p)   # [U_perp | V_perp]
        U_perp, V_perp = _split_uv(T_perp)
    else:
        U_perp = np.zeros((n2_sub, 0), dtype=np.int64)
        V_perp = np.zeros((n2_sub, 0), dtype=np.int64)

    U_W, V_W = _split_uv(T_blk)
    T_local = np.concatenate([U_W, U_perp, V_W, V_perp], axis=1)

    G = mod_p(T_local.T @ Omega_sub @ T_local, p)
    if not np.array_equal(G % p, Omega_sub % p):
        raise RuntimeError("Local completion is not symplectic (T_local^T Omega T_local ≠ Omega).")
    return T_local


def block_decompose(F: np.ndarray, p: int, min_block_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sector-aware decomposition: find S = T^{-1} F T with the largest block as small as possible.
    Strategy:
      1) RCF pre-pass -> primary/reciprocal sectors W_q with half-dimension floors.
      2) In each sector, run the existing Krylov-based minimal block finder constrained to that sector.
      3) Pick the globally-smallest nontrivial block (tie-break by rank(S_blk - I) descending).
      4) Omega-complement, assemble T in canonical [all U | all V], then S = T^{-1} F T.
      5) Symplectic mode permutation to pack blocks contiguously.
    """
    assert is_symplectic(F, p), "F must be symplectic"
    n2 = F.shape[0]
    n = n2 // 2
    Omega = omega_matrix(n, p)

    # -------------------------
    # 1) RCF pre-pass (sectorization with floors)
    # -------------------------
    rcf_info = rcf_prepass(F, p)
    sectors = rcf_info["sectors"]  # list of {"type","W_basis","half_dim_floor",...}

    # -------------------------
    # 2) Search minimal NONTRIVIAL block within each sector
    # -------------------------
    best = None  # tuple(size, score, T_blk, sector_idx)
    for s_idx, sec in enumerate(sectors):
        N = sec["W_basis"]                      # columns span the sector subspace
        if N.shape[1] == 0:
            continue
        # Prefer to keep this deterministic: set trials=0 (sector basis already reduces search)
        out = minimal_symplectic_block_in_complement(F, p, N, trials=0)
        if out is None:
            # Fallback: allow a few sector-local random seeds
            out = minimal_symplectic_block_in_complement(F, p, N, trials=32)
            if out is None:
                continue
        T_blk, S_blk = out
        size = T_blk.shape[1]                    # = 2k
        if size % 2 != 0:
            continue
        # Reject trivial blocks
        r_score = rank_mod(mod_p(S_blk - np.eye(size, dtype=np.int64), p), p)
        if r_score == 0:
            continue
        # Respect a sector-aware minimum if requested
        if size < max(2 * sec["half_dim_floor"], min_block_size):
            # keep going; this block is smaller than the sector’s theoretical floor (robustness)
            pass
        # Pick smallest size, break ties by larger r_score
        if (best is None) or (size < best[0]) or (size == best[0] and r_score > best[1]):
            best = (size, r_score, T_blk, s_idx)
            if size == 2 and r_score > 0:
                break  # cannot beat a 1-mode hyperbolic block

    # If nothing found in sectors (should be rare for symplectic F), fall back to global search
    if best is None:
        T_blk = minimal_symplectic_block_full(F, p, trials=128, min_block_size=min_block_size)
    else:
        T_blk = best[2]

    k2 = T_blk.shape[1]
    k = k2 // 2
    U_blk, V_blk = T_blk[:, :k], T_blk[:, k:]

    # -------------------------
    # 3) Exact Omega-orthogonal complement and canonical basis there
    # -------------------------
    A = mod_p(T_blk.T @ Omega, p)       # k2 x n2
    N_perp = nullspace_mod(A, p)      # n2 x d_perp
    if N_perp.shape[1] > 0:
        T_perp = symplectic_basis_from_span(N_perp, p)  # [U_perp | V_perp]
        kp2 = T_perp.shape[1] // 2
        U_perp, V_perp = T_perp[:, :kp2], T_perp[:, kp2:]
    else:
        U_perp = np.zeros((n2, 0), dtype=np.int64)
        V_perp = np.zeros((n2, 0), dtype=np.int64)

    # -------------------------
    # 4) Assemble global T in canonical [all U | all V]
    # -------------------------
    T = np.concatenate([U_blk, U_perp, V_blk, V_perp], axis=1)

    # Strict symplectic check
    assert np.array_equal(mod_p(T.T @ Omega @ T, p), Omega % p), "Constructed T is not symplectic"

    # -------------------------
    # 5) Compute S and apply mode permutation
    # -------------------------
    J = mod_p(T.T @ Omega @ T, p)
    J_inv = inv_mod_mat(J, p)
    L_inv = matmul_mod(J_inv, matmul_mod(T.T, Omega, p), p)
    S = matmul_mod(L_inv, matmul_mod(F, T, p), p)

    # Pack coupled modes contiguously and bring the smallest nontrivial first
    S, T = _apply_mode_permutation_to_ST(S, T, p)

    # Final sanity
    assert is_symplectic(T, p)
    assert np.array_equal(S, mod_p(inv_mod_mat(T, p) @ F @ T, p))
    return S, T


# =========================
# Nontrivial block finder with size floor (public API preserved)
# =========================

def minimal_symplectic_block_full(
    F: np.ndarray, p: int, trials: int = 64, min_block_size: int = 2
) -> np.ndarray:
    """
    Find a smallest-dimension invariant, non-degenerate, and nontrivial symplectic block W for F.
    min_block_size is even (2,4,6,...) and lets you prioritize e.g. 4 to "find coupling first".
    Returns T_blk (n2 x 2k) canonical: T_blk^T Omega T_blk = Omega_k.
    """
    n2 = F.shape[0]
    rng = np.random.default_rng(2025)
    seeds: List[np.ndarray] = [np.eye(n2, dtype=np.int64)[:, i: i + 1] for i in range(n2)]
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    T_blk = _minimal_block_from_seeds(F, p, seeds, min_block_size=min_block_size)
    if T_blk is None:
        raise RuntimeError("Failed to find a nontrivial symplectic block meeting the size floor")
    return T_blk


# =========================
# Diagnostics
# =========================

def ordered_block_sizes(S: np.ndarray, p: int) -> list[int]:
    """Return 2*n_modes for connected components, ordered with the same nontrivial-first policy."""
    n2 = S.shape[0]
    k = n2 // 2
    adj = _mode_graph_from_S(S, p)
    comps = _components(adj)
    M = mod_p(S - np.eye(n2, dtype=np.int64), p)

    def comp_rank(comp):
        idx = comp + [c + k for c in comp]
        return rank_mod(M[np.ix_(idx, idx)], p)

    info = [{"modes": c, "n_modes": len(c), "rank": comp_rank(c)} for c in comps]
    non_triv_ge2 = [d for d in info if d["rank"] > 0 and d["n_modes"] >= 2]
    non_triv_1 = [d for d in info if d["rank"] > 0 and d["n_modes"] == 1]
    triv = [d for d in info if d["rank"] == 0]

    non_triv_ge2.sort(key=lambda d: (d["n_modes"], -d["rank"]))
    non_triv_1.sort(key=lambda d: -d["rank"])
    triv.sort(key=lambda d: d["n_modes"])

    ordered = non_triv_ge2 + non_triv_1 + triv
    return [2 * d["n_modes"] for d in ordered]
