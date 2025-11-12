import numpy as np
from typing import Tuple, List
from .modular_helpers import (modp, rank_mod, _solve_linear, nullspace_mod, rref_mod, omega_matrix, inv_mod_mat,
                             inv_mod_scalar, matmul_mod, is_symplectic)
from .minimal_block_size import rcf_prepass
# =========================
# GF(p) linear algebra utils
# =========================


def independent_columns(B: np.ndarray, p: int) -> np.ndarray:
    """Return a column-subset of B with independent columns over GF(p) (single RREF pass)."""
    if B.size == 0:
        return B
    _, piv = rref_mod(modp(B, p), p)
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
    M = modp(S, p)
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

    M = modp(S - np.eye(n2, dtype=np.int64), p)

    def comp_rank(comp_modes: list[int]) -> int:
        idx = comp_modes + [c + k for c in comp_modes]
        return rank_mod(M[np.ix_(idx, idx)], p)

    info = []
    for comp in comps:
        r = comp_rank(comp)
        info.append({"modes": comp, "nmodes": len(comp), "rank": r})

    nontriv_ge2 = [c for c in info if c["rank"] > 0 and c["nmodes"] >= 2]
    nontriv_1   = [c for c in info if c["rank"] > 0 and c["nmodes"] == 1]
    triv        = [c for c in info if c["rank"] == 0]

    nontriv_ge2.sort(key=lambda c: (c["nmodes"], -c["rank"]))
    nontriv_1.sort(key=lambda c: -c["rank"])
    triv.sort(key=lambda c: c["nmodes"])

    order_modes: list[int] = []
    for bucket in (nontriv_ge2, nontriv_1, triv):
        for c in bucket:
            order_modes.extend(c["modes"])

    P = np.zeros((k, k), dtype=np.int64)
    for new_pos, old_mode in enumerate(order_modes):
        P[new_pos, old_mode] = 1
    return P

def _apply_mode_permutation_to_ST(S: np.ndarray, T: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permute modes with the same P in U and V: Π = diag(P, P).
    Preserves Ω; produces S' = Π^{-1} S Π, T' = T Π.
    """
    n2 = S.shape[0]
    k = n2 // 2
    P = _mode_permutation_for_blocks(S, p)
    Π = np.block([[P, np.zeros((k, k), dtype=np.int64)],
                  [np.zeros((k, k), dtype=np.int64), P]])
    Πinv = Π.T
    S2 = modp(Πinv @ S @ Π, p)
    T2 = modp(T @ Π, p)
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
    Input: B (n2 x s) spans an even-dimensional nondegenerate subspace.
    Output: T (n2 x 2k) canonical, T^T Ω T = Ω_k, ordered [U | V].
    """
    n2 = B.shape[0]
    Ω = omega_matrix(n2 // 2, p)

    # Independent spanning columns S (single RREF)
    S = independent_columns(B, p)
    dS = S.shape[1]
    if dS % 2 != 0:
        raise ValueError("Subspace dimension is odd; cannot form symplectic basis")

    def as_col(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.int64)
        return v.reshape(-1, 1)

    def pair(a: np.ndarray, b: np.ndarray) -> int:
        return _scalar((a.T @ Ω @ b) % p)

    def row_from_vec(vec: np.ndarray) -> np.ndarray:
        vec = as_col(vec)
        r = modp(S.T @ Ω @ vec, p).reshape(-1)
        return r

    def row_from_a(a: np.ndarray) -> np.ndarray:
        a = as_col(a)
        r = modp(a.T @ Ω @ S, p).reshape(-1)
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
            cand = S[:, j:j+1]
            if rank_mod(np.concatenate([M, cand], axis=1), p) > rank_mod(M, p):
                a = cand
                used[j] = True
                break
        if a is None:
            # fallback: random combo (rare)
            rng = np.random.default_rng(1337)
            for _ in range(128):
                coeffs = rng.integers(0, p, size=(dS, 1), dtype=np.int64)
                cand = modp(S @ coeffs, p)
                if rank_mod(np.concatenate([M, cand], axis=1), p) > rank_mod(M, p):
                    a = cand
                    break
            if a is None:
                raise RuntimeError("Failed to extend independent vector in span(B)")

        # Ω-orthogonalize a against existing pairs: a ← a - <a,v>u + <a,u>v
        for u, v in zip(U, V):
            av = pair(a, v)
            au = pair(a, u)
            if av:
                a = modp(a - av * u, p)
            if au:
                a = modp(a + au * v, p)

        # Solve for b in span(S): <b,u>=0, <b,v>=0, and <a,b>=1
        rows = []
        rhs = []
        for u, v in zip(U, V):
            rows.append(row_from_vec(u)); rhs.append(0)
            rows.append(row_from_vec(v)); rhs.append(0)
        rows.append(row_from_a(a)); rhs.append(1)

        A = modp(np.vstack(rows), p)
        bvec = np.array(rhs, dtype=np.int64).reshape(-1, 1)
        b = _solve_linear(A, bvec, p)
        b = modp(S @ b, p)

        ab = pair(a, b)
        if ab % p == 0:
            raise RuntimeError("Zero pairing for constructed (a,b)")
        if ab % p != 1 % p:
            b = modp(b * inv_mod_scalar(ab, p), p)

        U.append(a)
        V.append(b)

    T = np.column_stack([u for u in U] + [v for v in V])  # [U | V]
    G = modp(T.T @ Ω @ T, p)
    if rank_mod(G, p) != T.shape[1]:
        raise RuntimeError("Degenerate Gram in symplectic_basis_from_span")
    return T

# =========================
# Minimal symplectic block via Krylov + partner
# =========================

def krylov_closure(F: np.ndarray, v: np.ndarray, p: int, cap: int | None = None) -> np.ndarray:
    """Krylov span K = span{v, Fv, ..., F^{k-1}v} until rank stops increasing."""
    n2 = F.shape[0]
    if cap is None:
        cap = n2
    V = np.zeros((n2, 0), dtype=np.int64)
    w = v.reshape(-1, 1)
    r0 = 0
    for _ in range(cap):
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
    Efficient vector recurrence for rows r_b = (F^b)^T Ω v.
    """
    n2 = F.shape[0]
    Ω = omega_matrix(n2 // 2, p)
    k = K.shape[1]
    v0 = K[:, 0:1]  # column

    # rb_0 = Ω v0, rb_{b+1} = F^T rb_b
    rb = modp(Ω @ v0, p)
    rows = [rb.T.reshape(-1)]
    for _ in range(1, k):
        rb = modp(F.T @ rb, p)
        rows.append(rb.T.reshape(-1))
    A = np.stack(rows, axis=0)           # (k, n2)
    bvec = np.zeros((k, 1), dtype=np.int64)
    bvec[-1, 0] = 1                       # δ_{b,k-1}
    z = _solve_linear(A, bvec, p)         # (n2,1)
    return z

def _restricted_action(F: np.ndarray, Tblk: np.ndarray, p: int) -> np.ndarray:
    """Return Sblk = Tblk^{-1} F Tblk using J^{-1} T^T Ω."""
    n2 = F.shape[0]
    Ω = omega_matrix(n2 // 2, p)
    J = modp(Tblk.T @ Ω @ Tblk, p)  # should be Ω_k
    Jinv = inv_mod_mat(J, p)
    left_inv = matmul_mod(Jinv, matmul_mod(Tblk.T, Ω, p), p)
    return matmul_mod(left_inv, matmul_mod(F, Tblk, p), p)

def _minimal_block_from_seeds(
    F: np.ndarray,
    p: int,
    seeds: List[np.ndarray],
    min_block_size: int = 0
) -> np.ndarray | None:
    """
    Unified worker: tries seeds, returns best Tblk or None.
    Score: smallest size (>= min_block_size if >0), tie-break by rank(Sblk - I) descending.
    """
    n2 = F.shape[0]
    n = n2 // 2
    Ω = omega_matrix(n, p)

    best: tuple[int, int, np.ndarray] | None = None  # (size, rscore, Tblk)

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

        G = modp(B.T @ Ω @ B, p)
        if rank_mod(G, p) != B.shape[1]:
            continue

        try:
            Tblk = symplectic_basis_from_span(B, p)  # [U|V]
        except Exception:
            continue

        Sblk = _restricted_action(F, Tblk, p)
        rscore = rank_mod(modp(Sblk - np.eye(Sblk.shape[0], dtype=np.int64), p), p)
        if rscore == 0:
            continue  # reject trivial block

        size = Tblk.shape[1]
        if (best is None or
            size < best[0] or
            (size == best[0] and rscore > best[1])):
            best = (size, rscore, Tblk)
            if min_block_size and size == min_block_size and rscore > 0:
                break
            if not min_block_size and size == 2:
                break

    return None if best is None else best[2]

def minimal_symplectic_block_in_complement(
    F: np.ndarray, p: int, N: np.ndarray, trials: int = 64
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Find the smallest-dimension invariant nondegenerate symplectic block W for F,
    **constrained to the subspace span(N)** (columns of N).
    Returns (Tblk, Sblk) in ambient coordinates, or None if none found.
    """
    n2 = F.shape[0]
    Ω = omega_matrix(n2 // 2, p)
    rng = np.random.default_rng(2025)

    seeds: List[np.ndarray] = [N[:, i:i+1] for i in range(N.shape[1])]
    for _ in range(trials):
        coeffs = rng.integers(0, p, size=(N.shape[1], 1), dtype=np.int64)
        seeds.append(modp(N @ coeffs, p))

    Tblk = _minimal_block_from_seeds(F, p, seeds, min_block_size=0)
    if Tblk is None:
        return None
    Sblk = _restricted_action(F, Tblk, p)
    return Tblk, Sblk

def minimal_symplectic_block(F: np.ndarray, p: int, trials: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a smallest-dimension invariant, nondegenerate symplectic block W for F.
    Returns (Tblk, Sblk).
    """
    n2 = F.shape[0]
    Ω = omega_matrix(n2 // 2, p)
    rng = np.random.default_rng(2025)

    seeds: List[np.ndarray] = [np.eye(n2, dtype=np.int64)[:, i:i+1] for i in range(n2)]
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    Tblk = _minimal_block_from_seeds(F, p, seeds, min_block_size=0)
    if Tblk is None:
        raise RuntimeError("Failed to find a nondegenerate invariant symplectic block")
    Sblk = _restricted_action(F, Tblk, p)
    return Tblk, Sblk

# =========================
# Local symplectic completion and global peel
# =========================

def complete_symplectic_local(Tblk: np.ndarray, p: int) -> np.ndarray:
    """
    Given Tblk (n2_sub x k2) a canonical symplectic basis of a nondegenerate subspace W,
    complete to a full symplectic basis of the trailing subspace:
       T_local = [U_W | U_perp | V_W | V_perp], with T_local^T Ω_sub T_local = Ω_sub.
    """
    n2_sub = Tblk.shape[0]
    Ω_sub = omega_matrix(n2_sub // 2, p)

    # Ω-orthogonal complement: W_perp = {x | Tblk^T Ω_sub x = 0}
    A = modp(Tblk.T @ Ω_sub, p)           # k2 x n2_sub
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

    U_W, V_W = _split_uv(Tblk)
    T_local = np.concatenate([U_W, U_perp, V_W, V_perp], axis=1)

    G = modp(T_local.T @ Ω_sub @ T_local, p)
    if not np.array_equal(G % p, Ω_sub % p):
        raise RuntimeError("Local completion is not symplectic (T_local^T Ω T_local ≠ Ω).")
    return T_local

def block_decompose(F: np.ndarray, p: int, min_block_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sector-aware decomposition: find S = T^{-1} F T with the largest block as small as possible.
    Strategy:
      1) RCF pre-pass -> primary/reciprocal sectors W_q with half-dimension floors.
      2) In each sector, run the existing Krylov-based minimal block finder constrained to that sector.
      3) Pick the globally-smallest NONTRIVIAL block (tie-break by rank(Sblk - I) descending).
      4) Ω-complement, assemble T in canonical [all U | all V], then S = T^{-1} F T.
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
    best = None  # tuple(size, score, Tblk, sector_idx)
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
        Tblk, Sblk = out
        size = Tblk.shape[1]                    # = 2k
        if size % 2 != 0:
            continue
        # Reject trivial blocks
        rscore = rank_mod(modp(Sblk - np.eye(size, dtype=np.int64), p), p)
        if rscore == 0:
            continue
        # Respect a sector-aware minimum if requested
        if size < max(2 * sec["half_dim_floor"], min_block_size):
            # keep going; this block is smaller than the sector’s theoretical floor (robustness)
            pass
        # Pick smallest size, break ties by larger rscore
        if (best is None) or (size < best[0]) or (size == best[0] and rscore > best[1]):
            best = (size, rscore, Tblk, s_idx)
            if size == 2 and rscore > 0:
                break  # cannot beat a 1-mode hyperbolic block

    # If nothing found in sectors (should be rare for symplectic F), fall back to global search
    if best is None:
        Tblk = minimal_symplectic_block_full(F, p, trials=128, min_block_size=min_block_size)
    else:
        Tblk = best[2]

    k2 = Tblk.shape[1]
    k = k2 // 2
    U_blk, V_blk = Tblk[:, :k], Tblk[:, k:]

    # -------------------------
    # 3) Exact Ω-orthogonal complement and canonical basis there
    # -------------------------
    A = modp(Tblk.T @ Omega, p)       # k2 x n2
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
    assert np.array_equal(modp(T.T @ Omega @ T, p), Omega % p), "Constructed T is not symplectic"

    # -------------------------
    # 5) Compute S and apply mode permutation
    # -------------------------
    J = modp(T.T @ Omega @ T, p)                         # = Ω
    Jinv = inv_mod_mat(J, p)
    Linv = matmul_mod(Jinv, matmul_mod(T.T, Omega, p), p)
    S = matmul_mod(Linv, matmul_mod(F, T, p), p)

    # Pack coupled modes contiguously and bring the smallest nontrivial first
    S, T = _apply_mode_permutation_to_ST(S, T, p)

    # Final sanity
    assert is_symplectic(T, p)
    assert np.array_equal(S, modp(inv_mod_mat(T, p) @ F @ T, p))
    return S, T


# =========================
# Nontrivial block finder with size floor (public API preserved)
# =========================

def minimal_symplectic_block_full(
    F: np.ndarray, p: int, trials: int = 64, min_block_size: int = 2
) -> np.ndarray:
    """
    Find a smallest-dimension invariant, nondegenerate, and NONTRIVIAL symplectic block W for F.
    min_block_size is even (2,4,6,...) and lets you prioritize e.g. 4 to "find coupling first".
    Returns Tblk (n2 x 2k) canonical: Tblk^T Ω Tblk = Ω_k.
    """
    n2 = F.shape[0]
    rng = np.random.default_rng(2025)
    seeds: List[np.ndarray] = [np.eye(n2, dtype=np.int64)[:, i:i+1] for i in range(n2)]
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    Tblk = _minimal_block_from_seeds(F, p, seeds, min_block_size=min_block_size)
    if Tblk is None:
        raise RuntimeError("Failed to find a nontrivial symplectic block meeting the size floor")
    return Tblk

# =========================
# Diagnostics
# =========================

def ordered_block_sizes(S: np.ndarray, p: int) -> list[int]:
    """Return 2*nmodes for connected components, ordered with the same nontrivial-first policy."""
    n2 = S.shape[0]
    k = n2 // 2
    adj = _mode_graph_from_S(S, p)
    comps = _components(adj)
    M = modp(S - np.eye(n2, dtype=np.int64), p)

    def comp_rank(comp):
        idx = comp + [c + k for c in comp]
        return rank_mod(M[np.ix_(idx, idx)], p)

    info = [{"modes": c, "nmodes": len(c), "rank": comp_rank(c)} for c in comps]
    nontriv_ge2 = [d for d in info if d["rank"] > 0 and d["nmodes"] >= 2]
    nontriv_1 = [d for d in info if d["rank"] > 0 and d["nmodes"] == 1]
    triv = [d for d in info if d["rank"] == 0]

    nontriv_ge2.sort(key=lambda d: (d["nmodes"], -d["rank"]))
    nontriv_1.sort(key=lambda d: -d["rank"])
    triv.sort(key=lambda d: d["nmodes"])

    ordered = nontriv_ge2 + nontriv_1 + triv
    return [2 * d["nmodes"] for d in ordered]
