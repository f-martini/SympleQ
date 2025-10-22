import numpy as np
from typing import Tuple, List

# =========================
# GF(p) linear algebra utils
# =========================

def modp(A: np.ndarray, p: int) -> np.ndarray:
    return np.asarray(A % p, dtype=np.int64)

def matmul_mod(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    return modp(A @ B, p)

def inv_mod_scalar(a: int | np.integer, p: int) -> int:
    return pow(int(a) % p, p - 2, p)

def rref_mod(aug: np.ndarray, p: int) -> Tuple[np.ndarray, List[int]]:
    """RREF over GF(p). Returns (RREF_augmented, pivot_cols)."""
    A = modp(aug.copy(), p)
    m, n = A.shape
    r = 0
    c = 0
    piv_cols: List[int] = []
    while r < m and c < n:
        piv = None
        for i in range(r, m):
            if A[i, c] % p != 0:
                piv = i
                break
        if piv is None:
            c += 1
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = inv_mod_scalar(A[r, c], p)
        A[r, :] = modp(A[r, :] * inv, p)
        for i in range(m):
            if i != r and A[i, c] % p != 0:
                fac = A[i, c] % p
                A[i, :] = modp(A[i, :] - fac * A[r, :], p)
        piv_cols.append(c)
        r += 1
        c += 1
    return A, piv_cols

def inv_mod_mat(A: np.ndarray, p: int) -> np.ndarray:
    """Gauss-Jordan inverse over GF(p). Raises if singular."""
    n = A.shape[0]
    aug = np.concatenate([modp(A, p), np.eye(n, dtype=np.int64)], axis=1)
    R, _ = rref_mod(aug, p)
    left = R[:, :n]
    right = R[:, n:]
    if not np.array_equal(left % p, np.eye(n, dtype=np.int64)):
        raise ValueError("Matrix not invertible mod p")
    return modp(right, p)

def rank_mod(A: np.ndarray, p: int) -> int:
    R, _ = rref_mod(modp(A, p), p)
    # Count nonzero rows
    return int(np.sum(np.any(R % p != 0, axis=1)))

def nullspace_mod(A: np.ndarray, p: int) -> np.ndarray:
    """Right nullspace basis of A over GF(p); columns form a basis."""
    A = modp(A, p)
    m, n = A.shape
    aug = np.concatenate([A, np.zeros((m, 1), dtype=np.int64)], axis=1)
    R, piv_cols = rref_mod(aug, p)
    piv_set = set(piv_cols)
    free = [j for j in range(n) if j not in piv_set]
    if not free:
        return np.zeros((n, 0), dtype=np.int64)
    # Solve Ax=0 with free vars = standard basis
    basis = []
    for f in free:
        x = np.zeros((n, 1), dtype=np.int64)
        x[f, 0] = 1
        # back-substitution: R is in RREF on the left block
        # Each pivot row r has leading 1 at col pc = piv_cols[idx]
        row_idx = 0
        for pc in piv_cols:
            # R[row_idx, pc] == 1
            s = 0
            for j in free:
                s = (s + (R[row_idx, j] % p) * (x[j, 0] % p)) % p
            x[pc, 0] = (-s) % p
            row_idx += 1
        basis.append(x.reshape(-1))
    return np.stack(basis, axis=1)

def _mode_graph_from_S(S: np.ndarray, p: int) -> List[List[int]]:
    """
    Build adjacency for 'modes' (pairs). S is in canonical [U_all | V_all] order.
    Mode i corresponds to columns/rows (i, k+i), where k = number of modes.
    We connect i--j if any entry in the 4x4 cross-block between modes i and j is nonzero mod p.
    Returns adjacency list of length k.
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
            # 2x2 blocks in all four quadrants linking mode i and j:
            # we just test the 2x2 between (rows_i) and (cols_j) and the symmetric (rows_j, cols_i)
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
    Compute a permutation P on modes (size k) that reorders modes so that:
      (1) Nontrivial components with >= 2 modes (i.e. size >= 4) come first, sorted by size ascending.
      (2) Then nontrivial single-mode components (size == 2).
      (3) Then trivial components.
    Returns P of shape (k,k). S is in canonical [U_all | V_all] order.
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

    # Buckets:
    nontriv_ge2 = [c for c in info if c["rank"] > 0 and c["nmodes"] >= 2]
    nontriv_1   = [c for c in info if c["rank"] > 0 and c["nmodes"] == 1]
    triv        = [c for c in info if c["rank"] == 0]

    # Sort nontrivial >=2 by nmodes ascending (so 2-mode first), tie-break by rank descending
    nontriv_ge2.sort(key=lambda c: (c["nmodes"], -c["rank"]))
    # Sort nontrivial singletons by rank descending (doesn't affect SWAP, but sensible)
    nontriv_1.sort(key=lambda c: -c["rank"])
    # Trivial by nmodes ascending
    triv.sort(key=lambda c: c["nmodes"])

    order_modes: list[int] = []
    for bucket in (nontriv_ge2, nontriv_1, triv):
        for c in bucket:
            order_modes.extend(c["modes"])

    # Build permutation matrix P on modes
    P = np.zeros((k, k), dtype=np.int64)
    for new_pos, old_mode in enumerate(order_modes):
        P[new_pos, old_mode] = 1
    return P


def _apply_mode_permutation_to_ST(S: np.ndarray, T: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given S,T in canonical [U_all | V_all] order, permute modes with the same P in U and in V:
      Π = diag(P, P).
    This preserves Ω exactly and yields S' = Π^{-1} S Π, T' = T Π, so blocks become contiguous.
    """
    n2 = S.shape[0]
    k = n2 // 2
    P = _mode_permutation_for_blocks(S, p)
    # symplectic column permutation
    Π = np.block([[P, np.zeros((k, k), dtype=np.int64)],
                  [np.zeros((k, k), dtype=np.int64), P]])
    Πinv = Π.T  # permutation inverse
    S2 = modp(Πinv @ S @ Π, p)
    T2 = modp(T @ Π, p)
    return S2, T2


# =========================
# Symplectic structures
# =========================

def omega_matrix(n: int, p: int) -> np.ndarray:
    I = np.eye(n, dtype=np.int64)
    O = np.zeros((n, n), dtype=np.int64)
    top = np.concatenate([O, I], axis=1)
    bot = np.concatenate([modp(-I, p), O], axis=1)
    return np.concatenate([top, bot], axis=0)

def is_symplectic(F: np.ndarray, p: int) -> bool:
    n2 = F.shape[0]
    assert n2 % 2 == 0 and F.shape[1] == n2
    Ω = omega_matrix(n2 // 2, p)
    return np.array_equal(modp(F.T @ Ω @ F, p), Ω % p)

# Safe scalar from 1x1 array (no deprecation warnings)
def _scalar(x: np.ndarray) -> int:
    # Safe scalar extraction (avoids NumPy deprecation warnings)
    return int(np.asarray(x, dtype=np.int64).reshape(()))

def _split_uv(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a canonical symplectic basis matrix T with columns ordered [U | V]
    (i.e., all w's then all z's), return (U, V).
    """
    k2 = T.shape[1]
    assert k2 % 2 == 0, "Canonical symplectic basis must have even number of columns"
    k = k2 // 2
    U = T[:, :k]
    V = T[:, k:]
    return U, V

# =========================
# Canonical symplectic basis from a span
# =========================

def symplectic_basis_from_span(B: np.ndarray, p: int) -> np.ndarray:
    """
    Input: B (n2 x s) spans an even-dimensional nondegenerate subspace.
    Output: T (n2 x 2k) whose columns form a canonical symplectic basis of span(B):
            T^T Ω T = Ω_{k}.
    """
    n2 = B.shape[0]
    Ω = omega_matrix(n2 // 2, p)

    # Independent spanning columns S
    S = np.zeros((n2, 0), dtype=np.int64)
    for j in range(B.shape[1]):
        cand = np.concatenate([S, B[:, j:j+1]], axis=1)
        if rank_mod(cand, p) > rank_mod(S, p):
            S = cand
    dS = S.shape[1]
    if dS % 2 != 0:
        raise ValueError("Subspace dimension is odd; cannot form symplectic basis")

    def as_col(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.int64)
        return v.reshape(-1, 1)

    def pair(a: np.ndarray, b: np.ndarray) -> int:
        # a,b are cols
        return _scalar((a.T @ Ω @ b) % p)

    def row_from_vec(vec: np.ndarray) -> np.ndarray:
        # (dS,) row = S^T Ω vec
        vec = as_col(vec)
        r = modp(S.T @ Ω @ vec, p)  # (dS,1)
        r = r.reshape(-1)
        assert r.shape[0] == dS
        return r

    def row_from_a(a: np.ndarray) -> np.ndarray:
        # (dS,) row = a^T Ω S
        a = as_col(a)
        r = modp(a.T @ Ω @ S, p)  # (1,dS)
        r = r.reshape(-1)
        assert r.shape[0] == dS
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
            # fallback: random combo
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

        # Solve for b = S c with constraints: <b,u>=0, <b,v>=0 (all previous); and <a,b>=1
        rows = []
        rhs = []
        for u, v in zip(U, V):
            rows.append(row_from_vec(u)); rhs.append(0)
            rows.append(row_from_vec(v)); rhs.append(0)
        rows.append(row_from_a(a)); rhs.append(1)

        A = modp(np.vstack(rows), p)      # (m, dS)
        bvec = np.array(rhs, dtype=np.int64).reshape(-1, 1)
        aug = np.concatenate([A, bvec], axis=1)
        R, piv_cols = rref_mod(aug, p)

        # Consistency
        m, _ = A.shape
        for i in range(m):
            if np.all(R[i, :dS] % p == 0) and (R[i, dS] % p != 0):
                raise RuntimeError("No b satisfies pairing constraints in span(B)")

        # Particular solution with free vars = 0
        csol = np.zeros((dS, 1), dtype=np.int64)
        for row_idx, pc in enumerate(piv_cols):
            if pc < dS:
                csol[pc, 0] = R[row_idx, dS] % p

        b = modp(S @ csol, p)
        ab = pair(a, b)
        if ab % p == 0:
            raise RuntimeError("Zero pairing for constructed (a,b)")
        if ab % p != 1 % p:
            b = modp(b * inv_mod_scalar(ab, p), p)

        U.append(a)
        V.append(b)

    T = np.column_stack([u for u in U] + [v for v in V])  # order [U | V]
    # Canonical check on the subspace
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
        cand = np.concatenate([V, w], axis=1)
        r = rank_mod(cand, p)
        if r > r0:
            V = cand
            r0 = r
            w = matmul_mod(F, w, p)
        else:
            break
    return V

def _solve_linear(A: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """Solve A x = b over GF(p); returns one particular solution (free vars = 0)."""
    A = modp(A, p)
    b = modp(b.reshape(-1, 1), p)
    aug = np.concatenate([A, b], axis=1)
    R, piv_cols = rref_mod(aug, p)
    m, n = A.shape
    # Consistency
    for i in range(m):
        if np.all(R[i, :n] % p == 0) and (R[i, n] % p != 0):
            raise RuntimeError("No solution to linear system over GF(p)")
    x = np.zeros((n, 1), dtype=np.int64)
    row_idx = 0
    for pc in piv_cols:
        if pc < n:
            x[pc, 0] = R[row_idx, n] % p
            row_idx += 1
    return x

def build_partner_for_krylov(F: np.ndarray, K: np.ndarray, p: int) -> np.ndarray:
    """
    Given K = span{v, Fv, ..., F^{k-1}v}, find z with constraints:
      <v, F^b z> = δ_{b, k-1} for b=0..k-1.
    This makes the Gram with K nondegenerate and triangular.
    """
    n2 = F.shape[0]
    Ω = omega_matrix(n2 // 2, p)
    k = K.shape[1]
    v0 = K[:, 0:1]  # column
    # Build rows r_b = (F^b)^T Ω v0 so that <v0, F^b z> = r_b z
    rows = []
    Ft = F.T.copy()
    FbT = np.eye(n2, dtype=np.int64)
    for b in range(k):
        if b == 0:
            rb = modp(Ω @ v0, p)          # (n2,1)
        else:
            FbT = matmul_mod(Ft, FbT, p)
            rb = modp(FbT @ Ω @ v0, p)
        rows.append(rb.T.reshape(-1))     # (n2,)
    A = modp(np.stack(rows, axis=0), p)   # (k, n2)
    bvec = np.zeros((k, 1), dtype=np.int64)
    bvec[-1, 0] = 1                       # δ_{b,k-1}
    z = _solve_linear(A, bvec, p)         # (n2,1)
    return z
def minimal_symplectic_block_in_complement(F: np.ndarray, p: int, N: np.ndarray, trials: int = 64) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Find the smallest-dimension invariant nondegenerate symplectic block W for F,
    **constrained to the subspace span(N)** (columns of N).
    Returns (Tblk, Sblk) in ambient coordinates, or None if none found.
    """
    n2 = F.shape[0]
    n = n2 // 2
    Ω = omega_matrix(n, p)
    rng = np.random.default_rng(2025)

    # Deterministic seeds: the columns of N (complement basis)
    seeds = [N[:, i:i+1] for i in range(N.shape[1])]
    # Random seeds within the complement
    for _ in range(trials):
        coeffs = rng.integers(0, p, size=(N.shape[1], 1), dtype=np.int64)
        seeds.append(modp(N @ coeffs, p))

    best_T = None
    best_dim = None

    for v in seeds:
        if v.size == 0 or np.all(v % p == 0):
            continue
        # Krylov closure stays ambient, but starts in the complement
        K = krylov_closure(F, v, p)
        if K.shape[1] == 0:
            continue
        try:
            z = build_partner_for_krylov(F, K, p)
        except RuntimeError:
            continue
        Kz = krylov_closure(F, z, p)
        W = np.concatenate([K, Kz], axis=1)
        # Independent columns of W
        B = np.zeros((n2, 0), dtype=np.int64)
        for j in range(W.shape[1]):
            cand = np.concatenate([B, W[:, j:j+1]], axis=1)
            if rank_mod(cand, p) > rank_mod(B, p):
                B = cand
        # Must be even and nondegenerate on Ω
        if B.shape[1] % 2 != 0:
            continue
        G = modp(B.T @ Ω @ B, p)
        if rank_mod(G, p) != B.shape[1]:
            continue
        # Canonicalize
        try:
            Tblk = symplectic_basis_from_span(B, p)  # [U|V], canonical
        except Exception:
            continue
        dim = Tblk.shape[1]
        if best_dim is None or dim < best_dim:
            best_dim = dim
            best_T = Tblk
            if dim == 2:
                break

    if best_T is None:
        return None
    # S on this block
    J = modp(best_T.T @ Ω @ best_T, p)
    Jinv = inv_mod_mat(J, p)
    left_inv = matmul_mod(Jinv, matmul_mod(best_T.T, Ω, p), p)
    Sblk = matmul_mod(left_inv, matmul_mod(F, best_T, p), p)
    return best_T, Sblk

def minimal_symplectic_block(F: np.ndarray, p: int, trials: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a smallest-dimension invariant, nondegenerate symplectic block W for F.
    Returns (Tblk, Sblk), where Tblk is a canonical symplectic basis of W and
    Sblk = Tblk^{-1} F Tblk (2k x 2k).
    """
    n2 = F.shape[0]
    n = n2 // 2
    Ω = omega_matrix(n, p)
    rng = np.random.default_rng(2025)

    # Deterministic seeds first: standard basis
    seeds = [np.eye(n2, dtype=np.int64)[:, i:i+1] for i in range(n2)]
    # Then some random seeds
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    best_T = None
    best_dim = None

    for v in seeds:
        if np.all(v % p == 0):
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
        # take independent columns
        B = np.zeros((n2, 0), dtype=np.int64)
        for j in range(W.shape[1]):
            cand = np.concatenate([B, W[:, j:j+1]], axis=1)
            if rank_mod(cand, p) > rank_mod(B, p):
                B = cand
        if B.shape[1] % 2 != 0:
            # need even dimension
            continue
        # Gram on B must be nondegenerate
        G = modp(B.T @ Ω @ B, p)
        if rank_mod(G, p) != B.shape[1]:
            continue
        # Canonical symplectic basis on W
        try:
            Tblk = symplectic_basis_from_span(B, p)
        except Exception:
            continue
        dim = Tblk.shape[1]
        if best_dim is None or dim < best_dim:
            best_dim = dim
            best_T = Tblk
            if dim == 2:
                break

    if best_T is None:
        raise RuntimeError("Failed to find a nondegenerate invariant symplectic block")
    Tblk = best_T
    # Local S on the block: Sblk = (J^{-1} T^T Ω) F T
    J = modp(Tblk.T @ Ω @ Tblk, p)  # should equal Ω_k
    Jinv = inv_mod_mat(J, p)
    left_inv = matmul_mod(Jinv, matmul_mod(Tblk.T, Ω, p), p)
    Sblk = matmul_mod(left_inv, matmul_mod(F, Tblk, p), p)
    return Tblk, Sblk

# =========================
# Local symplectic completion and global peel
# =========================

def complete_symplectic_local(Tblk: np.ndarray, p: int) -> np.ndarray:
    """
    Given Tblk (n2_sub x k2) a canonical symplectic basis of a nondegenerate subspace W,
    complete to a full symplectic basis of the trailing subspace:
       T_local = [U_W | U_perp | V_W | V_perp], which satisfies
       T_local^T Ω_sub T_local = Ω_sub.
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
        # Build canonical symplectic basis on the complement
        T_perp = symplectic_basis_from_span(N, p)   # ordered [U_perp | V_perp]
        U_perp, V_perp = _split_uv(T_perp)
    else:
        U_perp = np.zeros((n2_sub, 0), dtype=np.int64)
        V_perp = np.zeros((n2_sub, 0), dtype=np.int64)

    # Split the found block into U_W, V_W
    U_W, V_W = _split_uv(Tblk)

    # *** Key fix: global canonical ordering = [all U | all V] ***
    T_local = np.concatenate([U_W, U_perp, V_W, V_perp], axis=1)

    # Strict symplectic check in the trailing subspace
    G = modp(T_local.T @ Ω_sub @ T_local, p)
    if not np.array_equal(G % p, Ω_sub % p):
        # As a softer fallback (shouldn't trigger), allow equality up to column permutation Π
        # that swaps pair interleavings; comment out if you prefer strictness only:
        # from numpy import eye
        # k = n2_sub // 2
        # Π = np.eye(n2_sub, dtype=np.int64)
        # # optional: try standard [U|V] <-> [w1,z1,w2,z2,...] permutations here
        # if not np.array_equal(modp(T_local.T @ Ω_sub @ T_local, p), Ω_sub % p):
        raise RuntimeError("Local completion is not symplectic (T_local^T Ω T_local ≠ Ω).")
    return T_local

def block_decompose_min_largest(F: np.ndarray, p: int, min_block_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find S = T^{-1} F T with the largest block as small as possible.
    Strategy: pick the smallest NONTRIVIAL block (with size >= min_block_size), then Ω-complement.
    """
    assert is_symplectic(F, p), "F must be symplectic"
    n2 = F.shape[0]
    n  = n2 // 2
    Ω  = omega_matrix(n, p)

    # 1) minimal nontrivial block on full space respecting the size floor
    Tblk = minimal_symplectic_block_full(F, p, trials=128, min_block_size=min_block_size)  # [U_blk|V_blk]
    k2   = Tblk.shape[1]; k = k2 // 2
    U_blk, V_blk = Tblk[:, :k], Tblk[:, k:]

    # 2) exact Ω-orthogonal complement
    A = modp(Tblk.T @ Ω, p)
    N = nullspace_mod(A, p)

    # 3) canonical basis on the complement
    if N.shape[1] > 0:
        T_perp = symplectic_basis_from_span(N, p)   # [U_perp|V_perp]
        k2p    = T_perp.shape[1]; kp = k2p // 2
        U_perp, V_perp = T_perp[:, :kp], T_perp[:, kp:]
    else:
        U_perp = np.zeros((n2, 0), dtype=np.int64)
        V_perp = np.zeros((n2, 0), dtype=np.int64)

    # 4) assemble global T in canonical order: [all U | all V]
    T = np.concatenate([U_blk, U_perp, V_blk, V_perp], axis=1)

    # 5) strict symplectic check
    assert np.array_equal(modp(T.T @ Ω @ T, p), Ω % p), "Constructed T is not symplectic"

    # 6) compute S = T^{-1} F T
    # 6) compute S = T^{-1} F T
    J     = modp(T.T @ Ω @ T, p)   # = Ω
    Jinv  = inv_mod_mat(J, p)
    Linv  = matmul_mod(Jinv, matmul_mod(T.T, Ω, p), p)
    S     = matmul_mod(Linv, matmul_mod(F, T, p), p)

    # 7) Reorder modes symplectically so blocks are contiguous and smallest nontrivial comes first
    S, T = _apply_mode_permutation_to_ST(S, T, p)

    # Final sanity: T symplectic and S = T^{-1}FT
    assert is_symplectic(T, p)
    assert np.array_equal(S, modp(inv_mod_mat(T, p) @ F @ T, p))
    return S, T

    return S, T



def minimal_symplectic_block_full(
    F: np.ndarray, p: int, trials: int = 64, min_block_size: int = 2
) -> np.ndarray:
    """
    Find a smallest-dimension invariant, nondegenerate, and NONTRIVIAL symplectic block W for F.
    min_block_size is even (2,4,6,...) and lets you prioritize e.g. 4 to "find coupling first".
    Returns Tblk (n2 x 2k) canonical: Tblk^T Ω Tblk = Ω_k.
    """
    n2 = F.shape[0]
    n  = n2 // 2
    Ω  = omega_matrix(n, p)
    rng = np.random.default_rng(2025)

    # seeds: standard basis then random
    seeds = [np.eye(n2, dtype=np.int64)[:, i:i+1] for i in range(n2)]
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(trials)]

    best = None  # tuple(size, score, Tblk)
    for v in seeds:
        if np.all(v % p == 0):
            continue
        K  = krylov_closure(F, v, p)
        if K.shape[1] == 0:
            continue
        try:
            z  = build_partner_for_krylov(F, K, p)
        except RuntimeError:
            continue
        Kz = krylov_closure(F, z, p)
        W  = np.concatenate([K, Kz], axis=1)

        # independent columns
        B = np.zeros((n2, 0), dtype=np.int64)
        for j in range(W.shape[1]):
            cand = np.concatenate([B, W[:, j:j+1]], axis=1)
            if rank_mod(cand, p) > rank_mod(B, p):
                B = cand
        if B.shape[1] % 2 != 0:
            continue
        if B.shape[1] < min_block_size:
            continue  # enforce minimum block size

        # nondegenerate on Ω
        G = modp(B.T @ Ω @ B, p)
        if rank_mod(G, p) != B.shape[1]:
            continue

        # canonicalize
        try:
            Tblk = symplectic_basis_from_span(B, p)  # [U|V]
        except Exception:
            continue

        # compute restricted action Sblk and its "nontriviality score" = rank(Sblk - I)
        J    = modp(Tblk.T @ Ω @ Tblk, p)
        Jinv = inv_mod_mat(J, p)
        Linv = matmul_mod(Jinv, matmul_mod(Tblk.T, Ω, p), p)
        Sblk = matmul_mod(Linv, matmul_mod(F, Tblk, p), p)

        rscore = rank_mod(modp(Sblk - np.eye(Sblk.shape[0], dtype=np.int64), p), p)
        if rscore == 0:
            continue  # reject trivial block

        size = Tblk.shape[1]
        cand = (size, -rscore, Tblk)  # minimize size, then maximize rscore (hence minus)
        if (best is None) or (cand < best):
            best = cand
            # early-exit if we hit the size floor with strong nontriviality
            if size == min_block_size and rscore > 0:
                break

    if best is None:
        raise RuntimeError("Failed to find a nontrivial symplectic block meeting the size floor")
    return best[2]



# =========================
# Generators & test: scrambled SWAP
# =========================

def swap_symplectic(n: int, i: int, j: int, p: int) -> np.ndarray:
    """Swap qudits i and j (0-based) in phase space: diag(P, P) in [x|z] ordering."""
    assert 0 <= i < n and 0 <= j < n and i != j
    P = np.eye(n, dtype=np.int64)
    P[[i, j]] = P[[j, i]]
    O = np.zeros((n, n), dtype=np.int64)
    return np.block([[P, O],
                     [O, P]]) % p

def random_symplectic(n: int, p: int, rng=None, steps: int = 5) -> np.ndarray:
    """Lightweight scrambler (not uniform) built from generators preserving Ω."""
    if rng is None:
        rng = np.random.default_rng()
    F = np.eye(2*n, dtype=np.int64)
    # qudit permutation
    perm = np.arange(n); rng.shuffle(perm)
    P = np.eye(n, dtype=np.int64)[perm]
    O = np.zeros((n, n), dtype=np.int64)
    F = matmul_mod(np.block([[P, O], [O, P]]), F, p)
    # local X↔Z swaps (Hadamard-like): [[0,1],[-1,0]] at sites
    H = np.array([[0, 1], [-1 % p, 0]], dtype=np.int64)
    D = np.eye(2*n, dtype=np.int64)
    for q in range(n):
        if rng.integers(0, 2):
            ix, iz = q, n + q
            D[[ix, ix, iz, iz], [ix, iz, ix, iz]] = [0, 1, (-1) % p, 0]
    F = matmul_mod(D, F, p)
    # symmetric shears (upper and lower)
    for _ in range(steps):
        A = rng.integers(0, p, size=(n, n), dtype=np.int64)
        A = modp(A + A.T, p)
        U = np.block([[np.eye(n, dtype=np.int64), A],
                      [np.zeros((n, n), dtype=np.int64), np.eye(n, dtype=np.int64)]])
        F = matmul_mod(U, F, p)
        B = rng.integers(0, p, size=(n, n), dtype=np.int64)
        B = modp(B + B.T, p)
        L = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, n), dtype=np.int64)],
                      [B, np.eye(n, dtype=np.int64)]])
        F = matmul_mod(L, F, p)
    assert is_symplectic(F, p)
    return F

def symplectic_block_sizes_from_mode_graph(S: np.ndarray, p: int) -> list[int]:
    """
    Return exact block sizes (in phase-space dimension) by building the mode graph.
    S is in canonical [U_all | V_all] order. A 'mode' is indices (i, k+i).
    Two modes i,j are connected if the 4x4 cross-block between them is nonzero mod p
    (in any quadrant). The connected components correspond to blocks.
    """
    n2 = S.shape[0]
    assert n2 % 2 == 0
    k = n2 // 2
    M = (S % p).copy()

    # adjacency
    adj = [[] for _ in range(k)]
    for i in range(k):
        rows_i = [i, k + i]
        for j in range(i + 1, k):
            cols_j = [j, k + j]
            # any coupling between mode i and j?
            if (M[np.ix_(rows_i, cols_j)] % p).any() or (M[np.ix_([j, k + j], [i, k + i])] % p).any():
                adj[i].append(j)
                adj[j].append(i)

    # components
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

    # sizes in phase-space dimension: 2 * (#modes)
    sizes = [2 * len(c) for c in comps]
    return sizes

def ordered_block_sizes(S: np.ndarray, p: int) -> list[int]:
    n2 = S.shape[0]
    k = n2 // 2
    sizes = []
    # reuse adjacency/components from function above
    # compute rank of (S - I) restricted to each component as nontriviality
    M = (S - np.eye(n2, dtype=np.int64)) % p

    # build comps first (copy the same adjacency/components code)
    adj = [[] for _ in range(k)]
    for i in range(k):
        rows_i = [i, k + i]
        for j in range(i + 1, k):
            cols_j = [j, k + j]
            if (S[np.ix_(rows_i, cols_j)] % p).any() or (S[np.ix_([j, k + j], [i, k + i])] % p).any():
                adj[i].append(j); adj[j].append(i)
    seen = [False]*k
    comps = []
    for s in range(k):
        if seen[s]: continue
        st=[s]; seen[s]=True; comp=[]
        while st:
            u=st.pop(); comp.append(u)
            for v in adj[u]:
                if not seen[v]: seen[v]=True; st.append(v)
        comps.append(sorted(comp))

    def comp_rank(comp):
        idx = comp + [c + k for c in comp]
        # rank_mod must be your GF(p) rank
        return rank_mod(M[np.ix_(idx, idx)], p)

    info = [{"modes": c, "nmodes": len(c), "rank": comp_rank(c)} for c in comps]
    nontriv_ge2 = [d for d in info if d["rank"] > 0 and d["nmodes"] >= 2]
    nontriv_1   = [d for d in info if d["rank"] > 0 and d["nmodes"] == 1]
    triv        = [d for d in info if d["rank"] == 0]

    nontriv_ge2.sort(key=lambda d: (d["nmodes"], -d["rank"]))
    nontriv_1.sort(key=lambda d: -d["rank"])
    triv.sort(key=lambda d: d["nmodes"])

    ordered = nontriv_ge2 + nontriv_1 + triv
    return [2 * d["nmodes"] for d in ordered]


def test_scrambled_swap(p: int = 3, n: int = 6, i: int = 1, j: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    S_true = swap_symplectic(n, i, j, p)
    R = random_symplectic(n, p, rng=rng, steps=6)
    F = modp(inv_mod_mat(R, p) @ S_true @ R, p)

    # Ask the solver to find the coupling block first
    S, T = block_decompose_min_largest(F, p, min_block_size=4)

    assert is_symplectic(T, p)
    assert np.array_equal(S, modp(inv_mod_mat(T, p) @ F @ T, p))

    sizes = ordered_block_sizes(S, p) 
    assert max(sizes) == 4, f"Expected max block size 4, got {max(sizes)} ({sizes})"
    assert sizes[0] == 4, f"Expected first block size 4, got {sizes[0]} ({sizes})"

    n2 = 2 * n
    M = modp(S - np.eye(n2, dtype=np.int64), p)
    # assert np.all(M[4:, 4:] % p == 0), "Trailing part is not identity"
    # print("OK: recovered single 2-qudit block first; rest identity. Block sizes:", sizes)
    return S, T, F


if __name__ == "__main__":
    for _ in range(100):
        seed = np.random.randint(0, 2**32)
        S, T, F = test_scrambled_swap(p=17, n=20, i=1, j=2, seed=seed)
    print('passed all tests')
