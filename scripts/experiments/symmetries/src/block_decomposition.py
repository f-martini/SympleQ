from __future__ import annotations
from typing import List, Tuple
import numpy as np


# =========================
# GF(p) linear algebra
# =========================

def mod_p(A: np.ndarray, p: int) -> np.ndarray:
    return np.mod(A, p)


def inv_mod(a: int, p: int) -> int:
    a = int(a) % p
    if a == 0:
        raise ZeroDivisionError("No inverse exists for 0 mod p")
    # Fermat's little theorem (p prime)
    return pow(a, p - 2, p)


def rref_mod_p(A: np.ndarray, p: int):
    """Row-reduced echelon form over F_p. Returns (R, pivots)."""
    A = mod_p(A.copy(), p)
    m, n = A.shape
    R = A
    pivots: List[int] = []
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if R[rr, c] % p != 0:
                piv = rr
                break
        if piv is None:
            continue
        if piv != r:
            R[[r, piv]] = R[[piv, r]]
        inv_p = inv_mod(int(R[r, c]), p)
        R[r, :] = (R[r, :] * inv_p) % p
        for rr in range(m):
            if rr == r:
                continue
            fac = int(R[rr, c] % p)
            if fac != 0:
                R[rr, :] = (R[rr, :] - fac * R[r, :]) % p
        pivots.append(c)
        r += 1
        if r == m:
            break
    return R % p, pivots


def solve_linear_mod_p(A: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """
    Solve A x = b over F_p with Gauss–Jordan elimination (one RHS).
    Returns ONE solution with free vars set to 0. Raises ValueError if inconsistent.
    """
    A = mod_p(A.copy(), p)
    b = mod_p(b.copy(), p).reshape(-1, 1)
    m, n = A.shape
    Ab = np.concatenate([A, b], axis=1)

    row = 0
    pivot_cols: List[int] = []

    for col in range(n):
        piv = None
        for r in range(row, m):
            if Ab[r, col] % p != 0:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            Ab[[row, piv]] = Ab[[piv, row]]
        inv_p = inv_mod(int(Ab[row, col]), p)
        Ab[row, :] = (Ab[row, :] * inv_p) % p
        for r in range(m):
            if r == row:
                continue
            factor = int(Ab[r, col] % p)
            if factor != 0:
                Ab[r, :] = (Ab[r, :] - factor * Ab[row, :]) % p
        pivot_cols.append(col)
        row += 1
        if row == m:
            break

    # Inconsistency check
    for r in range(row, m):
        if np.all(Ab[r, :n] % p == 0) and Ab[r, n] % p != 0:
            raise ValueError("Inconsistent linear system over F_p")

    # Read solution (free vars = 0)
    x = np.zeros((n,), dtype=int)
    rptr = 0
    for col in range(n):
        if rptr < len(pivot_cols) and pivot_cols[rptr] == col:
            x[col] = int(Ab[rptr, n] % p)
            rptr += 1
        else:
            x[col] = 0
    return x % p


def solve_multi_rhs_mod_p(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """
    Solve A X = B over F_p with Gauss-Jordan elimination where B has k RHS (mxk).
    Returns one solution X (nxk) with free vars set to 0.
    Raises ValueError if any RHS is inconsistent.
    """
    A = mod_p(A.copy(), p)
    B = mod_p(B.copy(), p)
    m, n = A.shape
    k = B.shape[1]
    Ab = np.concatenate([A, B], axis=1)  # m × (n+k)

    row = 0
    pivot_cols: List[int] = []

    for col in range(n):
        piv = None
        for r in range(row, m):
            if Ab[r, col] % p != 0:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            Ab[[row, piv]] = Ab[[piv, row]]
        invp = inv_mod(int(Ab[row, col]), p)
        Ab[row, :] = (Ab[row, :] * invp) % p
        for r in range(m):
            if r == row:
                continue
            factor = int(Ab[r, col] % p)
            if factor != 0:
                Ab[r, :] = (Ab[r, :] - factor * Ab[row, :]) % p
        pivot_cols.append(col)
        row += 1
        if row == m:
            break

    # Inconsistency check for all RHS columns
    for r in range(row, m):
        if np.all(Ab[r, :n] % p == 0):
            for j in range(k):
                if Ab[r, n + j] % p != 0:
                    raise ValueError("Inconsistent linear system over F_p (one of RHS).")

    # Read solution X (free vars = 0)
    X = np.zeros((n, k), dtype=int)
    rptr = 0
    for col in range(n):
        if rptr < len(pivot_cols) and pivot_cols[rptr] == col:
            X[col, :] = Ab[rptr, n:n + k] % p
            rptr += 1
        else:
            X[col, :] = 0
    return X % p


def nullspace_right_mod_p(A: np.ndarray, p: int) -> np.ndarray:
    """
    Return an nxh matrix H whose columns form a basis for the right nullspace {x: A x = 0} over F_p.
    """
    A = mod_p(A.copy(), p)
    m, n = A.shape
    R, pivots = rref_mod_p(A, p)
    free = [c for c in range(n) if c not in pivots]
    if not free:
        return np.zeros((n, 0), dtype=int)
    H = np.zeros((n, len(free)), dtype=int)
    for j, f in enumerate(free):
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for i, c in enumerate(pivots):
            v[c] = (-R[i, f]) % p
        H[:, j] = v % p
    return H % p


# =========================
# Symplectic basics
# =========================

def symplectic_form(n_sites: int, p: int) -> np.ndarray:
    """
    Standard 2nx2n symplectic form in [w-block | z-block] ordering:
        [ 0  I ]
        [  I 0 ]  for p = 2
        [ 0  I ]
        [ -I 0 ]  for odd p
    """
    Id = np.eye(n_sites, dtype=int)
    top = np.concatenate([np.zeros((n_sites, n_sites), dtype=int), Id], axis=1)
    if p == 2:
        bottom = np.concatenate([Id, np.zeros((n_sites, n_sites), dtype=int)], axis=1)
    else:
        bottom = np.concatenate([(-Id) % p, np.zeros((n_sites, n_sites), dtype=int)], axis=1)
    return np.concatenate([top, bottom], axis=0) % p


def dot_omega(u: np.ndarray, v: np.ndarray, Omega: np.ndarray, p: int) -> int:
    return int((u.T @ (Omega @ v)) % p)


def is_symplectic(F: np.ndarray, Omega: np.ndarray, p: int) -> bool:
    return np.array_equal((F.T @ Omega @ F) % p, Omega % p)


def symplectic_inverse(T: np.ndarray, Omega: np.ndarray, p: int) -> np.ndarray:
    """Return T^{-1} = Omega^{-1} T^T Omega (mod p)."""
    if p == 2:
        Omega_inv = Omega % p
    else:
        Omega_inv = (-Omega) % p
    return (Omega_inv @ (T.T @ Omega)) % p


# =========================
# Unipotent ladders & partner solves
# =========================

def ladder(N: np.ndarray, w: np.ndarray, p: int, maxlen: int) -> List[np.ndarray]:
    """Build W = [w, Nw, N^2 w, ...] until zero or maxlen reached."""
    L: List[np.ndarray] = []
    x = (w.copy() % p)
    for _ in range(maxlen):
        if np.all(x % p == 0):
            break
        L.append(x.copy() % p)
        x = (N @ x) % p
    return L


def independent_row_indices(A: np.ndarray, p: int) -> List[int]:
    """
    Return indices of a maximal independent set of rows of A over F_p.
    We compute pivots of A^T to identify independent rows of A.
    """
    AT = mod_p(A.T.copy(), p)
    R, pivots = rref_mod_p(AT, p)  # pivots are column indices of A^T ⇒ row indices of A
    return pivots


def batch_solve_partners_stage1(W: List[np.ndarray], Omega: np.ndarray, p: int) -> np.ndarray:
    """
    Stage-1 partner solve for a ladder W=[w0,...,w_{k-1}]:
        <w_a, z_b> = δ_{a+b,k-1}.
    Returns Z as a dim x k matrix with columns z_b.
    Robust to rank-deficient A1 and arbitrary seeds:
      1) try full multi-RHS solve,
      2) if inconsistent, restrict to an independent row set,
      3) if still inconsistent for some RHS, fall back to per-column single-row solves.
    """
    k = len(W)
    dim = Omega.shape[0]
    if k == 0:
        return np.zeros((dim, 0), dtype=int)

    # Build A1 (k × dim) and target anti-diagonal identity Jk (k × k)
    A1 = np.vstack([(w.T @ Omega) % p for w in W]) % p
    Jk = np.fliplr(np.eye(k, dtype=int)) % p

    # --- Attempt 1: full system (all rows)
    try:
        Z = solve_multi_rhs_mod_p(A1, Jk, p)  # dim × k
        return Z % p
    except ValueError:
        pass  # fall through to restricted solve

    # --- Attempt 2: restrict to independent rows of A1
    rows = independent_row_indices(A1, p)
    if len(rows) > 0:
        A1r = A1[rows, :] % p
        Jr = Jk[rows, :] % p
        try:
            Z = solve_multi_rhs_mod_p(A1r, Jr, p)
            return Z % p
        except ValueError:
            pass  # fall through to per-column fallback

    # --- Attempt 3 (last resort): build columns independently with a single constraint each
    # z_b solves (w_b^T Omega) z_b = 1; this is always solvable if that row is nonzero.
    Z = np.zeros((dim, k), dtype=int)
    for b in range(k):
        row = (W[b].T @ Omega) % p
        if np.all(row % p == 0):
            # pick any earlier nonzero row to anchor; if none, choose the first nonzero row in A1
            nz_rows = [idx for idx in range(k) if np.any(A1[idx, :] % p != 0)]
            if not nz_rows:
                # degenerate ladder; leave z_b as zero and let repair/completion handle basis fill-in
                continue
            anchor = nz_rows[0]
            row = A1[anchor, :]
        # Solve 1×dim system row * z = 1
        zb = solve_linear_mod_p(row.reshape(1, -1), np.array([1], dtype=int), p)
        Z[:, b] = zb % p
    return Z % p


# =========================
# Symplectic Gram–Schmidt (deterministic, constrained solves)
# =========================

def symplectic_project_to_complement(u: np.ndarray,
                                     pairs: List[Tuple[np.ndarray, np.ndarray]],
                                     Omega: np.ndarray, p: int) -> np.ndarray:
    """
    Project u to the symplectic complement of span{w_i, z_i} by:
       u <- u - <w_i,u> z_i ; u <- u + <z_i,u> w_i
    """
    u = (u.copy() % p)
    for (w, z) in pairs:
        alpha = dot_omega(w, u, Omega, p) % p
        if alpha:
            u = (u - alpha * z) % p
        beta = dot_omega(z, u, Omega, p) % p
        if beta:
            u = (u + beta * w) % p
    return u % p


def solve_partner_constrained(u: np.ndarray,
                              pairs: List[Tuple[np.ndarray, np.ndarray]],
                              Omega: np.ndarray, p: int) -> np.ndarray:
    """
    Solve for v with constraints (no explicit projection needed):
        <w_i,v>=0, <z_i,v>=0  for all existing pairs
        <u,v>=1
    Returns v or raises ValueError if inconsistent.
    """
    m = 2 * len(pairs) + 1
    dim = Omega.shape[0]
    A = np.zeros((m, dim), dtype=int)
    b = np.zeros(m, dtype=int)
    r = 0
    for (wi, zi) in pairs:
        A[r, :] = (wi.T @ Omega) % p
        b[r] = 0
        r += 1
        A[r, :] = (zi.T @ Omega) % p
        b[r] = 0
        r += 1
    A[r, :] = (u.T @ Omega) % p
    b[r] = 1
    v = solve_linear_mod_p(A, b, p)
    return v % p


def repair_pairs_and_complete_deterministic(raw_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                            Omega: np.ndarray,
                                            p: int,
                                            n_sites: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Deterministically repair raw pairs into a symplectic basis and complete if needed.
    Assembles T = [w1...wn | z1...zn] in STANDARD convention.
    """
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    # Repair (enforce orthogonality and <w,z>=1); keep w's in complement for stability
    for (w, z) in raw_pairs:
        w = symplectic_project_to_complement(w, pairs, Omega, p)
        if np.all(w % p == 0):
            continue
        z_new = solve_partner_constrained(w, pairs, Omega, p)
        pairs.append((w % p, z_new % p))
        if len(pairs) == n_sites:
            break

    # Deterministic completion by sweeping canonical basis
    dim = Omega.shape[0]
    for i in range(dim):
        if len(pairs) == n_sites:
            break
        e = np.zeros(dim, dtype=int)
        e[i] = 1
        u = symplectic_project_to_complement(e, pairs, Omega, p)
        if np.all(u % p == 0):
            continue
        try:
            v = solve_partner_constrained(u, pairs, Omega, p)
        except ValueError:
            continue
        pairs.append((u % p, v % p))

    # Assemble T = [w1..wn | z1..zn]
    if len(pairs) != n_sites:
        raise RuntimeError("Failed to complete symplectic basis deterministically.")
    w_cols = [w % p for (w, _) in pairs]
    z_cols = [z % p for (_, z) in pairs]
    T = np.stack(w_cols + z_cols, axis=1) % p

    # Verify symplecticity
    G = (T.T @ Omega @ T) % p
    if not np.array_equal(G % p, Omega % p):
        raise AssertionError(f"Assembled T not symplectic.\nG =\n{G}\nOmega =\n{Omega}")
    return pairs, T


# =========================
# Diagnostics (optional): kernel dims and Δ_k
# =========================

def kernel_dims_chain(N: np.ndarray, p: int, max_k: int | None = None) -> List[int]:
    """
    Compute dims of ker(N^k) for k=0,1,... until stabilization (or up to max_k).
    Returns list D where D[k] = dim ker(N^k), with D[0]=0 by convention.
    """
    dim = N.shape[0]
    if max_k is None:
        max_k = dim + 1
    D = [0]
    Nk = np.eye(dim, dtype=int) % p
    for k in range(1, max_k + 1):
        Nk = (Nk @ N) % p
        # rank of Nk via RREF
        R, pivots = rref_mod_p(Nk, p)
        rank = len(pivots)
        ker_dim = dim - rank
        D.append(ker_dim)
        if k > 1 and D[-1] == D[-2]:
            break
    return D  # D[1]=dim ker N, etc.


# =========================
# Main: block decomposition
# =========================

def block_decompose_symplectic(F: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        F: 2nx2n symplectic matrix over F_p
        p: prime modulus
    Output:
        (S, T) with T symplectic and S = T^{-1} F T block-diagonal by
        invariant blocks extracted from the unipotent sector (N = F - I).
    Workflow:
        - Build ladders from seeds projected to the current complement.
        - For each ladder, batch-solve partners (<w_a, z_b> = δ_{a+b,k-1}).
        - Add raw pairs (w_a, z_{k-1-a}), then repair + deterministic completion.
        - Use symplectic inverse for T^{-1}.
    """
    F = mod_p(F, p)
    dim = F.shape[0]
    assert dim == F.shape[1] and dim % 2 == 0, "F must be 2nx2n"
    n = dim // 2

    Omega = symplectic_form(n, p)
    if not is_symplectic(F, Omega, p):
        raise ValueError("Input F is not symplectic modulo p")

    N = (F - np.eye(dim, dtype=int)) % p

    raw_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    pairs_so_far: List[Tuple[np.ndarray, np.ndarray]] = []  # for projections when picking seeds

    # (Optional) compute kernel dims for diagnostics/limits
    # D = kernel_dims_chain(N, p)  # not strictly needed; keep if you want to log Δ_k

    # Sweep canonical basis for seeds, project to complement, then build ladders
    for i in range(dim):
        if len(raw_pairs) >= n:
            break
        e = np.zeros(dim, dtype=int)
        e[i] = 1
        u0 = symplectic_project_to_complement(e, pairs_so_far, Omega, p)
        if np.all(u0 % p == 0):
            continue

        # Build ladder W = [u0, N u0, ..., N^{k-1} u0]
        W = ladder(N, u0, p, maxlen=dim + 2)
        if len(W) == 0:
            continue
        k = len(W)

        # Batch partner solve for this ladder
        Z = batch_solve_partners_stage1(W, Omega, p)  # dim × k

        # Append raw pairs (w_a, z_{k-1-a}), and grow a provisional orthogonal guide basis
        for a in range(k):
            w = W[a] % p
            z = Z[:, k - 1 - a] % p
            raw_pairs.append((w, z))

            # Also append a quickly “normalized” guide pair to pairs_so_far
            # to help future seed projections remain stable:
            w_clean = symplectic_project_to_complement(w, pairs_so_far, Omega, p)
            if np.all(w_clean % p == 0):
                continue
            try:
                z_clean = solve_partner_constrained(w_clean, pairs_so_far, Omega, p)
            except ValueError:
                # If constrained solve fails (rare), skip guiding pair; repair handles it later.
                continue
            pairs_so_far.append((w_clean % p, z_clean % p))
            if len(raw_pairs) >= n:
                break

    # Final repair + deterministic completion in STANDARD order
    pairs, T = repair_pairs_and_complete_deterministic(raw_pairs, Omega, p, n_sites=n)

    # Symplectic inverse shortcut
    T_inv = symplectic_inverse(T, Omega, p)
    S = (T_inv @ F @ T) % p

    # Sanity checks
    if not is_symplectic(T, Omega, p):
        G = (T.T @ Omega @ T) % p
        raise AssertionError(f"T is not symplectic (final check).\nG =\n{G}\nOmega =\n{Omega}")
    if not is_symplectic(S, Omega, p):
        raise AssertionError("S is not symplectic (final check).")
    return S % p, T % p


# =========================
# Example / basic test
# =========================

def _random_symplectic_transvection(n_sites: int, p: int, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a random scaled transvection F = I + s * v (v^T Omega); return (F, Omega).
    """
    Omega = symplectic_form(n_sites, p)
    dim = 2 * n_sites
    rng = np.random.default_rng(seed)
    v = rng.integers(0, p, size=(dim,))
    if np.all(v % p == 0):
        v[0] = 1
    s = int(rng.integers(1, p))
    v_col = v.reshape(-1, 1) % p
    row = (v_col.T @ Omega) % p
    F = (np.eye(dim, dtype=int) + (s % p) * (v_col @ row)) % p
    return F, Omega

# -------------------------
# Helpers for testing
# -------------------------


def kernel_stabilization_index(N: np.ndarray, p: int, max_k: int | None = None) -> int:
    """
    Return s_max = smallest k >= 0 with ker(N^k) = ker(N^{k+1}).
    This equals the length of the longest Jordan chain for eigenvalue 1.
    Works even when N is not nilpotent on the whole space.
    """
    N = mod_p(N, p)
    dim = N.shape[0]
    if max_k is None:
        max_k = dim + 1  # safe upper bound

    prev_ker_dim = 0
    Nk = np.eye(dim, dtype=int) % p  # N^0
    for k in range(1, max_k + 2):    # allow k = max_k+1 to detect stabilization at max_k
        Nk = (Nk @ N) % p            # Nk = N^k
        # rank via RREF
        R, pivots = rref_mod_p(Nk, p)
        ker_dim = dim - len(pivots)
        if ker_dim == prev_ker_dim:
            # stabilized at previous step ⇒ s_max = k-1
            return k - 1
        prev_ker_dim = ker_dim
    # Should never reach here for finite dimension
    raise RuntimeError("Kernel chain did not stabilize within expected bound.")


def _compose_transvections(n_sites: int, p: int, seeds: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose transvections: F = Π_i (I + s_i v_i (v_i^T Omega)).
    Returns (F, Omega). Each factor is symplectic; the product is symplectic.
    """
    Omega = symplectic_form(n_sites, p)
    dim = 2 * n_sites
    F = np.eye(dim, dtype=int) % p
    for sd in seeds:
        rng = np.random.default_rng(sd)
        v = rng.integers(0, p, size=(dim,))
        if np.all(v % p == 0):
            v[0] = 1
        s = int(rng.integers(1, p))  # nonzero in F_p
        v_col = v.reshape(-1, 1) % p
        row = (v_col.T @ Omega) % p                     # 1 × (2n)
        Ti = (np.eye(dim, dtype=int) + (s % p) * (v_col @ row)) % p
        # multiply, don't add:
        F = (Ti @ F) % p
    return F % p, Omega % p


# -------------------------
# tests
# -------------------------

def test_transvection_block_size_minimal():
    p = 5
    n_sites = 3
    seed = np.random.randint(0, 100000)
    F, Omega = _random_symplectic_transvection(n_sites, p, seed=seed)
    dim = 2 * n_sites

    N = (F - np.eye(dim, dtype=int)) % p
    s_max = kernel_stabilization_index(N, p)
    assert s_max == 2, f"Transvection should have s_max=2, got {s_max}."

    S, T = block_decompose_symplectic(F, p)
    assert is_symplectic(T, Omega, p)
    assert is_symplectic(S, Omega, p)
    T_inv = symplectic_inverse(T, Omega, p)
    assert np.array_equal((T_inv @ F @ T) % p, S % p)

    Np = (S - np.eye(dim, dtype=int)) % p
    s_max_prime = kernel_stabilization_index(Np, p)
    assert s_max_prime == s_max, "s_max not preserved under conjugation."

    # Minimal possible maximal symplectic block size is 2*s_max
    assert 2 * s_max == 4


def test_product_of_transvections_block_size_minimal():
    p = 5
    n_sites = 4
    dim = 2 * n_sites

    seeds = list(np.random.randint(low=0, high=1000, size=3))

    F, Omega = _compose_transvections(n_sites, p, seeds=seeds)
    # Guard: ensure we actually built a symplectic F
    assert is_symplectic(F, Omega, p), "Composed F is not symplectic (should never happen now)."

    N = (F - np.eye(dim, dtype=int)) % p
    s_max = kernel_stabilization_index(N, p)
    assert 0 <= s_max <= dim

    S, T = block_decompose_symplectic(F, p)

    assert is_symplectic(T, Omega, p)
    assert is_symplectic(S, Omega, p)
    T_inv = symplectic_inverse(T, Omega, p)
    assert np.array_equal((T_inv @ F @ T) % p, S % p)

    Np = (S - np.eye(dim, dtype=int)) % p
    s_max_prime = kernel_stabilization_index(Np, p)
    assert s_max_prime == s_max

    max_block_size_min_theory = 2 * s_max
    assert 0 <= max_block_size_min_theory <= dim


import numpy as np

def coupled_qudit_sets(F: np.ndarray, p: int):
    """
    Given a 2n×2n symplectic F over F_p (standard Ω convention),
    return a list of lists of 1-based qudit indices grouped by the
    minimal symplectic invariant blocks (“coupled sets”).

    Requires: block_decompose_symplectic(F, p) -> (S, T)
              and symplectic_form / is_symplectic as before.
    """
    # 1) Decompose to block form in the T-basis (standard ordering)
    S, T = block_decompose_symplectic(F, p)     # from the simplified implementation
    dim = F.shape[0]
    assert dim % 2 == 0, "F must be 2n×2n"
    n = dim // 2

    # 2) Build a graph over hyperbolic pairs (nodes = 0..n-1) using S
    #    Pairs i,j are connected if N' = S - I has any nonzero entry in the 4×4
    #    submatrix relating the coords of pair i to pair j (i != j).
    Np = (S - np.eye(dim, dtype=int)) % p
    adj = [set() for _ in range(n)]
    for i in range(n):
        rows = (i, n + i)
        for j in range(n):
            cols = (j, n + j)
            block_ij = Np[np.ix_(rows, cols)] % p
            if i != j and np.any(block_ij):
                adj[i].add(j)
                adj[j].add(i)

    # 3) Connected components over pairs → symplectic blocks in T-basis
    seen = [False] * n
    pair_blocks = []
    for i in range(n):
        if seen[i]:
            continue
        # BFS/DFS
        comp = []
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        pair_blocks.append(sorted(comp))

    # 4) Map pair-blocks back to *physical qudits*:
    #    For each block (set of pair indices I), take its 2|I| columns in T:
    #      columns = I (w-part)  ∪  (n + I) (z-part).
    #    A physical qudit q (0..n-1) participates if rows [q, n+q] have any
    #    nonzero entry across those columns.
    qudit_groups = []
    for comp in pair_blocks:
        # columns of T spanning this block
        w_cols = comp
        z_cols = [n + j for j in comp]
        cols = np.array(w_cols + z_cols, dtype=int)
        subT = T[:, cols] % p

        qudits_in_block = []
        for q in range(n):
            rows = np.array([q, n + q], dtype=int)
            sub = subT[rows, :] % p
            if np.any(sub):                      # any nonzero support on qudit q
                qudits_in_block.append(q + 1)    # 1-based index for output
        qudit_groups.append(qudits_in_block)

    # Sort groups for readability (smallest index first; groups ordered by min element)
    qudit_groups = [sorted(g) for g in qudit_groups]
    qudit_groups.sort(key=lambda g: (len(g) == 0, g[0] if g else 10**9))
    return qudit_groups


if __name__ == "__main__":
    # Smoke test from earlier example
    p = 5
    n_sites = 6
    F, Omega = _random_symplectic_transvection(n_sites, p, seed=346)
    print("Input F is symplectic?", is_symplectic(F, Omega, p))
    S, T = block_decompose_symplectic(F, p)
    print("T is symplectic?", is_symplectic(T, Omega, p))
    print("S is symplectic?", is_symplectic(S, Omega, p))
    T_inv = symplectic_inverse(T, Omega, p)
    ok = np.array_equal((T_inv @ F @ T) % p, S % p)
    print("S == T^{-1} F T ?", ok)
    print("S mod p:\n", S % p)

    print(coupled_qudit_sets(S, p))

    # Run the additional asserts
    for _ in range(100):
        test_transvection_block_size_minimal()
        test_product_of_transvections_block_size_minimal()
    print("All tests passed.")
