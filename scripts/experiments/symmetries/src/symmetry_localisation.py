import numpy as np
import galois

# ---------- Basic GF(p) linear algebra helpers ----------

def _has_nonzero_pairing_in_L(Nw, v, Omega, GF):
    # returns True iff ∃ g ∈ L with <v,g> ≠ 0, i.e. Nw^T Ω v ≠ 0
    a = (Nw.T @ Omega @ v).reshape(-1)
    for ai in a:
        if ai != GF(0):
            return True
    return False
def partner_orth_to_W(e, W, Omega, GF):
    """
    Given e and the already-placed columns W = [E_0, G_0, ..., E_{i-1}, G_{i-1}],
    find g such that:
        <w, g> = 0  for all w in W     (i.e., g ∈ W^⊥)
        <e, g> = 1
    Solve the single linear system over GF:  A g = b
    where A has rows [w^T Ω] and the last row [e^T Ω], with b = [0,...,0,1]^T.
    """
    if not W:
        A = e.T @ Omega    # 1 x d
        b = GF([1])
    else:
        rows = [(w.T @ Omega) for w in W]          # each 1 x d
        rows.append(e.T @ Omega)                    # last row imposes <e,g>=1
        A = np.vstack(rows)                         # m x d
        b = GF.Zeros(A.shape[0]); b[-1] = GF(1)     # [0,...,0,1]^T
    y = gf_solve(A, b, GF).reshape(-1, 1)
    return y
def _partner_in_L_any(e, Nw, Omega, GF):
    # like partner_in_L, but assumes a solution exists (Nw^T Ω e has some nonzero)
    a = (Nw.T @ Omega @ e).reshape(-1)
    for i, ai in enumerate(a):
        if ai != GF(0):
            y = GF.Zeros(Nw.shape[1]); y[i] = GF(1) / ai
            return Nw @ y.reshape(-1, 1)
    raise RuntimeError("Internal: expected a nonzero pairing in L but found none.")

def _repair_tail_to_symplectic(E_cols, G_cols, s, Omega, GF, rng=None):
    """
    Keep the first s pairs (kernel pairs) fixed. Rebuild the remaining pairs
    by symplectic Gram–Schmidt inside successive orthogonal complements so that
    Fs becomes exactly symplectic.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(E_cols)
    E_out = [None]*n
    G_out = [None]*n

    # 1) Copy the first s (already-kernel) pairs
    for i in range(s):
        E_out[i] = E_cols[i]
        G_out[i] = G_cols[i]

    # Running set of placed columns
    W = []
    for i in range(s):
        W.extend([E_out[i], G_out[i]])

    # 2) Repair/build the remaining pairs
    for i in range(s, n):
        Nw = orth_complement_basis(W, Omega, GF)     # L = W^⊥

        # Pick an E_i ∈ L that has nonzero pairing within L
        candidates = []
        if E_cols[i] is not None: candidates.append(E_cols[i])
        if G_cols[i] is not None: candidates.append(G_cols[i])

        e = None
        for cand in candidates:
            # orthogonalize candidate to W
            v = orthogonalize_to_pairs(Omega, E_out[:i], G_out[:i], cand, GF)
            if np.all(v == GF(0)):  # collapsed; skip
                continue
            if _has_nonzero_pairing_in_L(Nw, v, Omega, GF):
                e = v
                break

        # If candidates fail, draw a random vector in L until it works
        if e is None:
            t = Nw.shape[1]
            for _ in range(256):
                y = GF(rng.integers(0, GF.characteristic, size=t))
                if np.all(y == GF(0)):
                    continue
                v = Nw @ y.reshape(-1,1)
                if _has_nonzero_pairing_in_L(Nw, v, Omega, GF):
                    e = v
                    break
            if e is None:
                # deterministic fallback: scan L’s basis
                for j in range(Nw.shape[1]):
                    v = Nw[:, [j]]
                    if _has_nonzero_pairing_in_L(Nw, v, Omega, GF):
                        e = v
                        break
        if e is None:
            raise RuntimeError("Repair: could not find an E vector with nonzero pairing in L.")

        # Partner g ∈ L with <e,g> = 1
        g = _partner_in_L_any(e, Nw, Omega, GF)

        E_out[i], G_out[i] = e, g
        W.extend([e, g])

    return E_out, G_out

def span_rank(W, GF):
    """Rank of the span of columns in list W (each d x 1), over GF."""
    if not W:
        return 0
    M = GF(np.hstack(W))
    return gf_rank(M, GF)

def in_span(W, u, GF):
    """Check if u is in span(W), over GF."""
    if not W:
        return False
    M = GF(np.hstack(W + [u]))
    return gf_rank(M, GF) == gf_rank(GF(np.hstack(W)), GF)

def pick_kernel_not_in_span(BK, W, GF, rng, max_tries=256):
    """
    BK: d x k matrix of a kernel basis. Return u in K \ span(W).
    Try basis vectors first, then random combinations.
    """
    # Try basis columns
    for j in range(BK.shape[1]):
        u = BK[:, [j]]
        if not in_span(W, u, GF):
            return u
    # Random combinations
    for _ in range(max_tries):
        coeffs = GF(rng.integers(0, GF.characteristic, size=BK.shape[1]))
        if np.all(coeffs == GF(0)):
            continue
        u = BK @ coeffs.reshape(-1, 1)
        if not np.all(u == GF(0)) and not in_span(W, u, GF):
            return u
    raise RuntimeError("pick_kernel_not_in_span: K ⊆ span(W); can’t place required kernel column now.")
def omega_matrix(n, GF):
    """Standard symplectic form Ω on 2n over GF(p)."""
    I, Z = GF.Identity(n), GF.Zeros((n, n))
    top = np.hstack((Z, I))
    bot = np.hstack((-I, Z))  # in char-2, -I == I
    return np.vstack((top, bot))

def gf_rref(A, GF):
    A = GF(A)
    m, n = A.shape
    R = A.copy()
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if R[rr, c] != GF(0):
                piv = rr; break
        if piv is None:
            continue
        if piv != r:
            R[[r, piv], :] = R[[piv, r], :]
        inv = GF(1) / R[r, c]
        R[r, :] *= inv
        for rr in range(m):
            if rr != r and R[rr, c] != GF(0):
                R[rr, :] -= R[rr, c] * R[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return R, pivots

def gf_rank(A, GF):
    _, piv = gf_rref(A, GF)
    return len(piv)

def gf_nullspace(A, GF):
    A = GF(A)
    m, n = A.shape
    R, piv = gf_rref(A, GF)
    piv = set(piv)
    free = [j for j in range(n) if j not in piv]
    if not free:
        return []
    basis = []
    # R is in RREF
    for f in free:
        x = GF.Zeros(n)
        x[f] = GF(1)
        row = 0
        for c in range(n):
            if c in piv:
                s = GF(0)
                for j in range(c + 1, n):
                    if R[row, j] != GF(0) and x[j] != GF(0):
                        s += R[row, j] * x[j]
                x[c] = -s
                row += 1
                if row == m:
                    break
        basis.append(x.reshape(n, 1))
    return basis

def gf_solve(A, b, GF):
    A = GF(A); b = GF(b).reshape(-1, 1)
    m, n = A.shape
    M = np.hstack((A, b))
    R, piv = gf_rref(M, GF)
    # consistency
    for i in range(m):
        if np.all(R[i, :n] == GF(0)) and R[i, n] != GF(0):
            raise ValueError("Inconsistent linear system over GF(p).")
    x = GF.Zeros(n)
    # reconstruct pivot variables (R is RREF)
    row = 0
    for c in range(n):
        # pivot row has leading 1 at column c and zeros before it
        if row < m and R[row, c] == GF(1) and np.all(R[row, :c] == GF(0)):
            s = GF(0)
            for j in range(c + 1, n):
                s += R[row, j] * x[j]
            x[c] = R[row, n] - s
            row += 1
    return x

# ---------- Symplectic utilities ----------

def is_symplectic(A, Omega):
    return (A.T @ Omega @ A == Omega).all()

def omega_pairing(Omega, u, v):
    return (u.T @ Omega @ v)[0, 0]

def orthogonalize_to_pairs(Omega, E_list, G_list, v, GF):
    """Make v orthogonal to all existing pairs <v,Gi>=0 and <v,Ei>=0."""
    if not E_list:
        return v
    vv = v.copy()
    for Ei, Gi in zip(E_list, G_list):
        vv -= omega_pairing(Omega, vv, Gi) * Ei
        vv += omega_pairing(Omega, vv, Ei) * Gi
    return vv

# ---------- Main construction ----------
# ---- Inside-K symplectic (Witt) decomposition --------------------------------

# --- Main constructor: S near identity; first r columns deviate ---
def decompose_kernel_symplectically(BK, Omega, GF):
    """
    BK: d x k GF matrix with columns a basis for K = ker(F - I).
    Returns (pairs, radicals) with pairs = [(u_i,v_i)], <u_i,v_i>=1, mutually orthogonal,
    and radicals = [w_j] with <w_j, BK> = 0.
    """
    V = [BK[:, [j]] for j in range(BK.shape[1])]
    pairs, radicals = [], []
    def pairing(a, b):
        return (a.T @ Omega @ b)[0, 0]

    while V:
        u = V.pop(0)
        idx = None
        for t, cand in enumerate(V):
            if pairing(u, cand) != GF(0):
                idx = t
                break
        if idx is None:
            radicals.append(u); continue
        v = V.pop(idx)
        c = pairing(u, v)
        v = (GF(1) / c) * v  # normalize to <u,v>=1
        # Orthogonalize the rest against span{u,v}
        for j in range(len(V)):
            x = V[j]
            V[j] = x - pairing(x, v) * u + pairing(x, u) * v
        pairs.append((u, v))
    return pairs, radicals

# ---------- Subspace helpers (GF-safe) ----------

def rows_constraints(W, Omega, GF):
    """Matrix whose rows are w^T Ω for w in W (possibly empty)."""
    if not W:
        return GF.Zeros((0, Omega.shape[0]))
    return np.vstack([(w.T @ Omega) for w in W])

def nullspace_matrix(A, GF):
    """Return a GF matrix whose columns are a basis of Null(A)."""
    basis = gf_nullspace(A, GF)  # list of (n x 1)
    if len(basis) == 0:
        # No constraints -> whole space; return Identity (caller will post-multiply)
        n = A.shape[1]
        return GF.Identity(n)
    return GF(np.hstack(basis))

def orth_complement_basis(W, Omega, GF):
    """
    Basis Nw (d x t) of L = W^⊥ = { x : <w, x> = 0 for all w in W }.
    """
    A = rows_constraints(W, Omega, GF)  # m x d
    return nullspace_matrix(A, GF)  # d x t

def kernel_in_subspace_basis(Nw, F, GF):
    """
    Basis Uk (d x m) of (K ∩ L) where K = ker(F - I), L = im(Nw).
    Solve (F - I) (Nw y) = 0 -> ((F - I) Nw) y = 0 for y, then Uk = Nw Y.
    """
    M = (F - GF.Identity(F.shape[0])) @ Nw  # d x t
    Y = nullspace_matrix(M, GF)             # t x m  (note: returns Identity(t) if M=0)
    return Nw @ Y                           # d x m

def partner_in_L(u, Nw, Omega, GF):
    """
    Find e in L = im(Nw) such that <e, u> = 1.
    Solve (Nw^T Ω u)^T y = 1 for y, then e = Nw y.
    """
    a = (Nw.T @ Omega @ u).reshape(-1)  # length t
    # Find any index with a[i] != 0
    for i, ai in enumerate(a):
        if ai != GF(0):
            y = GF.Zeros(Nw.shape[1])
            y[i] = GF(1) / ai
            return (Nw @ y.reshape(-1,1))
    # If all zeros, no vector in L pairs nontrivially with u -> u ∈ L^⊥; caller should change u
    raise RuntimeError("partner_in_L: u has zero pairing with L; choose a different u in L.")

def nonkernel_in_L(Nw, F, GF, rng, max_tries=128):
    """
    Pick e in L = im(Nw) with (F - I) e != 0.
    Try random y in GF^t until (F - I) Nw y != 0.
    """
    t = Nw.shape[1]
    for _ in range(max_tries):
        y = GF(rng.integers(0, GF.characteristic, size=t))
        if np.all(y == GF(0)):
            continue
        e = Nw @ y.reshape(-1,1)
        if not np.all(((F - GF.Identity(F.shape[0])) @ e) == GF(0)):
            return e
    # As a deterministic fallback, scan standard basis
    for i in range(t):
        y = GF.Zeros(t); y[i] = GF(1)
        e = Nw @ y.reshape(-1,1)
        if not np.all(((F - GF.Identity(F.shape[0])) @ e) == GF(0)):
            return e
    raise RuntimeError("nonkernel_in_L: L ⊆ ker(F - I); cannot place non-kernel here.")

def kernel_pair_in_L(Nk, Omega, GF):
    """
    Given Nk (d x m) whose columns span K ∩ L, find u,v in im(Nk) with <v,u>=1.
    Returns (u, v) or None if K ∩ L is totally isotropic or dim < 2.
    """
    m = Nk.shape[1]
    if m < 2:
        return None
    # Scan for nonzero pairing among columns
    for i in range(m):
        u = Nk[:, [i]]
        c = (Nk.T @ Omega @ u).reshape(-1)  # pairings with all Nk columns
        for j, cj in enumerate(c):
            if cj != GF(0):
                v = (GF(1) / cj) * Nk[:, [j]]
                return u, v
    return None  # totally isotropic inside L

# ---------- Main constructor: build Fs one pair at a time in L = W^⊥ ----------

def _pair_permute_move_first_s_pairs_to_tail(Fs, s, GF):
    d = Fs.shape[0]; n = d // 2
    order = list(range(s, n)) + list(range(0, s))  # move first s pairs to the tail
    ord_cols = order + [n + i for i in order]
    P = GF.Zeros((d, d))
    for new_j, old_j in enumerate(ord_cols):
        P[old_j, new_j] = GF(1)
    Omega = omega_matrix(n, GF)
    assert (P.T @ Omega @ P == Omega).all()
    return P

def partner_in_L_nonkernel(e, Nw, F, Omega, GF, rng):
    # Solve for g = Nw y with <e,g>=1 and (F - I)g != 0
    t = Nw.shape[1]
    A = (e.T @ Omega @ Nw)              # 1 x t
    y0 = gf_solve(A, GF([1]), GF)       # one particular solution to A y = 1
    Z_basis = gf_nullspace(A, GF)       # nullspace basis columns (t x 1 each)
    B = (F - GF.Identity(F.shape[0])) @ Nw   # d x t
    # Try random combinations y = y0 + Z c until B y != 0
    for _ in range(128):
        if Z_basis:
            coeffs = GF(rng.integers(0, GF.characteristic, size=len(Z_basis)))
            y = y0.copy()
            for a, z in zip(coeffs, Z_basis):
                if a != GF(0):
                    y += a * z.reshape(-1)
        else:
            y = y0
        if not np.all((B @ y.reshape(-1,1)) == GF(0)):
            return Nw @ y.reshape(-1,1)
    # Fallback: accept any partner in L (may create a lone identity column, harmless)
    return Nw @ y0.reshape(-1,1)

def near_identity_conjugate(F, p):
    """
    Build Fs ∈ Sp and S = Fs^{-1} F Fs so that, after pair-permutation,
    the last s qudits are identity. Returns (Fs, S, r, s).
    """
    GF = galois.GF(p)
    F = GF(F)
    d = F.shape[0]; assert d % 2 == 0
    n = d // 2
    Omega = omega_matrix(n, GF)
    assert is_symplectic(F, Omega), "Input F is not symplectic for the standard Ω."

    # Invariants
    r = gf_rank(F - GF.Identity(d), GF)
    # Kernel and its Witt decomposition
    # --- STEP A: removable qudits from K = ker(F - I) ---
    K_cols = gf_nullspace(F - GF.Identity(d), GF)
    k = len(K_cols)
    BK = GF(np.hstack(K_cols)) if k > 0 else GF.Zeros((d, 0))
    pairsK, radicalsK = decompose_kernel_symplectically(BK, Omega, GF)
    s = len(pairsK)                         # # idle qudits achievable

    # Put those s kernel pairs first (E=v, G=u so <E,G>=1)
    E_cols = [v for (u, v) in pairsK]
    G_cols = [u for (u, v) in pairsK]
    W = []
    for e, g in zip(E_cols, G_cols):
        W.extend([e, g])

    # --- STEP B: complete a symplectic basis on W^⊥ with NON-kernel pairs ---
    rng = np.random.default_rng(0)
    while len(E_cols) < n:
        # pick a random vector, orthogonalize to current pairs; reject if zero or kernel
        tries = 0
        while True:
            tries += 1
            # random seed in the full space
            v = GF(rng.integers(0, GF.characteristic, size=(d, 1)))
            if np.all(v == GF(0)):
                continue
            e = orthogonalize_to_pairs(Omega, E_cols, G_cols, v, GF)
            if np.all(e == GF(0)):
                if tries < 256: 
                    continue
                # deterministic fallback: scan standard basis
                e = None
                Eye_d = GF.Identity(d)
                for j in range(d):
                    cand = orthogonalize_to_pairs(Omega, E_cols, G_cols, Eye_d[:, [j]], GF)
                    if not np.all(cand == GF(0)):
                        e = cand; break
                if e is None:
                    raise RuntimeError("Could not find nonzero vector in W^⊥.")
            # avoid creating extra kernel columns in the active block
            if np.all(((F - GF.Identity(d)) @ e) == GF(0)):
                if tries < 256:
                    continue
            break

        # find g ∈ W^⊥ with <e,g>=1
        g = partner_orth_to_W(e, W, Omega, GF)

        E_cols.append(e); G_cols.append(g)
        W.extend([e, g])

    # Assemble Fs and verify symplectic BEFORE forming S
    Fs = GF(np.hstack(E_cols + G_cols))
    assert is_symplectic(Fs, Omega), "Constructed Fs is not symplectic."

    # Form S and then pair-permute to push the s idle qudits to the end
    Fs_inv = np.linalg.inv(Fs)        # your preferred inverse; stays in GF
    S = Fs_inv @ F @ Fs

    # Move the first s pairs to the tail so the last s qudits are identity
    P = _pair_permute_move_first_s_pairs_to_tail(Fs, s, GF)
    Fs = Fs @ P
    S  = P.T @ S @ P

    # Sanity: last s qudits are identity (both columns per qudit)
    Eye = GF.Identity(d)
    for j in range(n - s, n):
        assert np.all(S[:, j]     == Eye[:, j])
        assert np.all(S[:, n + j] == Eye[:, n + j])

    return Fs, S, int(r), int(s)
# ---------- Example usage & self-check (no prints; only asserts) ----------


def _self_test():
    for p in [2, 3, 5, 7]:
        GF = galois.GF(p)
        for d in [4, 6, 8]:
            # Create an F with controllable rank(F - I):
            # Conjugate a near-identity S0 by a random symplectic Y
            n = d // 2
            Omega = omega_matrix(n, GF)
            # Build S0 = diag(S*, I) where S* is a product of a few transvections acting on a small block
            S0 = GF.Identity(d)
            Tsmall = random_symplectic(d, p, num_transvections=min(5, d))
            # Limit support by projecting with a random idempotent mask M (keep only first r columns/rows)
            # For simplicity, just use Tsmall directly (it typically gives small rank(F-I))
            S0 = Tsmall
            Y = random_symplectic(d, p)
            F = Y @ S0 @ np.linalg.inv(Y)
            # Run construction
            Fs, S, r, s = near_identity_conjugate(F, p)
            assert is_symplectic(Fs, Omega)
            Eye = GF.Identity(d)
            # Active qudit count = n - s; the last s qudits are fully identity:
            for j in range(n - s, n):
                assert np.all(S[:, j]     == Eye[:, j])      # E-column of qudit j
                assert np.all(S[:, n + j] == Eye[:, n + j])  # G-column of qudit j
            # And F is indeed conjugate:
            assert (F == Fs @ S @ np.linalg.inv(Fs)).all()


def symplectic_transvection(u, Omega, GF):
    # T_u(w) = w + <w,u> u with <w,u> = w^T Ω u
    # Matrix: I + u (Ω u)^T
    u = u.reshape(-1, 1)
    Ou = Omega @ u
    return GF.Identity(len(u)) + (u @ Ou.T)


def random_symplectic(d, p, num_transvections=10, rng=None):
    GF = galois.GF(p)
    rng = np.random.default_rng() if rng is None else rng
    assert d % 2 == 0
    Omega = omega_matrix(d // 2, GF)
    T = GF.Identity(d)
    for _ in range(num_transvections):
        u = GF(rng.integers(0, p, size=d))
        # Ensure u not zero; (alternating form makes <u,u>=0 automatically)
        while np.all(u == 0):
            u = GF(rng.integers(0, p, size=d))
        Tu = symplectic_transvection(u, Omega, GF)
        T = Tu @ T
    return T


if __name__ == '__main__':
    _self_test()
