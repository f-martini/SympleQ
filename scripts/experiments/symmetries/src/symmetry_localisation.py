import numpy as np

# ---------- basic symplectic utilities (mod p) ----------

def omega(n: int, p: int) -> np.ndarray:
    """Standard 2n x 2n symplectic form over F_p."""
    I = np.eye(n, dtype=np.int64)
    top = np.concatenate([np.zeros((n, n), dtype=np.int64), I], axis=1)
    bot = np.concatenate([(-I) % p, np.zeros((n, n), dtype=np.int64)], axis=1)
    return np.concatenate([top, bot], axis=0) % p

def sp_inner(u: np.ndarray, v: np.ndarray, Ω: np.ndarray, p: int) -> int:
    """Symplectic inner product <u,v> = u^T Ω v mod p."""
    return int((u.T @ (Ω @ v)) % p)

def is_symplectic(F: np.ndarray, p: int) -> bool:
    n2 = F.shape[0]; n = n2 // 2
    Ω = omega(n, p)
    return np.array_equal((F.T @ Ω @ F) % p, Ω)

# ---------- transvection machinery ----------

def transvection_update(F: np.ndarray, v: np.ndarray, s: int, Ω: np.ndarray, p: int) -> np.ndarray:
    """
    Fast rank-2/1 conjugation update:
      F' = F + s(F v u^T - v u^T F) - s^2 * α * v u^T,  with u^T = v^T Ω, α = <v, F v>.
    """
    s %= p
    if s == 0:
        return F

    v = (v.reshape(-1, 1) % p).astype(np.int64)
    uT = ((v.T @ Ω) % p).astype(np.int64)          # shape (1, 2n)
    Fv = (F @ v) % p                                # shape (2n, 1)
    alpha = int(((uT @ Fv) % p).item())             # scalar α in F_p

    term1 = (Fv @ uT) % p                           # rank-1: F v u^T
    term2 = (v @ (uT @ F) % p) % p                  # rank-1: v u^T F
    scale = (s * s * alpha) % p
    vuT   = (v @ uT) % p                            # rank-1: v u^T
    F_new = (F + (s * term1) % p - (s * term2) % p - (scale * vuT) % p) % p
    return F_new

def transvection_matrix(v: np.ndarray, s: int, Ω: np.ndarray, p: int) -> np.ndarray:
    """Explicit T_{v,s} = I + s v (v^T Ω), useful if you want to accumulate T."""
    v = (v.reshape(-1, 1) % p).astype(np.int64)
    uT = ((v.T @ Ω) % p).astype(np.int64)
    N_v = (v @ uT) % p
    I = np.eye(len(v), dtype=np.int64)
    return (I + (s % p) * N_v) % p

# ---------- penalty & core management ----------

def reorder_indices_for_core(K: list[int], n: int) -> np.ndarray:
    """Return a permutation of phase-space indices that puts qudits K first."""
    K = list(dict.fromkeys(K))  # unique, keep order
    all_q = [i for i in range(n)]
    rest = [i for i in all_q if i not in K]
    order = [*K, *rest]
    # phase-space order: [X_0..X_{n-1}, Z_0..Z_{n-1}]
    xs = order
    zs = [i + n for i in order]
    return np.array([*xs, *zs], dtype=np.int64)

def penalty(F: np.ndarray, K: list[int], p: int) -> int:
    """Φ = nnz(off-core cross) + nnz(off-core bottom-right - I)."""
    n2 = F.shape[0]; n = n2 // 2
    perm = reorder_indices_for_core(K, n)
    Fp = F[np.ix_(perm, perm)]
    m = len(K); m2 = 2 * m
    # blocks
    Kbar2 = 2 * (n - m)
    if Kbar2 == 0:
        return 0
    B12 = Fp[:m2, m2:]
    B21 = Fp[m2:, :m2]
    B22 = Fp[m2:, m2:]
    I22 = np.eye(Kbar2, dtype=np.int64) % p
    return int(np.count_nonzero(B12 % p) + np.count_nonzero(B21 % p) + np.count_nonzero((B22 - I22) % p))

def qudit_block_indices(i: int, n: int) -> np.ndarray:
    """Return the 2 indices (X_i, Z_i) in phase-space for qudit i."""
    return np.array([i, i + n], dtype=np.int64)

def coupling_weights(F: np.ndarray, p: int) -> np.ndarray:
    """
    Build a simple qudit coupling graph: w_ij = nnz in the 2x2 qudit block (rows of i, cols of j)
    of (F - I), summed with its transpose block to symmetrise.
    """
    n2 = F.shape[0]; n = n2 // 2
    R = (F - np.eye(n2, dtype=np.int64)) % p
    W = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        rows = qudit_block_indices(i, n)
        for j in range(n):
            if i == j:
                continue
            cols = qudit_block_indices(j, n)
            block = R[np.ix_(rows, cols)]
            W[i, j] = np.count_nonzero(block)
    # symmetrise by max or sum; use sum for more sensitivity
    return (W + W.T)

def choose_core_by_weight(F: np.ndarray, p: int, k: int) -> list[int]:
    """Pick k qudits with largest total coupling weights."""
    W = coupling_weights(F, p)
    scores = W.sum(axis=1)  # total weight per node
    order = np.argsort(-scores)
    return list(map(int, order[:k]))


def prune_core_to_target(F: np.ndarray, K: list[int], p: int, target: int) -> list[int]:
    """Greedily drop qudits while keeping Φ=0 until |K|==target or no drop is possible."""
    K = K.copy()
    while len(K) > target:
        base = penalty(F, K, p)
        if base != 0:
            break  # can't prune while not perfectly localised
        # try removing any one qudit that preserves Φ=0
        removed = False
        for q in list(K):
            K_try = [x for x in K if x != q]
            if penalty(F, K_try, p) == 0:
                K = K_try
                removed = True
                break
        if not removed:
            break
    return K



def best_swap(F, K, p, n):
    """Find best 1-swap that reduces Φ the most; return (Δ, i_in, j_out)."""
    current = penalty(F, K, p)
    Kset = set(K)
    best = (0, None, None)
    W = coupling_weights(F, p).sum(axis=1)
    # try few promising candidates by weight
    nonK_sorted = [j for j in np.argsort(-W) if j not in Kset][:min(n, 2*len(K)+8)]
    K_sorted = [i for i in np.argsort(W) if i in Kset][:min(len(K), 8)]
    for i in K_sorted:
        for j in nonK_sorted:
            KK = K.copy()
            ii = KK.index(i)
            KK[ii] = j
            d = current - penalty(F, KK, p)
            if d > best[0]:
                best = (d, i, j)
    return best
# ---------- heuristic localiser ----------


def heuristic_transvection_localiser(
    F: np.ndarray,
    p: int,
    L_min: int,
    max_iters: int = 2000,
    candidate_neighbour_expand: int = 4,
    return_T: bool = False,
    allow_core_growth: bool = False,   # False: keep core fixed (for tests)
    prune_each_step: bool = True       # True: prune back to L_min whenever Φ==0
):
    """
    Greedy transvection localiser for concentrating a symplectic F onto the smallest qudit core.

    Uses existing helper functions:
      omega, is_symplectic, penalty, coupling_weights,
      choose_core_by_weight, transvection_update, transvection_matrix, best_swap.

    Returns:
        S, K, steps [, T]
    """

    n2 = F.shape[0]
    assert F.shape[0] == F.shape[1] and n2 % 2 == 0, "F must be square and even dimension."
    n = n2 // 2
    Ω = omega(n, p)
    F = (F % p).astype(np.int64)
    assert is_symplectic(F, p), "Input F must be symplectic."

    # ---------------- helper: prune back to target size ----------------
    def prune_core_to_target(F_: np.ndarray, K_: list[int], target: int) -> list[int]:
        """Greedily drop qudits while keeping Φ=0 until |K| == target or no drop possible."""
        K_ = K_.copy()
        while len(K_) > target:
            if penalty(F_, K_, p) != 0:
                break
            removed = False
            for q in list(K_):
                K_try = [x for x in K_ if x != q]
                if penalty(F_, K_try, p) == 0:
                    K_ = K_try
                    removed = True
                    break
            if not removed:
                break
        return K_

    # ---------------- initialisation ----------------
    K = choose_core_by_weight(F, p, max(L_min, 1))[:L_min]
    steps: list[tuple[np.ndarray, int]] = []
    T_acc = np.eye(2 * n, dtype=np.int64) if return_T else None

    # ---------------- main loop ----------------
    for _ in range(max_iters):
        Φ = penalty(F, K, p)
        if Φ == 0:
            if prune_each_step:
                K = prune_core_to_target(F, K, L_min)
            if len(K) == L_min:
                break

        # Candidate pool: core + top neighbours
        W = coupling_weights(F, p)
        neigh_scores = W[K, :].sum(axis=0)
        neigh_order = np.argsort(-neigh_scores)
        C = list(dict.fromkeys([*K, *map(int, neigh_order[:candidate_neighbour_expand])]))[:min(n, len(K) + candidate_neighbour_expand)]

        # Candidate vectors v: X_i, Z_i and optionally (F @ e_i)
        V_candidates = []
        for i in C:
            for idx in qudit_block_indices(i, n):
                e = np.zeros((2 * n, 1), dtype=np.int64)
                e[idx, 0] = 1
                V_candidates.append(e)
                V_candidates.append((F @ e) % p)  # adds F e to hit cross-terms

        baseΦ = Φ
        best_drop = 0
        best_vs = None

        # Try all (v, s)
        for v in V_candidates:
            for s in range(p):
                F_try = transvection_update(F, v, s, Ω, p)
                newΦ = penalty(F_try, K, p)
                drop = baseΦ - newΦ
                if drop > best_drop:
                    best_drop = int(drop)
                    best_vs = (v.copy(), int(s))
                    if best_drop == baseΦ:
                        break
            if best_drop == baseΦ:
                break

        # If found improvement, apply
        if best_drop > 0 and best_vs is not None:
            v, s = best_vs
            F = transvection_update(F, v, s, Ω, p)
            steps.append((v.flatten(), s))
            if return_T:
                T_step = transvection_matrix(v, s, Ω, p)
                T_acc = (T_step @ T_acc) % p
            continue

        # Try a single 1-swap (keeps |K| fixed)
        Δ, i_in, j_out = best_swap(F, K, p, n)
        if Δ > 0 and i_in is not None:
            K[K.index(i_in)] = j_out
            continue

        # Optionally allow growth if still stuck
        if allow_core_growth and len(K) < n:
            Wsum = W.sum(axis=1)
            add_order = [int(i) for i in np.argsort(-Wsum) if i not in set(K)]
            if add_order:
                K.append(add_order[0])
                continue

        # Stalled completely
        break

    # Final prune to exact L_min if possible
    if prune_each_step:
        K = prune_core_to_target(F, K, L_min)

    assert is_symplectic(F, p), "Internal error: result lost symplecticity."

    if return_T:
        return F % p, K, steps, T_acc % p
    else:
        return F % p, K, steps


# ---------- utilities needed by the test (pure NumPy, mod p) ----------

def transvection_matrix(v: np.ndarray, s: int, Ω: np.ndarray, p: int) -> np.ndarray:
    v = (v.reshape(-1, 1) % p).astype(np.int64)
    uT = ((v.T @ Ω) % p).astype(np.int64)
    N_v = (v @ uT) % p
    I = np.eye(len(v), dtype=np.int64)
    return (I + (s % p) * N_v) % p



def reorder_indices_for_core(K: list[int], n: int) -> np.ndarray:
    K = list(dict.fromkeys(K))
    rest = [i for i in range(n) if i not in K]
    order = [*K, *rest]
    xs = order
    zs = [i + n for i in order]
    return np.array([*xs, *zs], dtype=np.int64)


def penalty(F: np.ndarray, K: list[int], p: int) -> int:
    n2 = F.shape[0]; n = n2 // 2
    perm = reorder_indices_for_core(K, n)
    Fp = F[np.ix_(perm, perm)]
    m = len(K); m2 = 2 * m
    Kbar2 = 2 * (n - m)
    if Kbar2 == 0:
        return 0
    B12 = Fp[:m2, m2:]
    B21 = Fp[m2:, :m2]
    B22 = Fp[m2:, m2:]
    I22 = np.eye(Kbar2, dtype=np.int64) % p
    return int(np.count_nonzero(B12 % p) + np.count_nonzero(B21 % p) + np.count_nonzero((B22 - I22) % p))


def swap_symplectic(n: int, i: int, j: int, p: int) -> np.ndarray:
    """Symplectic that swaps qudits i and j (same permutation on X and Z)."""
    assert 0 <= i < n and 0 <= j < n and i != j
    perm = list(range(2 * n))
    # swap X coords
    perm[i], perm[j] = perm[j], perm[i]
    # swap Z coords
    zi, zj = i + n, j + n
    perm[zi], perm[zj] = perm[zj], perm[zi]
    P = np.eye(2 * n, dtype=np.int64)[perm, :]
    return P % p


def random_symplectic(n: int, p: int, rng: np.random.Generator, num_steps: int = None):
    """Random symplectic as product of random transvections; also returns its inverse."""
    if num_steps is None:
        num_steps = 20 * n
    Ω = omega(n, p)
    R = np.eye(2 * n, dtype=np.int64)
    steps = []
    for _ in range(num_steps):
        v = rng.integers(0, p, size=(2 * n, 1), dtype=np.int64)
        if np.all(v % p == 0):
            v[0, 0] = 1
        s = int(rng.integers(1, p, dtype=np.int64))
        T = transvection_matrix(v, s, Ω, p)
        R = (T @ R) % p
        steps.append((v.copy(), s))
    R_inv = np.eye(2 * n, dtype=np.int64)
    for v, s in reversed(steps):
        T_inv = transvection_matrix(v, (-s) % p, Ω, p)
        R_inv = (T_inv @ R_inv) % p
    assert is_symplectic(R, p)
    assert np.array_equal((R @ R_inv) % p, np.eye(2 * n, dtype=np.int64) % p)
    return R, R_inv


# ---------- MAIN TEST (drop-in) ----------


def run_scrambled_swap_localisation_test(
    heuristic_transvection_localiser,
    p: int = 3,
    n: int = 6,
    i: int = 1,
    j: int = 4,
    seed: int = 42,
    max_iters: int = 600,
):
    """
    Build F_in = R * SWAP_{i,j} * R^{-1}, run the heuristic with L_min=2,
    assert localisation to exactly 2 qudits (Φ=0, |K|=2), and print a summary.
    """
    rng = np.random.default_rng(seed)
    SW = swap_symplectic(n, i, j, p)
    R, R_inv = random_symplectic(n, p, rng, num_steps=20 * n)
    F_in = (R @ SW @ R_inv) % p
    assert is_symplectic(F_in, p), "Input F_in must be symplectic."

    L_min = 2
    out = heuristic_transvection_localiser(F_in, p, L_min=L_min, max_iters=max_iters, return_T=False)
    # Accept both (S,K,steps) and (S,K,steps,T)
    if len(out) == 3:
        S, K, steps = out
    else:
        S, K, steps, _T = out

    # Checks
    assert is_symplectic(S, p), "Output S must be symplectic."
    assert len(K) == 2, f"Expected 2-qudit core, got {len(K)}"
    Φ = penalty(S, K, p)
    assert Φ == 0, f"Expected zero off-core penalty, got Φ={Φ}"

    # Stronger structural check: block-diagonal after permuting core first
    perm = reorder_indices_for_core(K, n)
    Sp = S[np.ix_(perm, perm)]
    m2 = 2 * len(K)
    B12 = Sp[:m2, m2:]
    B21 = Sp[m2:, :m2]
    B22 = Sp[m2:, m2:]
    I22 = np.eye(2 * (n - len(K)), dtype=np.int64) % p
    assert np.count_nonzero(B12 % p) == 0
    assert np.count_nonzero(B21 % p) == 0
    assert np.array_equal(B22 % p, I22), "Outside-core block must be identity."
    assert is_symplectic(Sp[:m2, :m2], p), "Core (2-qudit) block must be symplectic."

    print(
        f"[PASS] SWAP localisation over F_{p}: n={n}, swap=({i},{j}), "
        f"|K|={len(K)}, K={K}, steps={len(steps)}"
    )


if __name__ == "__main__":

    # Run one or more test instances
    run_scrambled_swap_localisation_test(heuristic_transvection_localiser, p=3, n=6, i=1, j=4, seed=42)
    run_scrambled_swap_localisation_test(heuristic_transvection_localiser, p=5, n=5, i=0, j=3, seed=1234)
