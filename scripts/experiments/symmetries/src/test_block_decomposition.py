import numpy as np
from block_decomposition import matmul_mod, modp, is_symplectic, block_decompose, rank_mod, inv_mod_mat
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
    # compute rank of (S - I) restricted to each component as nontriviality
    M = (S - np.eye(n2, dtype=np.int64)) % p

    # build comps first (copy the same adjacency/components code)
    adj = [[] for _ in range(k)]
    for i in range(k):
        rows_i = [i, k + i]
        for j in range(i + 1, k):
            cols_j = [j, k + j]
            if (S[np.ix_(rows_i, cols_j)] % p).any() or (S[np.ix_([j, k + j], [i, k + i])] % p).any():
                adj[i].append(j)
                adj[j].append(i)
    seen = [False]*k
    comps = []
    for s in range(k):
        if seen[s]:
            continue
        st = [s]
        seen[s] = True
        comp = []
        while st:
            u = st.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    st.append(v)
        comps.append(sorted(comp))

    def comp_rank(comp):
        idx = comp + [c + k for c in comp]
        # rank_mod must be your GF(p) rank
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


def test_scrambled_swap(p: int = 3, n: int = 6, i: int = 1, j: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    S_true = swap_symplectic(n, i, j, p)
    R = random_symplectic(n, p, rng=rng, steps=6)
    F = modp(inv_mod_mat(R, p) @ S_true @ R, p)

    # Ask the solver to find the coupling block first
    S, T = block_decompose(F, p, min_block_size=4)

    assert is_symplectic(T, p)
    assert np.array_equal(S, modp(inv_mod_mat(T, p) @ F @ T, p))

    sizes = ordered_block_sizes(S, p)
    assert max(sizes) == 4, f"Expected max block size 4, got {max(sizes)} ({sizes})"
    assert sizes[0] == 4, f"Expected first block size 4, got {sizes[0]} ({sizes})"

    return S, T, F


if __name__ == "__main__":
    for _ in range(100):
        seed = np.random.randint(0, 2**32)
        S, T, F = test_scrambled_swap(p=17, n=20, i=1, j=2, seed=seed)
    print('passed all tests')
