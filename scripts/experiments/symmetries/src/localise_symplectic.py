import numpy as np

# ---------- helpers (as in your module) ----------

def gf2(A): return (A.astype(np.uint8) & 1)

def Omega(n):
    I = np.eye(n, dtype=np.uint8)
    Z = np.zeros((n, n), dtype=np.uint8)
    return np.block([[Z, I], [I, Z]])

def partner(i, n):
    return i + n if i < n else i - n

def is_symplectic(F):
    F = gf2(F)
    n2 = F.shape[0]; n = n2 // 2
    Om = Omega(n)
    return np.array_equal(gf2(F.T @ Om @ F), Om)

def qubit_neighbors_mask(F, q):
    F = gf2(F)
    n = F.shape[0] // 2
    a, b = q, partner(q, n)
    row_or = F[a, :] | F[b, :]
    lo, hi = row_or[:n], row_or[n:]
    neigh = (lo | hi).astype(bool)
    neigh[q] = False
    return neigh


def coupling_components(F):
    n = F.shape[0] // 2
    adj = [set(np.flatnonzero(qubit_neighbors_mask(F, q))) for q in range(n)]
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = {s}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.add(v)
        comps.append(comp)
    return comps


def max_component_size(F):
    comps = coupling_components(F)
    return max((len(c) for c in comps), default=0)


# ---------- transvections (fixed unit; general via rectangle rule) ----------

def apply_T_unit_inplace(F, zeta):
    F[:] = gf2(F)
    n2 = F.shape[0]
    n = n2 // 2
    k = partner(zeta, n)

    # snapshot from original F
    row_zeta = F[zeta, :].copy()
    col_k = F[:, k].copy()
    s = int(F[zeta, k] & 1)

    # stripes
    F[k, :] ^= row_zeta
    F[:, zeta] ^= col_k
    # uniform fill (1x1)
    if s:
        F[k, zeta] ^= 1
    return F


def apply_T_general_inplace(F, v):
    F[:] = gf2(F)
    v = gf2(v.reshape(1, -1))
    n2 = F.shape[0]
    n = n2 // 2
    Om = Omega(n)
    b = gf2(Om @ v.T)      # (2n,1)
    vF = gf2(v @ F)         # (1,2n)
    Fb = gf2(F @ b)         # (2n,1)
    s = int(np.bitwise_and(vF @ b, 1).sum() % 2)  # scalar in {0,1}
    F ^= gf2(b @ vF)
    F ^= gf2(Fb @ v)
    if s:
        F ^= gf2(b @ v)
    return F


# ---------- NEW: pair-transvection try step ----------

def try_pair_transvections(F, q):
    """
    Try v = e_zeta ^ e_eta with zeta in {q,bar q} and eta in {r,bar r} for neighbors r of q.
    Accept the first move that reduces (M, deg_q) lexicographically; otherwise keep the best
    non-worsening move if it doesn’t increase M. Returns (changed, (zeta,eta), F_new).
    """
    F = gf2(F)
    n2 = F.shape[0]; n = n2 // 2

    def bar(i): return partner(i, n)
    def deg_q(Fmat): return int(qubit_neighbors_mask(Fmat, q).sum())

    M0 = max_component_size(F)
    d0 = deg_q(F)
    best = None

    neigh = np.flatnonzero(qubit_neighbors_mask(F, q))
    if neigh.size == 0:
        return False, None, F

    for r in neigh:
        for zeta in (q, bar(q)):
            for eta in (r, bar(r)):
                v = np.zeros((1, 2 * n), dtype=np.uint8)
                v[0, zeta] = 1
                v[0, eta] = 1
                F_try = F.copy()
                apply_T_general_inplace(F_try, v)
                # safety: must remain symplectic
                if not is_symplectic(F_try):
                    continue
                M = max_component_size(F_try)
                d = deg_q(F_try)
                score = (M, d)
                if score < (M0, d0):
                    return True, (zeta, eta), F_try
                if best is None or score < best[0]:
                    best = (score, (zeta, eta), F_try)

    # fallback: apply best non-worsening w.r.t M
    if best and best[0][0] <= M0:
        return True, best[1], best[2]
    return False, None, F

# ---------- UPDATED: outer loop using pair-transvections ----------


def minimize_largest_component(F, max_passes=8):
    """
    Outer loop:
      - Focus on largest component(s).
      - For each qubit q in those components, try pair-transvections.
      - Accept improving (or non-worsening in M) moves, always checking symplecticity.
    Returns (moves, F_final) where moves is a list of ("pair", zeta, eta).
    """
    F = gf2(F.copy())
    assert is_symplectic(F), "Input F must be symplectic over GF(2)."

    moves = []
    best_M = max_component_size(F)

    for _ in range(max_passes):
        comps = coupling_components(F)
        if not comps:
            break
        # work on components in decreasing size
        comps.sort(key=lambda C: -len(C))
        improved_any = False

        # process only the largest size(s)
        max_size = len(comps[0])
        target_sets = [C for C in comps if len(C) == max_size]

        for C in target_sets:
            # Greedy order: highest-degree first inside the component
            degs = {q: int(qubit_neighbors_mask(F, q).sum()) for q in C}
            for q in sorted(C, key=lambda x: -degs[x]):
                changed, pair, F_try = try_pair_transvections(F, q)
                if not changed:
                    continue
                M_new = max_component_size(F_try)
                # accept if it doesn't worsen M
                if M_new <= best_M and is_symplectic(F_try):
                    F = F_try
                    best_M = M_new
                    moves.append(("pair", pair[0], pair[1]))
                    improved_any = improved_any or (M_new < best_M)

        if not improved_any:
            # if no strict M improvement in this pass, break to avoid cycles
            break

    return moves, F


if __name__ == "__main__":

    from SympleQ.core.circuits import Circuit, Gate
    # Example: a 3-qubit symplectic with some inter-qubit coupling
    n = 50
    Om = Omega(n)
    # Start from block-diagonal single-qubit (identity) and inject some couplings
    F = Gate.from_random(n, 2, 2).symplectic
    # print(F)
    # print(F)
    # print((F.T @ Om @ F) % 2)
    # Add a CNOT-like coupling between qubits 0 and 1, and between 1 and 2
    # (just for demo; F must remain symplectic — applying unit transvections is safer)
    # apply_T_unit_inplace(F, 1)   # shear at qubit 1's X
    # apply_T_unit_inplace(F, 3)   # shear at qubit 2's X (indexing 0..5)
    print("Initial F symplectic:", is_symplectic(F))
    print("Initial largest component size:", max_component_size(F))

    moves, F_final = minimize_largest_component(F)
    print("Moves (ζ indices) applied:", moves)
    print("Final largest component size:", max_component_size(F_final))
    print("Still symplectic:", is_symplectic(F_final))
    # print(F_final)
    # print(F_final)
