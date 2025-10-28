import numpy as np

# =========================
# Utilities (GF(2))
# =========================

def mod2(A): 
    return (A & 1).astype(np.uint8)

def mm2(A, B):
    return mod2(A @ B)

def is_symplectic_gf2(F):
    F = mod2(F)
    n2 = F.shape[0]
    assert F.shape[0] == F.shape[1] and n2 % 2 == 0
    n = n2 // 2
    J = np.block([[np.zeros((n,n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
                  [np.eye(n, dtype=np.uint8),       np.zeros((n,n), dtype=np.uint8)]])
    FtJ = mm2(F.T, J)
    lhs = mm2(mm2(FtJ, F), np.eye(n2, dtype=np.uint8))
    return np.array_equal(lhs, J)

def inv_gf2(A):
    A = mod2(A.copy())
    n = A.shape[0]
    I = np.eye(n, dtype=np.uint8)
    Aug = np.concatenate([A, I], axis=1)
    # Gauss-Jordan over GF(2)
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, n):
            if Aug[i, c]:
                piv = i; break
        if piv is None:
            raise np.linalg.LinAlgError("singular over GF(2)")
        if piv != r:
            Aug[[r, piv]] = Aug[[piv, r]]
        # eliminate others
        for i in range(n):
            if i != r and Aug[i, c]:
                Aug[i, :] ^= Aug[r, :]
        r += 1
    return Aug[:, n:]

# =========================
# Gate symplectics (GF(2))
# Naming matches qubit gates
# =========================

def gate_H(n, i):
    # Swap X_i and Z_i
    F = np.eye(2*n, dtype=np.uint8)
    F[i, i] = 0;     F[i, n+i] = 1
    F[n+i, i] = 1;   F[n+i, n+i] = 0
    return F

def gate_S(n, i):
    # X_i -> X_i, Z_i -> Z_i + X_i  (upper-right B_ii = 1)
    F = np.eye(2*n, dtype=np.uint8)
    F[i, n+i] = 1
    return F

def gate_CNOT(n, i, j):
    # control i, target j
    # A += E_{j,i}; D += E_{i,j}
    F = np.eye(2*n, dtype=np.uint8)
    F[i, i] = 1; F[j, j] = 1
    F[i, j] ^= 1           # A[i, j] = 1 so X_i picks up X_j
    F[n+i, n+i] = 1; F[n+j, n+j] = 1
    F[n+j, n+i] ^= 1       # D[j,i] = 1 so Z_j picks up Z_i
    return F

def gate_CZ(n, i, j):
    # Symmetric phase on i,j: B_{ij} = B_{ji} = 1
    F = np.eye(2*n, dtype=np.uint8)
    if i == j:
        F[i, n+i] ^= 1
    else:
        F[i, n+j] ^= 1
        F[j, n+i] ^= 1
    return F

def gate_SWAP(n, i, j):
    # swap X_i <-> X_j and Z_i <-> Z_j
    F = np.eye(2*n, dtype=np.uint8)
    if i == j: return F
    # swap columns/rows i<->j and n+i <-> n+j
    P = np.eye(n, dtype=np.uint8)
    P[[i,j]] = P[[j,i]]
    # Block permutation
    top = np.block([[P, np.zeros((n,n), dtype=np.uint8)],
                    [np.zeros((n,n), dtype=np.uint8), P]])
    return top

def H_all(n):
    F = np.eye(2*n, dtype=np.uint8)
    for i in range(n):
        F = mm2(gate_H(n, i), F)
    return F

# =========================
# Layer synthesis helpers
# =========================

def synth_upper_from_symmetric(n, S):
    """Return a CZ/S layer for [[I, S],[0, I]], S symmetric."""
    S = mod2(S)
    gates = []
    # Diagonals -> S(i)
    for i in range(n):
        if S[i, i]:
            gates.append(("S", i))
    # Off-diagonals -> CZ(i,j) for i<j with S_ij = 1
    for i in range(n):
        for j in range(i+1, n):
            if S[i, j]:
                gates.append(("CZ", i, j))
    return gates

def synth_lower_from_symmetric(n, Csym):
    """[[I,0],[Csym, I]] via H-all, upper-layer(Csym), H-all."""
    Csym = mod2(Csym)
    g = [("H", i) for i in range(n)]
    g += synth_upper_from_symmetric(n, Csym)
    g += [("H", i) for i in range(n)]
    return g

def synth_linear_A(n, A):
    """
    Synthesize [[A,0],[0,(A^T)^{-1}]] using SWAP + CNOT,
    but now via **row** operations since gates left-multiply.

    Row op "swap rows r<->c"  -> SWAP(r, c)
    Row op "row i ^= row c"   -> CNOT(c -> i)
    """
    A = mod2(A.copy())
    gates = []
    for c in range(n):
        # --- pivot in column c
        r = None
        for rr in range(c, n):
            if A[rr, c]:
                r = rr; break
        if r is None:
            # A is invertible, so if nothing below, there must be one above
            for rr in range(0, c):
                if A[rr, c]:
                    r = rr; break
        if r is None:
            raise RuntimeError("Unexpected: could not find pivot; A should be invertible.")

        # bring pivot to row c
        if r != c:
            A[[c, r], :] = A[[r, c], :]
            gates.append(("SWAP", c, r))

        # eliminate other 1s in column c by adding pivot row into them
        for i in range(n):
            if i != c and A[i, c]:
                A[i, :] ^= A[c, :]
                gates.append(("CNOT", i, c))

    # Now A == I
    return list(reversed(gates))


def blocks(F):
    n = F.shape[0] // 2
    A = F[:n, :n]
    B = F[:n, n:]
    C = F[n:, :n]
    D = F[n:, n:]
    return A, B, C, D

def compose_symplectic_from_gates(n, gates):
    F = np.eye(2*n, dtype=np.uint8)
    for g in gates:
        if g[0] == "H":
            Gi = gate_H(n, g[1])
        elif g[0] == "S":
            Gi = gate_S(n, g[1])
        elif g[0] == "CNOT":
            Gi = gate_CNOT(n, g[1], g[2])
        elif g[0] == "CZ":
            Gi = gate_CZ(n, g[1], g[2])
        elif g[0] == "SWAP":
            Gi = gate_SWAP(n, g[1], g[2])
        else:
            raise ValueError(f"Unknown gate {g}")
        F = mm2(Gi, F)  # prepend gate (physical order)
    return F

# =========================
# Main decomposition
# =========================

def decompose_symplectic_gf2(F):
    """
    Input: 2n x 2n binary symplectic F (Pauli rows act as p -> p F^T).
    Output: list of gates [("S",i), ("CZ",i,j), ("CNOT",i,j), ("H",i), ("SWAP",i,j), ...]
            in the **application order** (left-to-right equals time order).
    """
    F = mod2(F)
    assert is_symplectic_gf2(F), "Input F is not symplectic over GF(2)."
    n = F.shape[0] // 2
    A, B, C, D = blocks(F)

    # Ensure A is invertible; if not, raise with a friendly hint (can be extended with H/SWAP pivots)
    try:
        Ainv = inv_gf2(A)
    except np.linalg.LinAlgError:
        raise ValueError(
            "A is singular over GF(2). A simple extension is to conjugate by H/SWAP "
            "on selected qubits to move rank from C into A, then retry."
        )

    # Compute symmetric “upper” and “lower” forms
    # B' = A^{-1} B, C' = C A^{-1}
    Bp = mod2(Ainv @ B)
    Cp = mod2(C @ Ainv)

    # Sanity (they must be symmetric for symplectic F)
    if not np.array_equal(mod2(Bp), mod2(Bp.T)):
        raise ValueError("A^{-1} B is not symmetric — F is not symplectic or numeric bug.")
    if not np.array_equal(mod2(Cp), mod2(Cp.T)):
        raise ValueError("C A^{-1} is not symmetric — F is not symplectic or numeric bug.")

    # R layer: [[I, B'], [0,I]]  -> CZ/S
    gates_R = synth_upper_from_symmetric(n, Bp)

    # M layer: [[A,0],[0,(A^T)^{-1}]] -> linear reversible network (CNOT+SWAP)
    gates_M = synth_linear_A(n, A)

    # L layer: [[I,0],[C', I]] -> H_all + CZ/S(C') + H_all
    gates_L = synth_lower_from_symmetric(n, Cp)

    # Final gate list applies in order R -> M -> L (this multiplies to F)
    gates = []
    gates += gates_R
    gates += gates_M
    gates += gates_L

    # Verify reconstruction
    Frecon = compose_symplectic_from_gates(n, gates)
    if not np.array_equal(Frecon, F):
        raise AssertionError(F"Internal check failed: reconstructed F != input F.")

    return gates

# =========================
# Quick smoke test
# =========================

def random_symmetric(n, rng):
    M = rng.integers(0, 2, size=(n,n), dtype=np.uint8)
    return mod2(M + M.T)  # force symmetric (diag automatic mod2)

def random_invertible(n, rng):
    # Build via random CNOT+SWAP sequence so it's guaranteed invertible
    A = np.eye(n, dtype=np.uint8)
    k = 3*n
    for _ in range(k):
        if rng.random() < 0.5:
            i, j = rng.integers(0, n, 2)
            if i != j:
                # CNOT i->j (column add)
                A[:, j] ^= A[:, i]
        else:
            i, j = rng.integers(0, n, 2)
            if i != j:
                A[:, [i,j]] = A[:, [j,i]]
    # Ensure full rank (very likely); if not, retry
    try:
        _ = inv_gf2(A)
        return A
    except np.linalg.LinAlgError:
        return random_invertible(n, rng)

def build_F_from_LMR(n, A, Bsym, Csym):
    # F = [[I,0],[Csym, I]] [[A,0],[0,(A^T)^{-1}]] [[I,Bsym],[0,I]]
    AinvT = inv_gf2(A.T)
    L = np.block([[np.eye(n, dtype=np.uint8), np.zeros((n,n), dtype=np.uint8)],
                  [Csym,                         np.eye(n, dtype=np.uint8)]])
    M = np.block([[A,                          np.zeros((n,n), dtype=np.uint8)],
                  [np.zeros((n,n), dtype=np.uint8), AinvT]])
    R = np.block([[np.eye(n, dtype=np.uint8), Bsym],
                  [np.zeros((n,n), dtype=np.uint8), np.eye(n, dtype=np.uint8)]])
    return mm2(mm2(L, M), R)

def demo(n=4, seed=2025):
    rng = np.random.default_rng(seed)
    A = random_invertible(n, rng)
    Bsym = random_symmetric(n, rng)
    Csym = random_symmetric(n, rng)
    F = build_F_from_LMR(n, A, Bsym, Csym)
    assert is_symplectic_gf2(F)
    gates = decompose_symplectic_gf2(F)
    print(f"n={n}, #gates={len(gates)}")
    # quick check already done inside; print a readable circuit
    for g in gates:
        print(g)

if __name__ == "__main__":
    demo(n=3)
    demo(n=5)
