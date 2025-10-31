import numpy as np
from sympleq.core.circuits import Circuit
from sympleq.core.circuits.gates import Hadamard, PHASE, SUM, SWAP, Gate
from collections import deque

# ---------- GF(p) helpers - should remove most and put in better place ----------


def mod_p(A, p):
    return np.asarray(A, dtype=int) % p


def identity(n):
    return np.eye(n, dtype=int)


def zeros(shape):
    return np.zeros(shape, dtype=int)


def mm_p(A, B, p):
    return mod_p(A @ B, p)


def inv_gfp(A: np.ndarray, p: int) -> np.ndarray:
    """Inverse over GF(p) by Gauss–Jordan."""
    A = mod_p(A.copy(), p)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    Id = identity(n)
    aug = np.hstack((A, Id))
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, n):
            if aug[r, col] % p != 0:
                piv = r
                break
        if piv is None:
            raise ValueError("Singular over GF(p).")
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        inv_pivot = pow(int(aug[row, col] % p), -1, p)
        aug[row, :] = (aug[row, :] * inv_pivot) % p
        for r in range(n):
            if r == row:
                continue
            f = aug[r, col] % p
            if f != 0:
                aug[r, :] = (aug[r, :] - f * aug[row, :]) % p
        row += 1
    return aug[:, n:] % p


def is_symplectic_gfp(F: np.ndarray, p: int) -> bool:
    F = mod_p(F, p)
    n2 = F.shape[0]
    if n2 % 2 or F.shape[0] != F.shape[1]:
        return False
    n = n2 // 2
    Id = identity(n) % p
    J = np.block([[zeros((n, n)), Id], [(-Id) % p, zeros((n, n))]]) % p
    return np.array_equal(mod_p(F.T @ J @ F, p), J)


def blocks(F: np.ndarray, p: int):
    F = mod_p(F, p)
    n = F.shape[0] // 2
    return F[:n, :n], F[:n, n:], F[n:, :n], F[n:, n:]


def _is_invertible_mod(A: np.ndarray, p: int) -> bool:
    try:
        _ = inv_gfp(A, p)
        return True
    except ValueError:
        return False


# ---------- preconditioner (BFS over Gate objects) ----------

def ensure_invertible_A_circuit(F: np.ndarray, p: int, max_depth: int | None = None) -> Circuit:
    """
    Find a Circuit C_pre such that the A-block of (C_pre.full_symplectic(n) @ F) is invertible mod p.
    Returns the Circuit (does NOT modify F).
    """
    F = mod_p(F, p)
    assert is_symplectic_gfp(F, p), "F must be symplectic over GF(p)."
    n = F.shape[0] // 2

    A, _, _, _ = blocks(F, p)
    if _is_invertible_mod(A, p):
        return Circuit([p] * n, [])

    if max_depth is None:
        max_depth = max(1, 3 * n)

    # Candidate generators (Gate objects). Include PHASE^{-1} via repeating PHASE (p-1) times.
    candidates: list[object] = []
    for i in range(n):
        candidates.append(Hadamard(i, p))
        candidates.append(PHASE(i, p))
        candidates.append(("PHASE_INV", i))  # expand to (p-1) PHASEs when applying
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append(SWAP(i, j, p))
            # Both directions of SUM are useful during search
            candidates.append(SUM(i, j, p))
            candidates.append(SUM(j, i, p))

    def apply_left_once(F_mat: np.ndarray, g) -> np.ndarray:
        if isinstance(g, tuple) and g[0] == "PHASE_INV":
            i = g[1]
            for _ in range((p - 1) % p):
                F_mat = (PHASE(i, p).full_symplectic(n) @ F_mat) % p
            return F_mat
        return (g.full_symplectic(n) @ F_mat) % p

    # BFS with an index (no deque) for clarity and determinism
    seen = {tuple(F.flatten())}
    queue: list[tuple[np.ndarray, list[object]]] = [(F, [])]
    head = 0

    while head < len(queue):
        F_cur, ops = queue[head]
        head += 1
        if len(ops) >= max_depth:
            continue
        for g in candidates:
            F_new = apply_left_once(F_cur.copy(), g)
            key = tuple(F_new.flatten())
            if key in seen:
                continue
            new_ops = ops + [g]

            Anew, _, _, _ = blocks(F_new, p)
            if _is_invertible_mod(Anew, p):
                # Build the concrete Circuit with expanded PHASE_INV
                C_pre = Circuit([p] * n, [])
                for op in new_ops:
                    if isinstance(op, tuple) and op[0] == "PHASE_INV":
                        i = op[1]
                        for _ in range((p - 1) % p):
                            C_pre.add_gate(PHASE(i, p))
                    else:
                        C_pre.add_gate(op)
                return C_pre

            seen.add(key)
            queue.append((F_new, new_ops))

    raise RuntimeError("Unable to make A invertible within the depth limit.")

#  ############################################################################  #
#                                  M - Block                                     #
#  ############################################################################  #


# ---- synthesize local diagonal D(u) = [[u,0],[0,u^{-1}]] on wire `index` ----
_DU_CACHE: dict[int, dict[int, list]] = {}   # per p: u -> word (list[Gate] on wire 0)


def _compose_1q_word_as_matrix(p: int, word0: list[Gate]) -> np.ndarray:
    """Compose a 1-qudit word (on qudit index 0) into its 2x2 symplectic over GF(p)."""
    return Circuit([p], list(word0)).composite_gate().symplectic % p


def _all_Du_targets_1q(p: int) -> dict[int, np.ndarray]:
    targets = {}
    for u in range(2, p):
        inv_u = pow(u, -1, p)
        targets[u] = np.array([[u, 0], [0, inv_u]], dtype=int) % p
    return targets


def _build_Du_cache_for_p(p: int, bfs_depth: int = 12) -> None:
    if p in _DU_CACHE:
        return

    gens = [
        [PHASE(0, p)],
        [Hadamard(0, p)],
        [Hadamard(0, p, inverse=True)],
    ]

    I2 = np.eye(2, dtype=int) % p
    targets = _all_Du_targets_1q(p)  # now excludes u==1
    remaining = set(targets.keys())

    solutions: dict[int, list] = {}
    # Pre-seed identity: D(1) = I has empty word
    solutions[1] = []
    _DU_CACHE[p] = solutions  # write early so recursion-safe
    # BFS only for the nontrivial diagonals
    from collections import deque
    Q = deque([([], I2)])
    seen = {tuple(I2.flatten())}

    while Q and remaining:
        word, M = Q.popleft()
        if len(word) > bfs_depth:
            continue
        for g in gens:
            w2 = word + g
            M2 = _compose_1q_word_as_matrix(p, w2)
            key = tuple(M2.flatten())
            if key in seen:
                continue
            seen.add(key)
            # check if M2 matches any remaining D(u)
            hit = [u for u in list(remaining) if np.array_equal(M2, targets[u])]
            for u in hit:
                solutions[u] = w2
                remaining.remove(u)
            Q.append((w2, M2))

    if remaining:
        raise RuntimeError(f"1q BFS couldn’t reach D(u) for u={sorted(remaining)} over GF({p}) "
                           f"with depth={bfs_depth}; increase depth or add generators.")

    _DU_CACHE[p] = solutions  # finalize


def _emit_local_ops_for_D(index: int, u: int, p: int) -> list[Gate]:
    """
    Emit a minimal word for D(u) on the given wire `index`, using a cached BFS word on wire 0.
    """
    u %= p
    if u == 1 % p:
        return []
    _build_Du_cache_for_p(p, bfs_depth=12)  # build once per p
    word0 = _DU_CACHE[p][u]                 # gates on wire 0
    # remap the cached wire-0 gates to the requested wire `index`
    remapped: list[Gate] = []
    for g in word0:
        if isinstance(g, PHASE):
            remapped.append(PHASE(index, p))
        elif isinstance(g, Hadamard):
            inv = getattr(g, "is_inverse", (g.name == "H_inv"))
            remapped.append(Hadamard(index, p, inverse=bool(inv)))
        else:
            raise ValueError(f"Unexpected 1q gate in D(u) cache: {type(g)}")
    return remapped


# ---- MAIN: synthesize M = diag(A, (A^T)^{-1}) as a gate list (right-multiplication) ----
def synth_linear_A_to_gates(n: int, A: np.ndarray, p: int):
    """
    RIGHT-multiplication consistent synthesis of [[A,0],[0,(A^T)^{-1}]].

    We eliminate on A^T with LEFT row ops, mirroring each row-op L as a RIGHT column-op E on A:
      • Row swap on A^T (r <-> c)         -> SWAP(r, c)
      • Scale row c by λ on A^T           -> scale column c by λ^{-1} on the RIGHT
                                            BUT since we normalized A^T by inv(λ), we must apply D(λ) on A.  # see below
      • Row add: row i ← row i − f*row c  -> col i ← col i − f*col c (RIGHT)
                                            -> use SUM(src=c, dst=i) repeated (-f) mod p times.
    """
    A = (A % p).copy()
    At = A.T.copy()
    ops: list[Gate] = []

    for c in range(n):
        # --- pivot selection (prefer 1 to avoid scaling) ---
        pivot_row = None
        for r in range(c, n):
            if At[r, c] % p == 1 % p:
                pivot_row = r
                break
        if pivot_row is None:
            for r in range(c, n):
                if At[r, c] % p != 0:
                    pivot_row = r
                    break
        if pivot_row is None:
            for r in range(0, c):
                if At[r, c] % p != 0:
                    pivot_row = r
                    break
        if pivot_row is None:
            raise RuntimeError("A invertible, but no pivot found (unexpected).")

        # Row swap on A^T -> column swap on A
        if pivot_row != c:
            At[[c, pivot_row], :] = At[[pivot_row, c], :]
            ops.append(SWAP(pivot_row, c, p))

        # Scale row c of A^T to 1: multiply row c by inv(pv)
        # RIGHT effect: scale column c of A by pv (not inv_pv)  # <<< FIX
        pv = int(At[c, c] % p)
        if pv != 1 % p:
            inv_pv = pow(pv, -1, p)
            At[c, :] = (At[c, :] * inv_pv) % p
            ops += _emit_local_ops_for_D(c, pv, p)             # <<< FIX: use pv, not inv_pv

        # Eliminate other rows in column c on A^T:
        # row i <- row i - f * row c  =>  col i <- col i - f * col c (RIGHT)
        for i in range(n):
            if i == c:
                continue
            f = int(At[i, c] % p)
            if f != 0:
                At[i, :] = (At[i, :] - f * At[c, :]) % p
                reps = (f) % p
                for _ in range(reps):
                    ops.append(SUM(i, c, p))

    return ops

#  ############################################################################  #
#                        Symmetric L and R block synthesis                       #
#  ############################################################################  #


def _as_int_mod(A, p: int) -> np.ndarray:
    """Coerce A to a plain integer ndarray and reduce mod p.
    Accepts: ndarray-like, Circuit/Gate with .symplectic, lists of lists, etc.
    """
    # unwrap symplectic matrices if a Circuit/Gate-ish object was passed
    if hasattr(A, "symplectic"):
        A = A.symplectic
    # standardize to ndarray first
    A = np.asarray(A)
    # aggressively coerce dtype to integer
    if A.dtype.kind in ("O", "U", "S"):  # object/str/unicode
        try:
            A = A.astype(np.int64, copy=False)
        except Exception:
            # last resort: element-wise int()
            A = np.vectorize(lambda x: int(x))(A)
    else:
        A = A.astype(np.int64, copy=False)
    return A % int(p)


def _full_from_lower(n, C, p):
    Id = np.eye(n, dtype=int)
    Z = np.zeros((n, n), dtype=int)
    F = np.block([[Id, Z], [_as_int_mod(C, p), Id]])
    return _as_int_mod(F, p)


def _compose_symp(n, p, gates):
    F = Circuit([p] * n, list(gates)).composite_gate().symplectic
    return _as_int_mod(F, p)


def _gens_2q(i, j, p):
    """Small generating set on two wires i,j (right-multiplication consistent)."""
    return [
        [PHASE(i, p)], [PHASE(j, p)],
        [Hadamard(i, p)], [Hadamard(i, p, inverse=True)],
        [Hadamard(j, p)], [Hadamard(j, p, inverse=True)],
        [SUM(i, j, p)], [SUM(j, i, p)],
        [SWAP(i, j, p)],
    ]


def _bfs_2q_to_target(n, p, target_full, i, j, max_depth=7):
    """Breadth-first synth on the two wires i,j; returns a word (list[Gate])."""
    Id = np.eye(2 * n, dtype=int) % p
    if np.array_equal(target_full % p, Id):
        return []
    gens = _gens_2q(i, j, p)
    Q = deque([([], Id)])
    seen = {tuple(Id.flatten())}
    while Q:
        word, M = Q.popleft()
        if len(word) >= max_depth:
            continue
        for g in gens:
            new_word = word + g
            M2 = _compose_symp(n, p, new_word)
            key = tuple(M2.flatten())
            if key in seen:
                continue
            if np.array_equal(M2, target_full % p):
                return new_word
            seen.add(key)
            Q.append((new_word, M2))
    raise RuntimeError("2q BFS failed (increase depth or check generators).")


def synth_lower_from_symmetric(n: int, C_sym: np.ndarray, p: int) -> list[Gate]:
    """
    Build L(C) = [[I,0],[C,I]] with right-multiplication convention.
    - Diagonals: C_ii * PHASE(i)
    - Off-diagonals: for each i<j with c != 0, BFS a minimal 2-qudit word that
      realizes [[I,0],[c(E_ij + E_ji), I]].
    """
    C_sym = mod_p(C_sym, p)
    assert np.array_equal(C_sym, C_sym.T % p), "C_sym must be symmetric mod p."

    ops: list[Gate] = []

    # Diagonal entries via local PHASE
    for i in range(n):
        t = int(C_sym[i, i] % p)
        if t:
            # PHASE acts like adding to the lower-left block on wire i
            ops += [PHASE(i, p)] * t

    # Off-diagonals via tiny BFS on the 2-wire subspace
    for i in range(n):
        for j in range(i + 1, n):
            c = int(C_sym[i, j] % p)
            if c == 0:
                continue
            # Build the target lower-symmetric elementary
            E = np.zeros((n, n), dtype=int)
            E[i, j] = (E[i, j] + c) % p
            E[j, i] = (E[j, i] + c) % p
            T = _full_from_lower(n, E, p)
            ops += _bfs_2q_to_target(n, p, T, i, j, max_depth=7)

    return ops


def synth_upper_from_symmetric_via_H(n: int, S_sym: np.ndarray, p: int) -> list[Gate]:
    """
    Build R(S) = [[I,S],[0,I]] using the H-sandwich:
      R(S) = (H_all) · L(-S) · (H_all)^(-1)   (right-multiplication)
    """
    S_sym = mod_p(S_sym, p)
    assert np.array_equal(S_sym, S_sym.T % p), "S_sym must be symmetric mod p."
    H_all = [Hadamard(q, p) for q in range(n)]
    H_all_inv = [Hadamard(q, p, inverse=True) for q in range(n)]
    ops_L = synth_lower_from_symmetric(n, (-S_sym) % p, p)
    return H_all + ops_L + H_all_inv


# =========================
#  FINAL DECOMPOSITION
# =========================

def decompose_symplectic_to_circuit(F: np.ndarray, p: int, *, check: bool = True) -> Circuit:
    """
    Return a Circuit whose composite symplectic equals F (mod p),
    consistent with your RIGHT-multiplication convention for Pauli action.

      1) Find C_pre so that A' of F' := C_pre · F is invertible.
      2) Split F' into blocks A,B,C,D and form the symmetric Schur pieces:
           B' := A^{-1} B,   C' := C A^{-1}
         (both must be symmetric over GF(p) for symplectic F')
      3) Synthesize
           R ~ [[I, B'], [0, I]]
           M ~ [[A, 0], [0, (A^T)^{-1}]]
           L ~ [[I, 0], [C', I]]
      4) Since F' = L M R, we have
           F = C_pre^{-1} · L · M · R
         With your Circuit accumulation rule (left-multiply accumulator),
         the gate list must be [R, M, L, C_pre^{-1}].

    Returns
    -------
    Circuit  (gates ordered so that Circuit.composite_gate().symplectic == F mod p)
    """
    F = np.asarray(F, dtype=int) % p
    assert is_symplectic_gfp(F, p), "F must be symplectic over GF(p)."
    n = F.shape[0] // 2

    # 1) Preconditioner
    C_pre = ensure_invertible_A_circuit(F, p)
    F_prime = (C_pre.full_symplectic() @ F) % p
    A, B, C, D = blocks(F_prime, p)

    # 2) Symmetric Schur complements
    A_inv = inv_gfp(A, p)
    Bp = (A_inv @ B) % p
    Cp = (C @ A_inv) % p
    if check:
        assert np.array_equal(Bp % p, Bp.T % p), "A^{-1} B must be symmetric (bug)."
        assert np.array_equal(Cp % p, Cp.T % p), "C A^{-1} must be symmetric (bug)."

    # 3) Synthesize each block
    ops_R = synth_upper_from_symmetric_via_H(n, Bp, p)      # [[I, B'], [0, I]]
    ops_M = synth_linear_A_to_gates(n, A, p)                # [[A, 0], [0, (A^T)^{-1}]]
    ops_L = synth_lower_from_symmetric(n, Cp, p)            # [[I, 0], [C', I]]

    # 4) Assemble with your circuit accumulation rule (left-multiply stack),
    #    but aiming for F = C_pre^{-1} · L · M · R  => append [R, M, L, C_pre^{-1}]
    C_tot = Circuit([p] * n, [])
    C_tot.add_gate(ops_R)
    C_tot.add_gate(ops_M)
    C_tot.add_gate(ops_L)
    C_tot.add_gate(C_pre.inv().gates)

    if check:
        F_rec = C_tot.composite_gate().symplectic % p
        if not np.array_equal(F_rec, F):
            # Minimal, targeted debug
            A_rec, B_rec, C_rec, D_rec = blocks(F_rec, p)
            A_tgt, B_tgt, C_tgt, D_tgt = blocks(F, p)
            raise AssertionError(
                "Decomposition round-trip failed.\n"
                f"Got:\n{F_rec}\nExp:\n{F}\n"
                f"[A got vs exp]\n{A_rec}\n{A_tgt}\n"
                f"[B got vs exp]\n{B_rec}\n{B_tgt}\n"
                f"[C got vs exp]\n{C_rec}\n{C_tgt}\n"
                f"[D got vs exp]\n{D_rec}\n{D_tgt}\n"
            )

    return C_tot
