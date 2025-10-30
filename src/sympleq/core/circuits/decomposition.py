import numpy as np
from sympleq.core.circuits import Circuit, Gate, Hadamard, PHASE, SUM, SWAP, CZ
from sympleq.core.circuits.utils import is_symplectic
import numpy as np
from typing import List, Tuple


def first_diff(A, B, p):
    A = mod_p(A, p); B = mod_p(B, p)
    D = (A - B) % p
    nz = np.argwhere(D != 0)
    if nz.size == 0:
        print("✓ no diffs")
        return None
    r, c = nz[0]
    print(f"✗ first diff at ({r},{c}): got {A[r,c]}, want {B[r,c]}")
    print(" row A:", A[r, :])
    print(" row B:", B[r, :])
    return nz

def symp_of_gates(n, p, gates):
    # Use YOUR composition path to stay consistent
    Ctmp = Circuit([p]*n, list(gates))
    return Ctmp.composite_gate().symplectic % p

def check_upper_from_S(n, p, S):
    I = np.eye(n, dtype=int); Z = np.zeros((n,n), dtype=int)
    return mod_p(np.block([[I, S],[Z, I]]), p)

def check_lower_from_C(n, p, C):
    I = np.eye(n, dtype=int); Z = np.zeros((n,n), dtype=int)
    return mod_p(np.block([[I, Z],[C, I]]), p)

def check_linear_from_A(n, p, A):
    D = mod_p(inv_gfp(A, p).T, p)
    Z = np.zeros((n,n), dtype=int)
    return mod_p(np.block([[A, Z],[Z, D]]), p)


def mod_p(A: np.ndarray, p: int) -> np.ndarray:
    return np.asarray(A, dtype=int) % p

def identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=int)

def zeros(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=int)

def mm_p(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    return mod_p(A @ B, p)

def inv_gfp(A: np.ndarray, p: int) -> np.ndarray:
    A = mod_p(A.copy(), p)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    Id = identity(n)
    aug = np.hstack((A, Id))
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(p).")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
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
    if F.shape[0] != F.shape[1] or n2 % 2 != 0:
        return False
    n = n2 // 2
    Id = identity(n) % p
    J = np.block([[zeros((n, n)), Id], [(-Id) % p, zeros((n, n))]]) % p
    lhs = mm_p(mm_p(F.T, J, p), F, p)
    return np.array_equal(lhs, J)

def blocks(F: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = mod_p(F, p)
    n = F.shape[0] // 2
    A = F[:n, :n]
    B = F[:n, n:]
    C = F[n:, :n]
    D = F[n:, n:]
    return A, B, C, D

# --------------------------------------------------------------------
# 1) Single-qudit SL(2,p) synthesis (PHASE/Hadamard only)
# --------------------------------------------------------------------

def _sl2_generators(p: int):
    """Return the generator matrices in SL(2,p) for S(±1) and H,H^{-1}."""
    S1  = np.array([[1, 1], [0, 1]], dtype=int) % p
    S_1 = np.array([[1, -1 % p], [0, 1]], dtype=int) % p
    H   = np.array([[0, -1], [1, 0]], dtype=int) % p
    Hinv= np.array([[0, 1], [-1, 0]], dtype=int) % p
    return [
        ("S+1", S1),
        ("S-1", S_1),
        ("H",   H),
        ("Hinv",Hinv),
    ]

def _synthesize_local_sl2(target: np.ndarray, p: int) -> List[Tuple[str, int]]:
    """
    BFS over SL(2,p) (tiny for small p) to find a short word in {S±1, H, Hinv}
    that equals `target`. Returns a sequence of ('S+1'/'S-1'/'H'/'Hinv', i)
    where the int is a placeholder (we map later to a concrete qudit index).
    """
    target = mod_p(target, p)
    gens = _sl2_generators(p)

    # BFS state: 2x2 matrix -> sequence of generator names
    from collections import deque
    start = tuple((np.eye(2, dtype=int) % p).flatten())
    tgt   = tuple(target.flatten())
    if start == tgt:
        return []

    Q = deque([(np.eye(2, dtype=int) % p, [])])
    seen = {start}

    while Q:
        M, ops = Q.popleft()
        for name, G in gens:
            M2 = (G @ M) % p
            key = tuple(M2.flatten())
            if key in seen:
                continue
            ops2 = ops + [name]
            if key == tgt:
                # Return as (name, -1); we insert the actual qudit index later.
                return [(nm, -1) for nm in ops2]
            seen.add(key)
            Q.append((M2, ops2))

    raise RuntimeError("SL(2,p) synthesis failed (unexpected for small p).")

def _local_D(u: int, p: int) -> np.ndarray:
    """Diagonal scaling in SL(2,p): diag(u, u^{-1})."""
    inv = pow(int(u % p), -1, p)
    return np.array([[u % p, 0], [0, inv]], dtype=int) % p

def _emit_local_ops_for_D(index: int, u: int, p: int) -> List:
    """
    Return a short sequence of Gate objects on `index` that realizes
    D(u) = diag(u, u^{-1}) in SL(2,p) using only PHASE and Hadamard.
    """
    target2x2 = _local_D(u, p)
    word = _synthesize_local_sl2(target2x2, p)  # [('S+1',-1), ('H',-1), ...]
    gates = []
    for name, _ in word:
        if name == "S+1":
            gates.append(PHASE(index, p))
        elif name == "S-1":
            # inverse = (p-1) repeats of PHASE
            for _ in range((p - 1) % p):
                gates.append(PHASE(index, p))
        elif name == "H":
            gates.append(Hadamard(index, p, inverse=False))
        elif name == "Hinv":
            gates.append(Hadamard(index, p, inverse=True))
        else:
            raise ValueError(f"Unknown local op {name}")
    return gates

# --------------------------------------------------------------------
# 2) Preconditioning: ensure A is invertible (returns Circuit and F_new)
# --------------------------------------------------------------------

def ensure_invertible_A_circuit(n: int, F: np.ndarray, p: int, max_depth: int | None = None) -> Circuit:
    """
    Find a Circuit C_pre such that the A block of (C_pre.full_symplectic(n) @ F) is invertible (mod p).
    Return C_pre only (do not apply it here).
    """
    F = F % p
    if _is_invertible_mod(F[:n, :n], p):
        return Circuit([p]*n, [])

    if max_depth is None:
        max_depth = max(1, 3*n)

    # candidate gates (objects)
    candidates = []
    for i in range(n):
        candidates.append(Hadamard(i, p))
        candidates.append(PHASE(i, p))
        candidates.append(("PHASE_INV", i))   # := PHASE repeated (p-1) times
    for i in range(n):
        for j in range(i+1, n):
            candidates.append(SWAP(i, j, p))
            candidates.append(SUM(i, j, p))
            candidates.append(SUM(j, i, p))

    def apply_left(Fmat: np.ndarray, g) -> np.ndarray:
        if isinstance(g, tuple) and g[0] == "PHASE_INV":
            i = g[1]
            for _ in range((p-1) % p):
                Fmat = (PHASE(i, p).full_symplectic(n) @ Fmat) % p
            return Fmat
        return (g.full_symplectic(n) @ Fmat) % p

    start_key = tuple(F.flatten())
    seen = {start_key}
    queue = [(F, [])]
    head = 0

    while head < len(queue):
        Fcur, ops = queue[head]; head += 1
        if len(ops) >= max_depth:
            continue
        for g in candidates:
            Fnew = apply_left(Fcur.copy(), g)
            if tuple(Fnew.flatten()) in seen:
                continue
            new_ops = ops + [g]
            if _is_invertible_mod(Fnew[:n, :n], p):
                C_pre = Circuit([p]*n, [])
                for op in new_ops:
                    if isinstance(op, tuple) and op[0] == "PHASE_INV":
                        i = op[1]
                        for _ in range((p-1) % p):
                            C_pre.add_gate(PHASE(i, p))
                    else:
                        C_pre.add_gate(op)
                return C_pre
            seen.add(tuple(Fnew.flatten()))
            queue.append((Fnew, new_ops))

    raise ValueError("Unable to make A invertible within depth.")


def _is_invertible_mod(A: np.ndarray, p: int) -> bool:
    try:
        _ = inv_gfp(A, p)
        return True
    except ValueError:
        return False

# --------------------------------------------------------------------
# 3) Symmetric upper & lower block synthesis using gate classes
# --------------------------------------------------------------------

def synth_upper_from_symmetric_gates(n: int, S: np.ndarray, p: int) -> List:
    """
    Build gates for [[I, S],[0, I]] with S symmetric (n x n).
    Use PHASE for diagonals, CZ for off-diagonals; repeat to implement coeffs.
    """
    S = mod_p(S, p)
    gates: List = []
    # diagonals
    for i in range(n):
        coeff = int(S[i, i] % p)
        for _ in range(coeff):
            gates.append(PHASE(i, p))
    # off-diagonals
    for i in range(n):
        for j in range(i + 1, n):
            coeff = int(S[i, j] % p)
            for _ in range(coeff):
                gates.append(CZ(i, j, p))
    return gates

def synth_lower_from_symmetric_gates(n: int, Csym: np.ndarray, p: int) -> List:
    """
    Build gates for [[I,0],[Csym, I]] using H • Upper(-Csym) • H^{-1}.
    """
    Csym = mod_p(Csym, p)
    ops: List = []
    # H on all
    for i in range(n):
        ops.append(Hadamard(i, p, inverse=False))
    # Upper with -Csym
    ops += synth_upper_from_symmetric_gates(n, (-Csym) % p, p)
    # H^{-1} on all
    for i in range(n):
        ops.append(Hadamard(i, p, inverse=True))
    return ops

# --------------------------------------------------------------------
# 4) Linear block synthesis: diag(A, (A^T)^(-1)) with SUM/SWAP + local D(u)
# --------------------------------------------------------------------

def _sl2_gens_mats(p: int):
    """Return generator matrices in SL(2,p) consistent with your Gate classes."""
    S_plus  = np.array([[1, 1], [0, 1]], dtype=int) % p    # PHASE (+1)
    S_minus = np.array([[1, -1 % p], [0, 1]], dtype=int)   # PHASE inverse
    H       = np.array([[0, -1 % p], [1, 0]], dtype=int)   # Hadamard (your non-inverse)
    Hinv    = np.array([[0, 1], [-1 % p, 0]], dtype=int)   # Hadamard inverse
    return [
        ("S+1",  S_plus),
        ("S-1",  S_minus),
        ("H",    H),
        ("Hinv", Hinv),
    ]

def _local_D_matrix(u: int, p: int) -> np.ndarray:
    """D(u) = diag(u, u^{-1}) in SL(2,p)."""
    u = int(u % p)
    if u == 0:
        raise ValueError("u must be non-zero modulo p")
    inv = pow(u, -1, p)
    return np.array([[u, 0], [0, inv]], dtype=int) % p

def _synthesize_D_as_gates(index: int, u: int, p: int) -> list:
    """
    Synthesize D(u) on qudit `index` as a list of Gate objects (PHASE/Hadamard),
    using RIGHT-multiplication word in SL(2,p). For p=2, u==1 always, so returns [].
    """
    u %= p
    if u == 1 % p:
        return []

    target = _local_D_matrix(u, p)
    gens = _sl2_gens_mats(p)

    # BFS over SL(2,p), ACCUMULATING ON THE RIGHT: M_next = M @ G
    from collections import deque
    I = np.eye(2, dtype=int) % p
    start_key = tuple(I.flatten())
    tgt_key = tuple(target.flatten())

    if start_key == tgt_key:
        return []

    Q = deque([(I, [])])
    seen = {start_key}

    while Q:
        M, word = Q.popleft()
        for name, G in gens:
            M2 = (M @ G) % p  # RIGHT-multiplication
            key = tuple(M2.flatten())
            if key in seen:
                continue
            word2 = word + [name]
            if key == tgt_key:
                # Map word -> gate objects in the SAME order (right-multiplication in time order)
                return _map_sl2_word_to_gates(index, word2, p)
            seen.add(key)
            Q.append((M2, word2))

    # Should never happen for small p
    raise RuntimeError(f"Could not synthesize D({u}) in SL(2,{p}).")

def _map_sl2_word_to_gates(index: int, word: list[str], p: int) -> list:
    """
    Map a right-multiplication SL(2,p) word ['S+1','H',...] to your Gate objects, same order.
    - 'S+1'  -> PHASE(index, p)
    - 'S-1'  -> PHASE(index, p) repeated (p-1) times
    - 'H'    -> Hadamard(index, p, inverse=False)
    - 'Hinv' -> Hadamard(index, p, inverse=True)
    """
    gates = []
    for name in word:
        if name == "S+1":
            gates.append(PHASE(index, p))
        elif name == "S-1":
            for _ in range((p - 1) % p):
                gates.append(PHASE(index, p))
        elif name == "H":
            gates.append(Hadamard(index, p, inverse=False))
        elif name == "Hinv":
            gates.append(Hadamard(index, p, inverse=True))
        else:
            raise ValueError(f"Unknown SL(2,p) generator name: {name}")
    return gates


def _as_scalar_dim(dim) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    return int(np.asarray(dim).reshape(-1)[0])


def _emit_local_ops_for_D(index: int, u: int, p: int) -> list:
    # Realize D(u)=diag(u,u^{-1}) on qudit `index` using PHASE/Hadamard (short BFS or precomputed words).
    # For p=2, u=1 only, so it’s a no-op. For odd p, keep the BFS version you already have.
    # Here’s the minimal stub that does nothing for u==1:
    u %= p
    if u == 1 % p:
        return []
    # If you already implemented the BFS helper, call it here:
    return _synthesize_D_as_gates(index, u, p)  # <-- use your existing SL(2,p) local synthesizer

def synth_linear_A_to_gates(n: int, A: np.ndarray, p: int) -> list:
    """
    RIGHT-multiplication consistent synthesis of [[A,0],[0,(A^T)^{-1}]].
    We factor A by column ops by doing Gaussian elimination on A^T with row ops,
    mapping each row-op L to a right column-op E = L^T, and emitting gates in that same order.
    """
    A = (A % p).copy()
    At = A.T.copy()  # operate on rows (left ops)
    ops: list = []

    for c in range(n):
        # --- pivot selection on A^T (row ops) ---
        pivot_row = None
        # prefer a 1 to avoid scaling
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
            # try above
            for r in range(0, c):
                if At[r, c] % p != 0:
                    pivot_row = r
                    break
        if pivot_row is None:
            raise RuntimeError("Unexpected: could not find pivot; A should be invertible.")

        # row swap on A^T  => column swap on A  => SWAP(pivot_row, c)
        if pivot_row != c:
            At[[c, pivot_row], :] = At[[pivot_row, c], :]
            ops.append(SWAP(pivot_row, c, p))

        # scale row c to 1 on A^T  => scale column c by same factor on A => D(inv_pivot) on column c
        pivot = int(At[c, c] % p)
        if pivot != 1 % p:
            inv_pivot = pow(pivot, -1, p)
            At[c, :] = (At[c, :] * inv_pivot) % p
            ops += _emit_local_ops_for_D(c, inv_pivot, p)

        # eliminate other rows in column c (on A^T): row i <- row i - factor * row c
        # maps to column add on A: col i <- col i - factor * col c  => SUM(c -> i) repeated (-factor) times
        for i in range(n):
            if i == c:
                continue
            factor = int(At[i, c] % p)
            if factor != 0:
                reps = (-factor) % p
                # Apply to A^T (left): row i -= factor * row c
                At[i, :] = (At[i, :] - factor * At[c, :]) % p
                # Emit column-add on A (right): SUM(c -> i) reps times
                for _ in range(reps):
                    ops.append(SUM(i, c, p))

    # No inversion of ops: emitted ops already satisfy  I * (∏ ops) has A in the X-block.
    return ops


def _invert_gate_list(ops: list, p: int) -> list:
    """
    Invert a list of Gate objects under left-multiplication.
    - SWAP, CZ are self-inverse.
    - SUM^{-1} = SUM^{p-1} (repeat SUM p-1 times).
    - PHASE^{-1} = PHASE^{p-1} (repeat PHASE p-1 times).
    - Hadamard inverse toggles the 'inverse' flag.
    Fallback: use gate.inv() (returns a generic Gate; fine for composing symplectics).
    """
    inv_ops: list = []

    for g in reversed(ops):
        # Normalize a scalar dimension for constructors
        d = _as_scalar_dim(getattr(g, "dimensions", p))

        if isinstance(g, SWAP) or isinstance(g, CZ):
            inv_ops.append(g)  # self-inverse

        elif isinstance(g, SUM):
            i, j = g.qudit_indices
            # inverse is coefficient -1 -> repeat SUM (p-1) times
            for _ in range((p - 1) % p):
                inv_ops.append(SUM(i, j, d))

        elif isinstance(g, PHASE):
            i = g.qudit_indices[0]
            for _ in range((p - 1) % p):
                inv_ops.append(PHASE(i, d))

        elif isinstance(g, Hadamard):
            i = g.qudit_indices[0]
            # Your Hadamard class encodes inverse in the constructor flag and name:
            # name == "H"  -> inverse=False
            # name == "H_inv" -> inverse=True
            is_inv = getattr(g, "name", "H") != "H"
            inv_ops.append(Hadamard(i, d, inverse=not is_inv))

        else:
            # Safe fallback if a new gate type appears; keeps symplectic correct.
            inv_ops.append(g.inv())

    return inv_ops


# --------------------------------------------------------------------
# 5) Compose a Circuit to a full-system symplectic (for verification)
# --------------------------------------------------------------------

def circuit_full_symplectic(C: 'Circuit') -> np.ndarray:
    n = C.n_qudits()
    F_tot = np.eye(2 * n, dtype=int)
    for g in C.gates:
        F_tot = (g.full_symplectic(n) @ F_tot) % C.dimensions[0]  # all dims equal (your code assumes uniform p here)
    return F_tot

# --------------------------------------------------------------------
# 6) Main: decompose to Circuit
# --------------------------------------------------------------------

def decompose_symplectic_to_circuit(F: np.ndarray, p: int) -> Circuit:
    """
    Emit a Circuit of known gates (Hadamard, PHASE, SUM, SWAP, CZ) whose
    composite_gate().symplectic equals F (mod p).  100% aligned with
    right-multiplication convention used by Circuit/Gate.
    """
    F = F % p
    assert is_symplectic_gfp(F, p), "F must be symplectic."
    n = F.shape[0] // 2

    # 1) Precondition (find C_pre), then form F' = C_pre • F (virtually)
    C_pre = ensure_invertible_A_circuit(n, F, p)
    print(f"[dbg] p={p}, n={n}")
    print("[dbg] verifying preconditioner makes A invertible...")
    F_prime = (C_pre.full_symplectic() @ F) % p
    A, B, C, D = blocks(F_prime, p)
    print("  rank/invertible A:", _is_invertible_mod(A, p))
    A_inv = inv_gfp(A, p)
    Bp = (A_inv @ B) % p
    Cp = (C @ A_inv) % p
    print("  symmetry A^{-1}B:", np.array_equal(Bp, Bp.T % p))
    print("  symmetry C A^{-1}:", np.array_equal(Cp, Cp.T % p))

    if not np.array_equal(Bp, Bp.T % p):
        raise ValueError("A^{-1}B not symmetric (bug or non-symplectic input).")
    if not np.array_equal(Cp, Cp.T % p):
        raise ValueError("CA^{-1} not symmetric (bug or non-symplectic input).")

    # 3) Synthesize R, M, L as gate lists (right product)
    ops_R = synth_upper_from_symmetric_gates(n, Bp, p)      # [[I,B'],[0,I]]
    ops_M = synth_linear_A_to_gates(n, A, p)                # diag(A, (A^T)^(-1))
    ops_L = synth_lower_from_symmetric_gates(n, Cp, p)      # [[I,0],[C',I]]


    # --- Verify R ---
    print("[dbg] checking R synthesis...")
    FR = symp_of_gates(n, p, ops_R)
    FR_expect = check_upper_from_S(n, p, Bp)
    first_diff(FR, FR_expect, p)

    # --- Verify M ---
    print("[dbg] checking M synthesis...")
    FM = symp_of_gates(n, p, ops_M)
    FM_expect = check_linear_from_A(n, p, A)
    first_diff(FM, FM_expect, p)

    # --- Verify L ---
    print("[dbg] checking L synthesis...")
    FL = symp_of_gates(n, p, ops_L)
    FL_expect = check_lower_from_C(n, p, Cp)
    first_diff(FL, FL_expect, p)

    # 4) Assemble as: C_pre^{-1}, R, M, L
    C = Circuit([p]*n, [])
    C.add_gate(ops_L)              # last in math order
    C.add_gate(ops_M)
    C.add_gate(ops_R)
    C.add_gate(C_pre.inv().gates)  # first in math order
    print("[dbg] verifying final right-product order: C_pre^{-1} · R · M · L")

    F_pre_inv = symp_of_gates(n, p, C_pre.inv().gates)
    first_diff(F_pre_inv, np.linalg.inv(C_pre.full_symplectic()) % p, p)  # sanity (mod p inverse)

    FRML = mod_p(F_pre_inv @ FR @ FM @ FL, p)
    print("[dbg] comparing FRML to target F...")
    first_diff(FRML, F, p)

    # Build the *expected* product from the three verified blocks:
    F_pre_inv = symp_of_gates(n, p, C_pre.inv().gates)
    FR = symp_of_gates(n, p, ops_R)
    FM = symp_of_gates(n, p, ops_M)
    FL = symp_of_gates(n, p, ops_L)
    FRML = (F_pre_inv @ FR @ FM @ FL) % p

    # Compare to what the *single* Circuit C reports:
    F_rec = C.composite_gate().symplectic % p

    print("[cmp] FRML vs circuit.composite_gate:")
    nz = np.argwhere((FRML - F_rec) % p != 0)
    print("  diffs:", len(nz))
    if nz.size:
        r, c = nz[0]
        print(f"  first diff at ({r},{c}): FRML={FRML[r,c]}, C={F_rec[r,c]}")

    def summarize(label, gates):
        print(f"{label} (len={len(gates)}):")
        for k, g in enumerate(gates):
            print(f"  {k:3d}: {g.name:6s} idx={list(g.qudit_indices)} dim={int(np.asarray(g.dimensions).reshape(-1)[0])}")

    print()
    summarize("C_pre_inv", C_pre.inv().gates)
    summarize("ops_R",     ops_R)
    summarize("ops_M",     ops_M)
    summarize("ops_L",     ops_L)
    summarize("C.gates",   C.gates)        

    A_rec, B_rec, C_rec, D_rec = blocks(F_rec, p)
    A_tgt, B_tgt, C_tgt, D_tgt = blocks(F, p)

    print("[cmp] A block:"); first_diff(A_rec, A_tgt, p)
    print("[cmp] B block:"); first_diff(B_rec, B_tgt, p)
    print("[cmp] C block:"); first_diff(C_rec, C_tgt, p)
    print("[cmp] D block:"); first_diff(D_rec, D_tgt, p)

    # 5) Verify using YOUR right-multiplication path
    F_rec = C.composite_gate().symplectic % p
    if not np.array_equal(F_rec, F % p):
        print("==== DEBUG REPORT ====")
        n = F.shape[0] // 2
        # Use the checks above
        # 1) preconditioner and F'
        C_pre = ensure_invertible_A_circuit(n, F, p)
        F_prime = (C_pre.full_symplectic() @ F) % p
        A, B, C_, D = blocks(F_prime, p)
        A_inv = inv_gfp(A, p)
        Bp = (A_inv @ B) % p
        Cp = (C_ @ A_inv) % p
        print("[dbg] A invertible:", _is_invertible_mod(A, p))
        print("[dbg] sym(A^{-1}B):", np.array_equal(Bp, Bp.T % p))
        print("[dbg] sym(C A^{-1}):", np.array_equal(Cp, Cp.T % p))

        ops_R = synth_upper_from_symmetric_gates(n, Bp, p)
        ops_M = synth_linear_A_to_gates(n, A, p)
        ops_L = synth_lower_from_symmetric_gates(n, Cp, p)

        FR = symp_of_gates(n, p, ops_R)
        FM = symp_of_gates(n, p, ops_M)
        FL = symp_of_gates(n, p, ops_L)

        print("[dbg] R block:")
        first_diff(FR, check_upper_from_S(n, p, Bp), p)
        print("[dbg] M block:")
        first_diff(FM, check_linear_from_A(n, p, A), p)
        print("[dbg] L block:")
        first_diff(FL, check_lower_from_C(n, p, Cp), p)

        FRML = (symp_of_gates(n, p, C_pre.inv().gates) @ FR @ FM @ FL) % p
        print("[dbg] final product vs F:")
        first_diff(FRML, F, p)

        raise AssertionError(f"Internal check failed: reconstructed F != input F.\nF_rec:\n{F_rec}\nF:\n{F % p}")


    return C


def random_symmetric(n: int, rng: np.random.Generator, p: int) -> np.ndarray:
    M = rng.integers(0, p, size=(n, n), dtype=int)
    return mod_p(M + M.T, p)


def random_invertible(n: int, rng: np.random.Generator, p: int) -> np.ndarray:
    while True:
        A = rng.integers(0, p, size=(n, n), dtype=int)
        try:
            _ = inv_gfp(A, p)
            return mod_p(A, p)
        except ValueError:
            continue


def build_F_from_LMR(n: int, A: np.ndarray, Bsym: np.ndarray, Csym: np.ndarray, p: int) -> np.ndarray:
    A = mod_p(A, p)
    Bsym = mod_p(Bsym, p)
    Csym = mod_p(Csym, p)

    A_invT = inv_gfp(A.T, p)
    L = np.block([[identity(n), zeros((n, n))],
                  [Csym, identity(n)]])
    M = np.block([[A, zeros((n, n))],
                  [zeros((n, n)), A_invT]])
    R = np.block([[identity(n), Bsym],
                  [zeros((n, n)), identity(n)]])
    return mm_p(mm_p(L, M, p), R, p)


def random_symplectic(n: int, p: int, rng=None, steps: int = 5) -> np.ndarray:
    """Lightweight scrambler (not uniform) built from generators preserving Ω."""
    if rng is None:
        rng = np.random.default_rng()
    F = np.eye(2*n, dtype=np.int64)
    # qudit permutation
    perm = np.arange(n); rng.shuffle(perm)
    P = np.eye(n, dtype=np.int64)[perm]
    O = np.zeros((n, n), dtype=np.int64)
    F = mm_p(np.block([[P, O], [O, P]]), F, p)
    # local X↔Z swaps (Hadamard-like): [[0,1],[-1,0]] at sites
    H = np.array([[0, 1], [-1 % p, 0]], dtype=np.int64)
    D = np.eye(2*n, dtype=np.int64)
    for q in range(n):
        if rng.integers(0, 2):
            ix, iz = q, n + q
            D[[ix, ix, iz, iz], [ix, iz, ix, iz]] = [0, 1, (-1) % p, 0]
    F = mm_p(D, F, p)
    # symmetric shears (upper and lower)
    for _ in range(steps):
        A = rng.integers(0, p, size=(n, n), dtype=np.int64)
        A = mod_p(A + A.T, p)
        U = np.block([[np.eye(n, dtype=np.int64), A],
                      [np.zeros((n, n), dtype=np.int64), np.eye(n, dtype=np.int64)]])
        F = mm_p(U, F, p)
        B = rng.integers(0, p, size=(n, n), dtype=np.int64)
        B = mod_p(B + B.T, p)
        L = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, n), dtype=np.int64)],
                      [B, np.eye(n, dtype=np.int64)]])
        F = mm_p(L, F, p)
    assert is_symplectic(F, p)
    return F


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    # for p in (2, 3, 5, 7):
    #     for n in range(1, 5):
    #         for _ in range(20):
    #             A = random_invertible(n, rng, p)
    #             B = random_symmetric(n, rng, p)
    #             C = random_symmetric(n, rng, p)
    #             F = build_F_from_LMR(n, A, B, C, p)

    #             gates_tuple = decompose_symplectic_gfp(F, p)
    #             recon = compose_symplectic_from_gates(n, gates_tuple, p)
    #             assert np.array_equal(recon, F), "Symplectic reconstruction failed."

    #             try:
    #                 circuit = decompose_symplectic_to_circuit(F, p)
    #             except Exception:
    #                 circuit = None

    #             if circuit is not None:
    #                 allowed = {"H", "S", "SUM", "SWAP", "CZ"}
    #                 circuit_names = [gate.name for gate in circuit.gates]
    #                 assert all(name in allowed for name in circuit_names), f"Unexpected gate type in {circuit_names}."

    #             # Exercise preprocessing when A is singular
    #             for idx in range(n):
    #                 F_singular = mm_p(gate_H(n, idx, p), F, p)
    #                 if is_invertible_mod(F_singular[:n, :n], p):
    #                     continue
    #                 gates_tuple_sing = decompose_symplectic_gfp(F_singular, p)
    #                 recon_sing = compose_symplectic_from_gates(n, gates_tuple_sing, p)
    #                 assert np.array_equal(recon_sing, F_singular), "Preprocessing reconstruction failed."
    #                 break

    # # Additional tests targeting initially singular A blocks
    # for p in (2, 3, 5, 7):
    #     for n in range(2, 5):
    #         candidates = [("H", i) for i in range(n)]
    #         candidates.extend(("SWAP", i, j) for i in range(n) for j in range(i + 1, n))
    #         for _ in range(30):
    #             A = random_invertible(n, rng, p)
    #             B = random_symmetric(n, rng, p)
    #             C = random_symmetric(n, rng, p)
    #             F = build_F_from_LMR(n, A, B, C, p)

    #             F_singular = F.copy()
    #             made_singular = False
    #             for _ in range(3 * n):
    #                 gate = candidates[rng.integers(len(candidates))]
    #                 F_singular = _apply_gate_tuple(F_singular, gate, p)
    #                 if not is_invertible_mod(F_singular[:n, :n], p):
    #                     made_singular = True
    #                     break
    #             if not made_singular:
    #                 continue

    #             gates_tuple = decompose_symplectic_gfp(F_singular, p)
    #             recon = compose_symplectic_from_gates(n, gates_tuple, p)
    #             assert np.array_equal(recon, F_singular), "Singular-A preprocessing failed."

    # # Additional tests targeting initially singular A blocks
    # print('Done first two')

    # print(gate_CZ(2, 0, 1, 1, 2))
    # for p in (2, 3, 5, 7):
    #     for n in range(3, 6):

    #         for _ in range(10):

    p = 3
    n = 3
    F = random_symplectic(n, p, rng=rng)

    circuit = decompose_symplectic_to_circuit(F, p)
    print(f'tuple: len({len(circuit)})')
    print(circuit)
    F_reconstructed = circuit.composite_gate().symplectic
    print(F'circuit: len({len(circuit.gates)})')
    assert np.array_equal(F_reconstructed, F), f"Failed. Reconstructed:\n{F_reconstructed}\nOriginal:\n{F}"

    print("All GF(p) symplectic decomposition checks passed.")
