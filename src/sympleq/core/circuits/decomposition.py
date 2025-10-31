import numpy as np
from sympleq.core.circuits import Circuit, Gate, Hadamard, PHASE, SUM, SWAP
from sympleq.core.circuits.utils import is_symplectic
from collections import deque

def _compose_1q_on_wire(q: int, p: int, gates: list[Gate]) -> np.ndarray:
    """
    Remap a list of *single-qudit* gates that act on wire `q` to a 1-qudit
    circuit acting on wire 0, then return its 2x2 symplectic over GF(p).
    """
    remapped: list[Gate] = []
    for g in gates:
        # Rebuild the same gate type but on index 0 and scalar dim p
        if isinstance(g, Hadamard):
            inv = getattr(g, "is_inverse", (g.name == "H_inv"))
            remapped.append(Hadamard(0, int(p), inverse=bool(inv)))
        elif isinstance(g, PHASE):
            remapped.append(PHASE(0, int(p)))
        else:
            raise ValueError(f"_compose_1q_on_wire only supports single-qudit H/S; got {type(g)}")

    return Circuit([int(p)], remapped).composite_gate().symplectic % int(p)

def _compose_symp_full(n: int, p: int, gates: list[Gate]) -> np.ndarray:
    return Circuit([p]*n, list(gates)).composite_gate().symplectic % p

# --- generators on 1–2 chosen qudits (indices are 0-based wire ids) ---
def _gens_1q(q: int, p: int) -> list[list[Gate]]:
    return [
        [PHASE(q, p)],
        [Hadamard(q, p)],
        [Hadamard(q, p, inverse=True)],
    ]

def _gens_2q(i: int, j: int, p: int) -> list[list[Gate]]:
    G = []
    # single-qudit actions on both wires
    G += _gens_1q(i, p)
    G += _gens_1q(j, p)
    # two-qudit interactions in both directions
    G += [[SUM(i, j, p)], [SUM(j, i, p)]]
    # (optional) allow SWAP; harmless and can shorten words
    G += [[SWAP(i, j, p)]]
    return G

def _bfs_synth_2q_target(n: int, p: int, target_full: np.ndarray, i: int, j: int, max_depth: int = 7) -> list[Gate]:
    I = np.eye(2*n, dtype=int) % p
    if np.array_equal(target_full % p, I): return []
    gens = _gens_2q(i, j, p)
    Q = deque([([], I)])
    seen = {tuple(I.flatten())}
    while Q:
        word, M = Q.popleft()
        if len(word) >= max_depth: continue
        for g in gens:
            new_word = word + g
            M2 = _compose_symp_full(n, p, new_word)
            k = tuple(M2.flatten())
            if k in seen: continue
            if np.array_equal(M2, target_full % p): return new_word
            seen.add(k); Q.append((new_word, M2))
    raise RuntimeError("2q BFS failed to synthesize target (depth cap too small?)")


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


def _target_upper_pair(n: int, i: int, j: int, s: int, p: int) -> np.ndarray:
    I = np.eye(n, dtype=int); Z = np.zeros((n,n), dtype=int)
    S = np.zeros((n,n), dtype=int)
    s %= p
    S[i,j] = (S[i,j] + s) % p
    S[j,i] = (S[j,i] + s) % p
    return np.block([[I, S],[Z, I]]) % p

_cp_cache = {}
def _cp_macro(n, p, i, j, s):
    """Return a short word for [[I, s(E_ij+E_ji)],[0,I]] using H and SUM."""
    key = (n, p, i, j)
    if key not in _cp_cache:
        # two plausible sandwiches
        cand1 = [Hadamard(j, p), SUM(i, j, p), Hadamard(j, p, inverse=True)]
        cand2 = [Hadamard(i, p), SUM(i, j, p), Hadamard(i, p, inverse=True)]
        F1 = _compose_symp_full(n, p, cand1); sgn1 = int(F1[:n, n:][i, j] % p)
        F2 = _compose_symp_full(n, p, cand2); sgn2 = int(F2[:n, n:][i, j] % p)
        if sgn1 != 0: _cp_cache[key] = ("j", sgn1)
        elif sgn2 != 0: _cp_cache[key] = ("i", sgn2)
        else: raise RuntimeError("Could not calibrate CP from H/SUM for this convention.")
    which, sgn = _cp_cache[key]
    t = (s * pow(sgn, -1, p)) % p  # how many SUMs to get s
    if t == 0: return []
    if which == "j":
        return [Hadamard(j, p)] + [SUM(i, j, p)] * t + [Hadamard(j, p, inverse=True)]
    else:
        return [Hadamard(i, p)] + [SUM(i, j, p)] * t + [Hadamard(i, p, inverse=True)]


def synth_upper_from_symmetric_gates(n: int, S: np.ndarray, p: int) -> list[Gate]:
    S = (np.asarray(S, dtype=int) % p)
    ops: list[Gate] = []
    # diagonals: local PHASE
    for i in range(n):
        for _ in range(int(S[i, i] % p)):
            ops.append(PHASE(i, p))
    # off-diagonals: calibrated CP macro
    for i in range(n):
        for j in range(i+1, n):
            s = int(S[i, j] % p)
            if s: ops += _cp_macro(n, p, i, j, s)
    return ops

def synth_lower_from_symmetric_gates(n: int, Csym: np.ndarray, p: int) -> list[Gate]:
    Csym = (np.asarray(Csym, dtype=int) % p)
    ops: list[Gate] = [Hadamard(i, p) for i in range(n)]
    ops += synth_upper_from_symmetric_gates(n, (-Csym) % p, p)
    ops += [Hadamard(i, p, inverse=True) for i in range(n)]
    return ops


# --------------------------------------------------------------------
# 4) Linear block synthesis: diag(A, (A^T)^(-1)) with SUM/SWAP + local D(u)
# --------------------------------------------------------------------

def _emit_local_ops_for_D(index: int, u: int, p: int) -> list[Gate]:
    u %= p
    if u == 1 % p:
        return []
    target = np.array([[u, 0], [0, pow(int(u), -1, p)]], dtype=int) % p

    # candidate: S(u) ; H ; S(-u^{-1}) ; H ; S(u)
    inv_u = pow(int(u), -1, p)
    cand: list[Gate] = []
    cand += [PHASE(index, p) for _ in range(int(u % p))]
    cand += [Hadamard(index, p)]
    cand += [PHASE(index, p) for _ in range(int((-inv_u) % p))]
    cand += [Hadamard(index, p)]
    cand += [PHASE(index, p) for _ in range(int(u % p))]

    # verify on a remapped 1-qudit circuit
    if np.array_equal(_compose_1q_on_wire(index, p, cand), target):
        return cand

    # fallback BFS: also use the remapped composer inside it
    from collections import deque
    I2 = np.eye(2, dtype=int) % p
    gens = [[PHASE(index, p)], [Hadamard(index, p)], [Hadamard(index, p, inverse=True)]]
    Q = deque([([], I2)])
    seen = {tuple(I2.flatten())}
    max_depth = 6

    while Q:
        word, M = Q.popleft()
        if len(word) >= max_depth:
            continue
        for g in gens:
            new_word = word + g
            M2 = _compose_1q_on_wire(index, p, new_word)
            k = tuple(M2.flatten())
            if k in seen:
                continue
            if np.array_equal(M2, target):
                return new_word
            seen.add(k)
            Q.append((new_word, M2))

    raise RuntimeError(f"Could not synthesize D({u}) in SL(2,{p}) with current gates.")


# put near other helpers
_sum_dir_cache = {}
def _sum_is_col_add(n, p, src, dst) -> bool:
    """Return True iff SUM(src,dst) realizes col[dst] += col[src] in A (right-mult)."""
    F1 = _compose_symp_full(n, p, [SUM(src, dst, p)])
    A1 = F1[:n, :n]
    E  = np.eye(n, dtype=int); E[dst, src] = (E[dst, src] + 1) % p  # col-add on the right
    return np.array_equal(A1, (np.eye(n, dtype=int) @ E) % p)

def _col_add_ops(n, p, src, dst, reps):
    key = (n, p, src, dst)
    if key not in _sum_dir_cache:
        _sum_dir_cache[key] = _sum_is_col_add(n, p, src, dst)
    if _sum_dir_cache[key]:
        return [SUM(dst, src, p) for _ in range(reps)]
    else:
        # SUM class wired oppositely; fall back to flipped wiring
        return [SUM(src, dst, p) for _ in range(reps)]



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
                    ops += _col_add_ops(n, p, c, i, reps)   # col[i] += reps * col[c]

    return ops

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
    C.add_gate(ops_R)              # last in math order
    C.add_gate(ops_M)
    C.add_gate(ops_L)
    C.add_gate(C_pre.inv().gates)  # first in math order
    print("[dbg] verifying final right-product order: C_pre^{-1} · L · M · R")

    F_pre_inv = symp_of_gates(n, p, C_pre.inv().gates)
    def symp_inv(F, p):
        n = F.shape[0] // 2
        J = np.block([[np.zeros((n,n), int), np.eye(n, dtype=int)],
                    [(-np.eye(n,  dtype=int)) % p, np.zeros((n,n), int)]]) % p
        return (-J @ F.T @ J) % p

    first_diff(F_pre_inv, symp_inv(C_pre.full_symplectic(), p), p)


    FRML = mod_p(F_pre_inv @ FL @ FM @ FR, p)
    print("[dbg] comparing FRML to target F...")
    first_diff(FRML, F, p)

    # Build the *expected* product from the three verified blocks:
    F_pre_inv = symp_of_gates(n, p, C_pre.inv().gates)
    FR = symp_of_gates(n, p, ops_R)
    FM = symp_of_gates(n, p, ops_M)
    FL = symp_of_gates(n, p, ops_L)
    FRML = (F_pre_inv @ FL @ FM @ FR) % p

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
            print(
                f"  {k:3d}: {g.name:6s} idx={list(g.qudit_indices)} dim={int(np.asarray(g.dimensions).reshape(-1)[0])}")

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

    p = 2
    n = 3
    F = random_symplectic(n, p)  #, rng=rng

    circuit = decompose_symplectic_to_circuit(F, p)
    print(f'tuple: len({len(circuit)})')
    print(circuit)
    F_reconstructed = circuit.composite_gate().symplectic
    print(F'circuit: len({len(circuit.gates)})')
    assert np.array_equal(F_reconstructed, F), f"Failed. Reconstructed:\n{F_reconstructed}\nOriginal:\n{F}"

    print("All GF(p) symplectic decomposition checks passed.")
