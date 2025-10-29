import numpy as np
from collections import deque
from sympleq.core.circuits import Circuit, Gate, Hadamard, PHASE, SUM, SWAP, CZ
from sympleq.core.paulis import PauliSum, PauliString


def mod_p(A, p: int) -> np.ndarray:
    """Return A reduced modulo p with integer dtype."""
    return np.asarray(A, dtype=int) % p


def mm_p(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """Matrix multiplication modulo p."""
    return mod_p(A @ B, p)


def identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=int)


def zeros(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=int)


def is_invertible_mod(A: np.ndarray, p: int) -> bool:
    try:
        _ = inv_gfp(A, p)
        return True
    except ValueError:
        return False


def is_symplectic_gfp(F: np.ndarray, p: int) -> bool:
    F = mod_p(F, p)
    n2 = F.shape[0]
    if F.shape[0] != F.shape[1] or n2 % 2 != 0:
        return False
    n = n2 // 2
    Id = identity(n) % p
    minus_I = (-Id) % p
    J = np.block([[zeros((n, n)), Id], [minus_I, zeros((n, n))]]) % p
    lhs = mm_p(mm_p(F.T, J, p), F, p)
    return np.array_equal(lhs, J)


def inv_gfp(A: np.ndarray, p: int) -> np.ndarray:
    """Matrix inverse over GF(p) via Gauss-Jordan elimination."""
    A = mod_p(A.copy(), p)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to invert over GF(p).")
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
        pivot_val = aug[row, col] % p
        inv_pivot = pow(int(pivot_val), -1, p)
        aug[row, :] = (aug[row, :] * inv_pivot) % p
        for r in range(n):
            if r == row:
                continue
            factor = aug[r, col] % p
            if factor != 0:
                aug[r, :] = (aug[r, :] - factor * aug[row, :]) % p
        row += 1
    return aug[:, n:] % p


def gate_H(n: int, i: int, p: int) -> np.ndarray:
    """Generalised Fourier gate on qudit i."""
    F = identity(2 * n)
    F[i, i] = 0
    F[i, n + i] = 1 % p
    F[n + i, i] = (-1) % p
    F[n + i, n + i] = 0
    return mod_p(F, p)


def gate_S(n: int, i: int, coeff: int, p: int) -> np.ndarray:
    """Phase gate adding coeff * X_i into Z_i."""
    F = identity(2 * n)
    F[i, n + i] = (F[i, n + i] + coeff) % p
    return F


def gate_CZ(n: int, i: int, j: int, coeff: int, p: int) -> np.ndarray:
    """Symmetric phase interaction with strength coeff."""
    F = identity(2 * n)
    coeff %= p
    if i == j:
        F[i, n + i] = (F[i, n + i] + coeff) % p
    else:
        F[i, n + j] = (F[i, n + j] + coeff) % p
        F[j, n + i] = (F[j, n + i] + coeff) % p
    return F


def _gate_linear_from_A(A_block: np.ndarray, p: int) -> np.ndarray:
    """Build a symplectic matrix from a pure X-linear block."""
    n = A_block.shape[0]
    A_block = mod_p(A_block, p)
    A_inv = inv_gfp(A_block, p)
    D_block = mod_p(A_inv.T, p)
    top = np.hstack((A_block, zeros((n, n))))
    bottom = np.hstack((zeros((n, n)), D_block))
    return np.vstack((top, bottom)) % p


def gate_SUM(n: int, src: int, dst: int, coeff: int, p: int) -> np.ndarray:
    """Add coeff * row src into row dst on the X block."""
    if src == dst:
        raise ValueError("SUM gate requires src != dst.")
    coeff %= p
    A = identity(n)
    A[dst, src] = (A[dst, src] + coeff) % p
    return _gate_linear_from_A(A, p)


def gate_SCALE(n: int, i: int, scalar: int, p: int) -> np.ndarray:
    """Scale row i of the X block by scalar."""
    scalar %= p
    if scalar == 0:
        raise ValueError("Scaling factor must be non-zero modulo p.")
    A = identity(n)
    A[i, i] = (A[i, i] * scalar) % p
    return _gate_linear_from_A(A, p)


def gate_SWAP(n: int, i: int, j: int, p: int) -> np.ndarray:
    """Swap qudits i and j."""
    if i == j:
        return identity(2 * n)
    A = identity(n)
    A[[i, j], :] = A[[j, i], :]
    return _gate_linear_from_A(A, p)


def _apply_gate_tuple(F: np.ndarray, gate: tuple, p: int) -> np.ndarray:
    kind = gate[0]
    n = F.shape[0] // 2
    if kind == "H":
        Gi = gate_H(n, gate[1], p)
    elif kind == "SWAP":
        Gi = gate_SWAP(n, gate[1], gate[2], p)
    elif kind == "S":
        Gi = gate_S(n, gate[1], gate[2], p)
    elif kind == "SUM":
        Gi = gate_SUM(n, gate[1], gate[2], gate[3], p)
    else:
        raise ValueError(f"Unsupported gate in ensure invertible routine: {gate}")
    return mm_p(Gi, F, p)


def ensure_invertible_A(F: np.ndarray, p: int, max_depth: int | None = None) -> tuple[list[tuple], np.ndarray]:
    """
    Find a sequence of elementary symplectic gates that makes the A block invertible.

    Parameters
    ----------
    F : np.ndarray
        2n x 2n symplectic matrix over GF(p).
    p : int
        Prime modulus.
    max_depth : int | None
        Optional search depth limit.

    Returns
    -------
    ops : list[tuple]
        Gate tuples (using \"H\" and \"SWAP\") to left-multiply F.
    F_new : np.ndarray
        Updated matrix with invertible top-left block.
    """
    n = F.shape[0] // 2
    if is_invertible_mod(F[:n, :n], p):
        return [], F

    print('Not invertible')

    if max_depth is None:
        max_depth = max(1, 3 * n)

    candidates: list[tuple] = [("H", i) for i in range(n)]
    candidates += [("S", i, 1) for i in range(n)]
    candidates += [("S", i, (-1) % p) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append(("SWAP", i, j))
            candidates.append(("SUM", i, j, 1))
            candidates.append(("SUM", j, i, 1))

    queue: deque[tuple[np.ndarray, list[tuple]]] = deque()
    visited = {tuple(F.flatten())}
    queue.append((F, []))

    while queue:
        current_F, ops = queue.popleft()
        if len(ops) >= max_depth:
            continue
        for gate in candidates:
            new_ops = ops + [gate]
            new_F = _apply_gate_tuple(current_F, gate, p)
            key = tuple(new_F.flatten())
            if key in visited:
                continue
            if is_invertible_mod(new_F[:n, :n], p):
                return new_ops, new_F
            visited.add(key)
            queue.append((new_F, new_ops))

    raise ValueError("Unable to find symplectic preprocessing to make A invertible.")


def _invert_gate_tuples(ops: list[tuple], p: int) -> list[tuple]:
    inverse_ops: list[tuple] = []
    for gate in reversed(ops):
        kind = gate[0]
        if kind == "SWAP":
            inverse_ops.append(gate)
        elif kind == "H":
            idx = gate[1]
            neg_one = (-1) % p
            if neg_one != 1 % p:
                inverse_ops.append(("MUL", idx, neg_one))
            inverse_ops.append(("H", idx))
        elif kind == "S":
            idx, coeff = gate[1], gate[2]
            inverse_ops.append(("S", idx, (-coeff) % p))
        elif kind == "SUM":
            src, dst, coeff = gate[1], gate[2], gate[3]
            inverse_ops.append(("SUM", src, dst, (-coeff) % p))
        else:
            raise ValueError(f"Cannot invert gate {gate} in preprocessing.")
    return inverse_ops


def H_all(n: int, p: int) -> list[tuple[str, int]]:
    """Return a layer applying the Fourier gate to all qudits."""
    return [("H", i) for i in range(n)]


def synth_upper_from_symmetric(n: int, S: np.ndarray, p: int) -> list[tuple]:
    """Synthesise [[I, S], [0, I]] with symmetric S over GF(p)."""
    S = mod_p(S, p)
    gates: list[tuple] = []
    for i in range(n):
        coeff = int(S[i, i] % p)
        if coeff != 0:
            gates.append(("S", i, coeff))
    for i in range(n):
        for j in range(i + 1, n):
            coeff = int(S[i, j] % p)
            if coeff != 0:
                gates.append(("CZ", i, j, coeff))
    return gates


def synth_lower_from_symmetric(n: int, C_sym: np.ndarray, p: int) -> list[tuple]:
    """Synthesise [[I,0],[C_sym, I]] via Fourier conjugation."""
    C_sym = mod_p(C_sym, p)
    ops: list[tuple] = []
    ops += H_all(n, p)
    ops += synth_upper_from_symmetric(n, (-C_sym) % p, p)
    neg_one = (-1) % p
    for i in range(n):
        if neg_one != 1 % p:
            ops.append(("MUL", i, neg_one))
        ops.append(("H", i))
    return ops


def synth_linear_A(n: int, A: np.ndarray, p: int) -> list[tuple]:
    """
    Synthesize [[A,0],[0,(A^T)^{-1}]] using SWAP, SCALE, and SUM gates.
    Operations are recorded in the order they are applied.
    """
    A = mod_p(A.copy(), p)
    ops: list[tuple] = []
    for c in range(n):
        pivot_row = None
        for r in range(c, n):
            if A[r, c] % p != 0:
                pivot_row = r
                break
        if pivot_row is None:
            for r in range(0, c):
                if A[r, c] % p != 0:
                    pivot_row = r
                    break
        if pivot_row is None:
            raise RuntimeError("Unexpected: could not find pivot; A should be invertible.")
        if pivot_row != c:
            A[[c, pivot_row], :] = A[[pivot_row, c], :]
            ops.append(("SWAP", pivot_row, c))
        pivot = A[c, c] % p
        if pivot != 1 % p:
            inv_pivot = pow(int(pivot), -1, p)
            A[c, :] = (A[c, :] * inv_pivot) % p
            ops.append(("MUL", c, inv_pivot))
        for i in range(n):
            if i == c:
                continue
            factor = A[i, c] % p
            if factor != 0:
                A[i, :] = (A[i, :] - factor * A[c, :]) % p
                ops.append(("SUM", c, i, (-factor) % p))

    def invert_op(op: tuple[str, int, int, int | None]) -> tuple:
        kind = op[0]
        if kind == "SWAP":
            return op
        if kind == "MUL":
            idx, factor = op[1], op[2]
            inv_factor = pow(int(factor), -1, p)
            return ("MUL", idx, inv_factor)
        if kind == "SUM":
            src, dst, coeff = op[1], op[2], op[3]
            return ("SUM", src, dst, (-coeff) % p)
        raise ValueError(f"Unknown operation {op}")

    return [invert_op(op) for op in reversed(ops)]


def blocks(F: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = mod_p(F, p)
    n = F.shape[0] // 2
    A = F[:n, :n]
    B = F[:n, n:]
    C = F[n:, :n]
    D = F[n:, n:]
    return A, B, C, D


def compose_symplectic_from_gates(n: int, gates: list[tuple], p: int) -> np.ndarray:
    """Compose gate list into a symplectic matrix over GF(p)."""
    F = identity(2 * n)
    for gate in gates:
        kind = gate[0]
        if kind == "H":
            Gi = gate_H(n, gate[1], p)
        elif kind == "S":
            coeff = gate[2] if len(gate) > 2 else 1
            Gi = gate_S(n, gate[1], coeff, p)
        elif kind == "CZ":
            coeff = gate[3] if len(gate) > 3 else 1
            Gi = gate_CZ(n, gate[1], gate[2], coeff, p)
        elif kind == "SUM":
            coeff = gate[3] if len(gate) > 3 else 1
            Gi = gate_SUM(n, gate[1], gate[2], coeff, p)
        elif kind == "MUL":
            Gi = gate_SCALE(n, gate[1], gate[2], p)
        elif kind == "SWAP":
            Gi = gate_SWAP(n, gate[1], gate[2], p)
        else:
            raise ValueError(f"Unknown gate {gate}")
        F = mm_p(Gi, F, p)
    return F


def decompose_symplectic_gfp(F: np.ndarray, p: int) -> list[tuple]:
    """
    Decompose a 2n x 2n symplectic matrix F over GF(p) into elementary Clifford gates.
    Returns gates in application order (left-to-right equals time order).
    """
    F = mod_p(F, p)
    assert is_symplectic_gfp(F, p), "Input F is not symplectic over GF(p)."
    original_F = F.copy()
    pre_ops, F = ensure_invertible_A(F, p)
    n = F.shape[0] // 2
    A, B, C, _ = blocks(F, p)

    A_inv = inv_gfp(A, p)

    Bp = mod_p(A_inv @ B, p)
    Cp = mod_p(C @ A_inv, p)

    if not np.array_equal(Bp, Bp.T % p):
        raise ValueError("A^{-1} B is not symmetric — F is not symplectic or numeric bug.")
    if not np.array_equal(Cp, Cp.T % p):
        raise ValueError("C A^{-1} is not symmetric — F is not symplectic or numeric bug.")

    gates_R = synth_upper_from_symmetric(n, Bp, p)
    gates_M = synth_linear_A(n, A, p)
    gates_L = synth_lower_from_symmetric(n, Cp, p)

    gates = gates_R + gates_M + gates_L
    gates += _invert_gate_tuples(pre_ops, p)

    F_reconstructed = compose_symplectic_from_gates(n, gates, p)
    if not np.array_equal(F_reconstructed, original_F):
        raise AssertionError("Internal check failed: reconstructed F != input F.")

    return gates

def decompose_symplectic_to_circuit(F: np.ndarray, p: int) -> "Circuit":
    tuple_gates = decompose_symplectic_gfp(F, p)
    gate_objects = []

    for gate in tuple_gates:
        kind = gate[0]
        if kind == "H":
            gate_objects.append(Hadamard(gate[1], p))
        elif kind == "S":
            coeff = gate[2] if len(gate) > 2 else 1
            coeff_mod = coeff % p
            for _ in range(coeff_mod):
                gate_objects.append(PHASE(gate[1], p))
        elif kind == "CZ":
            coeff = gate[3] if len(gate) > 3 else 1
            coeff_mod = coeff % p
            for _ in range(coeff_mod):
                gate_objects.append(CZ(gate[1], gate[2], p))
        elif kind == "SUM":
            coeff = gate[3] if len(gate) > 3 else 1
            coeff_mod = coeff % p
            for _ in range(coeff_mod):
                gate_objects.append(SUM(gate[1], gate[2], p))
        elif kind == "SWAP":
            gate_objects.append(SWAP(gate[1], gate[2], p))
        elif kind == "MUL":
            scalar = gate[2] if len(gate) > 2 else 1
            scalar_mod = scalar % p
            if scalar_mod == 0:
                raise ValueError("MUL gate scalar must be non-zero modulo p.")
            if scalar_mod == 1:
                continue
            inv_scalar = pow(int(scalar_mod), -1, p)
            idx = gate[1]
            gate_objects.append(Hadamard(idx, p))
            for _ in range(inv_scalar % p):
                gate_objects.append(PHASE(idx, p))
            gate_objects.append(Hadamard(idx, p))
            for _ in range(scalar_mod):
                gate_objects.append(PHASE(idx, p))
            gate_objects.append(Hadamard(idx, p))
            for _ in range(inv_scalar % p):
                gate_objects.append(PHASE(idx, p))
        else:
            raise ValueError(f"Unknown gate type {gate}")

    n_qudits = F.shape[0] // 2
    circuit = Circuit([p] * n_qudits, gate_objects)
    return circuit

def _canonical_pauli_sum(dimensions: list[int] | np.ndarray) -> "PauliSum":

    dims = np.asarray(dimensions, dtype=int)
    n = len(dims)
    paulis: list[PauliString] = []
    zeros = np.zeros(n, dtype=int)

    for i in range(n):
        x = zeros.copy()
        x[i] = 1
        paulis.append(PauliString(x, zeros, dims, sanity_check=False))
    for i in range(n):
        z = zeros.copy()
        z[i] = 1
        paulis.append(PauliString(zeros, z, dims, sanity_check=False))

    weights = np.ones(2 * n, dtype=float)
    phases = np.zeros(2 * n, dtype=int)
    return PauliSum(paulis, weights=weights, phases=phases, dimensions=dims, standardise=False)


def decompose_gate_to_circuit(gate: "Gate") -> "Circuit":
    dims = np.asarray(gate.dimensions, dtype=int)
    if dims.ndim == 0:
        dims = np.array([int(dims)], dtype=int)
    if not np.all(dims == dims[0]):
        raise ValueError("decompose_gate_to_circuit currently supports uniform qudit dimensions.")
    p = int(dims[0])

    circuit = decompose_symplectic_to_circuit(gate.symplectic, p)
    canonical = _canonical_pauli_sum(dims)
    target_action = gate.act(canonical).standard_form()

    def circuit_action() -> "PauliSum":
        return circuit.act(canonical).standard_form()

    constructed_action = circuit_action()

    if constructed_action.standard_form().tableau().tolist() != target_action.tableau().tolist():
        constructed_gate = Gate.solve_from_target(
            "constructed",
            canonical.standard_form(),
            constructed_action,
            dimensions=dims,
        )
        residual = Gate.solve_from_target(
            "residual",
            constructed_action,
            target_action,
            dimensions=dims,
        )
        residual_symplectic = residual.symplectic % p
        if not np.array_equal(residual_symplectic, np.eye(2 * len(dims), dtype=int)):
            residual_circuit = decompose_symplectic_to_circuit(residual_symplectic, p)
            circuit.add_gate(residual_circuit.gates)

    return circuit


# =========================
# Quick smoke test helpers
# =========================

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


def demo(n: int = 3, p: int = 2, seed: int = 2025) -> None:
    rng = np.random.default_rng(seed)
    A = random_invertible(n, rng, p)
    Bsym = random_symmetric(n, rng, p)
    Csym = random_symmetric(n, rng, p)
    F = build_F_from_LMR(n, A, Bsym, Csym, p)
    assert is_symplectic_gfp(F, p)
    gates = decompose_symplectic_gfp(F, p)
    print(f"n={n}, p={p}, #gates={len(gates)}")
    for g in gates:
        print(g)


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
    assert is_symplectic_gfp(F, p)
    return F


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    for p in (2, 3, 5, 7):
        for n in range(1, 5):
            for _ in range(20):
                A = random_invertible(n, rng, p)
                B = random_symmetric(n, rng, p)
                C = random_symmetric(n, rng, p)
                F = build_F_from_LMR(n, A, B, C, p)

                gates_tuple = decompose_symplectic_gfp(F, p)
                recon = compose_symplectic_from_gates(n, gates_tuple, p)
                assert np.array_equal(recon, F), "Symplectic reconstruction failed."

                try:
                    circuit = decompose_symplectic_to_circuit(F, p)
                except Exception:
                    circuit = None

                if circuit is not None:
                    allowed = {"H", "S", "SUM", "SWAP", "CZ"}
                    circuit_names = [gate.name for gate in circuit.gates]
                    assert all(name in allowed for name in circuit_names), f"Unexpected gate type in {circuit_names}."

                # Exercise preprocessing when A is singular
                for idx in range(n):
                    F_singular = mm_p(gate_H(n, idx, p), F, p)
                    if is_invertible_mod(F_singular[:n, :n], p):
                        continue
                    gates_tuple_sing = decompose_symplectic_gfp(F_singular, p)
                    recon_sing = compose_symplectic_from_gates(n, gates_tuple_sing, p)
                    assert np.array_equal(recon_sing, F_singular), "Preprocessing reconstruction failed."
                    break

    # Additional tests targeting initially singular A blocks
    for p in (2, 3, 5, 7):
        for n in range(2, 5):
            candidates = [("H", i) for i in range(n)]
            candidates.extend(("SWAP", i, j) for i in range(n) for j in range(i + 1, n))
            for _ in range(30):
                A = random_invertible(n, rng, p)
                B = random_symmetric(n, rng, p)
                C = random_symmetric(n, rng, p)
                F = build_F_from_LMR(n, A, B, C, p)

                F_singular = F.copy()
                made_singular = False
                for _ in range(3 * n):
                    gate = candidates[rng.integers(len(candidates))]
                    F_singular = _apply_gate_tuple(F_singular, gate, p)
                    if not is_invertible_mod(F_singular[:n, :n], p):
                        made_singular = True
                        break
                if not made_singular:
                    continue

                gates_tuple = decompose_symplectic_gfp(F_singular, p)
                recon = compose_symplectic_from_gates(n, gates_tuple, p)
                assert np.array_equal(recon, F_singular), "Singular-A preprocessing failed."

    # Additional tests targeting initially singular A blocks
    print('Done first two')

    # print(gate_CZ(2, 0, 1, 1, 2))
    for p in (2, 3, 5, 7):
        for n in range(3, 6):

            for _ in range(10):
                print(p, n, _)
                F = random_symplectic(n, p)

                gates_tuple = decompose_symplectic_gfp(F, p)
                recon = compose_symplectic_from_gates(n, gates_tuple, p)
                circuit = decompose_symplectic_to_circuit(F, p)
                recon_from_circuit = circuit.composite_gate().symplectic
                assert np.array_equal(recon, F), "Random preprocessing failed."
                assert np.array_equal(recon_from_circuit, F), "Random Gate processing failed."

    # print("All GF(p) symplectic decomposition checks passed.")
