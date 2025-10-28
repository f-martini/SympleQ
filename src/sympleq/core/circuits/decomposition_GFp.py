import numpy as np
from sympleq.core.circuits import (
    CZ as CZGate,
    Hadamard,
    PHASE,
    SUM as SUMGate,
    SWAP as SWAPGate,
    Circuit,
    Gate,
)

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


def is_symplectic_gfp(F: np.ndarray, p: int) -> bool:
    """Check whether F is symplectic over GF(p)."""
    F = mod_p(F, p)
    n2 = F.shape[0]
    assert F.shape[0] == F.shape[1] and n2 % 2 == 0
    n = n2 // 2
    I = identity(n) % p
    minus_I = (-I) % p
    J = np.block([[zeros((n, n)), I], [minus_I, zeros((n, n))]]) % p
    lhs = mm_p(mm_p(F.T, J, p), F, p)
    return np.array_equal(lhs, J)


def inv_gfp(A: np.ndarray, p: int) -> np.ndarray:
    """Matrix inverse over GF(p) via Gauss-Jordan elimination."""
    A = mod_p(A.copy(), p)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to invert over GF(p).")
    I = identity(n)
    aug = np.hstack((A, I))
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


# ---------------------------------------------------------------------------
# Notes on the MUL gate
# ---------------------------------------------------------------------------
# The symbolic gate name "MUL" denotes the diagonal symplectic transformation
# diag(lambda, lambda^{-1}) acting on a single qudit (and leaving all other
# modes untouched).  Operationally it rescales X_i by lambda and Z_i by the
# inverse so that commutation relations are preserved.  Over GF(p) this map
# can be compiled entirely into the elementary generators already present in
# the circuit: a MUL(i, lambda) gate is equivalent to the sequence
#   H_i · S_i(lambda^{-1}) · H_i · S_i(lambda) · H_i · S_i(lambda^{-1})
# (with arithmetic modulo p and lambda^{-1} the multiplicative inverse), up
# to the special cases lambda = ±1 that can be shortened.  We keep MUL as an
# abstract gate while synthesising the linear block, and expand it into the
# above H/S combination only when required.

def gate_SWAP(n: int, i: int, j: int, p: int) -> np.ndarray:
    """Swap qudits i and j."""
    if i == j:
        return identity(2 * n)
    A = identity(n)
    A[[i, j], :] = A[[j, i], :]
    return _gate_linear_from_A(A, p)


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


def synth_lower_from_symmetric(n: int, Csym: np.ndarray, p: int) -> list[tuple]:
    """Synthesise [[I,0],[Csym, I]] via Fourier conjugation."""
    Csym = mod_p(Csym, p)
    ops: list[tuple] = []
    ops += H_all(n, p)
    ops += synth_upper_from_symmetric(n, (-Csym) % p, p)
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
    n = F.shape[0] // 2
    A, B, C, _ = blocks(F, p)

    Ainv = inv_gfp(A, p)

    Bp = mod_p(Ainv @ B, p)
    Cp = mod_p(C @ Ainv, p)

    if not np.array_equal(Bp, Bp.T % p):
        raise ValueError("A^{-1} B is not symmetric — F is not symplectic or numeric bug.")
    if not np.array_equal(Cp, Cp.T % p):
        raise ValueError("C A^{-1} is not symmetric — F is not symplectic or numeric bug.")

    gates_R = synth_upper_from_symmetric(n, Bp, p)
    gates_M = synth_linear_A(n, A, p)
    gates_L = synth_lower_from_symmetric(n, Cp, p)

    gates = gates_R + gates_M + gates_L

    Frecon = compose_symplectic_from_gates(n, gates, p)
    if not np.array_equal(Frecon, F):
        raise AssertionError("Internal check failed: reconstructed F != input F.")

    return gates


def decompose_symplectic_to_circuit(F: np.ndarray, p: int) -> Circuit:

    def repeat_phase(index: int, coeff: int) -> list:
        coeff_mod = coeff % p
        return [PHASE(index, p) for _ in range(coeff_mod)]

    def repeat_cz(index1: int, index2: int, coeff: int) -> list:
        coeff_mod = coeff % p
        return [CZGate(index1, index2, p) for _ in range(coeff_mod)]

    def repeat_sum(src: int, dst: int, coeff: int) -> list:
        coeff_mod = coeff % p
        return [SUMGate(src, dst, p) for _ in range(coeff_mod)]

    def expand_mul(index: int, scalar: int) -> list:
        scalar_mod = scalar % p
        if scalar_mod == 0:
            raise ValueError("MUL gate scalar must be non-zero modulo p.")
        if scalar_mod == 1:
            return []
        gates_local: list = []
        if scalar_mod == (p - 1) % p:
            gates_local.append(Hadamard(index, p))
            gates_local.append(Hadamard(index, p))
            return gates_local
        inv_scalar = pow(int(scalar_mod), -1, p)
        gates_local.append(Hadamard(index, p))
        gates_local.extend(repeat_phase(index, inv_scalar))
        gates_local.append(Hadamard(index, p))
        gates_local.extend(repeat_phase(index, scalar_mod))
        gates_local.append(Hadamard(index, p))
        gates_local.extend(repeat_phase(index, inv_scalar))
        return gates_local

    tuple_gates = decompose_symplectic_gfp(F, p)
    gate_objects: list = []
    for gate in tuple_gates:
        kind = gate[0]
        if kind == "H":
            gate_objects.append(Hadamard(gate[1], p))
        elif kind == "S":
            gate_objects.extend(repeat_phase(gate[1], gate[2] if len(gate) > 2 else 1))
        elif kind == "CZ":
            coeff = gate[3] if len(gate) > 3 else 1
            gate_objects.extend(repeat_cz(gate[1], gate[2], coeff))
        elif kind == "SUM":
            coeff = gate[3] if len(gate) > 3 else 1
            gate_objects.extend(repeat_sum(gate[1], gate[2], coeff))
        elif kind == "SWAP":
            gate_objects.append(SWAPGate(gate[1], gate[2], p))
        elif kind == "MUL":
            gate_objects.extend(expand_mul(gate[1], gate[2]))
        else:
            raise ValueError(f"Unknown gate type {gate}")
    n_qudits = F.shape[0] // 2
    C = Circuit([p for _ in range(n_qudits)], gate_objects)
    return C


def _canonical_pauli_sum(dimensions: list[int] | np.ndarray) -> "PauliSum":
    from sympleq.core.paulis import PauliSum, PauliString

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


def decompose_gate_to_circuit(gate: Gate) -> Circuit:
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

    AinvT = inv_gfp(A.T, p)
    L = np.block([[identity(n), zeros((n, n))],
                  [Csym, identity(n)]])
    M = np.block([[A, zeros((n, n))],
                  [zeros((n, n)), AinvT]])
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


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    for p in [2]:
        for n in [4]:
            for _ in range(1):
                A = random_invertible(n, rng, p)
                B = random_symmetric(n, rng, p)
                C = random_symmetric(n, rng, p)
                F = build_F_from_LMR(n, A, B, C, p)
                gates_tuple = decompose_symplectic_gfp(F, p)
                recon = compose_symplectic_from_gates(n, gates_tuple, p)
                assert np.array_equal(recon, F), "Symplectic reconstruction failed."

                circuit = decompose_symplectic_to_circuit(F, p)
                
                assert np.all(circuit.composite_gate().symplectic == F), "Symplectic decomposition failed."

                allowed = {"H", "S", "SUM", "SWAP", "CZ"}
                circuit_names = [gate.name for gate in circuit.gates]
                assert all(name in allowed for name in circuit_names), "Unexpected gate type in circuit output."
                assert all(gate.name != "MUL" for gate in circuit.gates), "MUL gate should be expanded."

                tuple_cz = sum(1 for g in gates_tuple if g[0] == "CZ")
                circuit_cz = sum(1 for name in circuit_names if name == "CZ")
                assert tuple_cz == circuit_cz, "CZ gate count mismatch."

    print("All GF(p) symplectic decomposition checks passed.")
