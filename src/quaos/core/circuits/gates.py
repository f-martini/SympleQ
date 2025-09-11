import numpy as np
from quaos.core.paulis import PauliString, PauliSum, Pauli
from typing import overload
from quaos.core.circuits.target import find_map_to_target_pauli_sum, get_phase_vector
from quaos.core.circuits.utils import transvection_matrix, symplectic_form
from quaos.core.circuits.random_symplectic import symplectic_gf2, symplectic_group_size
from quaos.utils import get_linear_dependencies
from quaos.core.circuits.random_symplectic import symplectic_random_transvection


class Gate:

    def __init__(self, name: str,
                 qudit_indices: list[int],
                 symplectic: np.ndarray,
                 dimensions: int | list[int] | np.ndarray,
                 phase_vector: np.ndarray | list[int]):

        if isinstance(dimensions, int) or isinstance(dimensions, np.signedinteger):
            dimensions = dimensions * np.ones(len(qudit_indices), dtype=int)
        elif len(qudit_indices) != len(dimensions):
            raise ValueError("Dimensions and qudit_indices must have the same length.")

        self.dimensions = dimensions
        self.name = name
        self.qudit_indices = qudit_indices
        self.n_qudits = len(qudit_indices)
        self.symplectic = symplectic
        self.phase_vector = phase_vector
        self.lcm = np.lcm.reduce(self.dimensions)

    @classmethod
    def solve_from_target(cls, name: str, input_pauli_sum: PauliSum, target_pauli_sum: PauliSum,
                          dimensions: int | list[int] | np.ndarray):
        """
        Create a gate that maps input_pauli_sum to target_pauli_sum.
        """

        independent_set, dependent_set = get_linear_dependencies(input_pauli_sum.tableau(), dimensions)

        if len(dependent_set) != 0:
            raise NotImplementedError("Input PauliSum is not linearly independent. Will be implemented dreckly.")

        symplectic, phase_vector, qudit_indices, dimension = find_map_to_target_pauli_sum(input_pauli_sum,
                                                                                          target_pauli_sum)
        return cls(name, qudit_indices, symplectic.T, dimension, phase_vector)

    @classmethod
    def from_random(cls, n_qudits: int, dimension: int, n_transvection=None, seed=None):
        np.random.seed(seed)
        if n_qudits < 4 and dimension == 2:
            symp_int = np.random.randint(symplectic_group_size(n_qudits))
            symplectic = symplectic_gf2(symp_int, n_qudits)
            phase_vector = get_phase_vector(symplectic, dimension)
            return cls(f"R{symp_int}", list(range(n_qudits)), symplectic.T, dimension, phase_vector)
        else:
            symplectic = symplectic_random_transvection(n_qudits, dimension=dimension, num_transvections=n_transvection)
            phase_vector = get_phase_vector(symplectic, dimension)
            return cls(f"R{n_transvection}", list(range(n_qudits)), symplectic, dimension, phase_vector)

    def _act_on_pauli_string(self, P: PauliString) -> tuple[PauliString, int]:
        if np.all(self.dimensions != P.dimensions[self.qudit_indices]):
            raise ValueError("Gate and PauliString have different dimensions.")
        local_symplectic = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])
        acquired_phase = self.acquired_phase(P)

        local_symplectic = (local_symplectic @ self.symplectic.T) % self.lcm
        P = P._replace_symplectic(local_symplectic, self.qudit_indices)
        return P, acquired_phase

    def _act_on_pauli_sum(self, pauli_sum: PauliSum) -> PauliSum:
        pauli_strings: list[PauliString] = []
        phases: list[int] = []
        for i, p in enumerate(pauli_sum.pauli_strings):
            ps, phase = self._act_on_pauli_string(p)
            pauli_strings.append(ps)
            phases.append(pauli_sum.phases[i] + phase)

        return PauliSum(pauli_strings, pauli_sum.weights, np.asarray(phases), pauli_sum.dimensions, False)

    def __repr__(self):
        return f"Gate(name={self.name}, qudit_indices={self.qudit_indices}, " \
            f"dimensions={self.dimensions}, phase_vector={self.phase_vector})"

    @overload
    def act(self, P: Pauli) -> PauliSum:
        ...

    @overload
    def act(self, P: PauliString) -> PauliString:
        ...

    @overload
    def act(self, P: PauliSum) -> PauliSum:
        ...

    def act(self, P: Pauli | PauliString | PauliSum):
        if isinstance(P, Pauli):
            P = PauliString.from_pauli(P)
        if isinstance(P, PauliString):
            return self._act_on_pauli_string(P)[0]
        elif isinstance(P, PauliSum):
            return self._act_on_pauli_sum(P)
        else:
            raise TypeError(f"Unsupported type {type(P)} for Gate.act. Expected Pauli, PauliString or PauliSum.")

    def acquired_phase(self, P: PauliString) -> int:
        """
        Returns the phase acquired by the PauliString P when acted upon by this gate.

        See PHYSICAL REVIEW A 71, 042315 (2005)

        """

        U = np.zeros((2 * self.n_qudits, 2 * self.n_qudits), dtype=int)
        U[self.n_qudits:, :self.n_qudits] = np.eye(self.n_qudits, dtype=int)

        C = self.symplectic

        U_conjugated = C.T @ U @ C
        h = self.phase_vector
        a = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])  # local symplectic
        p1 = np.dot(np.diag(U_conjugated), a)
        # negative sign in below as definition in paper is strictly upper diagonal, not including diagonal part
        p_part = 2 * np.triu(U_conjugated) - np.diag(np.diag(U_conjugated))
        p2 = np.dot(a.T, np.dot(p_part, a))

        return (np.dot(h, a) - p1 + p2) % (2 * P.lcm)

    def copy(self) -> 'Gate':
        """
        Returns a copy of the gate.
        """
        return Gate(self.name, self.qudit_indices.copy(), self.symplectic.copy(), self.dimensions,
                    self.phase_vector.copy())

    def transvection(self, transvection_vector: np.ndarray | list, transvection_weight: int = 1) -> 'Gate':
        """
        Returns a new gate that is the transvection of this gate by the given vector.
        The transvection vector should be a 2n-dimensional vector where n is the number of qudits.
        """
        if not np.all(self.dimensions == self.dimensions[0]):
            raise ValueError("Transvections only implemented for gates with equal dimensions.")
        dimension = self.dimensions[0]
        if transvection_weight >= dimension:
            raise ValueError("Transvection weight must be less than the gate dimension.")
        if not isinstance(transvection_weight, int) and not isinstance(transvection_weight, np.int64):
            raise TypeError("Transvection weight must be an integer.")
        if isinstance(transvection_vector, list):
            transvection_vector = np.array(transvection_vector)

        T = transvection_matrix(transvection_vector, multiplier=transvection_weight, p=dimension)
        if self.name[0] != "T":
            self.name = "T-" + self.name
        return Gate(self.name, self.qudit_indices, self.symplectic @ T, self.dimensions, self.phase_vector)

    def inv(self) -> 'Gate':
        print("Warning: inverse phase vector not working - PHASES MAY BE INCORRECT.")

        C = self.symplectic.T

        U = np.zeros((2 * self.n_qudits, 2 * self.n_qudits), dtype=int)
        U[self.n_qudits:, :self.n_qudits] = np.eye(self.n_qudits, dtype=int)
        Omega = symplectic_form(int(C.shape[0] / 2), p=self.lcm)

        C_inv = -(Omega.T @ C.T @ Omega) % self.lcm
        U_c = C_inv.T @ U @ C_inv % self.lcm

        p1 = - C_inv.T @ self.phase_vector
        p2 = - np.diag(C.T @ (2 * np.triu(U_c) - np.diag(np.diag(U_c))) @ C)
        p3 = C.T @ np.diag(U_c)

        phase_vector = (p1 + p2 + p3) % (2 * self.lcm)
        return Gate(self.name + "-inv", self.qudit_indices, C_inv.T, self.dimensions, phase_vector)

    def inverse(self) -> 'Gate':
        """
        Returns the inverse of this gate.

        The inverse of a gate G is another gate G' such that the composition G'G is the identity.

        The inverse of a gate is computed using the formulae presented in PHYSICAL REVIEW A 71, 042315 (2005)

        :return: A new Gate object, the inverse of this gate.
        """
        if not np.all(self.dimensions == self.dimensions[0]):
            raise NotImplementedError("Inverse only implemented for gates with equal dimensions.")

        d = self.dimensions[0]
        h = self.phase_vector
        n = len(self.qudit_indices)

        Id_n = np.eye(n)
        Zero_n = np.zeros((n, n))
        U = np.block([[Zero_n, Zero_n], [Id_n, Zero_n]])
        Omega = (U - U.T) % d

        C = self.symplectic.T
        C_inv = (Omega.T @ C.T @ Omega) % d

        U_trans = C_inv.T @ U @ C_inv
        T1 = np.diag(C.T @ (2 * np.triu(U_trans, k=1) + np.diag(np.diag(U_trans))) @ C)
        T2 = C.T @ np.diag(U_trans)

        h_inv = (- C_inv.T @ (h + T1 + T2)) % (2 * d)

        return Gate(self.name + '_inv', self.qudit_indices.copy(), C_inv.T, self.dimensions,
                    h_inv)


class SUM(Gate):
    def __init__(self, control, target, dimension):
        symplectic = np.array([
            [1, 1, 0, 0],   # image of X0:  X0 -> X0 X1
            [0, 1, 0, 0],   # image of X1:  X1 -> X1
            [0, 0, 1, 0],   # image of Z0:  Z0 -> Z0
            [0, 0, -1, 1]   # image of Z1:  Z1 -> Z0^-1 Z1
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], symplectic, dimensions=dimension, phase_vector=phase_vector)
        
    def unitary(self):
        # Implements |i, j> -> |i, j + i (mod d)| with basis ordering |i>⊗|j>,
        # linear index idx(i, j) = i * d + j.
        d = self.dimensions[0]
        mat = np.zeros((d * d, d * d), dtype=complex)
        for i in range(d):
            for j in range(d):
                row = i * d + ((j + i) % d)
                col = i * d + j
                mat[row, col] = 1
        return mat


class SWAP(Gate):
    def __init__(self, index1, index2, dimension):
        symplectic = np.array([
            [0, 1, 0, 0],  # image of X0:  X0 -> X1
            [1, 0, 0, 0],  # image of X1:  X1 -> X0
            [0, 0, 0, 1],  # image of Z0:  Z0 -> Z1
            [0, 0, 1, 0]   # image of Z1:  Z1 -> Z0
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SWAP", [index1, index2], symplectic, dimensions=dimension, phase_vector=phase_vector)

    def unitary(self):
        # General SWAP on two qudits: |i, j> -> |j, i>.
        # Basis ordering |i>⊗|j> with linear index idx(i, j) = i * d1 + j.
        d0 = int(self.dimensions[0])
        d1 = int(self.dimensions[1])
        dim = d0 * d1
        U = np.zeros((dim, dim), dtype=complex)
        for i in range(d0):
            for j in range(d1):
                row = j * d0 + i  # |j, i>
                col = i * d1 + j  # |i, j>
                U[row, col] = 1.0
        return U


class CNOT(Gate):
    def __init__(self, control, target):
        symplectic = np.array([
            [1, 1, 0, 0],   # image of X0:  X0 -> X0 X1
            [0, 1, 0, 0],   # image of X1:  X1 -> X1
            [0, 0, 1, 0],   # image of Z0:  Z0 -> Z0
            [0, 0, -1, 1]   # image of Z1:  Z1 -> Z0^-1 Z1
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], symplectic, dimensions=2, phase_vector=phase_vector)

    def unitary(self):
        # Standard qubit CNOT (control on the first, target on the second).
        d0 = int(self.dimensions[0])
        d1 = int(self.dimensions[1])
        if d0 != 2 or d1 != 2:
            raise NotImplementedError("CNOT unitary only defined for qubits (d=2). Use SUM for qudits.")
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)


class Hadamard(Gate):
    def __init__(self, index: int, dimension: int, inverse: bool = False):
        if inverse:
            symplectic = np.array([
                [0, 1],    # image of X:  X -> Z
                [-1, 0]    # image of Z:  Z -> -X
            ], dtype=int).T
        else:
            symplectic = np.array([
                [0, -1],   # image of X:  X -> -Z
                [1, 0]     # image of Z:  Z -> X
            ], dtype=int).T

        phase_vector = np.array([0, 0], dtype=int)

        name = "H" if not inverse else "H_inv"
        super().__init__(name, [index], symplectic, dimensions=dimension, phase_vector=phase_vector)

    def unitary(self):
        d = self.dimensions[0]
        omega = np.exp(2j * np.pi / d)
        factor = 1 / np.sqrt(d)
        mat = np.zeros((d, d), dtype=complex)
        for i in range(d):
            for j in range(d):
                mat[i, j] = omega ** (i * j)
        return factor * mat


class PHASE(Gate):

    def __init__(self, index: int, dimension: int):
        symplectic = np.array([
            [1, 1],  # image of X:  X -> XZ
            [0, 1]   # image of Z:  Z -> Z
        ], dtype=int).T

        phase_vector = np.array([dimension + 1, 0], dtype=int)

        super().__init__("S", [index], symplectic, dimensions=dimension, phase_vector=phase_vector)

    def unitary(self):
        # Qudit phase (Clifford) gate implementing X -> X Z, Z -> Z.
        # A unified construction for all d uses a 2d-th root of unity ζ = exp(2πi/(2d)) and
        # U = diag(ζ^{j^2}) for j in [0..d-1]. For d=2, this gives diag(1, i).
        d = int(self.dimensions[0])
        zeta = np.exp(1j * 2 * np.pi / (2 * d))
        diag = np.array([zeta ** (j * j) for j in range(d)], dtype=complex)
        return np.diag(diag)
