import numpy as np
from quaos.core.paulis import PauliString, PauliSum, Pauli
from typing import overload
from quaos.core.circuits.target import find_map_to_target_pauli_sum  # , get_phase_vector
# from quaos.core.circuits.utils import index_from_symplectic  # number_of_symplectics, symplectic_from_index,


class Gate:

    def __init__(self, name: str,
                 qudit_indices: list[int],
                 symplectic: np.ndarray,
                 dimension: int,
                 phase_vector: np.ndarray | list[int]):

        self.dimension = dimension
        self.name = name
        self.qudit_indices = qudit_indices
        self.n_qudits = len(qudit_indices)
        self.symplectic = symplectic
        self.phase_vector = phase_vector

    @classmethod
    def solve_from_target(cls, name: str, input_pauli_sum: PauliSum, target_pauli_sum: PauliSum):
        """
        Create a gate that maps input_pauli_sum to target_pauli_sum.
        """
        symplectic, phase_vector, qudit_indices, dimension = find_map_to_target_pauli_sum(input_pauli_sum, target_pauli_sum)
        return cls(name, qudit_indices, symplectic.T, dimension, phase_vector)

    # @classmethod
    # def from_random(cls, n_qudits: int, dimension: int, seed=None):
    #     np.random.seed(seed)
    #     if dimension != 2:
    #         raise NotImplementedError("Only implemented for dimension 2. GF(p) will be done asap.")
    #     symp_int = np.random.randint(number_of_symplectics(n_qudits))
    #     symplectic = symplectic_from_index(symp_int, n_qudits, dimension)
    #     phase_vector = get_phase_vector(symplectic, dimension)
    #     return cls(f"R{symp_int}", list(range(n_qudits)), symplectic.T, dimension, phase_vector)

    def _act_on_pauli_string(self, P: PauliString) -> tuple[PauliString, int]:
        if np.all(self.dimension != P.dimensions[self.qudit_indices]):
            raise ValueError("Gate and PauliString have different dimensions.")
        local_symplectic = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])
        acquired_phase = self.acquired_phase(P)

        local_symplectic = (local_symplectic @ self.symplectic.T) % self.dimension
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
               f"dimension={self.dimension}, phase_vector={self.phase_vector})"

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

        ctuc = C.T @ U @ C
        h = self.phase_vector
        a = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])  # local symplectic
        # V_diag(C^TUC)
        p1 = np.dot(np.diag(ctuc), a)
        # a^T P_upps(C^TUC) a a^T P_diag(C^TUC) a

        p_part = 2 * np.triu(ctuc) - np.diag(np.diag(ctuc))
        p2 = np.dot(a.T, np.dot(p_part, a))
        #

        return (np.dot(h, a) - p1 + p2) % (2 * P.lcm)

    def copy(self) -> 'Gate':
        """
        Returns a copy of the gate.
        """
        return Gate(self.name, self.qudit_indices.copy(), self.symplectic.copy(), self.dimension, self.phase_vector.copy())

    # def transvection(self, transvection_vector: np.ndarray) -> 'Gate':
    #     """
    #     Returns a new gate that is the transvection of this gate by the given vector.
    #     The transvection vector should be a 2n-dimensional vector where n is the number of qudits.
    #     """

    # def get_int(self) -> int:
    #     return index_from_symplectic(self.n_qudits, self.symplectic, self.dimension)


class SUM(Gate):
    def __init__(self, control, target, dimension):
        symplectic = np.array([
            [1, 1, 0, 0],   # image of X0:  X0 -> X0 X1
            [0, 1, 0, 0],   # image of X1:  X1 -> X1
            [0, 0, 1, 0],   # image of Z0:  Z0 -> Z0
            [0, 0, -1, 1]   # image of Z1:  Z1 -> Z0^-1 Z1
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], symplectic, dimension=dimension, phase_vector=phase_vector)


class SWAP(Gate):
    def __init__(self, index1, index2, dimension):
        symplectic = np.array([
            [0, 1, 0, 0],  # image of X0:  X0 -> X1
            [1, 0, 0, 0],  # image of X1:  X1 -> X0
            [0, 0, 0, 1],  # image of Z0:  Z0 -> Z1
            [0, 0, 1, 0]   # image of Z1:  Z1 -> Z0
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SWAP", [index1, index2], symplectic, dimension=dimension, phase_vector=phase_vector)


class CNOT(Gate):
    def __init__(self, control, target):
        symplectic = np.array([
            [1, 1, 0, 0],   # image of X0:  X0 -> X0 X1
            [0, 1, 0, 0],   # image of X1:  X1 -> X1
            [0, 0, 1, 0],   # image of Z0:  Z0 -> Z0
            [0, 0, -1, 1]   # image of Z1:  Z1 -> Z0^-1 Z1
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], symplectic, dimension=2, phase_vector=phase_vector)


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

        name = "H" if not inverse else "Hdag"
        super().__init__(name, [index], symplectic, dimension=dimension, phase_vector=phase_vector)


class PHASE(Gate):

    def __init__(self, index: int, dimension: int):
        symplectic = np.array([
            [1, 1],  # image of X:  X -> XZ
            [0, 1]   # image of Z:  Z -> Z
        ], dtype=int).T

        phase_vector = np.array([dimension + 1, 0], dtype=int)

        super().__init__("S", [index], symplectic, dimension=dimension, phase_vector=phase_vector)
