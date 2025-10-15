import numpy as np
from quaos.core.paulis import PauliString, PauliSum, Pauli
from typing import overload
from quaos.core.circuits.target import find_map_to_target_pauli_sum, get_phase_vector
from quaos.core.circuits.utils import (transvection_matrix, symplectic_form,
                                       CX_func, SWAP_func, S_mat, I_mat, tensor, H_mat)
# from quaos.core.circuits.random_symplectic import symplectic_gf2, symplectic_group_size
from quaos.core.finite_field_solvers import get_linear_dependencies
# from quaos.core.circuits.random_symplectic import symplectic_random_transvection
import scipy.sparse as sp


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
        self.phase_vector = np.asarray(phase_vector, dtype=int)
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
    def from_random(cls, n_qudits: int, dimension: int, n_transvection: int = 10, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        seed_vec = np.random.randint(0, 100000, size=n_transvection)
        # if n_qudits < 4 and dimension == 2:
        #     symp_int = np.random.randint(symplectic_group_size(n_qudits))
        #     symplectic = symplectic_gf2(symp_int, n_qudits)
        #     phase_vector = get_phase_vector(symplectic, dimension)
        #     return cls(f"R{symp_int}", list(range(n_qudits)), symplectic.T, dimension, phase_vector)
        # else:
        symplectic = np.eye(2 * n_qudits, dtype=int)

        for i in range(n_transvection):
            np.random.seed(seed_vec[i])
            Tv = transvection_matrix(np.random.randint(0, dimension, size=2 * n_qudits), dimension) % dimension
            symplectic = symplectic @ Tv % dimension

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
        p1 = np.dot(np.diag(U_conjugated).T, a)
        # negative sign in below as definition in paper is strictly upper diagonal, not including diagonal part

        p_part = 2 * np.triu(U_conjugated) - np.diag(np.diag(U_conjugated))
        p2 = np.dot(a.T, np.dot(p_part, a))
        return (np.dot(h, a.T) - p1 + p2) % (2 * P.lcm)

    def inv(self) -> 'Gate':
        n = self.n_qudits
        L = int(self.lcm)

        C = self.symplectic % L

        zero_block = np.zeros((n, n), dtype=int)
        identity_block = np.eye(n, dtype=int)
        Omega = np.block([[zero_block, identity_block], [-identity_block, zero_block]])

        C_inv = (Omega.T @ C.T @ Omega) % L
        C_inv = C_inv.astype(int)

        U = np.zeros((2 * n, 2 * n), dtype=int)
        U[n:, :n] = np.eye(n, dtype=int)

        U_conj = C.T @ U @ C
        U_inv_conj = C_inv.T @ U @ C_inv

        P = 2 * np.triu(U_conj) - np.diag(np.diag(U_conj))
        P_inv = 2 * np.triu(U_inv_conj) - np.diag(np.diag(U_inv_conj))

        modulus = 2 * L
        term1 = ((-(self.phase_vector % modulus) @ C_inv) % modulus).astype(int)
        term2 = (((np.diag(U_conj) % modulus).T @ C_inv) % modulus).astype(int)
        term3 = (np.diag((C_inv.T @ P @ C_inv) % modulus) % modulus).astype(int)
        term4 = (np.diag(U_inv_conj) % modulus).astype(int)
        term5 = (np.diag(P_inv) % modulus).astype(int)

        h_inv = (term1 + term2 - term3 + term4 - term5) % modulus

        return Gate(self.name + "-inv", self.qudit_indices, C_inv, self.dimensions, h_inv.astype(int))

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

    def unitary(self, dims=None):
        if dims is None:
            dims = self.dimensions
        raise NotImplementedError("Unitary not implemented for generic Gate. Use specific gate subclasses.")


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

    def unitary(self, dims=None):
        if dims is None:
            dims = self.dimensions
        D = np.prod(dims)
        aa = self.qudit_indices
        a0 = aa[0]
        a1 = aa[1]
        aa2 = np.array([1 for i in range(D)])
        aa3 = np.array([CX_func(i, a0, a1, dims) for i in range(D)])
        aa4 = np.array([i for i in range(D)])
        return sp.csr_matrix((aa2, (aa3, aa4)))

    def copy(self) -> 'Gate':
        """
        Returns a copy of the SUM gate.
        """
        return SUM(self.qudit_indices[0], self.qudit_indices[1], self.dimensions)


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

    def unitary(self, dims=None):
        # SWAP on two qudits of equal dimension: |i, j> -> |j, i>.
        # Basis ordering |i>⊗|j> with linear index idx(i, j) = i * d + j.
        if dims is None:
            dims = self.dimensions
        aa = self.qudit_indices
        q = len(dims)
        D = np.prod(dims)
        a0 = q - 1 - aa[0]
        a1 = q - 1 - aa[1]
        aa2 = np.array([1 for i in range(D)])
        aa3 = np.array([i for i in range(D)])
        aa4 = np.array([SWAP_func(i, a0, a1, dims) for i in range(D)])
        return sp.csr_matrix((aa2, (aa3, aa4)))

    def copy(self) -> 'Gate':
        """
        Returns a copy of the SWAP gate.
        """
        return SWAP(self.qudit_indices[0], self.qudit_indices[1], self.dimensions)


class CNOT(Gate):
    def __init__(self, control, target):
        symplectic = np.array([
            [1, 1, 0, 0],   # image of X0:  X0 -> X0 X1
            [0, 1, 0, 0],   # image of X1:  X1 -> X1
            [0, 0, 1, 0],   # image of Z0:  Z0 -> Z0
            [0, 0, 1, 1]   # image of Z1:  Z1 -> Z0^-1 Z1
        ], dtype=int).T

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], symplectic, dimensions=2, phase_vector=phase_vector)

    def unitary(self, dims=None):
        if dims is None:
            dims = self.dimensions
        D = np.prod(dims)
        aa = self.qudit_indices
        a0 = aa[0]
        a1 = aa[1]
        aa2 = np.array([1 for i in range(D)])
        aa3 = np.array([CX_func(i, a0, a1, dims) for i in range(D)])
        aa4 = np.array([i for i in range(D)])
        return sp.csr_matrix((aa2, (aa3, aa4)))

    def copy(self) -> 'Gate':
        """
        Returns a copy of the CNOT gate.
        """
        return CNOT(self.qudit_indices[0], self.qudit_indices[1])


class Hadamard(Gate):
    def __init__(self, index: int, dimension: int, inverse: bool = False):
        if inverse:
            symplectic = np.array([
                [0, 1],    # image of X:  X -> Z
                [-1, 0]    # image of Z:  Z -> -X
            ], dtype=int)
        else:
            symplectic = np.array([
                [0, -1],   # image of X:  X -> -Z
                [1, 0]     # image of Z:  Z -> X
            ], dtype=int)

        phase_vector = np.array([0, 0], dtype=int)

        name = "H" if not inverse else "H_inv"
        super().__init__(name, [index], symplectic, dimensions=dimension, phase_vector=phase_vector)

    def unitary(self, dims=None):
        if dims is None:
            dims = self.dimensions
        return tensor([H_mat(dims[i]) if i in self.qudit_indices else I_mat(dims[i]) for i in range(len(dims))])

    def copy(self) -> 'Gate':
        """
        Returns a copy of the Hadamard gate.
        """
        return Hadamard(self.qudit_indices[0], self.dimensions[0])


class PHASE(Gate):

    def __init__(self, index: int, dimension: int):
        symplectic = np.array([
            [1, 1],  # image of X:  X -> XZ
            [0, 1]   # image of Z:  Z -> Z
        ], dtype=int).T
        if dimension == 2:
            phase_vector = np.array([1, 0], dtype=int)
        else:
            phase_vector = np.array([0, 0], dtype=int)

        super().__init__("S", [index], symplectic, dimensions=dimension, phase_vector=phase_vector)

    def unitary(self, dims=None):
        if dims is None:
            dims = self.dimensions
        return tensor([S_mat(dims[i]) if i in self.qudit_indices else I_mat(dims[i]) for i in range(len(dims))])

    def copy(self) -> 'Gate':
        """
        Returns a copy of the PHASE gate.
        """
        return PHASE(self.qudit_indices[0], self.dimensions[0])


class PauliGate(Gate):
    def __init__(self, pauli: PauliString):
        n = pauli.n_qudits()
        lcm = int(pauli.lcm)
        symplectic = np.eye(2 * n, dtype=int)
        phase_vector = (2 * symplectic_form(n, lcm) @ np.concatenate([pauli.x_exp, pauli.z_exp])) % (2 * lcm)
        super().__init__("Pauli", list(range(n)), symplectic, dimensions=pauli.dimensions, phase_vector=phase_vector)
