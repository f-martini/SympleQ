import numpy as np
from sympleq.core.paulis import PauliString, PauliSum, Pauli
from typing import overload
from sympleq.core.circuits.target import find_map_to_target_pauli_sum, get_phase_vector
from sympleq.core.circuits.utils import (transvection_matrix, symplectic_form, tensor, I_mat, H_mat, S_mat, CX_func,
                                         SWAP_func)
from sympleq.utils import get_linear_dependencies
import scipy.sparse as sp


class Gate:

    def __init__(self, name: str,
                 qudit_indices: list[int] | np.ndarray,
                 symplectic: np.ndarray,
                 dimensions: int | list[int] | np.ndarray,
                 phase_vector: np.ndarray | list[int]):

        if len(qudit_indices) == 0:
            raise ValueError("Gate must act on at least one qudit_indices.")

        if isinstance(dimensions, int) or isinstance(dimensions, np.signedinteger):
            dimensions = dimensions * np.ones(len(qudit_indices), dtype=int)
        elif len(qudit_indices) != len(dimensions):
            raise ValueError("Dimensions and qudit_indices must have the same length.")

        self.dimensions = dimensions
        self.name = name

        if isinstance(qudit_indices, list):
            qudit_indices = np.asarray(qudit_indices, dtype=int)

        self.qudit_indices = qudit_indices
        self.n_qudits = len(qudit_indices)
        self.symplectic: np.ndarray = symplectic
        self.phase_vector = phase_vector
        self.lcm = np.lcm.reduce(self.dimensions)

        # U = [[0_n, 0_n],
        #      [I_n, 0_n]]
        U = np.zeros((2 * self.n_qudits, 2 * self.n_qudits), dtype=int)
        U[self.n_qudits:, :self.n_qudits] = np.eye(self.n_qudits, dtype=int)
        self.U_symplectic_conjugated = self.symplectic.T @ U @ self.symplectic

        self.V_diag = np.diag(self.U_symplectic_conjugated)
        # This is the part associated with the linear form.
        self.modified_phase_vector = self.phase_vector - self.V_diag  # h - V_diag
        # Remove diagonal part to match definition in Eq.[7] in PHYSICAL REVIEW A 71, 042315 (2005).
        # This is the part associated with the quadratic form.
        self.p_part = 2 * np.triu(self.U_symplectic_conjugated) - np.diag(self.V_diag)

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
        if np.any(self.dimensions != P.dimensions[self.qudit_indices]):
            raise ValueError("Gate and PauliString have different dimensions.")
        local_symplectic = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])
        acquired_phase = self.acquired_phase(P)

        local_symplectic = (local_symplectic @ self.symplectic.T) % self.lcm
        P = P._replace_symplectic(local_symplectic, list(self.qudit_indices))
        return P, acquired_phase

    def _act_on_pauli_sum(self, pauli_sum: PauliSum) -> PauliSum:
        """
        Returns the updated tableau and phases acquired by the PauliSum when acted upon by this gate.

        See Eq.[7] in PHYSICAL REVIEW A 71, 042315 (2005)

        """
        if not np.array_equal(self.dimensions, pauli_sum.dimensions[self.qudit_indices]):
            raise ValueError("Gate and PauliSum slice have different dimensions.")

        T = pauli_sum.tableau()

        # Precompute tableau mask. This will be applied to the PauliSum tableau to get
        # the subset of affected columns.
        tableau_mask = np.concatenate([self.qudit_indices, self.qudit_indices + pauli_sum.n_qudits()])

        T_affected = T[:, tableau_mask]
        relevant_dimensions = np.tile(pauli_sum.dimensions[self.qudit_indices], 2)
        updated_tableau = np.mod(T_affected @ self.symplectic.T, relevant_dimensions)
        new_tableau = T.copy()
        new_tableau[:, tableau_mask] = updated_tableau

        linear_terms = T_affected @ self.modified_phase_vector
        quadratic_terms = np.sum(T_affected * (T_affected @ self.p_part), axis=1)

        # FIXME: this is a but of a hack
        dimensional_factor = pauli_sum.lcm // np.lcm.reduce(pauli_sum.dimensions[self.qudit_indices])
        acquired_phases = (linear_terms + quadratic_terms) * dimensional_factor

        phases = (pauli_sum.phases + acquired_phases) % (2 * pauli_sum.lcm)

        return PauliSum.from_tableau(new_tableau, pauli_sum.dimensions, pauli_sum.weights, phases=phases)

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

        h = self.phase_vector
        a = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])  # local symplectic
        p1 = np.dot(np.diag(self.U_symplectic_conjugated), a)
        # negative sign in below as definition in paper is strictly upper diagonal, not including diagonal part
        p2 = np.dot(a.T, np.dot(self.p_part, a))
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
        # TODO: Test for mixed dimensions - not clear that the symplectic form here is correct.
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
