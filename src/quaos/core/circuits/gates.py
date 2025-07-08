import numpy as np
from quaos.core.paulis import PauliString, PauliSum, Pauli
from typing import overload


class Gate:

    def __init__(self, name: str,
                 qudit_indices: list[int],
                 images: list[np.ndarray],
                 dimension: int,
                 phase_vector: np.ndarray | list[int]):

        self.dimension = dimension
        self.name = name
        self.qudit_indices = qudit_indices
        self.images = images
        self.n_qudits = len(qudit_indices)
        self.symplectic = np.stack([v % dimension for v in images]).T
        self.phase_vector = phase_vector

    def _act_on_pauli_string(self, P: PauliString) -> tuple[PauliString, int]:
        if np.all(self.dimension != P.dimensions[self.qudit_indices]):
            raise ValueError("Gate and PauliString have different dimensions.")
        local_symplectic = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])
        acquired_phase = self.acquired_phase(P)

        local_symplectic = (self.symplectic @ local_symplectic) % self.dimension
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

        ha = phase_function(P) - treated separately in the act method
        """

        U = np.zeros((2 * self.n_qudits, 2 * self.n_qudits), dtype=int)
        U[self.n_qudits:, :self.n_qudits] = np.eye(self.n_qudits, dtype=int)

        C = self.symplectic
        h = self.phase_vector
        a = np.concatenate([P.x_exp[self.qudit_indices], P.z_exp[self.qudit_indices]])  # local symplectic
        # V_diag(C^TUC)
        p1 = np.dot(np.diag(C.T @ U @ C), a)
        # a^T P_upps(C^TUC) a a^T P_diag(C^TUC) a
        ctuc = C.T @ U @ C
        p_part = 2 * np.triu(ctuc) - np.diag(np.diag(ctuc))
        p2 = np.dot(a.T, np.dot(p_part, a))
        #

        return (np.dot(h, a) - p1 + p2) % (2 * P.lcm)

    def copy(self) -> 'Gate':
        """
        Returns a copy of the gate.
        """
        return Gate(self.name, self.qudit_indices.copy(), self.images.copy(), self.dimension, self.phase_vector.copy())


class SUM(Gate):
    def __init__(self, control, target, dimension):
        images = [np.array([1, 1, 0, 0]),  # image of X0:  X0 -> X0 X1
                  np.array([0, 1, 0, 0]),  # image of X1:  X1 -> X1
                  np.array([0, 0, 1, 0]),  # image of Z0:  Z0 -> Z0
                  np.array([0, 0, -1, 1])  # image of Z1:  Z1 -> Z0^-1 Z1
                  ]

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], images, dimension=dimension, phase_vector=phase_vector)


class SWAP(Gate):
    def __init__(self, index1, index2, dimension):
        images = [np.array([0, 1, 0, 0]),  # image of X0:  X0 -> X1
                  np.array([1, 0, 0, 0]),  # image of X1:  X1 -> X0
                  np.array([0, 0, 0, 1]),  # image of Z0:  Z0 -> Z2
                  np.array([0, 0, 1, 0])   # image of Z1:  Z1 -> Z0
                  ]

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SWAP", [index1, index2], images, dimension=dimension, phase_vector=phase_vector)


class CNOT(Gate):
    def __init__(self, control, target):
        images = [np.array([1, 1, 0, 0]),  # image of X0:  X0 -> X0 X1
                  np.array([0, 1, 0, 0]),  # image of X1:  X1 -> X1
                  np.array([0, 0, 1, 0]),  # image of Z0:  Z0 -> Z0
                  np.array([0, 0, -1, 1])  # image of Z1:  Z1 -> Z0^-1 Z1
                  ]

        phase_vector = np.array([0, 0, 0, 0], dtype=int)

        super().__init__("SUM", [control, target], images, dimension=2, phase_vector=phase_vector)


class Hadamard(Gate):
    def __init__(self, index: int, dimension: int, inverse: bool = False):
        if inverse:
            images = [np.array([0, 1]),  # image of X:  X -> Z
                      np.array([-1, 0]),  # image of Z:  Z -> -X
                      ]
        else:
            images = [np.array([0, -1]),  # image of X:  X -> -Z
                      np.array([1, 0]),  # image of Z:  Z -> X
                      ]

        phase_vector = np.array([0, 0], dtype=int)

        name = "H" if not inverse else "Hdag"
        super().__init__(name, [index], images, dimension=dimension, phase_vector=phase_vector)


class PHASE(Gate):

    def __init__(self, index: int, dimension: int):
        images = [np.array([1, 1]),  # image of X:  X -> XZ
                  np.array([0, 1]),  # image of Z:  Z -> Z
                  ]

        phase_vector = np.array([dimension + 1, 0], dtype=int)

        super().__init__("S", [index], images, dimension=dimension, phase_vector=phase_vector)
