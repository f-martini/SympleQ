from typing import Generator, overload
import numpy as np
from qiskit import QuantumCircuit
from .gates import Gate, Hadamard, PHASE, SUM, SWAP, CNOT
from sympleq.core.paulis import PauliSum, PauliString, Pauli
from .utils import embed_symplectic
import scipy.sparse as sp
from .gates import Hadamard as H, SUM as CX, PHASE as S
import random


class Circuit:
    def __init__(self, dimensions: list[int] | np.ndarray, gates: list[Gate] | None = None):
        """
        Initialize the Circuit with gates, indexes, and targets.

        If a multi-qubit gate has a target, the targets should be at the ent of the tuple of indexes
        e.g. a CNOT with control 1, target 3 is

        gate = 'CNOT'
        indexes = (1, 3)


        Parameters:
            dimensions (list[int] | np.ndarray): A list or array of integers representing the dimensions of the qudits.
            gates (list): A list of Gate objects representing the gates in the circuit.


        TODO: Remove dimensions as input - this can be obtained from the gates only - make this a method not attribute

        TODO: Perhaps store the composite gate as an attribute - it will allow gate.act to be significantly faster
        """
        if gates is None:
            gates = []
        self.dimensions = dimensions
        self.gates = gates
        self.indexes = [gate.qudit_indices for gate in gates]  # indexes accessible at the Circuit level

    @classmethod
    def from_random(cls, n_qudits: int, depth: int, dimensions: list[int] | np.ndarray) -> 'Circuit':
        """
        Creates a random circuit with the given number of qudits and depth.

        Parameters:
            n_qudits (int): The number of qudits in the circuit.
            depth (int): The depth of the circuit.

        Returns:
            Circuit: A new Circuit object.
        """
        # check if all dimensions are the different
        if len(set(dimensions)) != len(dimensions):
            g_max = 3  # only single qudit gates if not all dimensions are different (never selects CX)
        else:
            g_max = 2  # all gates possible

        gate_list = [H, S, CX]
        gg = []
        for i in range(depth):
            g_i = np.random.randint(g_max)
            if g_i == 2:
                aa = list(random.sample(range(n_qudits), 2))
                while aa[0] == aa[1] or dimensions[aa[0]] != dimensions[aa[1]]:
                    aa = list(random.sample(range(n_qudits), 2))
                gg += [gate_list[g_i](aa[0], aa[1], dimensions[aa[0]])]
            else:
                aa = list(random.sample(range(n_qudits), 1))
                gg += [gate_list[g_i](aa[0], dimensions[aa[0]])]

        return cls(dimensions, gg)

    def add_gate(self, gate: Gate | list[Gate]):
        """
        Appends a gate to qudit index with specified target (if relevant)

        If gate is a list indexes should be a list of integers or tuples
        """
        if isinstance(gate, list) or isinstance(gate, np.ndarray):
            for i, g in enumerate(gate):
                self.gates.append(g)
                self.indexes.append(g.qudit_indices)
        else:
            self.gates.append(gate)
            self.indexes.append(gate.qudit_indices)

    def remove_gate(self, index: int):
        """
        Removes a gate from the circuit at the specified index
        """
        self.gates.pop(index)
        self.indexes.pop(index)

    def n_qudits(self) -> int:
        """
        Returns the number of qudits in the circuit.
        """
        return len(self.dimensions)

    def __add__(self, other: "Circuit | Gate") -> "Circuit":
        """
        Adds two circuits together by concatenating their gates and indexes.
        """
        if not isinstance(other, Circuit) and not isinstance(other, Gate):
            raise TypeError("Can only add another Circuit or Gate object.")
        if isinstance(other, Gate):
            new_gates = self.gates + [other]
        else:
            new_gates = self.gates + other.gates
        return Circuit(self.dimensions, new_gates)

    def __eq__(self, other: 'Circuit') -> bool:
        if not isinstance(other, Circuit):
            return False
        if len(self.gates) != len(other.gates):
            return False
        for i in range(len(self.gates)):
            if self.gates[i] != other.gates[i]:
                return False
        return True

    def __getitem__(self, index: int) -> Gate:
        return self.gates[index]

    def __setitem__(self, index: int, value: Gate):
        self.gates[index] = value
        self.indexes[index] = value.qudit_indices

    def __len__(self) -> int:
        return len(self.gates)

    def __str__(self) -> str:
        str_out = ''
        for gate in self.gates:
            str_out += gate.name + ' ' + str(gate.qudit_indices) + '\n'
        return str_out

    @overload
    def act(self, pauli: Pauli | PauliString) -> PauliString:
        ...

    @overload
    def act(self, pauli: PauliSum) -> PauliSum:
        ...

    def act(self, pauli: Pauli | PauliString | PauliSum) -> PauliString | PauliSum:
        if isinstance(pauli, Pauli):
            if self.dimensions[0] != pauli.dimension or len(self.dimensions) != 1:
                raise ValueError("Pauli dimension does not match circuit dimensions")
            else:
                pauli = PauliString.from_pauli(pauli)

        elif np.any(self.dimensions != pauli.dimensions):
            raise ValueError("Pauli dimensions do not match circuit dimensions")
        for gate in self.gates:
            pauli = gate.act(pauli)
        return pauli

    def act_iter(self, pauli_sum:  PauliSum) -> Generator[PauliSum, None, None]:
        if np.any(self.dimensions != pauli_sum.dimensions):
            raise ValueError("Pauli dimensions do not match circuit dimensions")
        for gate in self.gates:
            pauli_sum = gate.act(pauli_sum)
            yield pauli_sum

    def show(self):
        if np.all(np.array(self.dimensions) != 2):
            print("Circuit dimensions are all 2, using Qiskit QuantumCircuit, some gates may not be supported")
        circuit = QuantumCircuit(len(self.dimensions))
        dict = {'X': circuit.x, 'H': circuit.h, 'S': circuit.s, 'SUM': circuit.cx, 'CNOT': circuit.cx,
                'Hdag': circuit.h}

        for gate in self.gates:
            name = gate.name
            if len(gate.qudit_indices) == 2:
                dict[name](gate.qudit_indices[0], gate.qudit_indices[1])
            else:
                dict[name](gate.qudit_indices[0])

        print(circuit)
        # return circuit

    def copy(self) -> 'Circuit':
        return Circuit(self.dimensions, self.gates.copy())

    def embed_circuit(self, circuit: 'Circuit', qudit_indices: list[int] | np.ndarray | None = None):
        """
        Embed a circuit into current circuit at the specified qudit indices.
        """

        if qudit_indices is not None:
            if len(qudit_indices) != circuit.n_qudits():
                raise ValueError("Number of qudit indices does not match number of qudits in circuit to embed")

        for gate in circuit.gates:
            new_gate = gate.copy()
            if qudit_indices is not None:
                new_indexes = [qudit_indices[j] for j in gate.qudit_indices]
                new_gate.qudit_indices = np.ndarray(new_indexes)
            self.add_gate(new_gate)

    def _composite_phase_vector(self, F_1: np.ndarray, F_2: np.ndarray, h_2: np.ndarray, lcm: int) -> np.ndarray:
        """
        Returns the vector to add to h_1 to obtain h'' in PHYSICAL REVIEW A 71, 042315 (2005) - Eq. (8)

        New phase vector is h_1 + h_c

        """
        U = np.zeros((2 * self.n_qudits(), 2 * self.n_qudits()), dtype=int)
        U[self.n_qudits():, :self.n_qudits()] = np.eye(self.n_qudits(), dtype=int)

        U_conjugated = F_2.T @ U @ F_2

        p1 = np.dot(F_1, h_2)
        # negative sign in below as definition in paper is strictly upper diagonal, not including diagonal part
        p2 = np.diag(np.dot(F_1, np.dot((2 * np.triu(U_conjugated) - np.diag(np.diag(U_conjugated))), F_1.T)))
        p3 = np.dot(F_1, np.diag(U_conjugated))

        h_c = (p1 + p2 - p3) % (2 * lcm)

        return h_c

    def composite_gate(self) -> Gate:
        """Composes the list of symplectics acting on all qudits to a single symplectic"""

        total_indexes = []
        total_symplectic = np.eye(2 * self.n_qudits(), dtype=np.uint8)
        lcm = np.lcm.reduce(self.dimensions)
        for i, gate in enumerate(self.gates):
            symplectic = gate.symplectic
            indexes = gate.qudit_indices
            phase_vector = gate.phase_vector

            F, h = embed_symplectic(symplectic, phase_vector, indexes, self.n_qudits())  #
            if i == 0:
                total_phase_vector = h
            else:
                total_phase_vector = np.mod(total_phase_vector + self._composite_phase_vector(total_symplectic, F, h,
                                                                                              lcm),
                                            2 * lcm)
            print(phase_vector, total_phase_vector)

            total_symplectic = np.mod(total_symplectic @ F.T, lcm)

            total_indexes.extend(indexes)

        total_indexes = list(set(np.sort(total_indexes)))
        total_symplectic = total_symplectic.T
        return Gate('CompositeGate', total_indexes, total_symplectic, self.dimensions, total_phase_vector)

    def unitary(self):
        known_unitaries = (Hadamard, PHASE, SUM, SWAP, CNOT)
        if not np.all([isinstance(gate, known_unitaries) for gate in self.gates]):
            print(self.gates)
            raise NotImplementedError("Unitary not implemented for all gates in the circuit.")
        q = self.dimensions
        m = sp.csr_matrix(([1] * (np.prod(q)), (range(np.prod(q)), range(np.prod(q)))))
        for g in self.gates:
            m = g.unitary(dims=self.dimensions) @ m
        return m
