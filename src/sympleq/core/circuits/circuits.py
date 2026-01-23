from __future__ import annotations
from typing import Generator, overload, TypeVar, TypeAlias, Unpack
import numpy as np
from .gates import Gate, Hadamard as H, PHASE as S, SUM as CX, SWAP, CNOT, PauliGate
from sympleq.core.paulis import PauliSum, PauliString, Pauli, PauliObject
from .utils import embed_symplectic
import scipy.sparse as sp
from collections import defaultdict

from .gates import Gate, GATES, _GenericGate
from .utils import embed_symplectic
from sympleq.core.paulis import PauliSum, PauliString, Pauli, PauliObject


# Type alias for gate + qudit indices tuple (used internally)
GateTuple: TypeAlias = tuple[Gate, tuple[int, ...]]

# Type alias for from_tuples input: (Gate, qudit_idx1, qudit_idx2, ...)
GateSpec: TypeAlias = tuple[Gate, *tuple[int, ...]]

# We define a type using TypeVar to let the type checker know that
# the input and output of the `act` function share the same type.
P = TypeVar("P", bound="PauliObject")


class Circuit:
    """
    A quantum circuit consisting of gates applied to specific qudits.

    The circuit stores:
    - dimensions: the dimension of each qudit (e.g., [2, 2, 2] for 3 qubits)
    - gates: list of Gate objects (dimension-independent)
    - qudits: list of tuples indicating which qudits each gate acts on

    Gates and qudits are stored separately because gates are now dimension-independent
    singletons that don't store their target qudits.
    """

    def __init__(self, dimensions: list[int] | np.ndarray,
                 gates: list[Gate] | None = None,
                 qudits: list[tuple[int, ...]] | None = None):
        """
        Initialize the Circuit.

        Parameters
        ----------
        dimensions : list[int] | np.ndarray
            The dimension of each qudit in the circuit.
        gates : list[Gate] | None
            List of Gate objects. If None, creates an empty circuit.
        qudits : list[tuple[int, ...]] | None
            List of tuples indicating which qudits each gate acts on.
            Must have the same length as gates.
        """
        self.dimensions = np.asarray(dimensions, dtype=int)

        if gates is None:
            gates = []
        if qudits is None:
            qudits = []

        if len(gates) != len(qudits):
            raise ValueError(f"gates and qudits must have the same length, "
                             f"got {len(gates)} gates and {len(qudits)} qudit tuples.")

        self._gates = gates
        self._qudits = list(qudits)

    @property
    def gates(self) -> list[Gate]:
        """List of gates in the circuit."""
        return self._gates

    @property
    def qudits(self) -> list[tuple[int, ...]]:
        """List of qudit index tuples for each gate."""
        return self._qudits

    @classmethod
    def from_random(cls, n_gates: int,
                    dimensions: list[int] | np.ndarray,
                    two_qudit_gate_ratio: float = 0.3) -> Circuit:
        """
        Creates a random circuit with the given number of gates.

        Parameters
        ----------
        n_gates : int
            Number of gates in the circuit.
        dimensions : list[int] | np.ndarray
            The dimension of each qudit.
        two_qudit_gate_ratio : float
            Probability of choosing a two-qudit gate vs single-qudit gate.

        Returns
        -------
        Circuit
            A new random Circuit.
        """
        def index_lists(lst):
            groups = defaultdict(list)
            for i, val in enumerate(lst):
                groups[val].append(i)
            return list(groups.values())

        dimensions = np.asarray(dimensions, dtype=int)
        index_sets = index_lists(dimensions)  # list of lists of indexes for each dimension
        n_dims = len(index_sets)

        single_qudit_gates = [GATES.H, GATES.S]
        two_qudit_gates = [GATES.SUM, GATES.SWAP]

        gates = []
        qudits = []

        for _ in range(n_gates):
            set_idx = np.random.randint(n_dims)
            if np.random.rand() < two_qudit_gate_ratio and len(index_sets[set_idx]) > 1:
                indices = tuple(np.random.choice(index_sets[set_idx], 2, replace=False))
                gate = np.random.choice(two_qudit_gates)
                gates.append(gate)
                qudits.append(indices)
            else:
                index = int(np.random.choice(index_sets[set_idx]))
                gate = np.random.choice(single_qudit_gates)
                gates.append(gate)
                qudits.append((index,))

        return cls(dimensions, gates, qudits)

    @classmethod
    def from_tuples(cls, dimensions: list[int] | np.ndarray,
                    data: list[GateSpec] | GateSpec) -> Circuit:
        """
        Creates a circuit from a list of (gate, qudit_indices...) tuples.

        Parameters
        ----------
        dimensions : list[int] | np.ndarray
            The dimension of each qudit.
        data : list of tuples
            Each tuple contains (Gate, qudit_idx1, qudit_idx2, ...).

        Returns
        -------
        Circuit
            A new Circuit.

        Example
        -------
        >>> Circuit.from_tuples([2, 2], [(GATES.H, 0), (GATES.SUM, 0, 1)])
        """
        if isinstance(data, tuple) and isinstance(data[0], Gate):
            data = [data]

        assert isinstance(data, list)

        gates = [d[0] for d in data]
        qudits = [d[1:] for d in data]
        return cls(dimensions, gates, qudits)

    def add_gate(self, gate: Gate, *qudit_indices: int):
        """
        Appends a gate acting on the specified qudits.

        Parameters
        ----------
        gate : Gate
            The gate to add.
        qudit_indices : int
            The indices of the qudits the gate acts on.
        """
        if len(qudit_indices) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} acts on {gate.n_qudits} qudits, "
                             f"but {len(qudit_indices)} indices provided.")

        for idx in qudit_indices:
            if idx < 0 or idx >= len(self.dimensions):
                raise IndexError(f"Qudit index {idx} out of range for circuit with {len(self.dimensions)} qudits.")

        self._gates.append(gate)
        self._qudits.append(tuple(qudit_indices))

    def remove_gate(self, index: int):
        """Removes the gate at the specified index."""
        self._gates.pop(index)
        self._qudits.pop(index)

    def n_qudits(self) -> int:
        """Returns the number of qudits in the circuit."""
        return len(self.dimensions)

    @property
    def lcm(self) -> int:
        """Returns the LCM of all qudit dimensions."""
        return int(np.lcm.reduce(self.dimensions))

    def __add__(self, other: Circuit) -> Circuit:
        """Concatenates two circuits."""
        if not isinstance(other, Circuit):
            raise TypeError("Can only add another Circuit object.")

        if not np.array_equal(self.dimensions, other.dimensions):
            raise ValueError("Cannot concatenate circuits with different dimensions.")

        new_gates = self._gates + other._gates
        new_qudits = self._qudits + other._qudits
        return Circuit(self.dimensions, new_gates, new_qudits)

    def __eq__(self, other: Circuit) -> bool:
        if not isinstance(other, Circuit):
            return False
        if not np.array_equal(self.dimensions, other.dimensions):
            return False
        if len(self._gates) != len(other._gates):
            return False
        for i in range(len(self._gates)):
            if self._gates[i] is not other._gates[i]:  # Compare by identity for singletons
                return False
            if self._qudits[i] != other._qudits[i]:
                return False
        return True

    def __getitem__(self, index: int) -> GateTuple:
        """Returns (gate, qudits) tuple at the given index."""
        return (self._gates[index], self._qudits[index])

    def __len__(self) -> int:
        return len(self._gates)

    def __str__(self) -> str:
        lines = [f"Circuit on {self.n_qudits()} qudits (dims={list(self.dimensions)}):"]
        for gate, qudits in zip(self._gates, self._qudits):
            lines.append(f"  {gate.name} {qudits}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Circuit(dimensions={list(self.dimensions)}, n_gates={len(self._gates)})"

    @overload
    def act(self, pauli: Pauli) -> Pauli:
        ...

    @overload
    def act(self, pauli: PauliString) -> PauliString:
        ...

    @overload
    def act(self, pauli: PauliSum) -> PauliSum:
        ...

    def act(self, pauli: P) -> P:
        """Apply all gates in the circuit to a Pauli object."""
        for gate, qudits in zip(self._gates, self._qudits):
            pauli = gate.act(pauli, qudits)
        return pauli

    @overload
    def act_iter(self, pauli: Pauli) -> Generator[Pauli, None, None]:
        ...

    @overload
    def act_iter(self, pauli: PauliString) -> Generator[PauliString, None, None]:
        ...

    @overload
    def act_iter(self, pauli: PauliSum) -> Generator[PauliSum, None, None]:
        ...

    def act_iter(self, pauli: P) -> Generator[P, None, None]:
        """Yields the Pauli object after each gate application."""
        for gate, qudits in zip(self._gates, self._qudits):
            pauli = gate.act(pauli, qudits)
            yield pauli

    def copy(self) -> 'Circuit':
        """Returns a shallow copy of the circuit."""
        return Circuit(self.dimensions.copy(), self._gates.copy(), self._qudits.copy())

    def _composite_phase_vector(self, F_1: np.ndarray, F_2: np.ndarray, h_2: np.ndarray) -> np.ndarray:
        """
        Computes the phase vector contribution when composing two symplectics.
        See Eq.(8) in PHYSICAL REVIEW A 71, 042315 (2005).
        """
        n = self.n_qudits()
        U = np.zeros((2 * n, 2 * n), dtype=int)
        U[n:, :n] = np.eye(n, dtype=int)

        U_conjugated = F_2.T @ U @ F_2

        p1 = np.dot(F_1, h_2)
        p2 = np.diag(np.dot(F_1, np.dot((2 * np.triu(U_conjugated) - np.diag(np.diag(U_conjugated))), F_1.T)))
        p3 = np.dot(F_1, np.diag(U_conjugated))

        return p1 + p2 - p3

    def composite_gate(self) -> Gate:
        """
        Composes all gates into a single equivalent gate.

        Returns a generic Gate (not a singleton) representing the full circuit transformation.
        """
        n_qudits = self.n_qudits()
        total_symplectic = np.eye(2 * n_qudits, dtype=int)
        lcm = self.lcm
        total_phase_vector = np.zeros(2 * n_qudits, dtype=int)

        for i, (gate, qudits) in enumerate(zip(self._gates, self._qudits)):
            # Get the phase vector for the relevant dimension
            relevant_dim = int(np.lcm.reduce(self.dimensions[list(qudits)]))
            phase_vec = gate.phase_vector(relevant_dim)

            # Embed the local symplectic into the full space
            F, h = embed_symplectic(gate.symplectic, phase_vec, list(qudits), n_qudits)

            if i == 0:
                total_phase_vector = h
            else:
                total_phase_vector = np.mod(
                    total_phase_vector + self._composite_phase_vector(total_symplectic, F, h),
                    2 * lcm
                )

            total_symplectic = np.mod(total_symplectic @ F.T, lcm)

        total_symplectic = total_symplectic.T
        return _GenericGate('CompositeGate', total_symplectic, total_phase_vector)

    def inverse(self) -> Circuit:
        """Returns the inverse circuit (gates in reverse order, each inverted)."""
        inv_gates = [g.inverse() for g in reversed(self._gates)]
        inv_qudits = list(reversed(self._qudits))
        return Circuit(self.dimensions, inv_gates, inv_qudits)

    def full_symplectic(self) -> np.ndarray:
        """Returns the full symplectic matrix of the composite gate."""
        return self.composite_gate().symplectic
