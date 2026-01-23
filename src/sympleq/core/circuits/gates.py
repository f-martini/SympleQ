from __future__ import annotations
from abc import ABC
import numpy as np
from typing import TypeVar, Self

from sympleq.core.paulis import PauliObject
from sympleq.core.circuits.utils import transvection_matrix

# We define a type using TypeVar to let the type checker know that
# the input and output of the `act` function share the same type.
P = TypeVar("P", bound="PauliObject")


class Gate(ABC):
    """
    Abstract base class for dimension-independent Clifford gates.

    Gates are defined purely by their symplectic matrix and phase vector(s).
    They do not store qudit indices or dimensions - these are passed at act-time.

    The symplectic matrix defines how Pauli operators transform under conjugation.
    The phase vector handles the phase acquired during the transformation.
    Some gates (like PHASE) have dimension-dependent phase vectors, which are
    stored in `_exceptional_phase_vectors`.
    """

    def __init__(self, name: str, symplectic: np.ndarray,
                 phase_vector: np.ndarray | None = None,
                 exceptional_phase_vectors: dict[int, np.ndarray] | None = None):
        self._name = name
        self._n_qudits = symplectic.shape[0] // 2
        self._symplectic = symplectic.astype(int)
        self._symplectic.setflags(write=False)

        if phase_vector is None:
            phase_vector = np.zeros(2 * self._n_qudits, dtype=int)
        self._phase_vector = np.asarray(phase_vector, dtype=int)
        self._phase_vector.setflags(write=False)

        # Store dimension-specific phase vectors (e.g., qubits are special for PHASE gate)
        self._exceptional_phase_vectors: dict[int, np.ndarray] = {}
        if exceptional_phase_vectors is not None:
            for dim, pv in exceptional_phase_vectors.items():
                pv_arr = np.asarray(pv, dtype=int)
                pv_arr.setflags(write=False)
                self._exceptional_phase_vectors[dim] = pv_arr

        # Precompute matrices for phase calculation (see Eq.[7] in PHYSICAL REVIEW A 71, 042315 (2005))
        # U = [[0_n, 0_n],
        #      [I_n, 0_n]]
        U = np.zeros((2 * self._n_qudits, 2 * self._n_qudits), dtype=int)
        U[self._n_qudits:, :self._n_qudits] = np.eye(self._n_qudits, dtype=int)
        self._U_symplectic_conjugated = self._symplectic.T @ U @ self._symplectic

        self._V_diag = np.diag(self._U_symplectic_conjugated)
        # This is the part associated with the quadratic form.
        # Remove diagonal part to match definition in Eq.[7].
        self._p_part = 2 * np.triu(self._U_symplectic_conjugated) - np.diag(self._V_diag)

        # Will be set by _Gates to link to inverse gate
        self._inverse: Self | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_qudits(self) -> int:
        return self._n_qudits

    @property
    def symplectic(self) -> np.ndarray:
        return self._symplectic

    def phase_vector(self, dimension: int = 0) -> np.ndarray:
        """
        Get the phase vector for a given dimension.

        Some gates have dimension-specific phase vectors (e.g., PHASE gate for qubits).
        If no exceptional phase vector exists for the given dimension, returns the default.
        """
        if dimension in self._exceptional_phase_vectors:
            return self._exceptional_phase_vectors[dimension]
        return self._phase_vector

    def __repr__(self) -> str:
        return f"Gate(name={self._name}, n_qudits={self._n_qudits})"

    def act(self, pauli: P, qudits: int | tuple[int, ...]) -> P:
        """
        Apply this gate to a Pauli object at the specified qudit indices.

        Returns the updated Pauli with transformed tableau and phases.
        See Eq.[7] in PHYSICAL REVIEW A 71, 042315 (2005).

        Parameters
        ----------
        pauli : Pauli | PauliString | PauliSum
            The Pauli object to transform.
        qudits : int | tuple[int, ...]
            The qudit index (for single-qudit gates) or tuple of indices
            (for multi-qudit gates) on which the gate acts.

        Returns
        -------
        The transformed Pauli object of the same type as the input.
        """
        if isinstance(qudits, int):
            qudits = (qudits,)

        affected_qudits = np.asarray(qudits, dtype=int)

        if len(affected_qudits) != self._n_qudits:
            raise ValueError(f"Gate acts on {self._n_qudits} qudits, but {len(affected_qudits)} indices provided.")

        T = pauli.tableau

        # Precompute tableau mask to select affected columns
        tableau_mask = np.concatenate([affected_qudits, affected_qudits + pauli.n_qudits()])
        T_affected = T[:, tableau_mask]

        pauli_dimensions = pauli.dimensions[affected_qudits]
        relevant_dimensions = np.tile(pauli_dimensions, 2)
        relevant_lcm = int(np.lcm.reduce(pauli_dimensions))

        # Apply symplectic transformation with modulo reduction
        updated_tableau = np.mod(T_affected @ self._symplectic.T, relevant_dimensions)
        new_tableau = T.copy()
        new_tableau[:, tableau_mask] = updated_tableau

        # Compute phase contribution
        phase_vec = self.phase_vector(relevant_lcm)
        modified_phase_vector = phase_vec - self._V_diag
        linear_terms = T_affected @ modified_phase_vector
        quadratic_terms = np.sum(T_affected * (T_affected @ self._p_part), axis=1)

        dimensional_factor = pauli.lcm // relevant_lcm
        acquired_phases = (linear_terms + quadratic_terms) * dimensional_factor

        new_phases = (pauli.phases + acquired_phases) % (2 * pauli.lcm)

        return pauli.__class__(tableau=new_tableau, dimensions=pauli.dimensions,
                               weights=pauli.weights, phases=new_phases)

    def inverse(self) -> Self:
        """Return the inverse of this gate.

        For singleton gates (from GATES), returns the pre-linked inverse.
        Otherwise computes the inverse symplectic matrix.
        """
        if self._inverse is not None:
            return self._inverse

        # Compute inverse symplectic: C^{-1} = -Omega^T @ C^T @ Omega
        n = self._n_qudits
        zero_block = np.zeros((n, n), dtype=int)
        identity_block = np.eye(n, dtype=int)
        Omega = np.block([[zero_block, identity_block], [-identity_block, zero_block]])

        C_inv = -Omega.T @ self._symplectic.T @ Omega

        # TODO: Compute inverse phase vector properly
        import warnings
        warnings.warn("Inverse phase vector computation not fully implemented for generic gates.")
        phase_vector_inv = np.zeros(2 * n, dtype=int)

        inv_name = self._name + "_inv" if not self._name.endswith("_inv") else self._name[:-4]
        return self.__class__(inv_name, C_inv, phase_vector_inv)

    def transvection(self, transvection_vector: np.ndarray | list, transvection_weight: int = 1) -> Gate:
        """
        Returns a new gate that is the transvection of this gate by the given vector.

        The transvection vector should be a 2n-dimensional vector where n is the number of qudits.
        Note: This returns a generic Gate, not a subclass instance.
        """
        if not isinstance(transvection_weight, int) and not isinstance(transvection_weight, np.int64):
            raise TypeError("Transvection weight must be an integer.")

        if isinstance(transvection_vector, list):
            transvection_vector = np.asarray(transvection_vector)

        T = transvection_matrix(transvection_vector, multiplier=transvection_weight)
        new_name = self._name if self._name.startswith("T-") else "T-" + self._name

        return _GenericGate(new_name, self._symplectic @ T, self._phase_vector.copy())


class _GenericGate(Gate):
    """
    A generic gate for computed gates (e.g., from transvection, composite gates).
    These are not singletons and don't have pre-computed inverses.
    Uses parent's inverse() which computes the inverse dynamically.
    """
    pass


class _Hadamard(Gate):
    """Hadamard gate: X -> -Z, Z -> X (or inverse: X -> Z, Z -> -X)"""

    def __init__(self, is_inverse: bool = False):
        self._is_inverse = is_inverse

        if is_inverse:
            symplectic = np.array([
                [0, 1],    # image of X:  X -> Z
                [-1, 0]    # image of Z:  Z -> -X
            ], dtype=int)
            name = "H_inv"
        else:
            symplectic = np.array([
                [0, -1],   # image of X:  X -> -Z
                [1, 0]     # image of Z:  Z -> X
            ], dtype=int)
            name = "H"

        super().__init__(name, symplectic)


class _PHASE(Gate):
    """Phase gate (S): X -> XZ, Z -> Z. Has special phase vector for qubits."""

    def __init__(self, is_inverse: bool = False):
        self._is_inverse = is_inverse

        if is_inverse:
            symplectic = np.array([
                [1, -1],  # image of X:  X -> XZ^{-1}
                [0, 1]    # image of Z:  Z -> Z
            ], dtype=int).T
            name = "S_inv"
        else:
            symplectic = np.array([
                [1, 1],   # image of X:  X -> XZ
                [0, 1]    # image of Z:  Z -> Z
            ], dtype=int).T
            name = "S"

        # Qubits (dimension=2) have a special phase vector
        if is_inverse:
            exceptional = {2: np.array([-1, 0], dtype=int)}
        else:
            exceptional = {2: np.array([1, 0], dtype=int)}

        super().__init__(name, symplectic, exceptional_phase_vectors=exceptional)


class _SUM(Gate):
    """SUM (CNOT) gate: X0 -> X0 X1, X1 -> X1, Z0 -> Z0, Z1 -> Z0^{-1} Z1"""

    def __init__(self, is_inverse: bool = False):
        self._is_inverse = is_inverse

        # SUM is self-inverse for qubits, but not in general
        if is_inverse:
            symplectic = np.array([
                [1, -1, 0, 0],   # image of X0:  X0 -> X0 X1^{-1}
                [0, 1, 0, 0],    # image of X1:  X1 -> X1
                [0, 0, 1, 0],    # image of Z0:  Z0 -> Z0
                [0, 0, 1, 1]     # image of Z1:  Z1 -> Z0 Z1
            ], dtype=int).T
            name = "SUM_inv"
        else:
            symplectic = np.array([
                [1, 1, 0, 0],    # image of X0:  X0 -> X0 X1
                [0, 1, 0, 0],    # image of X1:  X1 -> X1
                [0, 0, 1, 0],    # image of Z0:  Z0 -> Z0
                [0, 0, -1, 1]    # image of Z1:  Z1 -> Z0^{-1} Z1
            ], dtype=int).T
            name = "SUM"

        super().__init__(name, symplectic)


class _SWAP(Gate):
    """SWAP gate: X0 <-> X1, Z0 <-> Z1. Self-inverse."""

    def __init__(self):
        symplectic = np.array([
            [0, 1, 0, 0],  # image of X0:  X0 -> X1
            [1, 0, 0, 0],  # image of X1:  X1 -> X0
            [0, 0, 0, 1],  # image of Z0:  Z0 -> Z1
            [0, 0, 1, 0]   # image of Z1:  Z1 -> Z0
        ], dtype=int).T

        super().__init__("SWAP", symplectic)

    def inverse(self) -> _SWAP:
        # SWAP is self-inverse
        return self


class _CZ(Gate):
    """Controlled-Z gate. Self-inverse."""

    def __init__(self):
        symplectic = np.array([
            [1, 0, 0, 0],  # image of X0:  X0 -> X0
            [0, 1, 0, 0],  # image of X1:  X1 -> X1
            [0, 1, 1, 0],  # image of Z0:  Z0 -> X1 Z0
            [1, 0, 0, 1]   # image of Z1:  Z1 -> X0 Z1
        ], dtype=int).T

        super().__init__("CZ", symplectic)

    def inverse(self) -> _CZ:
        # CZ is self-inverse
        return self


class _Gates:
    """
    Singleton container for pre-instantiated gates.

    Gates are accessed as properties, e.g., GATES.H, GATES.S, GATES.SUM.
    Inverse gates are linked so that gate.inverse() returns the inverse singleton.
    """

    def __init__(self):
        # Single-qudit gates
        self._H = _Hadamard(is_inverse=False)
        self._H_inv = _Hadamard(is_inverse=True)
        self._H._inverse = self._H_inv
        self._H_inv._inverse = self._H

        self._S = _PHASE(is_inverse=False)
        self._S_inv = _PHASE(is_inverse=True)
        self._S._inverse = self._S_inv
        self._S_inv._inverse = self._S

        # Two-qudit gates
        self._SUM = _SUM(is_inverse=False)
        self._SUM_inv = _SUM(is_inverse=True)
        self._SUM._inverse = self._SUM_inv
        self._SUM_inv._inverse = self._SUM

        self._SWAP = _SWAP()
        # SWAP is self-inverse, already handled in the class

        self._CZ = _CZ()
        # CZ is self-inverse, already handled in the class

    # Hadamard
    @property
    def H(self) -> _Hadamard:
        return self._H

    @property
    def H_inv(self) -> _Hadamard:
        return self._H_inv

    @property
    def hadamard(self) -> _Hadamard:
        return self._H

    # Phase (S)
    @property
    def S(self) -> _PHASE:
        return self._S

    @property
    def S_inv(self) -> _PHASE:
        return self._S_inv

    @property
    def phase(self) -> _PHASE:
        return self._S

    # SUM / CNOT / CX
    @property
    def SUM(self) -> _SUM:
        return self._SUM

    @property
    def SUM_inv(self) -> _SUM:
        return self._SUM_inv

    @property
    def CNOT(self) -> _SUM:
        return self._SUM

    @property
    def CX(self) -> _SUM:
        return self._SUM

    # SWAP
    @property
    def SWAP(self) -> _SWAP:
        return self._SWAP

    # CZ
    @property
    def CZ(self) -> _CZ:
        return self._CZ


# Global singleton instance
GATES = _Gates()


class PauliGate(Gate):
    """
    A gate constructed from a PauliString.

    Unlike the singleton gates, PauliGate is dynamically created based on the input PauliString.
    The symplectic matrix is identity, and the phase vector encodes the Pauli conjugation effect.
    """

    def __init__(self, pauli):
        # Import here to avoid circular imports
        from sympleq.core.paulis import PauliString
        from sympleq.core.circuits.utils import symplectic_form

        if not isinstance(pauli, PauliString):
            raise TypeError("PauliGate requires a PauliString")

        self.pauli_string = pauli
        n = pauli.n_qudits()
        lcm = int(pauli.lcm)

        symplectic = np.eye(2 * n, dtype=int)
        phase_vector = (2 * symplectic_form(n, lcm) @ np.concatenate([pauli.x_exp, pauli.z_exp])) % (2 * lcm)

        # Store dimensions for this gate (needed for act method compatibility)
        self._dimensions = np.asarray(pauli.dimensions, dtype=int)

        super().__init__("Pauli", symplectic, phase_vector)

    @property
    def dimensions(self) -> np.ndarray:
        return self._dimensions

    def inverse(self) -> PauliGate:
        # Pauli gates are self-inverse (up to phase)
        return self

    def act(self, pauli: P, qudits: int | tuple[int, ...] | None = None) -> P:
        """
        Apply this PauliGate to a Pauli object.

        For PauliGate, qudits defaults to all qudits in order (0, 1, 2, ..., n-1)
        since the gate was constructed for a specific number of qudits.
        """
        if qudits is None:
            qudits = tuple(range(self._n_qudits))
        return super().act(pauli, qudits)


# Convenience aliases for backward compatibility
# These are the gate classes, not instances
Hadamard = _Hadamard
PHASE = _PHASE
SUM = _SUM
SWAP = _SWAP
CZ = _CZ
