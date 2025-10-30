from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Self

from .constants import DEFAULT_QUDIT_DIMENSION

P = TypeVar("P", bound="PauliObject")


class PauliObject(ABC):
    def __init__(self, tableau: np.ndarray, dimensions: int | list[int] | np.ndarray | None = None,
                 weights: int | float | complex | list[int | float | complex] | np.ndarray | None = None,
                 phases: int | list[int] | np.ndarray | None = None):
        """
        Constructor for Pauli objects represented in symplectic tableau form.
        Represents a sum of Pauli operators acting on multiple qudits.
        For more details, see the references:
        `Phys. Rev. A 71, 042315 (2005) <https://doi.org/10.1103/PhysRevA.71.042315>`_
        and
        `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_

        Parameters
        ----------
        tableau : np.ndarray
            Symplectic tableau representation of the object. A 2-D array with shape (n_paulis, 2 * n_qudits).
            For a single Pauli and a PauliString, n_paulis = 1.
        dimensions : int | list[int] | np.ndarray | None, optional
            Qudit dimension(s).
            - If int, a single qudit dimension is assumed or broadcast where appropriate.
            - If list/np.ndarray, must match the number of qudits implied by `tableau`.
            - If None, all dimensions are defaulted to DEFAULT_QUDIT_DIMENSION.
        weights : list | np.ndarray | None, optional
            Coefficients associated with each PauliString (for PauliSum).
            If None, it defaults to 1.
            For PauliString and Pauli, it is just a 1-D array of length 1.
        phases : list[int] | np.ndarray | None, optional
            Integer phases associated with each PauliString.
            Values are typically interpreted modulo (2 * lcm()).
            If None, it defaults to zero.

        Attributes
        ----------
        weights: list[int | float | complex] | np.ndarray | None = None
            The weights for each PauliString.
        phases: int | list[int] | np.ndarray | None = None
            The phases of the PauliStrings in the range [0, lcm(dimensions) - 1].
        dimensions : list[int] | np.ndarray | int, optional
            The dimensions of each qudit. If an integer is provided,
            all qudits are assumed to have the same dimension.
            If no value is provided, the default is `DEFAULT_QUDIT_DIMENSION`.
        lcm : int
            Least common multiplier of all qudit dimensions.
        """

        if tableau.ndim == 1:
            tableau = tableau.reshape(1, -1)

        n_pauli_strings = tableau.shape[0]
        n_qudits = tableau.shape[1] // 2

        if dimensions is None:
            dimensions = np.ones(n_qudits) * DEFAULT_QUDIT_DIMENSION
        else:  # Catches int but also list and arrays of length 1
            dimensions = np.asarray(dimensions, dtype=int)
            if dimensions.ndim == 0:
                dimensions = np.full(n_qudits, dimensions.item(), dtype=int)

        self._dimensions = dimensions
        self._lcm = int(np.lcm.reduce(self.dimensions()))

        self._tableau = tableau % np.tile(self._dimensions, 2)

        if weights is None:
            weights = np.ones(n_pauli_strings, dtype=complex)
        else:  # Catches scalars but also list and arrays of length 1
            weights = np.asarray(weights, dtype=complex)
            if weights.ndim == 0:
                weights = np.full(n_pauli_strings, weights.item(), dtype=complex)
        self._weights = weights

        if phases is None:
            phases = np.zeros(n_pauli_strings, dtype=int)
        else:  # Catches scalars but also list and arrays of length 1
            phases = np.asarray(phases, dtype=int)
            if phases.ndim == 0:
                phases = np.full(n_pauli_strings, phases.item(), dtype=int)
        self._phases = phases % (2 * self.lcm())

    def tableau(self) -> np.ndarray:
        """
        Returns the tableau representation of the Pauli object.
        The tableau representation is a vector of length 2 * n_qudits,
        where the first n_qudits entries correspond to the X exponents and the
        last n_qudits entries correspond to the Z exponents of the Pauli string.
        It is essential for efficient algebraic operations on Pauli strings, see
        `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_.

        Returns
        -------
        np.ndarray
            The tableau representation of the Pauli object.
        """
        return self._tableau

    def dimensions(self) -> np.ndarray:
        """
        Returns the dimensions of the Pauli object.

        Returns
        -------
        np.ndarray
            A 1D numpy array of length n_qudits().
        """
        return self._dimensions

    def lcm(self) -> int:
        """
        Returns the least common multiplier of the dimensions of the Pauli object.

        Returns
        -------
        int
            The Pauli object dimensions least common multiplier as integer.
        """
        return self._lcm

    def n_qudits(self) -> int:
        """
        Returns the number of qudits represented by the Pauli object.

        Returns
        -------
        int
            The number of qudits.
        """
        return len(self.dimensions())

    def n_paulis(self) -> int:
        """
        Returns the number of Pauli strings represented by the Pauli object.

        Returns
        -------
        int
            The number of Pauli strings.
        """
        return len(self.tableau())

    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the Pauli object.

        Returns
        -------
        tuple[int, int]
            The number of Pauli operators and the number of qudits.
        """
        return self.n_paulis(), self.n_qudits()

    @abstractmethod
    def phases(self) -> np.ndarray:
        # FIXME: improve docstring
        """
        Returns the phases associated with the Pauli object.
        These phases represent the numerator, the denominator is 2 * self.lcm()

        Returns
        -------
        np.ndarray
            The phases as a 1d-vector.
        """
        pass

    def set_phases(self, new_phases: list[int] | np.ndarray):
        # FIXME: improve docstring
        """
        Set the phases associated with the Pauli object.

        Parameters
        -------
        new_phases: list[int] | np.ndarray
            The new phases as a 1d-vector.
        """
        pass

    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Returns the weights associated with the Pauli object.

        Returns
        -------
        np.ndarray
            The weights as a 1d-vector.
        """
        pass

    def set_weights(self, new_weights: list[int] | np.ndarray):
        # FIXME: improve docstring
        """
        Set the weights associated with the Pauli object.

        Parameters
        -------
        new_weights: list[int] | np.ndarray
            The new weights as a 1d-vector.
        """
        pass

    def hermitian_conjugate(self) -> Self:
        # FIXME: add reference
        """
        Returns the Hermitian conjugate of the Pauli object.

        Returns
        -------
        PauliObject
            The PauliObject Hermitian conjugate
        """
        conjugate_weights = np.conj(self.weights())
        conjugate_tableau = (-self.tableau()) % np.tile(self.dimensions(), 2)

        acquired_phases = []
        for i in range(self.n_paulis()):
            hermitian_conjugate_phase = 0
            for j in range(self.n_qudits()):
                r = self.tableau()[i, j]
                s = self.tableau()[i, j + self.n_qudits()]
                hermitian_conjugate_phase += ((r * s) % self.dimensions()[j]) * self.lcm() / self.dimensions()[j]
            acquired_phases.append(2 * hermitian_conjugate_phase)
        acquired_phases = np.asarray(acquired_phases, dtype=int)

        conjugate_initial_phases = (-self.phases()) % (2 * self.lcm())
        conjugate_phases = (conjugate_initial_phases + acquired_phases) % (2 * self.lcm())

        return self.__class__(tableau=conjugate_tableau, dimensions=self.dimensions(),
                              weights=conjugate_weights, phases=conjugate_phases)

    H = hermitian_conjugate

    def is_hermitian(self) -> bool:
        """
        Checks if the PauliObject is Hermitian

        Returns
        -------
        bool
            True if the PauliObject is Hermitian, False otherwise
        """
        return self == self.H()

    def _sanity_check(self):
        """
        Validates the consistency of the PauliSum's internal representation.

        Raises
        ------
        ValueError
            If the lengths of `tableau`, and `dimensions` are not consistent
            or if any exponent is not valid for its corresponding dimension.
        """
        if len(self.weights()) != self.n_paulis():
            # FIXME: Improve error message
            raise ValueError("The weights and tableau have inconsistent shapes.")

        if len(self.phases()) != self.n_paulis():
            # FIXME: Improve error message
            raise ValueError("The phases and tableau have inconsistent shapes.")

        if len(self.tableau()[0]) != 2 * self.n_qudits():
            raise ValueError(f"Tableau ({len(self.tableau())}) should be twice as long as"
                             f"dimensions ({len(self.dimensions())}).")

        if np.any(self.dimensions() < DEFAULT_QUDIT_DIMENSION):
            bad_dims = self.dimensions()[self.dimensions() < DEFAULT_QUDIT_DIMENSION]
            raise ValueError(f"Dimensions {bad_dims} are less than {DEFAULT_QUDIT_DIMENSION}")

        d = np.tile(self.dimensions(), 2)
        if np.any((self.tableau() >= d)):
            bad_indices = np.where((self.tableau() >= d))[0]
            raise ValueError(
                f"Exponents at indices {bad_indices} are too large:"
                f"tableau={self.tableau()[bad_indices]}"
            )

        if np.any((self.tableau() < 0)):
            bad_indices = np.where((self.tableau() < 0))[0]
            raise ValueError(
                f"Exponents at indices {bad_indices} are negative:"
                f"tableau={self.tableau()[bad_indices]}"
            )
