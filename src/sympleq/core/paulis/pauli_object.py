from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Self, Union

from .constants import DEFAULT_QUDIT_DIMENSION

P = TypeVar("P", bound="PauliObject")

ScalarType = Union[float, complex, int]
PauliOrScalarType = Union['PauliObject', ScalarType]


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
            Coefficients associated with each PauliString (for Pauli object).
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
        dimensions: list[int] | np.ndarray | int, optional
            The dimensions of each qudit. If an integer is provided,
            all qudits are assumed to have the same dimension.
            If no value is provided, the default is `DEFAULT_QUDIT_DIMENSION`.
        lcm: int
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

    def is_close(self, other_pauli: Self, threshold: int = 10) -> bool:
        """
        Determine if two Pauli object objects are (almost) equal.

        Parameters
        ----------
        other_pauli : Pauli object
            The Pauli object instance to compare against.

        threshold: int


        Returns
        -------
        bool
            True if both Pauli object instances have identical PauliStrings, weights, phases, and dimensions;
            False otherwise.
        """
        if not isinstance(other_pauli, self.__class__):
            return False

        if not np.array_equal(self.tableau(), other_pauli.tableau()):
            return False

        if not np.all(np.isclose(self.weights(), other_pauli.weights(), 10**(-threshold))):
            return False

        if not np.array_equal(self.phases(), other_pauli.phases()):
            return False

        if not np.array_equal(self.dimensions(), other_pauli.dimensions()):
            return False

        return True

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

    # FIXME: Is this only for Pauli object?
    def is_hermitian(self) -> bool:
        """
        Checks if the PauliObject is Hermitian

        Returns
        -------
        bool
            True if the PauliObject is Hermitian, False otherwise
        """
        # FIXME: this is wonrg. maybe phase_to_weight standard_form would solve it for Pauli object
        # NOTE: rounding errors could make this fail.
        return self.to_standard_form().is_close(self.H().to_standard_form())

    def _sanity_check(self):
        """
        Validates the consistency of the Pauli object's internal representation.

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

    def __eq__(self, other_pauli: Self) -> bool:
        """
        Determine if two Pauli object objects are equal.

        Parameters
        ----------
        other_pauli : Pauli object
            The Pauli object instance to compare against.

        Returns
        -------
        bool
            True if both Pauli object instances have identical PauliStrings, weights, phases, and dimensions;
            False otherwise.
        """
        if not isinstance(other_pauli, self.__class__):
            return False

        if not np.array_equal(self.tableau(), other_pauli.tableau()):
            return False

        if not np.array_equal(self.weights(), other_pauli.weights()):
            return False

        if not np.array_equal(self.phases(), other_pauli.phases()):
            return False

        if not np.array_equal(self.dimensions(), other_pauli.dimensions()):
            return False

        return True

    def __ne__(self, other_pauli: Self) -> bool:
        """
        Determine if two Pauli object objects are different.

        Parameters
        ----------
        other_pauli : Pauli object
            The Pauli object instance to compare against.

        Returns
        -------
        bool
            True if the Pauli object instances do not have identical PauliStrings, weights, phases, and dimensions;
            False otherwise.
        """
        return not self == other_pauli

    def __gt__(self, other_pauli: Self) -> bool:
        """
        Compare this PauliString with another PauliString for the greater-than relationship.
        This method overrides the `>` operator to compare two PauliString objects by converting
        them to their integer representations (with bits reversed) and checking if this instance
        is greater than the other.

        Parameters
        ----------
        other_pauli : PauliString
            The other PauliString instance to compare against.

        Returns
        -------
        bool
            True if this PauliString is greater than `other_pauli`, False otherwise.

        Examples
        --------
        >>> ps1 = PauliString.from_string("x1z0 x0z1", [2, 2])
        >>> ps2 = PauliString.from_string("x0z1 x1z0", [2, 2])
        >>> ps1 > ps2
        True
        """

        # FIXME: can we compare PauliStrings with different n_qudits/dimensions?
        if self.n_qudits() != other_pauli.n_qudits():
            raise Exception("Cannot compare PauliStrings with different number of qudits.")

        # Flatten tableaus to 1D-vectors
        self_tableau = self.tableau().ravel()
        other_tableau = other_pauli.tableau().ravel()

        for i in range(len(self_tableau)):
            if self_tableau[i] == other_tableau[i]:
                continue
            # FIXME: is this really the intended behaviour?
            if self_tableau[i] < other_tableau[i]:
                return True
            return False

        # they are equal
        return False

    def __lt__(self, other_pauli: Self) -> bool:
        """
        Compare this PauliString with another PauliString for the greater-than relationship.
        This method overrides the `<` operator to compare two PauliString objects by converting
        them to their integer representations (with bits reversed) and checking if this instance
        is smaller than the other.

        Parameters
        ----------
        other_pauli : PauliString
            The other PauliString instance to compare against.

        Returns
        -------
        bool
            True if this PauliString is smaller than `other_pauli`, False otherwise.

        Examples
        --------
        >>> ps1 = PauliString.from_string("x1z0 x0z1", [2, 2])
        >>> ps2 = PauliString.from_string("x0z1 x1z0", [2, 2])
        >>> ps1 > ps2
        False
        """
        return not self.__gt__(other_pauli) and not self.__eq__(other_pauli)

    def __pow__(self, A: int) -> Self:
        """
        Raises the Pauli object to the power of an integer exponent.

        Parameters
        ----------
        A : int
            The integer exponent to which the Pauli object is to be raised.

        Returns
        -------
        Pauli object
            A new Pauli object instance representing the result of the exponentiation.

        Examples
        --------
        >>> ps = PauliString(x_exp, z_exp, dimensions)
        >>> ps_squared = ps ** 2
        """

        tableau = np.mod(self.tableau() * A, np.tile(self.dimensions(), 2))
        return self.__class__(tableau, self.dimensions().copy(), self.weights().copy(), self.phases().copy())

    def __hash__(self) -> int:
        """
        Return the hash value of the Pauli object object. That is a unique identifier.

        Returns
        -------
        int
            The hash value of the Pauli object instance.
        """
        return hash(
            (tuple(self.tableau()),
             tuple(self.weights()),
             tuple(self.phases()),
             tuple(self.dimensions()))
        )

    def __dict__(self) -> dict:
        """
        Returns a dictionary representation of the object's attributes.

        Returns
        -------
        dict
            A dictionary containing the values of `x_exp`, `z_exp`, `weights`, `phases` and `dimensions`.
        """
        return {'tableau': self.tableau(),
                'dimensions': self.dimensions(),
                'weights': self.weights(),
                'phases': self.phases()}

    def copy(self) -> Self:
        """
        Creates a copy of the Pauli object.

        Returns
        -------
        Pauli object
            A copy of the Pauli object.
        Pauli object
            A copy of the Pauli object.
        """
        return self.__class__(self.tableau().copy(), self.dimensions().copy(),
                              self.weights().copy(), self.phases().copy())

    def phase_to_weight(self):
        """
        Include the phases into the weights of the Pauli object.
        This method modifies the weights of the Pauli object by multiplying them with the phases,
        and reset the phases to all zeros.
        """
        new_weights = np.zeros(self.n_paulis(), dtype=np.complex128)
        for i in range(self.n_paulis()):
            phase = self.phases()[i]
            omega = np.exp(2 * np.pi * 1j * phase / (2 * self.lcm()))
            new_weights[i] = self.weights()[i] * omega
        self._phases = np.zeros(self.n_paulis(), dtype=int)
        self._weights = new_weights

    def to_standard_form(self) -> Self:
        """
        Get the Pauli object in standard form.

        Returns
        -------
        Pauli object
            The Pauli object in standard form.
        """
        ps_out = self.copy()
        ps_out.standardise()  # CHECK: american or british spelling?
        return ps_out

    def standardise(self):
        """
        Standardises the Pauli object object by combining equivalent Paulis and
        adding phase factors to the weights then resetting the phases.
        """
        self.phase_to_weight()
        T = self.tableau()
        W = self.weights()

        # FIXME: Lexicographic sort, I'm not sure why this works, but it does.
        order = np.lexsort(T.T)

        self._tableau = T[order]
        self._weights = W[order]
