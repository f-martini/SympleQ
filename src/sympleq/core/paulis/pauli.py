from __future__ import annotations
import numpy as np
from typing import Any

from .pauli_sum import PauliSum
from .pauli_string import PauliString
from .pauli_object import PauliObject
from .constants import DEFAULT_QUDIT_DIMENSION


class Pauli(PauliObject):
    f"""
    Constructor for Pauli class. This represent a single Pauli operator acting on a quDit in symplectic form.
    For more details, see the references:
    `Phys. Rev. A 71, 042315 (2005) <https://doi.org/10.1103/PhysRevA.71.042315>`_
    and
    `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_

   Parameters
    ----------
    tableau : np.ndarray
        Symplectic tableau representation of the object. A 1-D array of length 2 (x_exp, z_exp).
    dimensions : int | list[int] | np.ndarray | None, optional
        Qudit dimension(s).
        - If int, a single qudit dimension is assumed or broadcast where appropriate.
        - If list/np.ndarray, must match the number of qudits implied by `tableau`.
        - If None, all dimensions are defaulted to {DEFAULT_QUDIT_DIMENSION}.
    weights : list | np.ndarray | None, optional
        For a Pauli, this is currently unused, but it is kept to conmform to the PauliObject interface.
    phases : list[int] | np.ndarray | None, optional
        For a Pauli, this is currently unused, but it is kept to conmform to the PauliObject interface.
    """

    def __init__(self, tableau: np.ndarray, dimensions: int | list[int] | np.ndarray | None = None,
                 _weights: list[int] | np.ndarray | None = None, _phases: list[int] | np.ndarray | None = None):

        if isinstance(dimensions, (int, list)):
            dimensions = np.asarray(dimensions, dtype=int)
        elif dimensions is None:
            dimensions = np.asarray(DEFAULT_QUDIT_DIMENSION, dtype=int)

        self._dimensions = dimensions

        # TODO: should we silently take the modulo for the tableau or rise an error?
        self._tableau = tableau % self._dimensions

    @classmethod
    def from_exponents(cls, x_exp: int | None = None, z_exp: int | None = None,
                       dimension: int = DEFAULT_QUDIT_DIMENSION) -> Pauli:
        f"""
        Create a Pauli object from given x and z exponents.

        Args:
            x_exp : int | None
                Exponent of X part of Pauli in symplectic form. If None, this is set to 0.
            z_exp : int | None
                Exponent of Z part of Pauli in symplectic form. If None, this is set to 0.
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class constructed from the given exponents.
        """

        if x_exp is None:
            x_exp = 0
        if z_exp is None:
            z_exp = 0

        tableau = np.asarray([x_exp, z_exp], dtype=int) % dimension
        P = cls(tableau, dimensions=dimension)
        P._sanity_check()

        return P

    @classmethod
    def from_string(cls, pauli_str: str, dimension: int = DEFAULT_QUDIT_DIMENSION) -> Pauli:
        f"""
        Create a Pauli object from a string representation.

        Args:
            pauli_str (str): String representation of the Pauli operator,
                expected to contain the x and z exponents at specific positions, e.g. "x1z0".
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class constructed from the given string.

        Raises:
            ValueError: If the input string does not have the expected format or cannot be parsed.
        """
        tableau = np.empty(2, dtype=int)
        try:
            tableau[0] = int(pauli_str[1])
            tableau[1] = int(pauli_str[3])
        except Exception as e:
            raise ValueError(f"Could not format tableau from input string: {e}.")

        P = cls(tableau, dimensions=dimension)
        P._sanity_check()

        return P

    @classmethod
    def Xnd(cls, x_exp: int, dimension: int):
        """
        Create a Pauli X operator.

        Args:
            x_exp : int | None
                Exponent of X part of Pauli in symplectic form. If None, this is set to 0.
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class constructed from the given x exponent.
        """
        cls.from_exponents(x_exp, 0, dimension)

    @classmethod
    def Ynd(cls, y_exp: int, dimension: int):
        """
        Create a Pauli Y operator. This is represented as a Pauli operator with the same X and Z powers.

        Args:
            y_exp : int | None
                Exponent for both X and Z part of Pauli in symplectic form. If None, this is set to 0.
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class constructed from the given y exponent.
        """
        cls.from_exponents(y_exp, y_exp, dimension)

    @classmethod
    def Znd(cls, z_exp: int, dimension: int):
        """
        Create a Pauli Z operator.

        Args:
            z_exp : int | None
                Exponent of Z part of Pauli in symplectic form. If None, this is set to 0.
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class constructed from the given z exponent.
        """
        cls.from_exponents(z_exp, 0, dimension)

    @classmethod
    def Idnd(cls, dimension: int):
        """
        Create an n-dimensional Identity operator.

        Args:
            dimension (int, optional): The dimension of the Pauli operator.
                Defaults to {DEFAULT_QUDIT_DIMENSION}.

        Returns:
            Pauli: An instance of the Pauli class with 0 exponents.
        """
        cls.from_exponents(0, 0, dimension)

    def tableau(self) -> np.ndarray:
        """
        Returns the tableau representation of the Pauli.
        The tableau representation is a vector of length 2,
        where the first entry correspond to the X exponent and the
        second entry correspond to the Z exponent of the Pauli.

        Returns
        -------
        np.ndarray
            A 1D numpy array of length 2 representing the tableau
            form of the Pauli.
        """
        return self._tableau

    def dimensions(self) -> np.ndarray:
        """
        Returns the dimensions of the Pauli.

        Returns
        -------
        np.ndarray
            A 1D numpy array of length 1.
        """
        return self._dimensions

    def lcm(self) -> int:
        """
        Returns the least common multiplier of the dimensions of the Pauli,
        i.e. just the dimensions of the Pauli.

        Returns
        -------
        int
            The Pauli least common multiplier as integer.
        """
        return self.dimensions()[0]

    def n_qudits(self) -> int:
        """
        Returns the number of qudits represented by the Pauli operator (always 1).

        Returns
        -------
        int
            The number of qudits.
        """
        return 1

    def phases(self) -> np.ndarray:
        """
        Returns the phases associated with the Pauli object.
        For a Pauli operator, this is just the trivial phase.

        Returns
        -------
        np.ndarray
            The phases as a 1d-vector.
        """
        return np.asarray(0, dtype=int)

    def weights(self) -> np.ndarray:
        """
        Returns the weights associated with the Pauli object.
        For a Pauli operator, this is just 1.

        Returns
        -------
        np.ndarray
            The weights as a 1d-vector.
        """
        return np.asarray(1, dtype=complex)

    def to_pauli_sum(self) -> PauliSum:
        """
        Converts the Pauli to a PauliSum.

        Returns
        -------
        PauliSum
            A PauliSum instance representing the given Pauli operator.
        """
        return PauliSum(self.tableau(), self.dimensions(), self.weights(), self.phases())

    def to_pauli_string(self) -> PauliString:
        """
        Converts the Pauli to a PauliString.

        Returns
        -------
        PauliString
            A PauliString instance representing the given Pauli operator.
        """
        return PauliString(self.tableau(), self.dimensions(), self.weights(), self.phases())

    def copy(self) -> Pauli:
        """
        Creates a copy of the current Pauli operator.
        """
        return Pauli(tableau=self.tableau(), dimensions=self.dimensions())

    def __mul__(self, A: str | Pauli) -> Pauli:
        """
        Multiplies the current Pauli operator with another Pauli operator or a string representation.

        Args:
            A (str | Pauli): The Pauli operator or its string representation to multiply with.

        Returns:
            Pauli: The resulting Pauli operator after multiplication.

        Raises:
        Exception: If the multiplication cannot be performed (e.g., incompatible dimensions).
        """
        if isinstance(A, str):
            return self * Pauli.from_string(A)

        if not isinstance(A, Pauli):
            raise Exception(f"Cannot multiply Pauli with type {type(A)}")

        if not np.array_equal(self.dimensions(), A.dimensions()):
            raise Exception("To multiply two Paulis, their dimensions"
                            f" {A.dimensions()} and {self.dimensions()} must be equal")

        new_tableau = (self.tableau() + A.tableau()) % self.lcm()

        return Pauli(new_tableau, dimensions=self.dimensions())

    def __pow__(self, power: int) -> Pauli:
        """
        Raises the Pauli operator to a given power.

        Args:
            power (int): The power to raise the Pauli operator to.

        Returns:
            Pauli: The resulting Pauli operator after exponentiation.

        Raises:
            TypeError: If the power is not an integer.
            ValueError: If the power is negative.
        """
        if not isinstance(power, int):
            raise TypeError("Power must be an integer.")

        new_tableau = (self.tableau() * power) % self.lcm()
        return Pauli(new_tableau, dimensions=self.dimensions())

    def __str__(self) -> str:
        """
        Returns a string representation of the Pauli operator in the form 'x{self.x_exp}z{self.z_exp}'.
        """
        return f'x{self.x_exp}z{self.z_exp}'

    def __eq__(self, other_pauli: Any) -> bool:
        """
        Checks if two Pauli operators are equal.

        Args:
            other_pauli (Any): The other Pauli operator to compare with.

        Returns:
            bool: True if the Pauli operators are equal, False otherwise.
        """

        # FIXME: should we allow comparison between any PauliObject as long as the dimensions match?
        # We could simply remove this check and it would all flow automatically.
        if not isinstance(other_pauli, Pauli):
            return False

        return np.array_equal(self.tableau(), other_pauli.tableau()) and \
            np.array_equal(self.dimensions(), other_pauli.dimensions())

    def __ne__(self, other_pauli: Any) -> bool:
        """
        Checks if two Pauli operators are not equal.

        Args:
            other_pauli (Any): The other Pauli operator to compare with.

        Returns:
            bool: True if the Pauli operators are not equal, False otherwise.
        """
        return not self.__eq__(other_pauli)

    def __dict__(self) -> dict:
        """
        Returns a dictionary representation of the Pauli operator,
        in the form {'x_exp': ..., 'z_exp': ..., 'dimension': ...}.
        """
        return {'x_exp': self.x_exp, 'z_exp': self.z_exp, 'dimension': self.lcm()}

    def __gt__(self, other_pauli: Pauli) -> bool:
        """
        Compares two Pauli operators based on their x and z exponents.
        """
        d = self.lcm()
        # TODO: Ask @charlie why we are including "d-*_exp" in the comparison
        x_measure = min(self.x_exp % d, (d - self.x_exp) % d)
        x_measure_new = min(other_pauli.x_exp % d, (d - other_pauli.x_exp) % d)
        z_measure = min(self.z_exp % d, (d - self.z_exp) % d)
        z_measure_new = min(other_pauli.z_exp % d, (d - other_pauli.z_exp) % d)

        if x_measure > x_measure_new:
            return True
        if x_measure == x_measure_new:
            if z_measure > z_measure_new:
                return True
            if z_measure == z_measure_new:
                return False

        return False

    @property
    def x_exp(self) -> int:
        """
        x_exp : int
        The x exp of the Pauli.
        """
        return self._tableau[0]

    @property
    def z_exp(self) -> int:
        """
        z_exp : int
        The z exp of the Pauli.
        """
        return self._tableau[1]

    def _sanity_check(self):
        """
        Validates the consistency of the Pauli's internal representation.
        """

        if len(self.dimensions()) != 1:
            raise ValueError(f"Dimensions must have length 1 (got {len(self.dimensions())}).")

        if self.dimensions()[0] < DEFAULT_QUDIT_DIMENSION:
            raise ValueError(f"Dimension is less than {DEFAULT_QUDIT_DIMENSION}")

        if self.tableau().shape != (2, ):
            raise ValueError(f"Tableau has invalid shape ({self.tableau().shape}).")
