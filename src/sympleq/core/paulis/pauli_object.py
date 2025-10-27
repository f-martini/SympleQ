from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar

P = TypeVar("P", bound="PauliObject")


class PauliObject(ABC):
    @abstractmethod
    def __init__(self, tableau: np.ndarray, dimensions: int | list[int] | np.ndarray | None = None,
                 weights: int | float | complex | list[int | float | complex] | np.ndarray | None = None,
                 phases: int | list[int] | np.ndarray | None = None):
        """
        Abstract initializer for Pauli-like objects represented in symplectic tableau form.

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

        Notes
        -----
        - This method is abstract: concrete subclasses must implement construction and validation logic.
        - Implementations should validate shapes and consistency between `tableau` and `dimensions` and
          normalize exponents modulo the corresponding qudit dimensions where appropriate.
        - The `weights` and `phases` arguments are optional because single-Pauli/PauliString objects may not use them,
          whereas PauliSum does.
        """
        pass

    @abstractmethod
    def tableau(self) -> np.ndarray:
        """
        Returns the tableau representation of the Pauli-like object.


        Returns
        -------
        np.ndarray
            The tableau representation of the Pauli-like object.
        """
        pass

    @abstractmethod
    def dimensions(self) -> np.ndarray:
        """
        Returns the dimensions of the Pauli-like object.

        Returns
        -------
        np.ndarray
            A 1D numpy array of length n_qudits().
        """
        pass

    @abstractmethod
    def lcm(self) -> int:
        """
        Returns the least common multiplier of the dimensions of the Pauli-like object.

        Returns
        -------
        int
            The Pauli-like object dimensions least common multiplier as integer.
        """
        pass

    @abstractmethod
    def n_qudits(self) -> int:
        """
        Returns the number of qudits represented by the Pauli-like object.

        Returns
        -------
        int
            The number of qudits.
        """
        pass

    @abstractmethod
    def phases(self) -> np.ndarray:
        # FIXME: improve docstring
        """
        Returns the phases associated with the Pauli-like object.
        These phases represent the numerator, the denominator is 2 * self.lcm()

        Returns
        -------
        np.ndarray
            The phases as a 1d-vector.
        """
        pass

    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Returns the weights associated with the Pauli-like object.

        Returns
        -------
        np.ndarray
            The weights as a 1d-vector.
        """
        pass
