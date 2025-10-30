from __future__ import annotations
from typing import Union, overload, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp

from .pauli_object import PauliObject
from .pauli_string import PauliString
from .pauli import Pauli

if TYPE_CHECKING:
    PauliType = Union[Pauli, PauliString, 'PauliSum']
    ScalarType = Union[float, complex, int]
    PauliOrScalarType = Union[PauliType, ScalarType]


class PauliSum(PauliObject):
    @classmethod
    def from_tableau(cls, tableau: np.ndarray, dimensions: int | list[int] | np.ndarray | None = None,
                     weights: int | float | complex | list[int | float | complex] | np.ndarray | None = None,
                     phases: int | list[int] | np.ndarray | None = None
                     ) -> PauliSum:
        """
        Create a PauliSum instance from a tableau.

        Parameters
        ----------
        tableau : np.ndarray
            The tableau to convert into a PauliSum.
        dimensions : list[int] | np.ndarray | int, optional
            The dimensions of each qudit. If an integer is provided,
            all qudits are assumed to have the same dimension.
            If no value is provided, the default is `DEFAULT_QUDIT_DIMENSION`.
        weights: list[int | float | complex] | np.ndarray | None = None
            The weights for each PauliString.
        phases: int | list[int] | np.ndarray | None = None
            The phases of the PauliStrings in the range [0, lcm(dimensions) - 1].

        Returns
        -------
        PauliString
            A PauliString instance initialized with the exponents and dimensions from the input tableau.
        """

        P = cls(tableau, dimensions, weights, phases)
        P._sanity_check()

        return P

    @classmethod
    def from_pauli(cls, pauli: Pauli) -> 'PauliSum':
        """
        Create a PauliSum instance from a single Pauli object.

        Parameters
        ----------
        pauli : Pauli
            The Pauli object to convert into a PauliSum.

        Returns
        -------
        PauliSum
            A PauliSum instance representing the given Pauli operator.
        """
        P = cls(pauli.tableau(), pauli.dimensions(), pauli.weights(), pauli.phases())
        P._sanity_check()

        return P

    @classmethod
    def from_pauli_strings(cls, pauli_string: PauliString | list[PauliString],
                           weights: int | float | complex | list[int | float | complex] | np.ndarray | None = None,
                           phases: int | list[int] | np.ndarray | None = None,) -> PauliSum:
        """
        Create a PauliSum instance from a (list of) PauliString object.

        Parameters
        ----------
        pauli_string : PauliString | list[PauliString]
            The PauliString object(s) to convert into a PauliSum.

        Returns
        -------
        PauliSum
            A PauliSum instance representing the given Pauli operator.
        """
        if isinstance(pauli_string, PauliString):
            pauli_string = [pauli_string]
        elif isinstance(pauli_string, list) and len(pauli_string) == 0:
            raise ValueError("At least one PauliString must be provided.")

        dimensions = pauli_string[0].dimensions()
        if len(pauli_string) > 1:
            for ps in pauli_string[1:]:
                if not np.array_equal(ps.dimensions(), dimensions):
                    raise ValueError("The dimensions of all Pauli strings must be equal.")

        tableau = np.vstack([p.tableau() for p in pauli_string])
        return cls(tableau, dimensions, weights, phases)

    @classmethod
    def from_string(cls, pauli_str: str | list[str], dimensions: int | list[int] | np.ndarray,
                    weights: int | float | complex | list[int | float | complex] | np.ndarray | None = None,
                    phases: int | list[int] | np.ndarray | None = None
                    ) -> PauliSum:
        """
        Create a PauliSum instance from a string representation.

        Parameters
        ----------
        pauli_str : str | list[str]
            The string representation of the Pauli string, where exponents are separated by 'x' and 'z'.
        dimensions : list[int] | np.ndarray
            The dimensions parameter to be passed to the PauliSum constructor.

        Returns
        -------
        PauliSum
            An instance of PauliSum initialized with the exponents and dimensions parsed from the input string.

        Examples
        --------
        >>> PauliSum.from_string("x2z3 x4z1 x0z0", [4, 5, 2]])
        <PauliSum ...>
        """

        if isinstance(pauli_str, str):
            pauli_str = [pauli_str]

        pauli_strings = [PauliString.from_string(s, dimensions) for s in pauli_str]
        return cls.from_pauli_strings(pauli_strings, weights, phases)

    @classmethod
    def from_random(cls,
                    n_paulis: int,
                    dimensions: int | list[int] | np.ndarray,
                    rand_weights: bool = True,
                    seed: int | None = None) -> 'PauliSum':
        """
        Create a random PauliSum object.

        Parameters
        ----------
        n_pauli : int
            The number of Pauli operators to include in the sum.
        n_qudits : int
            The number of qudits in each Pauli operator.
        dimensions : list[int] | np.ndarray
            The dimensions of the qudits.
        rand_weights : bool
            Whether to use random weights for the Pauli operators.

        Returns
        -------
        PauliSum
            A PauliSum object.
        """
        # TODO: Eliminate n_qudits and set dimensions directly from len(dimensions)
        if seed is not None:
            np.random.seed(seed)
        weights = 2 * (np.random.rand(n_paulis) - 0.5) if rand_weights else np.ones(n_paulis)
        string_seeds = np.random.randint(1000000, size=1000)
        # ensure no duplicate strings
        strings = []
        for i in range(n_paulis):
            ps = PauliString.from_random(dimensions, seed=string_seeds[i])
            j = 0
            while ps in strings:
                j += 1
                ps = PauliString.from_random(dimensions, seed=string_seeds[j])
            strings.append(ps)

        return cls.from_pauli_strings(strings, weights=weights, phases=[0] * n_paulis).to_standard_form()

    def phases(self) -> np.ndarray:
        """
        Returns the phases associated with the Pauli-like object.
        These phases represent the numerator, the denominator is 2 * self.lcm()

        Returns
        -------
        np.ndarray
            The phases as a 1d-vector.
        """
        return self._phases

    def set_phases(self, new_phases: list[int] | np.ndarray):
        if isinstance(new_phases, list):
            new_phases = np.asarray(new_phases, dtype=int)

        if len(new_phases) != self.n_paulis():
            raise ValueError(
                f"New phases ({len(new_phases)}) length must equal the number of Pauli strings ({self.n_paulis()}.")

        self._phases = new_phases

    def weights(self) -> np.ndarray:
        """
        Returns the weights associated with the Pauli-like object.

        Returns
        -------
        np.ndarray
            The weights as a 1d-vector.
        """
        return self._weights

    def set_weights(self, new_weights: list[int] | np.ndarray):
        if isinstance(new_weights, list):
            new_weights = np.asarray(new_weights, dtype=int)

        if len(new_weights) != self.n_paulis():
            raise ValueError(
                f"New phases ({len(new_weights)}) length must equal the number of Pauli strings ({self.n_paulis()}.")

        self._weights = new_weights

    def phase_to_weight(self):
        """
        Include the phases into the weights of the PauliSum.
        This method modifies the weights of the PauliSum by multiplying them with the phases,
        and reset the phases to all zeros.
        """
        new_weights = np.zeros(self.n_paulis(), dtype=np.complex128)
        for i in range(self.n_paulis()):
            phase = self.phases()[i]
            omega = np.exp(2 * np.pi * 1j * phase / (2 * self.lcm()))
            new_weights[i] = self.weights()[i] * omega
        self._phases = np.zeros(self.n_paulis(), dtype=int)
        self._weights = new_weights

    def to_standard_form(self) -> PauliSum:
        """
        Get the PauliSum in standard form.

        Returns
        -------
        PauliSum
            The PauliSum in standard form.
        """
        ps_out = self.copy()
        ps_out.standardise()
        return ps_out

    @overload
    def __getitem__(self,
                    key: tuple[int, int]) -> Pauli:
        ...

    @overload
    def __getitem__(self,
                    key: int | tuple[int, slice | list[int] | np.ndarray]) -> PauliString:
        ...

    @overload
    def __getitem__(self,
                    key: slice | np.ndarray | list[int] | tuple[slice, int] | tuple[slice, slice] |
                    tuple[slice, int] |
                    tuple[slice, list[int]] | tuple[slice, np.ndarray] | tuple[list[int], int] |
                    tuple[np.ndarray, int] | tuple[np.ndarray, slice] | tuple[np.ndarray, list[int]] |
                    tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, list[int]] | tuple[list[int], list[int]] |
                    tuple[list[int], np.ndarray]) -> PauliSum:
        ...

    # FIXME: add typing
    def __getitem__(self, key) -> Pauli | PauliString | PauliSum:
        """
        Retrieve a Pauli,  PauliString, or (smaller) PauliSum from the PauliSum.

        Parameters
        ----------
        key : int | slice | np.ndarray | list[int]
            The index or indices specifying which Pauli(s) to retrieve. If an int, returns a single Pauli.
            If a slice, numpy array, or list, returns a new PauliString containing the selected Paulis.

        Returns
        -------
        PauliString or Pauli
            The selected Pauli operator(s). Returns a single Pauli if `key` is an int, otherwise returns a PauliString.

        Raises
        ------
        ValueError
            If `key` is not an int, slice, numpy.ndarray, or list.

        Examples
        --------
        >>> ps = PauliString(...)
        >>> ps[0]  # Returns a single Pauli
        >>> ps[1:3]  # Returns a PauliString with selected Paulis
        >>> ps[[0, 2]]  # Returns a PauliString with Paulis at indices 0 and 2
        """
        # TODO: tidy
        if isinstance(key, int):
            # Here we don't copy and return a view.
            tableau = self.tableau()[key]
            return PauliString(tableau, self.dimensions())

        if isinstance(key, (list, np.ndarray, slice)):
            return PauliSum(self.tableau()[key], self.dimensions(), self.weights()[key], self.phases()[key])

        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must be of length 2")

            pauli_indices, qudit_indices = key

            # Single PauliString
            if isinstance(pauli_indices, int):
                if isinstance(qudit_indices, int):
                    # Single Pauli
                    return Pauli(self.tableau()[pauli_indices][qudit_indices],
                                 self.dimensions()[qudit_indices])

                # Sub-PauliString
                if isinstance(qudit_indices, (list, np.ndarray, slice)):
                    sub_tableau = self.tableau()[pauli_indices, :][qudit_indices]
                    sub_dims = self.dimensions()[qudit_indices]
                    return PauliString(sub_tableau, sub_dims)

            return self.get_subspace(qudit_indices, pauli_indices)

        raise TypeError(f"Key must be int or slice, not {type(key)}")

    @overload
    def __setitem__(self,
                    key: tuple[int, int],
                    value: 'Pauli'):
        ...

    @overload
    def __setitem__(self,
                    key: int | slice | tuple[int, slice],
                    value: 'PauliString'):
        ...

    @overload
    def __setitem__(self,
                    key: tuple[slice, int] | tuple[slice, slice],
                    value: 'PauliSum'):
        ...

    def __setitem__(self, key, value):
        """
        Change PauliString within PauliSum at position `key`. It takes the PauliString
        identified by `key` and substitutes it with the PauliString `value`.

        Parameters
        ----------
        key : int | slice | tuple
            The key identifying the PauliString operator to change.
        value : PauliString | PauliSum
            The value to set the Pauli operator to.

        Raises
        -------
        ValueError
            If the key is not an int, slice, or tuple of length 2.
        """
        # TODO: Error messages here could be improved
        # FIXME: we should check that the dimensions are compatible.
        if isinstance(key, int):  # key indexes the pauli_string to be replaced by value
            if isinstance(value, PauliString):
                self._tableau[key, :] = value.tableau()[0]  # Update only the affected row in the tableau

        elif isinstance(key, slice):
            for i, v in zip(range(*key.indices(self.n_paulis())), value):
                if isinstance(value, PauliString):
                    self._tableau[i, :] = v.tableau()[0]
        elif isinstance(key, tuple):
            self._tableau[key] = value
            # TODO: if the previous line works, just remove this commented line and the function overrides
            # self._setitem_tuple(key, value)

    def __add__(self,
                A: PauliType) -> PauliSum:
        """
        Implements the addition of PauliSum objects.

        Parameters
        ----------
        A : PauliType
            The Pauli operator to add.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the sum of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> p2 = PauliSum.from_pauli_strings("x2z1 x1z1", [3, 2])
        >>> p1 + p2
        PauliSum(...)

        Raises
        ------
        ValueError
            If the dimensions of `self` and `A` do not match.

        Notes
        -----
        - Dimensions must agree!
        """

        if not np.array_equal(self.dimensions(), A.dimensions()):
            raise ValueError(f"The dimensions of the PauliSums do not match ({self.dimensions()}, {A.dimensions()}).")

        if isinstance(A, Pauli):
            A_sum = PauliSum(A.tableau(), A.dimensions())
        elif isinstance(A, PauliString):
            A_sum = PauliSum(A.tableau(), A.dimensions())
        elif isinstance(A, PauliSum):
            A_sum = A
        else:
            raise ValueError(f"Cannot add Pauli with type {type(A)}")

        new_tableau = np.vstack([self.tableau(), A_sum.tableau()])
        new_weights = np.concatenate([self.weights(), A_sum.weights()])
        new_phases = np.concatenate([self.phases(), A_sum.phases()])
        return PauliSum(new_tableau, self.dimensions(), new_weights, new_phases)

    def __radd__(self,
                 A: PauliType) -> 'PauliSum':
        """
        Implements the addition of PauliSum objects.

        Parameters
        ----------
        A : PauliType
            The Pauli operator to add.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the sum of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> p2 = PauliSum.from_pauli_strings("x2z1 x1z1", [3, 2])
        >>> p1 + p2
        PauliSum(...)

        Raises
        ------
        ValueError
            If the dimensions of `self` and `A` do not match.

        Notes
        -----
        - Dimensions must agree!
        """

        # FIXME: avoid copying and use tableaus
        # TODO: why is this not good enough?
        return self + A

    def __sub__(self,
                A: PauliSum) -> PauliSum:
        """
        Implements the subtraction of PauliSum objects.

        Parameters
        ----------
        A : PauliType
            The Pauli operator to subtract.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the difference of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> p2 = PauliSum.from_pauli_strings("x2z1 x1z1", [3, 2])
        >>> p1 - p2
        PauliSum(...)

        Raises
        ------
        ValueError
            If the dimensions of `self` and `A` do not match.

        Notes
        -----
        - Dimensions must agree!
        """
        # FIXME: why is this treated differently from __add__?
        new_tableau = np.vstack([self.tableau(), A.tableau()])
        new_weights = np.concatenate([self.weights(), -np.array(A.weights())])
        new_phases = np.concatenate([self.phases(), A.phases()])
        return PauliSum(new_tableau, self.dimensions(), new_weights, new_phases)

    def __rsub__(self,
                 A: PauliType) -> 'PauliSum':
        """
        Implements the subtraction of PauliSum objects.

        Parameters
        ----------
        A : PauliType
            The Pauli operator to subtract.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the difference of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> p2 = PauliSum.from_pauli_strings("x2z1 x1z1", [3, 2])
        >>> p1 - p2
        PauliSum(...)

        Raises
        ------
        ValueError
            If the dimensions of `self` and `A` do not match.

        Notes
        -----
        - Dimensions must agree!
        """
        ps1 = self.copy()
        if isinstance(A, Pauli):
            ps2 = PauliSum.from_pauli_strings(PauliString.from_pauli(A))
        elif isinstance(A, PauliString):
            ps2 = PauliSum.from_pauli_strings(A)
        elif isinstance(A, PauliSum):
            ps2 = A
        else:
            raise Exception(f"Cannot add Pauli with type {type(A)}")
        return ps1 - ps2

    def __matmul__(self,
                   A: PauliType) -> PauliSum:
        """
        Implements the tensor product between a PauliSum and a PauliType objects.
        It corresponds to operator tensor product (`@`).
        The resulting PauliSum has the exponents of both strings concatenated.

        Parameters
        ----------
        A : PauliType Pauli | PauliString | PauliSum
            The PauliType instance to be tensored with `self`.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the tensor product of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliSum.from_pauli_strings("x1z0 x0z1", [2, 2])
        >>> p2 = PauliSum.from_pauli_strings("x2z1", [3])
        >>> p1 @ p2
        PauliSum(...)

        Notes
        -----
        - This is NOT a product between two PauliSum objects!
        """
        if isinstance(A, PauliString):
            A = PauliSum.from_pauli_strings(A)
        elif isinstance(A, Pauli):
            A = PauliSum.from_pauli(A)

        new_dimensions = np.concatenate((self.dimensions(), A.dimensions()))
        new_lcm = np.lcm.reduce(new_dimensions)

        n1, n2 = self.n_qudits(), A.n_qudits()
        p1, p2 = self.n_paulis(), A.n_paulis()

        # Combined dimensions
        new_dimensions = np.concatenate((self.dimensions(), A.dimensions()))
        new_lcm = np.lcm.reduce(new_dimensions)

        # Allocate tableau: (p1*p2) rows, 2*(n1+n2) columns
        new_tableau = np.empty((p1 * p2, 2 * (n1 + n2)), dtype=int)
        new_weights = np.empty(p1 * p2, dtype=complex)
        new_phases = np.empty(p1 * p2, dtype=int)

        for i in range(p1):
            for j in range(p2):
                idx = i * p2 + j

                t1 = self.tableau()[i]
                t2 = A.tableau()[j]

                # NOTE: This mirrors PauliString.__matmul__
                new_tableau[idx, :n1] = t1[:n1]           # self.x_exp
                new_tableau[idx, n1:n1 + n2] = t2[:n2]    # A.x_exp
                new_tableau[idx, n1 + n2:n1 + n2 + n1] = t1[n1:]  # self.z_exp
                new_tableau[idx, n1 + n2 + n1:] = t2[n2:]         # A.z_exp

                new_weights[idx] = self.weights()[i] * A.weights()[j]
                new_phases[idx] = (self.phases()[i] + A.phases()[j]) % (2 * new_lcm)

        return PauliSum(new_tableau, new_dimensions, new_weights, new_phases)

    def __mul__(self,
                A: PauliOrScalarType) -> PauliSum:
        """
        Multiply a PauliSum and a PauliType (or scalar) objects element-wise.
        It corresponds to operator multiplication (`*`).
        It adds the tableaus of the two PauliSums modulo their dimensions.

        Parameters
        ----------
        A : PauliType Pauli | PauliString | PauliSum | float | int
            The PauliType or scalar instance to be multiplied with `self`.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the product of `self` and `A`.

        Raises
        ------
        ValueError
            If `A` is not an instance of a Pauli, PauliSum, PauliString, or scalar.

        Examples
        --------
        >>> ps1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> ps2 = PauliSum.from_pauli_strings("x2z1 x0z0", [3, 2])
        >>> ps3 = ps1 * ps2
        PauliSum(...)
        """
        if isinstance(A, (int, float)):
            return PauliSum(self.tableau(), self.dimensions(), self.weights() * A, self.phases())

        if isinstance(A, PauliString):
            return self * PauliSum.from_pauli_strings(A)

        if not isinstance(A, PauliSum):
            raise ValueError("Multiplication only supported with Pauli, PauliSum, PauliString, or scalar")

        # TODO: check if this check is necessary
        if not np.array_equal(self.dimensions(), A.dimensions()):
            raise ValueError(f"The dimensions of the PauliSums do not match ({self.dimensions()}, {A.dimensions()}).")

        w1 = self.weights()[:, None]
        w2 = A.weights()[None, :]
        new_weights = (w1 * w2).reshape(-1)

        p1 = self.phases()[:, None]
        p2 = A.phases()[None, :]

        # Extract z- and x-parts from tableau
        n1, n2 = self.n_qudits(), A.n_qudits()
        a_z = self.tableau()[:, n1:]
        b_x = A.tableau()[:, :n2]

        # Compute acquired phases via symplectic form
        factors = (self.lcm() // self.dimensions())
        acquired_phases = 2 * factors * a_z  @ b_x.T

        # Combine with existing phases and flatten
        new_phases = (p1 + p2 + acquired_phases) % (2 * self.lcm())
        new_phases = new_phases.reshape(-1)

        # Multiplication between PauliString corresponds to summing the tableaus
        new_tableau = np.asarray([ps1 + ps2 for ps1 in self.tableau() for ps2 in A.tableau()], dtype=int)

        return PauliSum(new_tableau, self.dimensions(), new_weights, new_phases)

    def __rmul__(self,
                 A: PauliOrScalarType) -> 'PauliSum':
        """
        Multiply a PauliSum and a PauliType (or scalar) objects element-wise.
        It corresponds to operator multiplication (`*`).
        It adds the tableaus of the two PauliSums modulo their dimensions.

        Parameters
        ----------
        A : PauliType Pauli | PauliString | PauliSum | float | int
            The PauliType or scalar instance to be multiplied with `self`.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the product of `self` and `A`.

        Raises
        ------
        ValueError
            If `A` is not an instance of PauliString or a scalar.

        Examples
        --------
        >>> ps1 = PauliSum.from_pauli_strings("x1z0 x0z1", [3, 2])
        >>> ps2 = PauliSum.from_pauli_strings("x2z1 x0z0", [3, 2])
        >>> ps3 = ps1 * ps2
        PauliSum(...)
        """
        if isinstance(A, (Pauli, PauliString, PauliSum, float, int, complex)):
            return self * A
        else:
            raise ValueError(f"Cannot multiply PauliString with type {type(A)}")

    def __truediv__(self,
                    A: PauliType) -> 'PauliSum':
        """
        Divide a PauliSum by a scalar. It corresponds to operator division (`/`).

        Parameters
        ----------
        A : float | int
            The scalar instance to be divided with `self`.

        Returns
        -------
        PauliSum
            A new PauliSum instance representing the quotient of `self` and `A`.

        Raises
        ------
        ValueError
            If `A` is not a scalar.
        """
        if not isinstance(A, (int, float)):
            raise ValueError("Division only supported with scalar")
        return self * (1 / A)

    def __eq__(self,
               other_pauli: 'PauliSum') -> bool:
        """
        Determine if two PauliSum objects are equal.

        Parameters
        ----------
        other_pauli : PauliSum
            The PauliSum instance to compare against.

        Returns
        -------
        bool
            True if both PauliSum instances have identical PauliStrings, weights, phases, and dimensions;
            False otherwise.
        """
        if not isinstance(other_pauli, PauliSum):
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

    def __ne__(self,
               other_pauli: 'PauliSum') -> bool:
        """
        Determine if two PauliSum objects are different.

        Parameters
        ----------
        other_pauli : PauliSum
            The PauliSum instance to compare against.

        Returns
        -------
        bool
            True if the PauliSum instances do not have identical PauliStrings, weights, phases, and dimensions;
            False otherwise.
        """
        return not self == other_pauli

    def __hash__(self) -> int:
        """
        Return the hash value of the PauliSum object. That is a unique identifier.

        Returns
        -------
        int
            The hash value of the PauliSum instance.
        """
        return hash(
            (tuple(self.tableau()),
             tuple(self.weights()),
             tuple(self.phases()),
             tuple(self.dimensions()))
        )

    def __dict__(self) -> dict:
        """`
        Returns a dictionary representation of the object's attributes.

        Returns
        -------
        dict
            A dictionary containing the values of `x_exp`, `z_exp`, `weights`, `phases` and `dimensions`.
        """
        return {'tableau': self.tableau(),
                'dimensions': self.dimensions,
                'weights': self.weights,
                'phases': self.phases}

    @property
    def x_exp(self) -> np.ndarray:
        """
        x_exp : np.ndarray
        Array of X exponents for each qudit.
        """
        return self._tableau[:, :self.n_qudits()]

    @property
    def z_exp(self) -> np.ndarray:
        """
        z_exp : np.ndarray
        Array of Z exponents for each qudit.
        """
        return self._tableau[:, self.n_qudits():]

    def standardise(self):
        """
        Standardises the PauliSum object by combining equivalent Paulis and
        adding phase factors to the weights then resetting the phases.
        """
        self.phase_to_weight()
        T = self.tableau()
        W = self.weights()

        # FIXME: Lexicographic sort, I'm not sure why this works, but it does.
        order = np.lexsort(T.T)

        self._tableau = T[order]
        self._weights = W[order]

    def combine_equivalent_paulis(self):
        """
        Combines equivalent Pauli operators in the sum by summing their coefficients and deleting duplicates.
        """
        self.standardise()  # makes sure all phases are 0
        # combine equivalent Paulis
        to_delete = []
        for i in reversed(range(self.n_paulis())):
            ps1 = self.select_pauli_string(i)
            for j in range(i + 1, self.n_paulis()):
                ps2 = self.select_pauli_string(j)
                if ps1 == ps2:
                    self._weights[i] = self._weights[i] + self._weights[j]
                    to_delete.append(j)
        self._delete_paulis(to_delete)

        # remove zero weight Paulis
        to_delete = []
        for i in range(self.n_paulis()):
            if self._weights[i] == 0:
                to_delete.append(i)
        self._delete_paulis(to_delete)

    def remove_trivial_paulis(self):
        """
        Removes trivial Pauli strings (those that are identity operators) from the sum.
        """
        # If entire Pauli string is x0z0, remove it
        to_delete = np.where(~self.tableau().any(axis=1))[0]
        self._delete_paulis(to_delete)

    def remove_trivial_qudits(self):
        """
        Removes trivial qudits (those that are identity operators) from the sum.
        """
        # If entire qudit is I, remove it
        to_delete = []
        for i in range(self.n_qudits()):
            x_col = self.tableau()[:, i]
            z_col = self.tableau()[:, self.n_qudits() + i]
            if np.all(x_col == 0) and np.all(z_col == 0):
                to_delete.append(i)
        self._delete_qudits(to_delete)

    def remove_zero_weight_paulis(self):
        """
        Removes zero weight Pauli strings from the sum.
        """
        to_delete = []
        for i in range(self.n_paulis()):
            # FIXME: move the magic number to a constant
            if np.abs(self.weights()[i]) <= 1e-14:
                to_delete.append(i)
        self._delete_paulis(to_delete)

    def is_x(self) -> bool:
        """
        Checks whether the PauliSum has only (i.e., all PauliStrings therein) X components.

        Returns
        -------
        bool
            True if the PauliSum has only X components, False otherwise.
        """
        return not np.any(self.z_exp)

    def is_z(self) -> bool:
        """
        Checks whether the PauliSum has only (i.e., all PauliStrings therein) Z components.

        Returns
        -------
        bool
            True if the PauliSum has only Z components, False otherwise.
        """
        return not np.any(self.x_exp)

    def is_commuting(self,
                     pauli_string_indexes: list[int] | None = None) -> bool:
        """
        Checks whether the PauliStrings elements identified by `pauli_string_indexes` are pairwise commuting.

        Parameters
        ----------
        pauli_string_indexes : list[int] | None
            The indices of the PauliStrings to check for commutativity. If None, checks all PauliStrings.

        Returns
        -------
        bool
            True if the PauliSum has pairwise commuting PauliStrings, False otherwise.
        """
        spm = self.symplectic_product_matrix()
        if pauli_string_indexes is None:
            return not np.any(spm)
        else:
            i, j = pauli_string_indexes[0], pauli_string_indexes[1]
            return not spm[i, j]

    def is_quditwise_commuting(self,
                               pauli_string_indexes: list[int] | None = None) -> bool:
        """
        Checks whether the PauliStrings elements identified by `pauli_string_indexes` are pairwise
        qudit-wise commuting.

        Parameters
        ----------
        pauli_string_indexes : list[int] | None
            The indices of the PauliStrings to check for commutativity. If None, checks all PauliStrings.

        Returns
        -------
        bool
            True if the PauliSum has pairwise commuting PauliStrings, False otherwise.
        """
        spm = self.quditwise_symplectic_product_matrix()
        if pauli_string_indexes is None:
            return not np.any(spm)
        else:
            i, j = pauli_string_indexes[0], pauli_string_indexes[1]
            return not spm[i, j]

    def select_pauli_string(self, index: int) -> PauliString:
        """
        Selects a PauliString from the PauliSum.

        Parameters
        ----------
        pauli_index : int
            The index of the PauliString to select.

        Returns
        -------
        PauliString
            The selected PauliString.
        """
        # FIXME: We pass a view to the tableau row and the dimensions,
        # meaning that they could be modified from the PauliString.
        return PauliString(self.tableau()[index], self.dimensions())

    def select_pauli(self, index: tuple[int, int]) -> Pauli:
        """
        Selects a Pauli from the PauliSum.

        Parameters
        ----------
        pauli_index : (int, int)
            The indices of the Pauli to select.

        Returns
        -------
        Pauli
            The selected Pauli.
        """
        # FIXME: We pass a view to the tableau row and the dimensions,
        # meaning that they could be modified from the PauliString.
        return Pauli(self.tableau()[index], self.dimensions())

    def _delete_paulis(self, pauli_indices: int | list[int] | np.ndarray):
        """
        Deletes PauliStrings from the PauliSum.

        Parameters
        ----------
        pauli_indices : list[int] | int
            The indices of the PauliStrings to delete.
        """
        if isinstance(pauli_indices, int):
            pauli_indices = [pauli_indices]

        self._weights = np.delete(self._weights, pauli_indices)
        self._phases = np.delete(self._phases, pauli_indices)
        self._tableau = np.delete(self._tableau, pauli_indices, axis=0)

    def _delete_qudits(self, qudit_indices: list[int] | int):
        """
        Deletes qudits from the PauliSum.

        Parameters
        ----------
        qudit_indices : list[int] | int
            The indices of the qudits to delete.
        """
        if isinstance(qudit_indices, int):
            qudit_indices = [qudit_indices]

        mask = np.ones(self.n_qudits(), dtype=bool)
        mask[qudit_indices] = False

        # Note: we first delete the rightmost indecies, so they are not shifted.
        self._tableau = np.delete(self._tableau, [idx + self.n_qudits() for idx in qudit_indices], axis=1)
        self._tableau = np.delete(self._tableau, qudit_indices, axis=1)

        self._dimensions = self._dimensions[mask]
        self._lcm = int(np.lcm.reduce(self._dimensions))

    def copy(self) -> 'PauliSum':
        """
        Creates a copy of the PauliSum.

        Returns
        -------
        PauliSum
            A copy of the PauliSum.
        PauliSum
            A copy of the PauliSum.
        """
        return PauliSum(self.tableau().copy(), self.dimensions().copy(), self.weights().copy(), self.phases().copy())

    def symplectic_product_matrix(self) -> np.ndarray:
        """
        Scalar (phase-preserving) symplectic product matrix S (n x n), entries mod LCM(dimensions).
        S[i, j] = sum_j (L/d_j) * r_j( P_i, P_j )  (mod L), symmetric with zeros on diagonal.
        """
        n = self.n_paulis()
        L = self.lcm()
        S = np.zeros((n, n), dtype=int)

        for i in range(n):
            ps1 = self.select_pauli_string(i)
            for j in range(i):  # fill lower triangle
                ps2 = self.select_pauli_string(j)
                val = ps1.symplectic_product(ps2, as_scalar=True)
                S[i, j] = val

        S = (S + S.T) % L
        # optional: ensure diagonal zeros (they should be)
        np.fill_diagonal(S, 0)
        return S

    def symplectic_residue_tensor(self) -> np.ndarray:
        """
        Per-qudit residue tensor R of shape (n_paulis, n_paulis, n_qudits),
        where R[i, j, k] = r_k(P_i, P_j) mod d_k. Symmetric in (i, j) and zero on diagonal.
        """
        n = self.n_paulis()
        nq = self.n_qudits()
        R = np.zeros((n, n, nq), dtype=int)

        for i in range(n):
            ps1 = self.select_pauli_string(i)
            for j in range(i):  # lower triangle
                ps2 = self.select_pauli_string(j)
                r = ps1.symplectic_residues(ps2)  # length-nq
                R[i, j, :] = r

        # symmetrize and clear diagonal
        R = R + np.transpose(R, (1, 0, 2))
        for i in range(n):
            R[i, i, :] = 0
        # optional
        return R

    def quditwise_symplectic_product_matrix(self) -> np.ndarray:
        """
        The qudit-wise symplectic product matrix Q associated to the PauliSum.
        It is an n x n matrix, n being the number of Paulis.
        The entry Q[i, j] is 1 if the ith Pauli and the jth Pauli do not commute qudit-wise, 0 otherwise.

        Returns
        -------
        np.ndarray
            The qudit-wise symplectic product matrix Q.
        """
        n = self.n_paulis()
        # list_of_symplectics = self.symplectic_matrix()

        qspm = np.zeros([n, n], dtype=int)
        for i in range(n):
            ps1 = self.select_pauli_string(i)
            for j in range(i):
                ps2 = self.select_pauli_string(j)
            qspm[i, j] = ps1.quditwise_product(ps2)
        qspm = qspm + qspm.T
        return qspm

    # TODO: What's the difference between the next two functions?
    def __str__(self) -> str:
        """
        Returns a more readable string representation of the PauliSum.

        Returns
        -------
        str
            A string representation of the PauliSum with weights, PauliStrings, and phases.
        """
        p_string = ''
        max_str_len = max([len(f'{self.weights()[i]}') for i in range(self.n_paulis())])
        for i in range(self.n_paulis()):
            pauli_string = self.tableau()[i]
            qudit_string = ''.join(['x' + f'{pauli_string[j]}' + 'z' +
                                   f'{pauli_string[j + self.n_qudits()]} ' for j in range(self.n_qudits())])
            n_spaces = max_str_len - len(f'{self.weights()[i]}')
            p_string += f'{self.weights()[i]}' + ' ' * n_spaces + '|' + qudit_string + f'| {self.phases()[i]} \n'
        return p_string

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the PauliSum.

        Returns
        -------
        str
            A string representation of the PauliSum with tableau, dimensions, weights, and phases.
        """
        return f'PauliSum({self.tableau()}, {self.dimensions()}, {self.weights()}, {self.phases()})'

    def get_subspace(self,
                     qudit_indices: int | list[int] | np.ndarray,
                     pauli_indices: int | list[int] | np.ndarray | None = None):
        """
        Get the subspace of the PauliSum corresponding to the qudit indices `qudit_indices` for the given Paulis.
        Not strictly speaking a subspace if we restrict the Pauli indices via `pauli_indices` -- pardon the terminology.

        Parameters
        ----------
        qudit_indices : list[int] | np.ndarray
            The indices of the qudits to include in the subspace.
        pauli_indices : list[int] | np.ndarray | None
            The indices of the Paulis to include in the subspace. If None, all Paulis are included.

        Returns
        -------
        PauliSum
            The subspace of the PauliSum.
        """

        if isinstance(qudit_indices, int):
            qudit_indices = [qudit_indices]
        qudit_indices = np.asarray(qudit_indices, dtype=int)

        if pauli_indices is None:
            pauli_indices = np.arange(self.n_paulis(), dtype=int)
        elif isinstance(pauli_indices, int):
            pauli_indices = [pauli_indices]
        pauli_indices = np.asarray(pauli_indices, dtype=int)

        # Create mask for tableau
        mask = np.concatenate([qudit_indices, qudit_indices + self.n_qudits()])

        # Extract sub-tableau (subset of rows and columns)
        sub_tableau = self.tableau()[pauli_indices][:, mask]

        sub_weights = self.weights()[pauli_indices]
        sub_phases = self.phases()[pauli_indices]
        sub_dims = self.dimensions()[qudit_indices]

        return PauliSum(tableau=sub_tableau, dimensions=sub_dims,
                        weights=sub_weights, phases=sub_phases)

    # FIXME: do we really need this return type?
    def matrix_form(self, pauli_string_index: int | None = None) -> sp.csr_matrix:
        """
        Get the matrix form of the PauliSum as a sparse matrix. This is inclusive of the weights.

        Parameters
        ----------
        pauli_string_index : int | None
            The index of the Pauli string to get the matrix form for.
            If None, the matrix form for the entire PauliSum is returned.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix representation of input Pauli.
        """
        if pauli_string_index is not None:
            ps = PauliSum(self.tableau()[pauli_string_index], self.dimensions(), self.weights(), self.phases())
            return ps.matrix_form()

        list_of_pauli_matrices = []
        for i in range(self.n_paulis()):
            X, Z, dim, phase = int(self.x_exp[i, 0]), int(self.z_exp[i, 0]), self.dimensions()[0], self.phases()[i]
            h = self.xz_mat(dim, X, Z)

            for n in range(1, self.n_qudits()):
                X, Z, dim, phase = int(self.x_exp[i, n]), int(self.z_exp[i, n]), self.dimensions()[n], self.phases()[i]
                h_next = self.xz_mat(dim, X, Z)
                h = sp.csr_matrix(sp.kron(h, h_next, format="csr"))

            e = np.exp(phase * 2 * np.pi * 1j / (2 * self.lcm())) * self.weights()[i]
            list_of_pauli_matrices.append(e * h)

        h = list_of_pauli_matrices[0]
        for m in list_of_pauli_matrices[1:]:
            h = h + m

        return h

    def acquire_phase(self,
                      phases: list[int],
                      pauli_index: int | list[int] | None = None):
        """
        Set new phases for the given PauliSum. If `pauli_index` is None,
        all phases are updated according to the provided list `phases`.
        The update is performed by adding the new and old phases together, modulo `2 * self.lcm`.

        Parameters
        ----------
        phases : list[int]
            The new phases to set.
        pauli_index : int | list[int] | None
            The index of the Pauli string to update. If None, all phases are updated.

        Raises
        ------
        ValueError
            If the number of phases does not match the number of Paulis.
        ValueError
            If `pauli_index` is not an int, list, or np.ndarray.
        """
        if pauli_index is not None:
            if isinstance(pauli_index, int):
                pauli_index = [pauli_index]
            elif len(pauli_index) != len(phases):
                raise ValueError(
                    f"Number of phases ({len(phases)}) must be equal to number of Paulis ({len(pauli_index)})")
            else:
                raise ValueError(f"pauli_index must be int, list, or np.ndarray, not {type(pauli_index)}")
            for i in pauli_index:
                self._phases[i] = (self.phases()[i] + phases) % (2 * self.lcm())
        else:
            if len(phases) != self.n_paulis():
                raise ValueError(
                    f"Number of phases ({len(phases)}) must be equal to number of Paulis ({self.n_paulis()})")
            new_phases = (self.phases() + np.array(phases)) % (2 * self.lcm())
            self._phases = new_phases

    def reorder(self,
                order: list[int]):
        """
        Reorder the Paulis in the PauliSum. If a set of indices are not in the list, they are
        appended to the end in the original order.
        For example, reorder([10, 42]) will put 10th Pauli first and 42nd Pauli second, followed by the remaining paulis
        in their original order.

        Parameters
        ----------
        order : list[int]
            The new order for the Paulis. The length of the list must be at most the number of Paulis.
            If less, leftover Paulis will be appended in their original order.
        """
        if len(order) != self.n_paulis():
            for i in range(self.n_paulis()):
                if i not in order:
                    order.append(i)

        self._weights = np.array([self.weights()[i] for i in order])
        self._phases = np.array([self.phases()[i] for i in order])
        self._tableau = np.array([self._tableau[i] for i in order])

    def swap_paulis(self, index_1: int, index_2: int):
        self._weights[index_1], self._weights[index_2] = self.weights()[index_2], self.weights()[index_1]
        self._phases[index_1], self._phases[index_2] = self.phases()[index_2], self.phases()[index_1]
        self._tableau[index_1], self._tableau[index_2] = self._tableau[index_2], self._tableau[index_1]

    # TODO: Move this to a better location and amend where it is used in the Pauli reduction code
    @staticmethod
    def xz_mat(d: int, aX: int, aZ: int) -> sp.csr_matrix:
        """
        Temporary function for pauli reduction.

        Function for creating generalized Pauli matrix.

        Parameters
        ----------
        d : int
            Dimension of the qudit
        aX : int
            X-part of the Pauli matrix
        aZ : int
            Z-part of the Pauli matrix

        Returns
        -------
        scipy.sparse.csr_matrix
            Generalized Pauli matrix
        """
        omega = np.exp(2 * np.pi * 1j / d)
        aa0 = np.array([1 for i in range(d)])
        aa1 = np.array([i for i in range(d)])
        aa2 = np.array([(i - aX) % d for i in range(d)])
        X = sp.csr_matrix((aa0, (aa1, aa2)))
        aa0 = np.array([omega**(i * aZ) for i in range(d)])
        aa1 = np.array([i for i in range(d)])
        aa2 = np.array([i for i in range(d)])
        Z = sp.csr_matrix((aa0, (aa1, aa2)))
        # if (d == 2) and (aX % 2 == 1) and (aZ % 2 == 1):
        #    return 1j * (X @ Z)
        return X @ Z
