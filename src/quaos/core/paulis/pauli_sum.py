from typing import Union, overload
import numpy as np
import scipy
from .pauli import Pauli
from .pauli_string import PauliString

PauliStringDerivedType = Union[list[PauliString], list[Pauli], list[str], PauliString, Pauli]
PauliType = Union[Pauli, PauliString, 'PauliSum']
ScalarType = Union[float, complex, int]
PauliOrScalarType = Union[PauliType, ScalarType]


class PauliSum:
    '''
    Constructor for PauliSum class.
    Represents a sum of Pauli operators acting on multiple qudits.
    For more details, see the references:
    `Phys. Rev. A 71, 042315 (2005) <https://doi.org/10.1103/PhysRevA.71.042315>`_
    and
    `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_

    Parameters
    ----------
    pauli_list : PauliStringDerivedType
        The list of PauliStrings representing the operators.
    weights: list[float | complex] | np.ndarray | None = None
        The weights for each PauliString.
    phases: list[int] | np.ndarray | None = None
        The phases of the PauliStrings in the range [0, lcm(dimensions) - 1].
    dimensions : list[int] | np.ndarray | int, optional
        The dimensions of each qudit. If an integer is provided,
        all qudits are assumed to have the same dimension (default is 2).

    Attributes
    ----------
    pauli_list : PauliStringDerivedType
        The list of PauliStrings representing the operators.
    weights: list[float | complex] | np.ndarray | None = None
        The weights for each PauliString.
    phases: list[int] | np.ndarray | None = None
        The phases of the PauliStrings in the range [0, lcm(dimensions) - 1].
    dimensions : list[int] | np.ndarray | int, optional
        The dimensions of each qudit. If an integer is provided,
        all qudits are assumed to have the same dimension (default is 2).
    lcm : int
        Least common multiplier of all qudit dimensions.

    Raises
    ------
    ValueError
        If the length of pauli_list and weights do not match.
    '''
    # TODO: I think this would be better if we didn't store the pauli strings, but a GF(d) matrix,
    #       then had a method to return the pauli strings. LD: agree, but to be honest it is OK as it is right now...
    #       If we have to work with the symplectic,
    #       we can simply get it at the beginning of whatever function and use it from there.
    # TODO: Change everything possible to numpy arrays.
    # TODO: Remove self.xz_mat - should be in a utils module
    def __init__(self,
                 pauli_list: PauliStringDerivedType,
                 weights: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None,
                 phases: list[int] | np.ndarray | None = None,
                 dimensions: list[int] | np.ndarray | None = None,
                 standardise: bool = True):
        sanitized_pauli_list, sanitized_dimensions, sanitized_phases, sanitized_weights = self._sanity_checks(
            pauli_list, weights, phases, dimensions
        )

        self.pauli_strings = sanitized_pauli_list
        self.weights = np.asarray(sanitized_weights, dtype=np.complex128)
        self.dimensions = sanitized_dimensions
        self.lcm = np.lcm.reduce(self.dimensions)
        self.phases = np.asarray(sanitized_phases, dtype=int) % (2 * self.lcm)

        self._set_exponents()

        if standardise:
            self.standardise()

    @classmethod
    def from_pauli(cls,
                   pauli: Pauli) -> 'PauliSum':
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
        return cls([PauliString.from_pauli(pauli)], standardise=False)

    @classmethod
    def from_pauli_strings(cls,
                           pauli_string: PauliString) -> 'PauliSum':
        """
        Create a PauliSum instance from a PauliString object.

        Parameters
        ----------
        pauli_string : PauliString
            The PauliString object to convert into a PauliSum.

        Returns
        -------
        PauliSum
            A PauliSum instance representing the given Pauli operator.
        """
        return cls(pauli_string,
                   weights=[1],
                   phases=[0],
                   dimensions=pauli_string.dimensions,
                   standardise=False)

    @classmethod
    def from_random(cls,
                    n_paulis: int,
                    n_qudits: int,
                    dimensions: list[int] | np.ndarray,
                    rand_weights: bool = True) -> 'PauliSum':
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
        weights = 2 * (np.random.rand(n_paulis) - 0.5) if rand_weights else np.ones(n_paulis)

        # ensure no duplicate strings
        strings = []
        for _ in range(n_paulis):
            ps = PauliString.from_random(n_qudits, dimensions)
            while ps in strings:
                ps = PauliString.from_random(n_qudits, dimensions)
            strings.append(ps)

        return cls(strings, weights=weights, phases=[0] * n_paulis, dimensions=dimensions, standardise=False)

    def _set_exponents(self):
        """
        Set the exponents for the Pauli strings in the sum, based on the PauliString objects in pauli_list.
        """
        x_exp = np.zeros((len(self.pauli_strings), len(self.dimensions)))  # we can always index [pauli #, qudit #]
        z_exp = np.zeros((len(self.pauli_strings), len(self.dimensions)))

        for i, p in enumerate(self.pauli_strings):
            x_exp[i, :] = p.x_exp
            z_exp[i, :] = p.z_exp

        self.x_exp = x_exp
        self.z_exp = z_exp

    @staticmethod
    def _sanitize_pauli_list(pauli_list: PauliStringDerivedType,
                             dimensions: list[int] | np.ndarray | None) -> list[PauliString]:
        """
        Validates the consistency of the PauliSum internal representation.
        This check is based only on the PauliString objects provided in pauli_list.

        Raises
        ------
        SyntaxError
            If the dimensions of any PauliString do not match.
        TypeError
            If the input is not a valid PauliString or compatible type.
        """
        if isinstance(pauli_list, Pauli):
            pauli_list = [pauli_list]
        if isinstance(pauli_list, PauliString):
            pauli_list = [pauli_list]
        if isinstance(pauli_list, str):
            pauli_list = [pauli_list]

        sanitised_pauli_list = []
        for p in pauli_list:
            if isinstance(p, PauliString):
                sanitised_pauli_list.append(p)
            elif isinstance(p, Pauli):
                sanitised_pauli_list.append(p)
            elif isinstance(p, str):
                if dimensions is None:
                    raise SyntaxError("Input of strings into PauliSum requires explicit dimensions input")
                sanitised_pauli_list.append(PauliString.from_string(p, dimensions=dimensions))
            else:
                raise TypeError("Pauli list must be a list of PauliString or Pauli objects or strings")

        return sanitised_pauli_list

    @staticmethod
    def _sanitize_dimensions(pauli_list: list[PauliString],
                             dimensions: list[int] | np.ndarray | None = None) -> np.ndarray:
        """
        Validates the consistency of the PauliSum internal representation.
        This check is for the dimensions.

        Raises
        ------
        ValueError
            If the dimensions of all Pauli strings are not equal.
        """
        if dimensions is None and len(pauli_list) == 0:
            return np.empty(0, dtype=int)

        if dimensions is None:
            for i in range(1, len(pauli_list)):
                if not np.array_equal(pauli_list[i].dimensions, pauli_list[0].dimensions):
                    raise ValueError("The dimensions of all Pauli strings must be equal.")
            dimensions = pauli_list[0].dimensions

        return np.array(dimensions)

    @staticmethod
    def _sanitize_phases(pauli_list: list[PauliString],
                         phases: list[int] | np.ndarray | None) -> np.ndarray:
        """
        Validates the consistency of the PauliSum internal representation.
        This check is for the phases. If None are given, they are initialized to zero.
        """
        if phases is None:
            return np.zeros(len(pauli_list), dtype=int)

        return np.asarray(phases, dtype=int)

    @staticmethod
    def _sanitize_weights(pauli_list: list[PauliString],
                          weights: list[float | complex] | list[float] | list[complex] |
                          np.ndarray | float | complex | None) -> np.ndarray:
        """
        Validates the consistency of the PauliSum internal representation.
        This check is for the coefficients. If None are given, they are initialized to ones.

        Raises
        ------
        ValueError
            If there are too many or too little coefficients.
        """
        if weights is None:
            return np.ones(len(pauli_list))

        if isinstance(weights, (float, complex)):
            return np.ones(len(pauli_list)) * weights

        if isinstance(weights, (np.ndarray, list)) and len(pauli_list) != len(weights):
            raise ValueError(f"Length of Pauli list ({len(pauli_list)}) and weights ({len(weights)}) must be equal.")

        return np.asarray(weights, dtype=complex)

    def _sanity_checks(self,
                       pauli_list: PauliStringDerivedType,
                       weights: list[float | complex] | list[float] | list[complex] | np.ndarray |
                       float | complex | None,
                       phases: list[int] | np.ndarray | None,
                       dimensions: list[int] | np.ndarray | None) -> tuple[list[PauliString],
                                                                           np.ndarray,
                                                                           np.ndarray,
                                                                           np.ndarray]:
        """
        Validates the consistency of the PauliSum internal representation.
        """
        sanitized_pauli_list = self._sanitize_pauli_list(pauli_list, dimensions)
        sanitized_dimensions = self._sanitize_dimensions(sanitized_pauli_list, dimensions)
        sanitized_phases = self._sanitize_phases(sanitized_pauli_list, phases)
        sanitized_weights = self._sanitize_weights(sanitized_pauli_list, weights)

        return sanitized_pauli_list, sanitized_dimensions, sanitized_phases, sanitized_weights

    def n_paulis(self) -> int:
        """
        Get the number of Pauli operators in the PauliSum.

        Returns
        -------
        int
            The number of Pauli operators.
        """
        return len(self.pauli_strings)

    def n_qudits(self) -> int:
        """
        Get the number of qudits in the PauliSum.

        Returns
        -------
        int
            The number of qudits.
        """
        return len(self.dimensions)

    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the PauliSum.

        Returns
        -------
        tuple[int, int]
            The number of Pauli operators and the number of qudits.
        """
        return self.n_paulis(), self.n_qudits()

    # TODO: Not sure I follow what this function is doing... Is it supposed to tell which are the
    # number of identities (intended as IIII...III) in the PauliSum?
    def n_identities(self):
        """
        Get the number of identities in the PauliSum.

        Returns
        -------
        int
            The number of identities.
        """
        n_is = []
        for i in range(self.n_paulis()):
            n_is.append(self.pauli_strings[i].n_identities())
        # TODO: I included this return... does it make sense?
        return sum(n_is)

    def phase_to_weight(self):
        """
        Include the phases into the weights of the PauliSum.
        This method modifies the weights of the PauliSum by multiplying them with the phases,
        and reset the phases to all zeros.
        """
        new_weights = np.zeros(self.n_paulis(), dtype=np.complex128)
        for i in range(self.n_paulis()):
            phase = self.phases[i]
            omega = np.exp(2 * np.pi * 1j * phase / (2 * self.lcm))
            new_weights[i] = self.weights[i] * omega
        self.phases = np.zeros(self.n_paulis(), dtype=int)
        self.weights = new_weights

    @overload
    def __getitem__(self,
                    key: tuple[int, int]) -> Pauli:
        ...

    @overload
    def __getitem__(self,
                    key: int | tuple[int, slice] | tuple[int, list[int]]) -> PauliString:
        ...

    @overload
    def __getitem__(self,
                    key: slice | np.ndarray | list[int] | tuple[slice, int] | tuple[slice, slice] | tuple[slice, int] |
                    tuple[slice, list[int]] | tuple[slice, np.ndarray] | tuple[list[int], int] |
                    tuple[np.ndarray, int] | tuple[np.ndarray, slice] | tuple[np.ndarray, list[int]] |
                    tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, list[int]] | tuple[list[int], list[int]] |
                    tuple[list[int], np.ndarray]) -> 'PauliSum':
        ...

    def __getitem__(self,
                    key):
        # TODO: tidy
        if isinstance(key, int):
            return self.pauli_strings[key]
        elif isinstance(key, slice):
            return PauliSum(self.pauli_strings[key], self.weights[key], self.phases[key], self.dimensions, False)
        elif isinstance(key, np.ndarray) or isinstance(key, list):
            new_pauli_strings = [self.pauli_strings[i] for i in key]
            new_weights = np.array([self.weights[i] for i in key])
            new_phases = np.array([self.phases[i] for i in key])
            return PauliSum(new_pauli_strings, new_weights, new_phases, self.dimensions, False)
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must be of length 2")
            if isinstance(key[0], int):
                return self.pauli_strings[key[0]][key[1]]
            if isinstance(key[0], slice):
                pauli_strings_all_qubits = self.pauli_strings[key[0]]
                pauli_strings = [p[key[1]] for p in pauli_strings_all_qubits]
                if isinstance(key[1], int):
                    return PauliSum(pauli_strings,
                                    self.weights[key[0]],
                                    self.phases[key[0]],
                                    np.asarray([self.dimensions[key[1]]]), False)
                elif isinstance(key[1], slice):
                    return PauliSum(pauli_strings, self.weights[key[0]],
                                    self.phases[key[0]], self.dimensions[key[1]], False)
                elif isinstance(key[1], list) or isinstance(key[1], np.ndarray):
                    return PauliSum(pauli_strings, self.weights[key[0]],
                                    self.phases[key[0]], self.dimensions[key[1]], False)
            if isinstance(key[0], list) or isinstance(key[0], np.ndarray):
                if isinstance(key[1], int):
                    return self.get_subspace([key[1]], key[0])
                return self.get_subspace(key[1], key[0])
        else:
            raise TypeError(f"Key must be int or slice, not {type(key)}")

    @overload
    def _setitem_tuple(self,
                       key: tuple[int, int],
                       value: 'Pauli'):
        ...

    @overload
    def _setitem_tuple(self,
                       key: tuple[int, slice],
                       value: 'PauliString'):
        ...

    @overload
    def _setitem_tuple(self,
                       key: tuple[slice, int] | tuple[slice, slice],
                       value: 'PauliSum'):
        ...

    def _setitem_tuple(self,
                       key,
                       value):
        """
        Set a value in the PauliSum using a tuple `key`. It takes the PauliString
        identified by the first element of the `key` and substitutes the Pauli
        identified by the second with `value`.

        Parameters
        ----------
        key : tuple
            The key identifying which Pauli operator to change.
        value : Pauli | PauliString | PauliSum
            The value to set the Pauli operator to.

        Raises
        -------
        ValueError
            If the key is not of length 2.
        """
        if len(key) != 2:
            raise ValueError("Tuple key must be of length 2")
        if isinstance(key[0], int):
            if isinstance(key[1], int):  # key[0] indexes the pauli string, key[1] indexes the qudit
                self.pauli_strings[key[0]][key[1]] = value
            elif isinstance(key[1], slice):  # key[0] indexes the pauli string, key[1] indexes the qudits
                self.pauli_strings[key[0]][key[1]] = value
        if isinstance(key[0], slice):
            if isinstance(key[1], int):  # key[0] indexes the pauli strings, key[1] indexes the qudit
                for i in np.arange(self.n_paulis())[key[0]]:
                    self.pauli_strings[i][key[1]] = value
            elif isinstance(key[1], slice):  # key[0] indexes the pauli strings, key[1] indexes the qudits
                for i_val, i in enumerate(np.arange(self.n_paulis())[key[0]]):
                    print(i, value[int(i_val)])
                    self.pauli_strings[i][key[1]] = value[int(i_val)]

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

    def __setitem__(self,
                    key,
                    value):
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
        if isinstance(key, int):  # key indexes the pauli_string to be replaced by value
            self.pauli_strings[key] = value
        elif isinstance(key, slice):
            self.pauli_strings[key] = value
        elif isinstance(key, tuple):
            self._setitem_tuple(key, value)
        self._set_exponents()  # update exponents x_exp and z_exp

    def __add__(self,
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
        if isinstance(A, Pauli):
            A_sum = PauliSum.from_pauli(A)
        elif isinstance(A, PauliString):
            A_sum = PauliSum.from_pauli_strings(A)
        elif isinstance(A, PauliSum):
            A_sum = A
        else:
            raise ValueError(f"Cannot add Pauli with type {type(A)}")

        new_pauli_list = self.pauli_strings + A_sum.pauli_strings
        new_weights = np.concatenate([self.weights, A_sum.weights])
        new_phases = np.concatenate([self.phases, A_sum.phases])
        return PauliSum(list(new_pauli_list), new_weights, new_phases, self.dimensions, False)

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
        ps1 = self.copy()
        if isinstance(A, Pauli):
            ps2 = PauliString.from_pauli(A)
        elif isinstance(A, PauliString):
            ps2 = PauliSum.from_pauli_strings(A)
        elif isinstance(A, PauliSum):
            ps2 = A
        else:
            raise ValueError(f"Cannot add Pauli with type {type(A)}")
        return ps1 + ps2

    def __sub__(self,
                A: 'PauliSum') -> 'PauliSum':
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
        new_pauli_list = self.pauli_strings + A.pauli_strings
        new_weights = np.concatenate([self.weights, -np.array(A.weights)])
        new_phases = np.concatenate([self.phases, A.phases])
        return PauliSum(list(new_pauli_list), new_weights, new_phases, self.dimensions, False)

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
                   A: PauliType) -> 'PauliSum':
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

        new_dimensions = np.hstack((self.dimensions, A.dimensions))
        new_lcm = np.lcm.reduce(new_dimensions)
        new_pauli_list = []
        new_weights = []
        new_phases = []
        for i in range(self.n_paulis()):
            for j in range(A.n_paulis()):
                new_pauli_list.append(self.pauli_strings[i] @ A.pauli_strings[j])
                new_weights.append(self.weights[i] * A.weights[j])
                new_phases.append(((self.phases[i] + A.phases[j]) % new_lcm))
        output_pauli = PauliSum(new_pauli_list, new_weights, new_phases, new_dimensions, False)
        return output_pauli

    def __mul__(self,
                A: PauliOrScalarType) -> 'PauliSum':
        """
        Multiply a PauliSum and a PauliType (or scalar) objects element-wise.
        It corresponds to operator multiplication (`*`).
        It adds the `x_exp` and `z_exp` exponents of the two PauliSums modulo their dimensions.

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
        if isinstance(A, (int, float)):
            return PauliSum(list(self.pauli_strings), np.array(self.weights) * A, self.phases)
        elif isinstance(A, PauliString):
            return self * PauliSum.from_pauli_strings(A)
        elif not isinstance(A, PauliSum):
            raise ValueError("Multiplication only supported with SymplecticPauli objects or scalar")

        new_p_sum = []
        new_weights = []
        new_phases = []
        for i in range(self.n_paulis()):
            for j in range(A.n_paulis()):
                new_p_sum.append(self.pauli_strings[i] * A.pauli_strings[j])
                new_weights.append(self.weights[i] * A.weights[j])
                acquired_phase = self.pauli_strings[i].acquired_phase(A.pauli_strings[j])
                new_phases.append((self.phases[i] + A.phases[j] + acquired_phase) % (2 * self.lcm))
        output_pauli = PauliSum(new_p_sum, new_weights, new_phases, self.dimensions, False)

        return output_pauli

    def __rmul__(self,
                 A: PauliOrScalarType) -> 'PauliSum':
        """
        Multiply a PauliSum and a PauliType (or scalar) objects element-wise.
        It corresponds to operator multiplication (`*`).
        It adds the `x_exp` and `z_exp` exponents of the two PauliSums modulo their dimensions.

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
        if not isinstance(other_pauli,
                          PauliSum):
            return False
        t1 = np.all(self.pauli_strings == other_pauli.pauli_strings)
        t2 = np.all(self.weights == other_pauli.weights)
        t3 = np.all(self.phases == other_pauli.phases)
        t4 = np.all(self.dimensions == other_pauli.dimensions)
        return bool(t1 and t2 and t3 and t4)

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
            (tuple(self.pauli_strings),
             tuple(self.weights),
             tuple(self.phases),
             tuple(self.dimensions))
        )

    def __dict__(self) -> dict:
        """
        Returns a dictionary representation of the object's attributes.

        Returns
        -------
        dict
            A dictionary containing the values of `x_exp`, `z_exp`, `weights`, `phases` and `dimensions`.
        """
        return {'pauli_strings': self.pauli_strings,
                'weights': self.weights,
                'phases': self.phases,
                'dimensions': self.dimensions}

    # TODO: does it combine equivalent Paulis?
    def standardise(self):
        """
        Standardises the PauliSum object by combining equivalent Paulis and
        adding phase factors to the weights then resetting the phases.
        """
        # combine equivalent
        # self.combine_equivalent_paulis()
        # sort
        self.phase_to_weight()
        self.weights = [x for _, x in sorted(zip(self.pauli_strings, self.weights))]
        # self.phases = [x for _, x in sorted(zip(self.pauli_strings, self.phases))]
        self.pauli_strings = sorted(self.pauli_strings)

    # TODO: Maybe switch self.standardise() -> self.phase_to_weight() here and
    #       include self.combine_equivalent_paulis() in standardize() above?
    def combine_equivalent_paulis(self):
        """
        Combines equivalent Pauli operators in the sum by summing their coefficients and deleting duplicates.
        """
        self.standardise()  # makes sure all phases are 0
        # combine equivalent Paulis
        to_delete = []
        for i in reversed(range(self.n_paulis())):
            for j in range(i + 1, self.n_paulis()):
                if self.pauli_strings[i] == self.pauli_strings[j]:
                    self.weights[i] = self.weights[i] + self.weights[j]
                    to_delete.append(j)
        self._delete_paulis(to_delete)

        # remove zero weight Paulis
        to_delete = []
        for i in range(self.n_paulis()):
            if self.weights[i] == 0:
                to_delete.append(i)
        self._delete_paulis(to_delete)

    def remove_trivial_paulis(self):
        """
        Removes trivial Pauli strings (those that are identity operators) from the sum.
        """
        # If entire Pauli string is I, remove it
        to_delete = []
        for i in range(self.n_paulis()):
            if np.all(self.x_exp[i, :] == 0) and np.all(self.z_exp[i, :] == 0):
                to_delete.append(i)
        self._delete_paulis(to_delete)

    def remove_trivial_qudits(self):
        """
        Removes trivial qudits (those that are identity operators) from the sum.
        """
        # If entire qudit is I, remove it
        to_delete = []
        for i in range(self.n_qudits()):
            if np.all(self.x_exp[:, i] == 0) and np.all(self.z_exp[:, i] == 0):
                to_delete.append(i)
        self._delete_qudits(to_delete)

    def symplectic(self) -> np.ndarray:
        """
        Returns the symplectic representation of the PauliSum.
        That is, a tableau with all symplectic representations of
        the PauliStrings.

        Returns
        -------
        np.ndarray
            The symplectic representation of the PauliSum.
        """
        symplectic = np.zeros([self.n_paulis(), 2 * self.n_qudits()],
                              dtype=int)
        for i, p in enumerate(self.pauli_strings):
            symplectic[i, :] = p.symplectic()
        return symplectic

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
        Checks whether the PauliSum has only (i.e., all PauliStrings therein) X components.

        Returns
        -------
        bool
            True if the PauliSum has only X components, False otherwise.
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

    def select_pauli_string(self,
                            pauli_index: int) -> PauliString:
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
        return self.pauli_strings[pauli_index]

    def _delete_paulis(self,
                       pauli_indices: list[int] | int):
        """
        Deletes PauliStrings from the PauliSum.

        Parameters
        ----------
        pauli_indices : list[int] | int
            The indices of the PauliStrings to delete.
        """
        if isinstance(pauli_indices, int):
            pauli_indices = [pauli_indices]

        new_weights = np.delete(self.weights, pauli_indices)
        new_phases = np.delete(self.phases, pauli_indices)
        new_x_exp = np.delete(self.x_exp, pauli_indices, axis=0)
        new_z_exp = np.delete(self.z_exp, pauli_indices, axis=0)

        for i in sorted(pauli_indices, reverse=True):  # sort in reverse order to avoid index shifting # Convert to list
            del self.pauli_strings[i]

        self.weights = new_weights
        self.phases = new_phases
        self.x_exp = new_x_exp
        self.z_exp = new_z_exp

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

        new_pauli_strings = []
        for p in self.pauli_strings:
            new_pauli_strings.append(p._delete_qudits(qudit_indices))

        self.pauli_strings = new_pauli_strings
        self.x_exp = np.delete(self.x_exp, qudit_indices, axis=1)
        self.z_exp = np.delete(self.z_exp, qudit_indices, axis=1)
        self.dimensions = np.delete(self.dimensions, qudit_indices)

        self.lcm = np.lcm.reduce(self.dimensions)

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
        return PauliSum([ps.copy() for ps in self.pauli_strings],
                        self.weights.copy(),
                        self.phases.copy(),
                        self.dimensions.copy(),
                        False)

    def symplectic_product_matrix(self) -> np.ndarray:
        """
        The symplectic product matrix S associated to the PauliSum.
        It is an n x n matrix, n being the number of Paulis.
        The entry S[i, j] is the symplectic product of the ith Pauli and the jth Pauli.

        Returns
        -------
        np.ndarray
            The symplectic product matrix S.
        """
        n = self.n_paulis()
        # list_of_symplectics = self.symplectic_matrix()

        spm = np.zeros([n, n], dtype=int)
        for i in range(n):
            for j in range(n):
                if i > j:
                    spm[i, j] = self.pauli_strings[i].symplectic_product(self.pauli_strings[j])
        spm = spm + spm.T
        return spm

    # TODO: What's the difference between the next two functions?
    def __str__(self) -> str:
        """
        Returns a string representation of the PauliSum.

        Returns
        -------
        str
            A string representation of the PauliSum with weights, PauliStrings, and phases.
        """
        p_string = ''
        max_str_len = max([len(f'{self.weights[i]}') for i in range(self.n_paulis())])
        for i in range(self.n_paulis()):
            pauli_string = self.pauli_strings[i]
            qudit_string = ''.join(['x' + f'{pauli_string.x_exp[j]}' + 'z' +
                                   f'{pauli_string.z_exp[j]} ' for j in range(self.n_qudits())])
            n_spaces = max_str_len - len(f'{self.weights[i]}')
            p_string += f'{self.weights[i]}' + ' ' * n_spaces + '|' + qudit_string + f'| {self.phases[i]} \n'
        return p_string

    def __repr__(self) -> str:
        """
        Returns a string representation of the PauliSum.

        Returns
        -------
        str
            A string representation of the PauliSum with weights, PauliStrings, and phases.
        """
        return f'PauliSum({self.pauli_strings}, {self.weights}, {self.phases}, {self.dimensions})'

    def get_subspace(self,
                     qudit_indices: list[int] | np.ndarray,
                     pauli_indices: list[int] | np.ndarray | None = None):
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
        if pauli_indices is None:
            indices = np.arange(self.n_paulis()).tolist()
        else:
            indices = np.asarray(pauli_indices)

        dimensions = self.dimensions[qudit_indices]
        pauli_list = []
        for i in indices:
            p = self.pauli_strings[i]
            p = p.get_subspace(qudit_indices)
            pauli_list.append(p)
        return PauliSum(pauli_list, self.weights[indices], self.phases[pauli_indices], dimensions, False)

    def matrix_form(self,
                    pauli_string_index: int | None = None) -> scipy.sparse.csr_matrix:
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
            ps = self.select_pauli_string(pauli_string_index)
            return PauliSum(ps).matrix_form()
        else:
            list_of_pauli_matrices = []
            for i in range(self.n_paulis()):
                X, Z, dim, phase = int(self.x_exp[i, 0]), int(self.z_exp[i, 0]), self.dimensions[0], self.phases[i]
                h = self.xz_mat(dim, X, Z)

                for n in range(1, self.n_qudits()):
                    X, Z, dim, phase = int(self.x_exp[i, n]), int(self.z_exp[i, n]), self.dimensions[n], self.phases[i]
                    h_next = self.xz_mat(dim, X, Z)

                    h = scipy.sparse.kron(h, h_next, format="csr")
                list_of_pauli_matrices.append(np.exp(phase * 2 * np.pi * 1j / (2 * self.lcm)) * self.weights[i] * h)
            m = sum(list_of_pauli_matrices)

        return m

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
                self.phases[i] = (self.phases[i] + phases) % (2 * self.lcm)
        else:
            if len(phases) != self.n_paulis():
                raise ValueError(
                    f"Number of phases ({len(phases)}) must be equal to number of Paulis ({self.n_paulis()})")
            new_phase = (np.array(self.phases) + np.array(phases)) % (2 * self.lcm)
        self.phases = new_phase

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
        self.pauli_strings = [self.pauli_strings[i] for i in order]
        self.weights = np.array([self.weights[i] for i in order])
        self.phases = np.array([self.phases[i] for i in order])
        self.x_exp = np.array([self.x_exp[i] for i in order])
        self.z_exp = np.array([self.z_exp[i] for i in order])

    def swap_paulis(self, index_1: int, index_2: int):
        self.pauli_strings[index_1], self.pauli_strings[index_2] = (self.pauli_strings[index_2],
                                                                    self.pauli_strings[index_1])
        self.weights[index_1], self.weights[index_2] = self.weights[index_2], self.weights[index_1]
        self.phases[index_1], self.phases[index_2] = self.phases[index_2], self.phases[index_1]
        self.x_exp[index_1], self.x_exp[index_2] = self.x_exp[index_2], self.x_exp[index_1]
        self.z_exp[index_1], self.z_exp[index_2] = self.z_exp[index_2], self.z_exp[index_1]

    # TODO: Move this to a better location and amend where it is used in the Pauli reduction code
    @staticmethod
    def xz_mat(d: int,
               aX: int,
               aZ: int) -> scipy.sparse.csr_matrix:
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
        X = scipy.sparse.csr_matrix((aa0, (aa1, aa2)))
        aa0 = np.array([omega**(i * aZ) for i in range(d)])
        aa1 = np.array([i for i in range(d)])
        aa2 = np.array([i for i in range(d)])
        Z = scipy.sparse.csr_matrix((aa0, (aa1, aa2)))
        if (d == 2) and (aX % 2 == 1) and (aZ % 2 == 1):
            return 1j * (X @ Z)
        return X @ Z
