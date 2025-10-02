from __future__ import annotations
from typing import overload
import numpy as np
import functools
import re
from .pauli import Pauli
from .bases_manipulation import bases_to_int


@functools.total_ordering
class PauliString:
    '''
    Tensor product of Pauli operators acting on multiple qudits.
    This class supports qudits of arbitrary dimensions and provides methods for construction,
    manipulation, and algebraic operations on Pauli strings.
    For more details, see the references:
    `Phys. Rev. A 71, 042315 (2005) <https://doi.org/10.1103/PhysRevA.71.042315>`_
    and
    `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_

    Parameters
    ----------
    x_exp : list[int] | np.ndarray | str
        The exponents of the X-type Pauli operators for each qudit, or a string representation.
    z_exp : list[int] | np.ndarray
        The exponents of the Z-type Pauli operators for each qudit.
    dimensions : list[int] | np.ndarray | int, optional
        The dimensions of each qudit. If an integer is provided,
        all qudits are assumed to have the same dimension (default is 2).

    Attributes
    ----------
    x_exp : np.ndarray
        Array of X exponents for each qudit.
    z_exp : np.ndarray
        Array of Z exponents for each qudit.
    dimensions : np.ndarray
        Array of dimensions for each qudit.
    lcm : int
        Least common multiplier of all qudit dimensions.

    Notes
    -----
    - The class assumes that the Pauli operator exponents are always reduced modulo their respective qudit dimensions.
    - The string representation for initialization should follow the format: 'x0z0 x1z1 ...', where xi and zi are
      integers within [0, d-1] for each qudit of dimension d.
    '''

    def __init__(self,
                 x_exp: list[int] | np.ndarray | str,
                 z_exp: list[int] | np.ndarray,
                 dimensions: list[int] | np.ndarray | int = 2):

        if isinstance(x_exp, str):
            if z_exp is not None:
                raise Warning('If input string is provided, z_exp is unnecessary')
            xz_exponents = re.split('x|z', x_exp)[1:]
            z_exp = np.array(xz_exponents[1::2], dtype=int)
            x_exp = np.array(xz_exponents[0::2], dtype=int)
        elif isinstance(x_exp, list):
            z_exp = np.array(z_exp)
            x_exp = np.array(x_exp)
        elif isinstance(x_exp, int):
            z_exp = np.array([z_exp])
            x_exp = np.array([x_exp])

        if type(dimensions) is int:
            self.dimensions = dimensions * np.ones(len(x_exp), dtype=int)
        else:
            self.dimensions = np.asarray(dimensions)

        self.x_exp = x_exp % self.dimensions
        self.z_exp = z_exp % self.dimensions
        self.lcm = np.lcm.reduce(self.dimensions)
        self._sanity_check()

    def _sanity_check(self):
        """
        Validates the consistency of the PauliString's internal representation.

        Raises
        ------
        ValueError
            If the lengths of `x_exp`, `z_exp`, and `dimensions` do not match,
            or if any exponent is not valid for its corresponding dimension.
        """
        if len(self.x_exp) != len(self.dimensions):
            raise ValueError(f"Number of x exponents ({len(self.x_exp)})"
                             f" and dimensions ({len(self.dimensions)}) must be equal.")

        if len(self.x_exp) != len(self.z_exp):
            raise ValueError(f"Number of x and z exponents ({len(self.x_exp)}"
                             f" and {len(self.z_exp)}) must be equal.")

        if len(self.dimensions) != len(self.z_exp):
            raise ValueError(f"Number of dimensions ({len(self.dimensions)})"
                             f" and z exponents ({len(self.z_exp)}) must be equal.")

        for i in range(len(self.x_exp)):
            if self.dimensions[i] - 1 < self.x_exp[i] or self.dimensions[i] - 1 < self.z_exp[i]:
                raise ValueError(f"Dimension {self.dimensions[i]} is too small for"
                                 f" exponents {self.x_exp[i]} and {self.z_exp[i]}")

    @classmethod
    def from_pauli(cls, pauli: Pauli) -> PauliString:
        """
        Create a PauliString instance from a single Pauli object.

        Parameters
        ----------
        pauli : Pauli
            The Pauli object to convert into a PauliString.

        Returns
        -------
        PauliString
            A PauliString instance representing the given Pauli operator.
        """

        return PauliString(x_exp=[pauli.x_exp], z_exp=[pauli.z_exp], dimensions=[pauli.dimension])

    @classmethod
    def from_string(cls, pauli_str: str, dimensions) -> PauliString:
        """
        Create a PauliString instance from a string representation.

        Parameters
        ----------
        pauli_str : str
            The string representation of the Pauli string, where exponents are separated by 'x' and 'z'.
        dimensions : Any
            The dimensions parameter to be passed to the PauliString constructor.

        Returns
        -------
        PauliString
            An instance of PauliString initialized with the exponents and dimensions parsed from the input string.

        Examples
        --------
        >>> PauliString.from_string("x2z3 x4z1 x0z0", [4, 5, 2]])
        <PauliString ...>
        """

        xz_exponents = re.split('x|z', pauli_str)[1:]
        z_exp = np.array(xz_exponents[1::2], dtype=int)
        x_exp = np.array(xz_exponents[0::2], dtype=int)
        return cls(x_exp=x_exp, z_exp=z_exp, dimensions=dimensions)

    @classmethod
    def from_random(cls, n_qudits: int, dimensions: list[int] | np.ndarray, seed=None) -> PauliString:
        """
        Generate a random PauliString instance for a given number of qudits and their dimensions.

        Parameters
        ----------
        n_qudits : int
            The number of qudits in the Pauli string.
        dimensions : list[int] or np.ndarray
            The dimensions of each qudit. Should be a list or array of integers specifying the dimension for each qudit.
        seed : int or None, optional
            Seed for the random number generator to ensure reproducibility. Default is None.

        Returns
        -------
        PauliString
            A randomly generated PauliString instance with random x and z exponents for each qudit.
        """
        if seed is not None:
            np.random.seed(seed)
        return cls(x_exp=np.random.randint(dimensions, dtype=int),
                   z_exp=np.random.randint(dimensions, dtype=int),
                   dimensions=dimensions)

    def __repr__(self) -> str:
        """
        Return the string representation of the Pauli object
        (in a format that is helpful for debugging).

        Returns
        -------
        str
            A string in the format "Pauli(x_exp=..., z_exp=..., dimensions=...)".
        """

        return f"Pauli(x_exp={self.x_exp}, z_exp={self.z_exp}, dimensions={self.dimensions})"

    def __str__(self) -> str:
        """
        Return the string representation of the Pauli object
        (in the format also employed for constructing an instance of the class).

        Returns
        -------
        str
            The string representation of the Pauli object, with each qudit's
            exponents shown as 'x{exp}z{exp} x{exp}z{exp} x{exp}z{exp} ...'.
        """

        p_string = ''
        for i in range(self.n_qudits()):
            p_string += 'x' + f'{self.x_exp[i]}' + 'z' + f'{self.z_exp[i]} '
        return p_string

    def __matmul__(self, A: PauliString) -> PauliString:
        """
        Implements the tensor product between PauliString objects.
        It corresponds to operator tensor product (`@`).
        The resulting PauliString has the exponents of both strings concatenated.

        Parameters
        ----------
        A : PauliString
            The PauliString instance to be tensored with `self`.

        Returns
        -------
        PauliString
            A new PauliString instance representing the tensor product of `self` and `A`.

        Examples
        --------
        >>> p1 = PauliString.from_string("x1z0 x0z1", [2, 2])
        >>> p2 = PauliString.from_string("x2z1", [3])
        >>> p1 @ p2
        Pauli(x_exp=[1 0 2], z_exp=[0 1 1], dimensions=[2 2 3])

        Notes
        -----
        - This is NOT a product between two PauliString objects!
        """
        new_x_exp = np.concatenate((self.x_exp, A.x_exp))
        new_z_exp = np.concatenate((self.z_exp, A.z_exp))
        new_dims = np.concatenate((self.dimensions, A.dimensions))
        return PauliString(new_x_exp,
                           new_z_exp,
                           new_dims)

    def __rmatmul__(self, A: Pauli) -> PauliString:
        """
        Implements the tensor product between a PauliString (`self`) and a Pauli (`A`) objects.
        It corresponds to operator tensor product (`@`).
        The resulting PauliString has the exponents of both strings concatenated.

        Parameters
        ----------
        A : Pauli
            The Pauli object to be right-multiplied with this PauliString.

        Returns
        -------
        PauliString
            A new PauliString instance with updated `x_exp`, `z_exp`, and `dimensions` arrays,
            resulting from concatenating the corresponding attributes of the operands.

        Examples
        --------
        >>> ps = PauliString(...)
        >>> p = Pauli(...)
        >>> result = p @ ps
        """
        new_x_exp = np.concatenate(self.x_exp, np.array([A.x_exp]))
        new_z_exp = np.concatenate(self.z_exp, np.array([A.z_exp]))
        new_dims = np.concatenate(self.dimensions, np.array([A.dimension]))
        return PauliString(new_x_exp,
                           new_z_exp,
                           new_dims)

    def __mul__(self, A: PauliString) -> PauliString:
        """
        Multiply two PauliString objects element-wise. It corresponds to operator multiplication (`*`).
        It adds the `x_exp` and `z_exp` exponents of the two PauliStrings modulo their dimensions.

        Parameters
        ----------
        A : PauliString
            The PauliString to multiply with `self`.

        Returns
        -------
        PauliString
            A new PauliString representing the product of `self` and `A`.

        Raises
        ------
        Exception
            If the dimensions of the two PauliStrings are not equal.
        ValueError
            If `A` is not an instance of PauliString.

        Examples
        --------
        >>> ps1 = PauliString(x_exp1, z_exp1, dims)
        >>> ps2 = PauliString(x_exp2, z_exp2, dims)
        >>> ps3 = ps1 * ps2
        """
        if isinstance(A, PauliString):
            if np.any(self.dimensions != A.dimensions):
                raise Exception("To multiply two PauliStrings, their dimensions"
                                f" {self.dimensions} and {A.dimensions} must be equal")
            x_new = np.mod(self.x_exp + A.x_exp, (self.dimensions))
            z_new = np.mod(self.z_exp + A.z_exp, (self.dimensions))
            return PauliString(x_new, z_new, self.dimensions)
        else:
            raise ValueError(f"Cannot multiply PauliString with type {type(A)}")

    def __pow__(self, A: int) -> PauliString:
        """
        Raises the PauliString to the power of an integer exponent.

        Parameters
        ----------
        A : int
            The integer exponent to which the PauliString is to be raised.

        Returns
        -------
        PauliString
            A new PauliString instance representing the result of the exponentiation.

        Examples
        --------
        >>> ps = PauliString(x_exp, z_exp, dimensions)
        >>> ps_squared = ps ** 2
        """
        return PauliString(self.x_exp * A, self.z_exp * A, self.dimensions)

    def __eq__(self, other_pauli: PauliString) -> bool:
        """
        Determine if two PauliString objects are equal.

        Parameters
        ----------
        other_pauli : PauliString
            The PauliString instance to compare against.

        Returns
        -------
        bool
            True if both PauliString instances have identical x_exp, z_exp, and dimensions;
            False otherwise.
        """
        if not isinstance(other_pauli, PauliString):
            return False

        return bool(np.all(self.x_exp == other_pauli.x_exp) and
                    np.all(self.z_exp == other_pauli.z_exp) and
                    np.all(self.dimensions == other_pauli.dimensions))

    def __ne__(self,
               other_pauli: PauliString
               ) -> bool:
        """
        Check if this PauliString is not equal to another PauliString.

        Parameters
        ----------
        other_pauli : PauliString
            The PauliString instance to compare against.

        Returns
        -------
        bool
            True if the two PauliString instances are not equal, False otherwise.
        """
        return not self.__eq__(other_pauli)

    def __gt__(self,
               other_pauli: PauliString
               ) -> bool:
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
        # compare x first
        for i in range(self.n_qudits()):
            if self.x_exp[i] == other_pauli.x_exp[i]:
                pass
            elif self.x_exp[i] < other_pauli.x_exp[i]:
                return True
            else:
                return False
        # x are equal, compare z
        for i in range(self.n_qudits()):
            if self.z_exp[i] == other_pauli.z_exp[i]:
                pass
            elif self.z_exp[i] < other_pauli.z_exp[i]:
                return True
            else:
                return False
        # they are equal
        return False

    def __lt__(self,
               other_pauli: PauliString
               ) -> bool:
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

    def _to_int(self, reverse=False):
        """
        Convert the Pauli string exponents to an integer representation.

        Parameters
        ----------
        reverse : bool, optional
            If True, the order of X and Z exponents is swapped and the resulting
            base and dimensions arrays are reversed before conversion. Default is False.

        Returns
        -------
        int
            The integer representation of the Pauli string.

        Notes
        -----
        - The conversion uses the `bases_to_int` function to map the base array. Look there for an example.
        - IMPORTANT: if too many qudits are provided, the output could be exceedingly large.
        """
        dims = self.dimensions
        if self.n_qudits() > 15:
            raise ValueError(f"Cannot convert PauliString with more than {self.n_qudits()} qudits to int, "
                             "as it may exceed the maximum integer size. Current max set to 15 qudits.")
        dims_double = [d for d in dims for _ in range(2)]
        base = np.zeros(len(dims_double), dtype=int)
        base[:len(dims)] = self.x_exp
        base[len(dims):] = self.z_exp
        if not reverse:
            return bases_to_int(base, dims_double)
        else:
            base[:len(dims)] = self.z_exp
            base[len(dims):] = self.x_exp
            return bases_to_int(base[::-1], dims_double[::-1])

    def __hash__(self) -> int:
        """
        Return the hash value of the PauliString object.
        That is a unique identifier, similar to the characterizing int.

        Returns
        -------
        int
            The hash value of the PauliString instance.
        """
        if self.n_qudits() > 15:
            raise ValueError(f"Cannot convert PauliString with more than {self.n_qudits()} qudits to hash, "
                             "as it may exceed the maximum integer size. Current max set to 15 qudits.")
        return hash(
            (tuple(self.x_exp),
             tuple(self.z_exp),
             tuple(self.dimensions))
        )

    def __dict__(self) -> dict:
        """
        Returns a dictionary representation of the object's attributes.

        Returns
        -------
        dict
            A dictionary containing the values of `x_exp`, `z_exp`, and `dimensions`.
        """
        return {'x_exp': self.x_exp,
                'z_exp': self.z_exp,
                'dimensions': self.dimensions}

    def n_qudits(self) -> int:
        """
        Returns the number of qudits represented by this PauliString.

        Returns
        -------
        int
            The number of qudits.
        """
        return len(self.x_exp)

    def n_identities(self) -> int:
        """
        Returns the number of identities within this PauliString.

        Returns
        -------
        int
            The number of identities.
        """
        return np.sum(np.logical_and(self.x_exp == 0, self.z_exp == 0))

    def get_paulis(self) -> list[Pauli]:
        """
        Returns a list of Pauli objects corresponding to the PauliString.

        Returns
        -------
        list[Pauli]
            A list of Pauli objects.
        """
        return [Pauli(x_exp=self.x_exp[i], z_exp=self.z_exp[i], dimension=self.dimensions[i])
                for i in range(len(self.x_exp))]

    def tableau(self) -> np.ndarray:
        """
        Returns the tableau representation of the Pauli string.
        The tableau representation is a binary vector of length 2 * n_qudits,
        where the first n_qudits entries correspond to the X exponents and the
        last n_qudits entries correspond to the Z exponents of the Pauli string.
        It is essential for efficient algebraic operations on Pauli strings, see
        `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_.

        Returns
        -------
        np.ndarray
            A 1D numpy array of length 2 * n_qudits representing the tableau
            form of the Pauli string.
        """
        symp = np.zeros(2 * self.n_qudits())
        symp[0:self.n_qudits()] = self.x_exp
        symp[self.n_qudits():2 * self.n_qudits()] = self.z_exp
        return symp

    def symplectic_product(self, A: PauliString) -> np.ndarray:
        """
        Compute the symplectic product between this PauliString and another.
        The symplectic product is defined as the sum over all qudits of the difference between
        the product of the X component of this PauliString and the Z component of the other,
        and the product of the Z component of this PauliString and the X component of the other,
        modulo the least common multiple (lcm) of the qudit dimensions. See
        `Phys. Rev. A 70, 052328 (2004) <https://doi.org/10.1103/PhysRevA.70.052328>`_
        for more details.

        Parameters
        ----------
        A : PauliString
            The other PauliString to compute the symplectic product with.

        Returns
        -------
        np.ndarray
            The symplectic product as a NumPy array, modulo the lcm.

        Notes
        -----
        The symplectic representation is used to efficiently compute commutation relations
        between Pauli operators in quantum information theory.
        """
        if self.n_qudits() != A.n_qudits() or not np.array_equal(self.dimensions, A.dimensions):
            raise ValueError(
                f"Incompatible PauliStrings: must have the same number of qudits (currently {self.n_qudits()} and {A.n_qudits()}) and dimensions (currently {self.dimensions} and {A.dimensions}).")
        n = self.n_qudits()
        symp = self.tableau()
        symp_A = A.tableau()
        prod = sum([symp[i] * symp_A[i + n] - symp[i + n] * symp_A[i] for i in range(n)]) % self.lcm
        return prod

    def quditwise_product(self, A: PauliString) -> int:
        """
        Compute the quditwise product between this PauliString and another.
        The quditwise product is defined as the elementwise product of the X and Z exponents.

        Parameters
        ----------
        A : PauliString
            The other PauliString to compute the quditwise product with.

        Returns
        -------
        np.ndarray
            The quditwise product as a NumPy array.
        """
        if self.n_qudits() != A.n_qudits() or not np.array_equal(self.dimensions, A.dimensions):
            raise ValueError(
                f"Incompatible PauliStrings: must have the same number of qudits (currently {self.n_qudits()} and {A.n_qudits()}) and dimensions (currently {self.dimensions} and {A.dimensions}).")
        if any(np.sum(self.x_exp[i] * A.z_exp[i] - self.z_exp[i] * A.x_exp[i]) % self.dimensions[i] for i in range(self.n_qudits())):
            return 1
        else:
            return 0

    def amend(self, qudit_index: int, new_x: int, new_z: int) -> PauliString:
        """
        Change the X and Z exponents for a specific qudit in the Pauli string.

        Parameters
        ----------
        qudit_index : int
            The index of the qudit to change.
        new_x : int
            The new exponent for the X operator at the specified qudit.
        new_z : int
            The new exponent for the Z operator at the specified qudit.

        Returns
        -------
        PauliString
            The modified PauliString instance.

        Raises
        ------
        ValueError
            If either `new_x` or `new_z` is greater than the dimension of the specified qudit.
        """
        if new_x > self.dimensions[qudit_index] or new_z > self.dimensions[qudit_index]:
            raise ValueError(f"Exponents ({new_x, new_z}) cannot be larger than qudit dimension"
                             f" ({self.dimensions[qudit_index]})")
        self.x_exp[qudit_index] = new_x
        self.z_exp[qudit_index] = new_z
        return self

    def acquired_phase(self, other_pauli: PauliString) -> int:
        """
        Compute the phase acquired when multiplying two Pauli strings self * other_pauli.
        This method calculates the phase factor resulting from the multiplication of two Pauli strings,
        using their symplectic representations. The phase is computed modulo twice the least common multiple (lcm)
        of the underlying dimensions. For details, see:
        `IEEE International Symposium on Information Theory (ISIT), pp. 791-795. IEEE (2018)
        <https://doi.org/10.1109/ISIT.2018.8437652>`_

        Parameters
        ----------
        other_pauli : PauliString
            The Pauli string to multiply with this Pauli string.

        Returns
        -------
        int
            The acquired phase, as an integer modulo ``2 * self.lcm``.
        """
        # phases acquired when multiplying two Pauli strings self * other_pauli
        # phi = 2.  # / self.dimensions
        # phase = 0
        # for i in range(self.n_qudits()):
        #     phase += phi * (self.x_exp[i] * other_pauli.z_exp[i] + self.z_exp[i] * other_pauli.x_exp[i])
        # return phase % (2 * self.lcm)

        # identity on lower diagonal of U
        U = np.zeros((2 * self.n_qudits(), 2 * self.n_qudits()), dtype=int)
        U[self.n_qudits():, :self.n_qudits()] = np.eye(self.n_qudits(), dtype=int)
        a = self.tableau()
        b = other_pauli.tableau()
        return int(2 * a.T @ U @ b) % (2 * self.lcm)

    def _replace_symplectic(self, symplectic: np.ndarray, qudit_indices: list[int]) -> PauliString:
        """
        Change a PauliString via its symplectic representation.
        This method updates the `x_exp` and `z_exp` exponents of the Pauli string for the given
        `qudit_indices` using the provided `symplectic` array. The `symplectic` array is expected
        to be a concatenation of the new x and z exponents for the specified qudits.

        Parameters
        ----------
        symplectic : np.ndarray
            A 1D numpy array containing the new symplectic representation (x and z exponents)
            for the specified qudits. The first half corresponds to x exponents, and the second
            half to z exponents.
        qudit_indices : list[int]
            List of indices specifying which qudits in the Pauli string should be replaced.

        Returns
        -------
        PauliString
            A new PauliString object with the updated symplectic representation for the specified qudits.

        Notes
        -----
        The length of `symplectic` must be exactly twice the length of `qudit_indices`.
        """
        x_exp_replace = symplectic[0:len(qudit_indices)]
        z_exp_replace = symplectic[len(qudit_indices):2 * len(qudit_indices)]

        x_exp = self.x_exp.copy()
        z_exp = self.z_exp.copy()
        for i, index in enumerate(qudit_indices):
            x_exp[index] = x_exp_replace[i]
            z_exp[index] = z_exp_replace[i]

        return PauliString(x_exp=x_exp, z_exp=z_exp, dimensions=self.dimensions)

    # TODO: not sure if here it is best to return a new object or not
    def _delete_qudits(self, qudit_indices: list[int], return_new: bool = True) -> PauliString:
        """
        Delete specified qudits from the PauliString.
        Removes the qudits at the given indices from the internal representations
        (`x_exp`, `z_exp`, and `dimensions`). Optionally returns a new PauliString
        instance with the specified qudits removed, or modifies the current instance
        in place.

        Parameters
        ----------
        qudit_indices : list of int
            Indices of the qudits to be deleted.
        return_new : bool, optional
            If True (default), returns a new PauliString instance with the specified
            qudits removed. If False, modifies the current instance in place and
            returns self.

        Returns
        -------
        PauliString
            A new PauliString instance with the specified qudits removed if
            `return_new` is True, otherwise returns self after modification.
        """
        x_exp = np.delete(self.x_exp, qudit_indices)
        z_exp = np.delete(self.z_exp, qudit_indices)
        dimensions = np.delete(self.dimensions, qudit_indices)
        if return_new:
            return PauliString(x_exp=x_exp, z_exp=z_exp, dimensions=dimensions)
        else:
            self.x_exp = x_exp
            self.z_exp = z_exp
            self.dimensions = dimensions
            self._sanity_check()
            return self

    @overload
    def __getitem__(self, key: int) -> Pauli:
        ...

    @overload
    def __getitem__(self, key: slice | np.ndarray | list) -> PauliString:
        ...

    def __getitem__(self, key: int | slice | np.ndarray | list) -> 'PauliString | Pauli':
        """
        Retrieve a Pauli or a (smaller) PauliString from the PauliString.

        Parameters
        ----------
        key : int, slice, np.ndarray, or list
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
        if isinstance(key, int):
            return PauliString.from_pauli(self.get_paulis()[key])
        elif isinstance(key, slice) or isinstance(key, np.ndarray) or isinstance(key, list):
            return PauliString(x_exp=self.x_exp[key], z_exp=self.z_exp[key], dimensions=self.dimensions[key])
        else:
            raise ValueError(f"Cannot get item with key {key}. Key must be an int or a slice.")

    def __setitem__(self, key: int | slice, value: 'Pauli | PauliString'):
        """
        Set the value(s) of the PauliString at the specified index or slice.

        Parameters
        ----------
        key : int or slice
            The index or slice at which to set the value. If an integer, a single Pauli is set.
            If a slice, a PauliString is set over the specified range.
        value : Pauli or PauliString
            The value to set at the specified key. Must be a Pauli if key is an int,
            or a PauliString if key is a slice.

        Raises
        ------
        ValueError
            If the key and value types do not match the expected combinations.
        """
        # TODO: is it necessary to distinguish the two cases in the if... elif... loop?

        if isinstance(key, int):
            self.x_exp[key] = value.x_exp
            self.z_exp[key] = value.z_exp
            if isinstance(value, Pauli):
                self.dimensions[key] = value.dimension
            else:
                self.dimensions[key] = value.dimensions
        elif isinstance(key, slice) and not isinstance(value, Pauli):
            self.x_exp[key] = value.x_exp
            self.z_exp[key] = value.z_exp
            self.dimensions[key] = value.dimensions
        else:
            raise ValueError(f"Cannot set item with key {key} and value {value}.")

    def get_subspace(self, qudit_indices: list[int] | int) -> PauliString:
        """
        Extracts a subspace of the PauliString corresponding to the specified qudit indices.

        Parameters
        ----------
        qudit_indices : list[int] or int
            The indices of the qudits to extract. Can be a single integer or a list of integers.

        Returns
        -------
        PauliString
            A new PauliString object representing the subspace defined by the selected qudit indices.

        Examples
        --------
        >>> ps = PauliString(...)
        >>> sub_ps = ps.get_subspace([0, 2])
        """
        return PauliString(x_exp=self.x_exp[qudit_indices], z_exp=self.z_exp[qudit_indices],
                           dimensions=self.dimensions[qudit_indices])

    def copy(self) -> PauliString:
        """
        Create a deep copy of the current PauliString instance.

        Returns
        -------
        PauliString
            A new instance of PauliString with copied `x_exp`, `z_exp`, and `dimensions` attributes.
        """
        return PauliString(x_exp=self.x_exp.copy(), z_exp=self.z_exp.copy(), dimensions=self.dimensions.copy())

    def commute(self, other_pauli: PauliString) -> bool:
        """
        Determine whether this Pauli string commutes with another Pauli string.
        It is efficiently checked via the symplectic product.

        Parameters
        ----------
        other_pauli : PauliString
            The other Pauli string to check commutation with.

        Returns
        -------
        bool
            True if the two Pauli strings commute, False otherwise.

        Notes
        -----
        Two Pauli strings commute if their symplectic product is zero.
        """
        return self.symplectic_product(other_pauli) == 0

    def hermitian(self) -> PauliString:
        """
        Computes the hermitian conjugate of the Pauli string.

        Returns
        -------
        PauliString
            The hermitian conjugate of the Pauli string.
        """
        return PauliString(x_exp=(-self.x_exp) % self.dimensions,
                           z_exp=(-self.z_exp) % self.dimensions,
                           dimensions=self.dimensions)

    def is_identity(self) -> bool:
        """
        Check if the PauliString represents the identity operator.

        Returns
        -------
        bool
            True if the PauliString is the identity operator, False otherwise.
        """
        return bool(np.all(self.x_exp == 0) and np.all(self.z_exp == 0))
