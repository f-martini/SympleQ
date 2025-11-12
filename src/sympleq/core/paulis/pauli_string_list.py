from typing import overload, Iterable
from .pauli_string import PauliString
from .pauli_sum import PauliSum


class PauliStringList(list):
    '''
    Constructor for PauliStringList class.
    This class is a wrapper used for tje list of PauliString in a PauliSum.
    It allows to update the PauliSum tableau any time a PauliString is updated.

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
    def __init__(self, data: Iterable["PauliString"], parent: 'PauliSum'):
        super().__init__(data)
        self._parent = parent

    @overload
    def __setitem__(self, key: int, value: 'PauliString') -> None:
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable['PauliString']) -> None:
        ...

    def __setitem__(self, key, value):
        """Replace one or more PauliStrings and update the parent's tableau accordingly."""
        super().__setitem__(key, value)

        if isinstance(key, int):
            self._parent._tableau[key, :] = self[key].tableau()
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            for i in indices:
                self._parent._tableau[i, :] = self[i].tableau()

    # TODO: Add append and extend methods
    # TODO: Convert to np.ndarray?
