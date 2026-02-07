import numpy as np
from .constants import DEFAULT_QUDIT_DIMENSION
from .typing import IntNDArray, FloatNDArray, ComplexNDArray, IntArrayVariant, IntMatrixVariant, ComplexArrayVariant


class PauliNumpyFormatter:
    @staticmethod
    def get_phases(phases: IntArrayVariant | None, lcm: int, n_pauli_strings: int) -> IntNDArray:
        if phases is None:
            phases = np.zeros(n_pauli_strings, dtype=int)
        else:  # Catches scalars but also list and arrays of length 1
            phases = np.asarray(phases, dtype=int)
            if phases.ndim == 0:
                phases = np.full(n_pauli_strings, phases.item(), dtype=int)

        return phases % (2 * lcm)

    @staticmethod
    def get_weights(weights: ComplexArrayVariant | None, n_pauli_strings: int) -> ComplexNDArray:
        if weights is None:
            weights = np.ones(n_pauli_strings, dtype=np.complex128)
        else:
            # Catches scalars but also list and arrays of length 1
            weights = np.asarray(weights, dtype=np.complex128)
            if weights.ndim == 0:
                weights = np.full(n_pauli_strings, weights.item(), dtype=np.complex128)

        return weights

    @staticmethod
    def get_tableau(tableau: IntNDArray) -> IntNDArray:
        if tableau.ndim == 1:
            tableau = tableau.reshape(1, -1)
        elif tableau.ndim != 2:
            raise ValueError(f"Invalid tableau shape ({tableau.shape}). Tableaus should be two dimensional.")

        return tableau

    @staticmethod
    def get_dimensions(dimensions: IntArrayVariant | None, n_qudits: int) -> IntNDArray:
        if dimensions is None:
            dimensions = np.ones(n_qudits, dtype=int) * DEFAULT_QUDIT_DIMENSION
        else:  # Catches int but also list and arrays of length 1
            dimensions = np.asarray(dimensions, dtype=int)
            if dimensions.ndim == 0:
                dimensions = np.full(n_qudits, dimensions.item(), dtype=int)

        return dimensions
