from typing import Union
import numpy as np
from numpy.typing import NDArray

ScalarType = Union[float, complex, int]
IntNDArray = NDArray[np.integer]
FloatNDArray = NDArray[np.floating]
ComplexNDArray = NDArray[np.complexfloating]

IntArrayVariant = Union[int | list[int] | IntNDArray]

FloatArrayVariant = Union[float | IntArrayVariant | list[float] | FloatNDArray]

ComplexArrayVariant = Union[ScalarType | FloatArrayVariant | list[complex] | ComplexNDArray]
