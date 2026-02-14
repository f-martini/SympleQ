from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')


class Converter(ABC, Generic[T, U]):
    """Abstract base for item transformations."""

    def __call__(self, source: T) -> U:
        return self._convert(source)

    @abstractmethod
    def _convert(self, source: T) -> U:
        ...

    @abstractmethod
    def can_convert(self, source: T) -> bool:
        return False
