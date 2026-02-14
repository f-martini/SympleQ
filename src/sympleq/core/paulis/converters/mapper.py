from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')


class Mapper(ABC, Generic[T, U]):
    """Abstract base for mappers."""
    @abstractmethod
    def map(self, x): ...
