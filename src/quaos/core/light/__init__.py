import os
os.add_dll_directory(os.path.dirname(__file__))

from .light import dot, kron

__all__ = ["dot, kron"]
