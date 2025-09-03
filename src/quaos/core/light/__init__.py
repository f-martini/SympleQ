import os
os.add_dll_directory(os.path.dirname(__file__))

from .light import cuda_add

__all__ = ["cuda_add"]
