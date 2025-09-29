import os
import ctypes
import glob

lib_dir = os.path.dirname(__file__)
if os.name == "nt":
    os.add_dll_directory(lib_dir)
else:
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    for so_file in glob.glob(os.path.join(lib_dir, "*.so")):
        ctypes.CDLL(so_file)

from .light import Aquire

__all__ = ["Aquire"]
