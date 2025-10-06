import os
import ctypes
import glob

lib_dir = os.path.dirname(__file__)
deps_dir = lib_dir + "\\.libs"
if os.name == "nt":
    os.add_dll_directory(lib_dir)
    os.add_dll_directory(deps_dir)
else:
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + deps_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    for so_file in glob.glob(os.path.join(lib_dir, "*.so")) + glob.glob(os.path.join(deps_dir, "*.so")):
        ctypes.CDLL(so_file)

from .light import Aquire

__all__ = ["Aquire"]
