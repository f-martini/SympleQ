import os
import sys
import ctypes
import pefile

def find_pyd_module(folder, base_name="light"):
    """Find the .pyd file matching the base name."""
    for fname in os.listdir(folder):
        if fname.startswith(base_name) and fname.endswith(".pyd"):
            return os.path.join(folder, fname)
    return None

def list_imports(pyd_path):
    """List DLLs that the .pyd depends on."""
    try:
        pe = pefile.PE(pyd_path)
        imports = []
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                imports.append(entry.dll.decode())
        return imports
    except Exception as e:
        print(f"Failed to parse {pyd_path}: {e}")
        return []

def add_dll_path(folder):
    """Add folder to DLL search path."""
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(folder)
    else:
        # fallback for older Python versions
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")

def load_dll(dll_path):
    """Try to load a DLL and report errors."""
    try:
        ctypes.WinDLL(dll_path)
        print(f"Loaded: {dll_path}")
        return True
    except OSError as e:
        print(f"Failed to load: {dll_path}\n  Error: {e}")
        return False

if __name__ == "__main__":
    # Folder where your .pyd and Base.dll live
    folder = r"D:\Projects_D\QuAOS-Lab\copies\quaos_fork\src\quaos\core\light"

    # Find the .pyd
    pyd_path = find_pyd_module(folder, base_name="light")
    if not pyd_path:
        print("No light*.pyd found in folder:", folder)
        sys.exit(1)

    print("Found .pyd:", pyd_path)

    # Add folder to DLL search path
    add_dll_path(folder)

    # List and check imported DLLs
    imports = list_imports(pyd_path)
    print("Dependencies:")
    for dll in imports:
        dll_full_path = os.path.join(folder, dll)
        if os.path.exists(dll_full_path):
            print(f" - {dll} [found]")
        else:
            print(f" - {dll} [MISSING]")

    # First try loading each dependency manually
    for dll in imports:
        dll_full_path = os.path.join(folder, dll)
        if os.path.exists(dll_full_path):
            load_dll(dll_full_path)

    # Finally, try to import the Python extension
    try:
        import light
        print("Successfully imported 'light'")
    except ImportError as e:
        print("Failed to import 'light':", e)
