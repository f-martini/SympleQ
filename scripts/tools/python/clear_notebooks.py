import sys
import os
import nbformat
from pathlib import Path

ROOT_DIR = Path()
NOTEBOOKS_DIR = []


def clear_notebook_output(nb_path: str) -> None:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    changed = False

    # Clear notebook-level metadata
    if nb.get('metadata'):
        nb['metadata'] = {}
        changed = True

    for cell in nb.cells:
        if cell.cell_type == 'code':
            if cell.get('outputs'):
                cell['outputs'] = []
                changed = True
            if cell.get('execution_count') is not None:
                cell['execution_count'] = None
                changed = True
                # Clear cell-level metadata
        if cell.get('metadata'):
            cell['metadata'] = {}
            changed = True

    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Cleared outputs: {nb_path}")


def find_ipynb_files(dirs: list[Path]) -> list[str]:
    ipynb_files = []
    for d in dirs:
        for root, _, files in os.walk(str(d.as_posix())):
            for file in files:
                if file.endswith('.ipynb'):
                    ipynb_files.append(os.path.join(root, file))
    return ipynb_files


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ROOT_DIR = Path(sys.argv[1]).resolve()
    else:
        raise ValueError("Missing ROOT_DIR argument.")

    NOTEBOOKS_DIR = [
        ROOT_DIR / 'scripts' / 'examples',
        ROOT_DIR / 'scripts' / 'experiments',
        ROOT_DIR / 'scripts' / 'tools',
        ROOT_DIR / 'docs',
    ]

    for notebooks_root in NOTEBOOKS_DIR:
        ipynb_files = find_ipynb_files([notebooks_root])
        for notebook in ipynb_files:
            clear_notebook_output(notebook)
