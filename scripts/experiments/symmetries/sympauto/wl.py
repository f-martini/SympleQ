from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

def wl1_colors(S_mod: np.ndarray, p: int, coeffs: Optional[np.ndarray] = None, max_rounds: int = 10) -> np.ndarray:
    n = S_mod.shape[0]
    hist = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts = np.bincount(S_mod[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette: Dict[Tuple, int] = {}
    color = np.empty(n, dtype=np.int64)
    for i in range(n):
        coeff_key = None if coeffs is None else int(coeffs[i])
        seed_key = (coeff_key, tuple(hist[i]))
        color[i] = palette.setdefault(seed_key, len(palette))

    for _ in range(max_rounds):
        new_keys = []
        for i in range(n):
            d: Dict[Tuple[int,int], int] = {}
            row = S_mod[i]
            for j in range(n):
                key = (int(color[j]), int(row[j]))
                d[key] = d.get(key, 0) + 1
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2: Dict[Tuple, int] = {}
        new_color = np.empty(n, dtype=np.int64)
        changed = False
        for i, key in enumerate(new_keys):
            c = palette2.setdefault(key, len(palette2))
            new_color[i] = c
            changed |= (c != color[i])
        color = new_color
        if not changed:
            break
    return color

def color_classes(color: np.ndarray) -> Dict[int, List[int]]:
    classes: Dict[int, List[int]] = {}
    for i, c in enumerate(color):
        classes.setdefault(int(c), []).append(i)
    for c in classes:
        classes[c].sort()
    return classes

if __name__ == "__main__":
    # smoke
    S = np.array([[0,1],[1,0]])
    col = wl1_colors(S%2, 2, None)
    cls = color_classes(col)
    assert set(map(tuple, cls.values())) == {(0,), (1,)}
    print("[wl] ok")
