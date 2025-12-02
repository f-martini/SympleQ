import numpy as np
from typing import Dict, List, Optional, Tuple


def _wl_colors_from_S(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray] = None,
    col_invariants: Optional[np.ndarray] = None,  # shape (n, t) ints; optional extras for seeding
    max_rounds: int = 10
) -> np.ndarray:
    """
    1-WL color refinement on the complete edge-colored graph with edge color S[i,j] in GF(p).
    Seed key: (coeff[i], col_invariants[i,*], row-histogram-of-S[i,*]).
    This is a safe isomorphism invariant; we use it as a *base* partition (hard constraint).
    """
    n = S_mod.shape[0]
    hist = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts = np.bincount(S_mod[i], minlength=p)
        hist[i, :p] = counts[:p]

    palette = {}
    color = np.empty(n, dtype=np.int64)
    for i in range(n):
        coeff_key = None if coeffs is None else (coeffs[i].item() if hasattr(coeffs[i], "item") else coeffs[i])
        inv_key = () if col_invariants is None else tuple(int(x) for x in np.atleast_1d(col_invariants[i]))
        seed_key = (coeff_key, inv_key, tuple(hist[i]))
        color[i] = palette.setdefault(seed_key, len(palette))

    for _ in range(max_rounds):
        new_keys = []
        # count pairs (neighbor_color, edge_value)
        for i in range(n):
            d = {}
            row = S_mod[i]
            for j in range(n):
                key = (int(color[j]), int(row[j]))
                d[key] = d.get(key, 0) + 1
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2 = {}
        new_color = np.empty(n, dtype=np.int64)
        changed = False
        for i, key in enumerate(new_keys):
            c = palette2.setdefault(key, len(palette2))
            new_color[i] = c
            if c != color[i]:
                changed = True
        color = new_color
        if not changed:
            break
    return color


def _color_classes(color: np.ndarray) -> Dict[int, List[int]]:
    classes: Dict[int, List[int]] = {}
    for i, c in enumerate(color):
        classes.setdefault(int(c), []).append(i)
    for c in classes:
        classes[c].sort()
    return classes


def _build_base_partition(
    S_mod: np.ndarray,
    p: int,
    *,
    coeffs: Optional[np.ndarray],
    col_invariants: Optional[np.ndarray],
    max_rounds: int = 10,
    color_mode: str = "wl",      # "wl" | "coeffs_only" | "none"
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Build the base colors & classes:
      - "wl":          WL-1 on S (optionally seeded with coeffs/invariants)
      - "coeffs_only": colors = coefficient IDs only (strict weight-preservation, no WL)
      - "none":        everyone in one color (true brute-force; only weights + S-consistency prune)
    """
    n = S_mod.shape[0]

    if color_mode == "none":
        base_colors = np.zeros(n, dtype=np.int64)

    elif color_mode == "coeffs_only":
        if coeffs is None:
            base_colors = np.zeros(n, dtype=np.int64)
        else:
            # compress coeffs to stable int IDs
            _, inv = np.unique(np.asarray(coeffs), return_inverse=True)
            base_colors = inv.astype(np.int64, copy=False)

    elif color_mode == "wl":
        base_colors = _wl_colors_from_S(
            S_mod, p, coeffs=coeffs, col_invariants=col_invariants, max_rounds=max_rounds
        )
    else:
        raise ValueError("color_mode must be 'wl', 'coeffs_only', or 'none'.")

    return base_colors, _color_classes(base_colors)
