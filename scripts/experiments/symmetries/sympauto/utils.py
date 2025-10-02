from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

Label = int

def is_identity_perm_idx(pi: np.ndarray) -> bool:
    return np.array_equal(pi, np.arange(pi.size, dtype=pi.dtype))

def perm_index_to_map(pi: np.ndarray) -> Dict[int, int]:
    return {i: int(pi[i]) for i in range(pi.size)}

def discretize_coeffs(
    coeffs: Optional[np.ndarray],
    rel: float = 1e-8,
    abs_tol: float = 1e-12,
) -> Optional[np.ndarray]:
    """
    Map scalar/complex coefficients to stable integer buckets for “colour” equality (hard filter).
    """
    if coeffs is None:
        return None
    c = np.asarray(coeffs)

    def q(x: np.ndarray) -> np.ndarray:
        step = np.maximum(np.abs(x) * rel, abs_tol)
        return np.rint(x / step).astype(np.int64)

    if np.iscomplexobj(c):
        qr = q(c.real)
        qi = q(c.imag)
        return (qr.astype(np.int64) << 32) ^ (qi.astype(np.int64) & ((1 << 32) - 1))
    else:
        return q(c)

def almost_zero(z: complex, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    return abs(z) <= max(atol, rtol)

def i_pow(k: int) -> complex:
    k &= 3
    return (1+0j, 0+1j, -1+0j, 0-1j)[k]

if __name__ == "__main__":
    # smoke
    assert is_identity_perm_idx(np.arange(4))
    assert not is_identity_perm_idx(np.array([1,0,2,3]))
    v = discretize_coeffs(np.array([1.0, 1.0 + 1e-12, 2.0]))
    assert v[0] == v[1] and v[0] != v[2]
    print("[utils] ok")
