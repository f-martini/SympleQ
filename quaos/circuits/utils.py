# TODO: Refine quaos.circuit_utils and move here
from math import gcd
import numpy as np


def solve_modular_linear(x: int, z: int, d: int) -> int:
    """
    Find smallest non-negative integer n such that (x + z*n) % d == 0.
    Returns None if no solution exists
    """
    x, z, d = int(x), int(z), int(d)
    g = gcd(z, d)
    if x % g != 0:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    # Reduce the equation modulo d // g
    z_ = z // g
    d_ = d // g
    x_ = (-x // g) % d_

    # compute modular inverse
    try:
        z_inv = pow(z_, -1, d_)
    except ValueError:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    n = (x_ * z_inv) % d_
    return n


def is_symplectic(symplectic: np.ndarray, dimension: int = 2) -> bool:
    """
    Check if the gate is symplectic, i.e., it preserves the symplectic structure of the Pauli group.
    """
    
    n_q = symplectic.shape[0] // 2

    id = np.eye(n_q, dtype=int)
    J = np.zeros([2 * n_q, 2 * n_q], dtype=int)
    J[n_q:, :n_q] = id
    J[:n_q, n_q:] = id

    return bool(np.all(J == symplectic.T @ J @ symplectic % dimension))


if __name__ == "__main__":
    # Example usage
    symplectic_matrix = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
    print(is_symplectic(symplectic_matrix))
