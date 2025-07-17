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


def embed_symplectic_single_pauli_string(symplectic_local, qudit_indices, n_qudits):
    """Embed a local Clifford (F, h) into full 2n space"""
    full_F = np.zeros(2 * n_qudits, dtype=int)

    n_qudits_local = len(qudit_indices)

    if 2 * n_qudits_local != len(symplectic_local):
        raise ValueError(
            f"symplectic_local must have 2 * n_qudits_local = {n_qudits_local}, but has length {len(symplectic_local)}")

    x_in = symplectic_local[:n_qudits_local]
    z_in = symplectic_local[n_qudits_local:]

    x_image = np.zeros(n_qudits, dtype=int)
    z_image = np.zeros(n_qudits, dtype=int)

    for i in range(n_qudits_local):
        x_image[qudit_indices[i]] = x_in[i]
        z_image[qudit_indices[i]] = z_in[i]

    full_F[:n_qudits] = x_image
    full_F[n_qudits:] = z_image

    return full_F


def embed_symplectic(symplectic_local, phase_vector_local, qudit_indices, n_qudits, dimension):
    """Embed a local Clifford (F, h) into full 2n space"""
    full_F = np.eye(2 * n_qudits, dtype=int)
    full_v = np.zeros(2 * n_qudits, dtype=int)

    n_loc_qudits = len(qudit_indices)
    x_in = symplectic_local[:n_loc_qudits, :]
    z_in = symplectic_local[n_loc_qudits:, :]

    for image_row_index in range(n_loc_qudits):

        full_F[image_row_index, :] = embed_symplectic_single_pauli_string(x_in[image_row_index, :],
                                                                          qudit_indices, n_qudits)
        full_F[n_qudits + image_row_index, :] = embed_symplectic_single_pauli_string(z_in[image_row_index, :],
                                                                                     qudit_indices, n_qudits)

    for i, ind in enumerate(qudit_indices):  # check
        full_v[ind] = phase_vector_local[i]

    return np.mod(full_F, dimension), np.mod(full_v, dimension)


if __name__ == '__main__':
    loc_sym = [1, 2, 3, 4]
    print(embed_symplectic_single_pauli_string(loc_sym, [1, 3], 5))

