# TODO: Refine quaos.circuit_utils and move here
import numpy as np
import galois


def is_symplectic(F, p):
    GF = galois.GF(p)
    n = F.shape[0] // 2
    Id = GF.Identity(n)
    if p == 2:
        Omega = GF.Zeros((2 * n, 2 * n))
        Omega[:n, n:] = Id
        Omega[n:, :n] = Id
    else:
        Omega = GF.Zeros((2 * n, 2 * n))
        Omega[:n, n:] = Id
        Omega[n:, :n] = -Id
    lhs = F.T @ Omega @ F
    return np.array_equal(lhs, Omega)


def symplectic_product(u: np.ndarray, v: np.ndarray, p: int = 2) -> int:
    """
    Compute the symplectic inner product of two binary vectors.

    Args:
        u, v: Binary vectors of length 2n

    Returns:
        Symplectic inner product modulo p
    """
    n = len(u) // 2
    return (np.sum(u[:n] * v[n:] + u[n:] * v[:n])) % p


def symplectic_product_matrix(pauli_sum):
    m = len(pauli_sum)
    spm = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            spm[i, j] = symplectic_product(pauli_sum[i], pauli_sum[j])

    return spm


def construct_omega(n: int, p: int = 2) -> np.ndarray:
    """
    Construct the symplectic matrix Omega for a given dimension n over GF(p).

    Args:
        n: Half the length of the vectors (2n is the full length)
        p: Modulus (default 2)

    Returns:
        Omega matrix as a 2n x 2n numpy array
    """
    Id = np.eye(n, dtype=int)
    Z = np.zeros((n, n), dtype=int)

    if p == 2:
        return np.block([[Z, Id], [-Id, Z]])
    else:
        return np.block([[Z, Id], [Id, Z]])


def transvection_matrix(h, p=2):
    """
    Compute the transvection matrix corresponding to the vector h.

    Args:
        h: Binary vector of length 2n
        p: Modulus (default 2)

    Returns:
        The transvection matrix as a 2n x 2n matrix over integers modulo p
    """
    n = len(h) // 2
    Omega = construct_omega(n, p)

    F_h = (np.eye(2 * n, dtype=int) + Omega @ (np.outer(h.T, h))) % p
    return F_h


def transvection(h, x, p=2):
    return (x + symplectic_product(x, h.T, p) * h) % p


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
