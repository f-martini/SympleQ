from __future__ import annotations
import numpy as np


# def is_symplectic(F, p):
#     GF = galois.GF(p)
#     if isinstance(F, np.ndarray):
#         F = GF(F)
#     n = F.shape[0] // 2
#     Id = GF.Identity(n)
#     if p == 2:
#         Omega = GF.Zeros((2 * n, 2 * n))
#         Omega[:n, n:] = Id
#         Omega[n:, :n] = Id
#     else:
#         Omega = GF.Zeros((2 * n, 2 * n))
#         Omega[:n, n:] = Id
#         Omega[n:, :n] = -Id
#     lhs = F.T @ Omega @ F
#     return np.array_equal(lhs, Omega)

def is_symplectic(F, p: int) -> bool:
    """
    Check if matrix F is symplectic over GF(p).

    Args:
        F: (2n x 2n) numpy array, entries in {0, 1, ..., p-1}.
        p: prime modulus.

    Returns:
        True if F is symplectic over GF(p), False otherwise.
    """
    n = F.shape[0] // 2
    Omega = np.zeros((2 * n, 2 * n), dtype=int)
    Omega[:n, n:] = np.eye(n, dtype=int)
    Omega[n:, :n] = -np.eye(n, dtype=int)

    Omega = Omega % p

    lhs = (F.T @ Omega @ F) % p
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
    return (np.sum(u[:n] * v[n:] - u[n:] * v[:n])) % p


def symplectic_product_matrix(pauli_sum: np.ndarray) -> np.ndarray:
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
        return np.block([[Z, Id], [Id, Z]])
    else:
        return np.block([[Z, Id], [-Id, Z]])


def transvection_matrix(h: np.ndarray, p=2, multiplier=1):
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

    F_h = (np.eye(2 * n, dtype=int) + multiplier * Omega @ (np.outer(h.T, h))) % p
    return F_h


def transvection(h, x, p=2):
    return (x + symplectic_product(x, h.T, p) * h) % p


def embed_symplectic(symplectic_local, phase_vector_local, qudit_indices, n_qudits):
    """
    Embed a local Clifford (F_local, h_local) into a larger 2n-dimensional space,
    correctly handling arbitrary qudit index ordering.
    """
    m = len(qudit_indices)
    if symplectic_local.shape != (2 * m, 2 * m):
        raise ValueError("symplectic_local must be 2m x 2m")
    if len(phase_vector_local) != 2 * m:
        raise ValueError("phase_vector_local must have length 2m")

    # Full 2n x 2n identity
    F_full = np.eye(2 * n_qudits, dtype=int)
    h_full = np.zeros(2 * n_qudits, dtype=int)

    qudit_indices = np.array(qudit_indices, dtype=int)

    # Build row/column index mapping for the full space
    # First X rows/columns
    row_indices = np.concatenate([qudit_indices, n_qudits + qudit_indices])
    col_indices = np.concatenate([qudit_indices, n_qudits + qudit_indices])

    # Place the full local symplectic block into the full system
    F_full[np.ix_(row_indices, col_indices)] = symplectic_local

    # Embed phase vector
    h_full[row_indices] = phase_vector_local

    return F_full, h_full
