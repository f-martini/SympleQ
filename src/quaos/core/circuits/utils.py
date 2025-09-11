from __future__ import annotations
import numpy as np
import scipy.sparse as sp


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


def symplectic_product_matrix(pauli_sum: np.ndarray, p: int=2) -> np.ndarray:
    m = len(pauli_sum)
    spm = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            spm[i, j] = symplectic_product(pauli_sum[i], pauli_sum[j], p)

    return spm


def symplectic_form(n: int, p: int = 2) -> np.ndarray:
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
    Omega = symplectic_form(n, p)

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


def _multi_index_to_linear(index: list[int] | np.ndarray, dims: list[int] | np.ndarray) -> int:
    """Convert a mixed-radix index to a linear index using row-major order.

    idx(i0, i1, ..., iN-1) = sum_k i_k * prod_{l>k} dims[l]
    """
    dims = list(map(int, dims))
    idx = 0
    stride = 1
    # Compute strides from right to left
    strides = [1] * len(dims)
    for k in range(len(dims) - 2, -1, -1):
        strides[k] = strides[k + 1] * dims[k + 1]
    for k, ik in enumerate(index):
        idx += int(ik) * strides[k]
    return idx


def embed_unitary(U_local: np.ndarray,
                  qudit_indices: list[int] | np.ndarray,
                  total_dimensions: list[int] | np.ndarray) -> np.ndarray:
    """
    Embed a local unitary acting on a subset of qudits into the full Hilbert space.

    - Basis ordering: |q0> ⊗ |q1> ⊗ ... ⊗ |qN-1>
    - Linear index mapping: idx(q) = sum_k q[k] * prod_{l>k} d[l]

    The local unitary is assumed to act on qudits in the order given by `qudit_indices`.

    Args:
        U_local: Local unitary of shape (D_loc, D_loc) where D_loc = prod(d[qudit_indices]).
        qudit_indices: Indices of the qudits the local unitary acts on.
        total_dimensions: Dimensions of each qudit in the full system.

    Returns:
        Full unitary of shape (D_total, D_total) with D_total = prod(total_dimensions).
    """
    qudit_indices = list(map(int, qudit_indices))
    dims = list(map(int, total_dimensions))
    N = len(dims)
    sel = qudit_indices
    rest = [k for k in range(N) if k not in sel]

    # Validate local size matches product of selected dims
    D_loc_expected = int(np.prod([dims[k] for k in sel]))
    if U_local.shape != (D_loc_expected, D_loc_expected):
        raise ValueError(
            f"U_local shape {U_local.shape} doesn't match selected dims product {D_loc_expected}"
        )

    D_rest = int(np.prod([dims[k] for k in rest]) if rest else 1)
    D_total = int(np.prod(dims))

    # Build permutation matrix P that reorders tensor factors to [sel..., rest...]
    dims_perm = [dims[k] for k in sel + rest]
    P = np.zeros((D_loc_expected * D_rest, D_total), dtype=complex)

    # Iterate over all basis states
    for q in np.ndindex(*dims):
        q = list(q)
        old_idx = _multi_index_to_linear(q, dims)
        q_perm = [q[k] for k in (sel + rest)]
        new_idx = _multi_index_to_linear(q_perm, dims_perm)
        P[new_idx, old_idx] = 1.0

    # Construct full operator: P^T (U_local ⊗ I_rest) P
    U_kron = np.kron(U_local, np.eye(D_rest, dtype=complex))
    return P.conj().T @ U_kron @ P


def left_multiply_local_unitary(U_full: np.ndarray,
                                U_local: np.ndarray,
                                qudit_indices: list[int] | np.ndarray,
                                total_dimensions: list[int] | np.ndarray) -> np.ndarray:
    """
    Left-multiply a full unitary U_full by a local unitary U_local acting
    on `qudit_indices`.

    Shapes:
      - U_full: (D_total, D_total), with D_total = prod(total_dimensions)
      - U_local: (D_sel, D_sel), with D_sel = prod(total_dimensions[i] for i in qudit_indices)

    Returns a dense ndarray of shape (D_total, D_total). Convert to sparse outside if desired.
    """
    dims = list(map(int, total_dimensions))
    N = len(dims)
    sel = list(map(int, qudit_indices))
    rest = [i for i in range(N) if i not in sel]

    if len(sel) == 0:
        return U_full

    d_sel = [dims[i] for i in sel]
    D_sel = int(np.prod(d_sel))
    if U_local.shape != (D_sel, D_sel):
        raise ValueError(
            f"U_local shape {U_local.shape} doesn't match product of selected dims {D_sel}"
        )

    # Reshape U_full to [out_dims..., in_dims...]
    U = U_full.reshape([*dims, *dims])

    # Permute output axes so selected come first: move axes in order (sel + rest) to positions 0..N-1
    perm_out = sel + rest
    U = np.moveaxis(U, perm_out, list(range(N)))

    # Now U shape is [d_sel..., d_rest..., in_dims...]
    rest_out_dims = [dims[i] for i in rest]
    U = U.reshape([D_sel, int(np.prod(rest_out_dims)) if rest_out_dims else 1, int(np.prod(dims))])

    # Apply U_local on the first axis (selected outputs): V(y, r, in) = sum_x U_local(y, x) * U(x, r, in)
    V = np.tensordot(U_local, U, axes=(1, 0))  # shape (D_sel, R, D_total)

    # Reshape back to tensor form with outputs in permuted order, then invert permutation
    V = V.reshape([*d_sel, *rest_out_dims, *dims])
    # Restore original output axis order
    V = np.moveaxis(V, list(range(N)), perm_out)

    return V.reshape(int(np.prod(dims)), int(np.prod(dims)))
