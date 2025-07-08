"""Codes for finding target Paulis and gates which map a given Pauli to a target Pauli."""
from quaos.paulis import PauliString, PauliSum
import numpy as np
from collections import defaultdict
from itertools import product
from sympy import Matrix


def find_map_to_target_pauli_sum(input_pauli: PauliSum, target_pauli: PauliSum) -> tuple[list[np.ndarray], np.ndarray,
                                                                                         list[int], int]:
    """
    Find a gate that maps Pauli P to target Pauli.

    Args:
        P (Pauli): The Pauli to be mapped.
        target (Pauli): The target Pauli.
        dimension (int): The dimension of the qudit.

    Returns:
        images (list[np.ndarray]): The images of the gate.
        h (np.ndarray): The phase vector of the gate.
        qudit_indices (list[int]): The indices of the qudits acted upon by the gate.
        gate_dimension (int): The dimension of the gate.
        """
    if np.all(input_pauli.dimensions != target_pauli.dimensions):
        raise ValueError("PauliSum and gate must have the same dimension.")

    n_qudits = input_pauli.n_qudits()
    if n_qudits != target_pauli.n_qudits():
        raise ValueError("PauliSum and target must have the same number of qudits.")

    # get list of qudits where input and target differ
    qudit_indices = []
    for i in range(n_qudits):
        if input_pauli[:, i] != target_pauli[:, i]:
            qudit_indices.append(i)
    gate_dimension = input_pauli.dimensions[qudit_indices][0]

    if not np.all(input_pauli.dimensions[qudit_indices] == gate_dimension):
        raise ValueError("PauliSum must have the same dimension for all qudits acted upon by the gate.")

    if np.all(input_pauli.symplectic_product_matrix() != target_pauli.symplectic_product_matrix()):
        raise ValueError("Input and target PauliSum must be symplectically equivalent.")

    input_symplectic = input_pauli.symplectic()
    target_symplectic = target_pauli.symplectic()

    C = find_symplectic_map(input_symplectic, target_symplectic)
    images = []
    for i in range(2 * len(qudit_indices)):
        basis_element = np.zeros(2 * len(qudit_indices), dtype=int)
        basis_element[i] = 1
        image = (C @ basis_element) % gate_dimension
        images.append(image)

    h = get_phase_vector(C, gate_dimension)

    return images, h, qudit_indices, gate_dimension


def find_allowed_target(pauli_sum, target_pauli_list):
    pauli_list = [target_pauli_list[i][0] for i in range(len(target_pauli_list))]
    pauli_list = str_to_int(pauli_list)
    string_indices = [target_pauli_list[i][1] for i in range(len(target_pauli_list))]
    qudit_indices = [target_pauli_list[i][2] for i in range(len(target_pauli_list))]

    dims = pauli_sum.dimensions
    combined_indices = list(zip(string_indices, qudit_indices))
    index_dict = defaultdict(list)

    for idx, tup in enumerate(combined_indices):
        index_dict[tup].append(idx)

    underdetermined_pauli_indices = [combined_indices[idxs[0]] for idxs in index_dict.values() if len(idxs) > 1]
    underdetermined_pauli_options = []
    for indices in underdetermined_pauli_indices:
        options = []
        for i in range(len(target_pauli_list)):
            if target_pauli_list[i][1] == indices[0] and target_pauli_list[i][2] == indices[1]:
                if target_pauli_list[i][0] not in options:
                    options.append(target_pauli_list[i][0])
        underdetermined_pauli_options.append(options)
    underdetermined_pauli_options = [str_to_int(l) for l in underdetermined_pauli_options]

    determined_pauli_indices = [combined_indices[idxs[0]] for idxs in index_dict.values() if len(idxs) == 1]

    options_matrix = np.empty([pauli_sum.n_paulis(), pauli_sum.n_qudits()], dtype=object)
    for i in range(pauli_sum.n_paulis()):
        for j in range(pauli_sum.n_qudits()):
            if (i, j) in underdetermined_pauli_indices:
                options_matrix[i, j] = underdetermined_pauli_options[underdetermined_pauli_indices.index((i, j))]
            elif (i, j) in determined_pauli_indices:
                options_matrix[i, j] = [pauli_list[determined_pauli_indices.index((i, j))]]
            else:
                options_matrix[i, j] = [0, 1, 2, 3]

    flag_matrix = np.zeros([pauli_sum.n_paulis(), pauli_sum.n_qudits()], dtype=int)  # flag matrix to track which indices are determined
    # 0 for not determined, 1 for underdetermined, -1 for determined
    for idx, tup in enumerate(combined_indices):
        if tup in underdetermined_pauli_indices:
            flag_matrix[tup[0], tup[1]] = -1  # -1 for underdetermined
        else:
            flag_matrix[tup[0], tup[1]] = 1  # 1 for determined

    spm = pauli_sum.symplectic_product_matrix()

    possible_targets = []
    for combo in product(*options_matrix.flatten()):
        combo = np.reshape(combo, (pauli_sum.n_paulis(), pauli_sum.n_qudits()))
        pauli_sum_candidate = matrix_to_pauli_sum(combo, pauli_sum.weights, pauli_sum.phases, dims)
        candidate_spm = pauli_sum_candidate.symplectic_product_matrix()

        if np.all(spm == candidate_spm):
            possible_targets.append(pauli_sum_candidate)
    return possible_targets


def get_phase_vector(gate_symplectic: np.ndarray, dimension: int) -> np.ndarray:
    """
    Calculate the phase vector for a gate given its symplectic matrix.

    See PRA 71, 042315 (2005) Eq. (10).
    Solves for h

    Args:
        gate_symplectic (np.ndarray): The symplectic matrix of the gate.
        dimension (int): The dimension of the qudit.

    Returns:
        np.ndarray: The phase vector of the gate.
    """
    n_qudits = gate_symplectic.shape[0] // 2

    U = np.zeros((2 * n_qudits, 2 * n_qudits), dtype=int)
    U[n_qudits:, :n_qudits] = np.eye(n_qudits, dtype=int)
    lhs = (dimension - 1) * np.diag(gate_symplectic.T @ U @ gate_symplectic) % 2
    return lhs


def str_to_int(string):
    output = []

    for s in string:
        if s == 'I':
            output.append(0)
        elif s == 'X':
            output.append(1)
        elif s == 'Y':
            output.append(2)
        elif s == 'Z':
            output.append(3)
    return output


def int_to_pauli(integer):
    if integer == 0:
        return [0, 0]
    elif integer == 1:
        return [1, 0]
    elif integer == 2:
        return [1, 1]
    elif integer == 3:
        return [0, 1]
    else:
        raise ValueError(f"Invalid integer for Pauli representation: {integer}. Must be in [0, 3].")


def matrix_to_pauli_sum(matrix, weights, phases, dimensions):

    n_paulis = matrix.shape[0]
    n_qudits = matrix.shape[1]

    pauli_strings = []
    for i in range(n_paulis):
        x_exp = np.zeros(n_qudits, dtype=int)
        z_exp = np.zeros(n_qudits, dtype=int)
        for j in range(n_qudits):
            x_exp[j] = int_to_pauli(matrix[i, j])[0]
            z_exp[j] = int_to_pauli(matrix[i, j])[1]
        ps = PauliString(x_exp, z_exp, dimensions)
        pauli_strings.append(ps)

    return PauliSum(pauli_strings, weights, phases, dimensions, standardise=False)



def symplectic_inner_product(u, v):
    """Compute symplectic inner product ⟨u, v⟩ over GF(2)."""
    n = len(u) // 2
    return (np.dot(u[:n], v[n:]) + np.dot(u[n:], v[:n])) % 2


def fftshift(x, axis=-1):
    x = np.asarray(x)
    n = x.shape[axis]
    return np.roll(x, shift=n // 2, axis=axis)


def gf2_solve(A, b):
    """Solve Ax = b over GF(2)."""
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    aug = np.hstack([A, b.reshape(-1, 1)]).astype(np.uint8)

    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1

    x = np.zeros(n, dtype=np.uint8)
    for i in range(min(m, n)):
        pivot_cols = np.flatnonzero(aug[i, :n])
        if len(pivot_cols) == 1:
            x[pivot_cols[0]] = aug[i, -1]
    return x


def find_symplectic_map(H, H_prime):
    """
    Find symplectic matrix C such that H_prime = C @ H over GF(2).
    """
    m, dim = H.shape
    n = dim // 2
    C = np.eye(2 * n, dtype=int)

    def Z_h(h):
        """Right-multiplied symplectic transvection."""
        h = h % 2
        outer = np.outer(h, fftshift(h)) % 2
        return (np.eye(2 * n, dtype=int) + outer) % 2

    def find_w(x, y, Ys):
        A = fftshift(np.vstack([x, y, Ys]), axis=1)
        b = np.concatenate([[1], [1], [symplectic_inner_product(y_j, y) for y_j in Ys]])
        return gf2_solve(A, b)

    for i in range(m):
        x_i = H[i]
        y_i = H_prime[i]
        x_it = (C @ x_i) % 2

        if np.array_equal(x_it, y_i):
            continue
        if symplectic_inner_product(x_it, y_i) == 1:
            h_i = (x_it + y_i) % 2
            C = (Z_h(h_i) @ C) % 2
        else:
            Ys_prev = H_prime[:i]
            w_i = find_w(x_it, y_i, Ys_prev)
            h_i1 = (w_i + y_i) % 2
            h_i2 = (x_it + w_i) % 2
            C = (Z_h(h_i1) @ C) % 2
            C = (Z_h(h_i2) @ C) % 2

    return C
