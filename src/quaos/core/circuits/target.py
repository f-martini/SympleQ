"""Codes for finding target Paulis and gates which map a given Pauli to a target Pauli."""
from quaos.core.paulis import PauliString, PauliSum
import numpy as np
from collections import defaultdict
from itertools import product, combinations
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


########## Symplectic Map Finding ##########



def is_symplectic(F):
    """Check if F is symplectic over GF(2)."""
    n = F.shape[0] // 2
    Omega = np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, n, dtype=int), np.zeros((n, n), dtype=int)]
    ])
    return np.array_equal((F.T @ Omega @ F) % 2, Omega)


def random_symplectic(n, seed=None):
    """Generate a random symplectic matrix over GF(2) of size 2n x 2n."""
    if seed is not None:
        np.random.seed(seed)

    while True:
        A = np.random.randint(0, 2, size=(n, n))
        if np.linalg.matrix_rank(A) < n:
            continue

        B = np.random.randint(0, 2, size=(n, n))
        B = (B + B.T) % 2  # Force symmetry

        C = np.random.randint(0, 2, size=(n, n))
        try:
            AinvT = np.linalg.inv(A.T) % 2
        except np.linalg.LinAlgError:
            continue

        D = (AinvT @ (C.T @ A + B)) % 2
        F = np.block([[A, B], [C, D]]) % 2

        if is_symplectic(F):
            return F.astype(int)


def symplectic_inner_product(u, v, d=2):
    """Symplectic inner product over Z_d."""
    n = len(u) // 2
    u = np.array(u) % d
    v = np.array(v) % d
    top = np.dot(u[:n], v[n:]) % d
    bottom = np.dot(u[n:], v[:n]) % d
    return (top - bottom) % d

def gram_schmidt_symplectic(B, d=2):
    """
    Given linearly independent rows B (k x 2n), complete to full symplectic basis (2n x 2n).
    """
    B = np.array(B, dtype=int) % d
    k, dim = B.shape
    n = dim // 2
    assert dim % 2 == 0, "Input must have even number of columns"

    basis = list(B)
    symp_duals = []

    # Construct symplectic duals for each vector in B
    for i, b in enumerate(B):
        # Find w such that <b, w> = 1 and <b_j, w> = 0 for all j < i
        for trial in range(1 << dim):
            w = np.array([int(x) for x in format(trial, f'0{dim}b')], dtype=int)
            if symplectic_inner_product(b, w, d) != 1:
                continue
            if all(symplectic_inner_product(basis[j], w, d) == 0 for j in range(i)):
                symp_duals.append(w)
                break
        else:
            raise ValueError("Could not find symplectic dual for vector")

    basis.extend(symp_duals)

    # Add arbitrary symplectic pairs to fill remaining dimension
    while len(basis) < dim:
        for trial in range(1 << dim):
            v = np.array([int(x) for x in format(trial, f'0{dim}b')], dtype=int)
            if all(symplectic_inner_product(v, b, d) == 0 for b in basis):
                # Find a dual for v
                for dual_trial in range(1 << dim):
                    w = np.array([int(x) for x in format(dual_trial, f'0{dim}b')], dtype=int)
                    if symplectic_inner_product(v, w, d) == 1 and all(
                        symplectic_inner_product(w, b, d) == 0 for b in basis
                    ):
                        basis.append(v)
                        basis.append(w)
                        break
                break
        else:
            raise ValueError("Could not complete basis")

    return np.array(basis[:dim]) % d


def find_symplectic_map(H, H_prime):
    """Recover symplectic matrix C from H, H_prime."""
    H = np.array(H, dtype=int) % 2
    H_prime = np.array(H_prime, dtype=int) % 2

    F1 = gram_schmidt_symplectic(H)
    F2 = gram_schmidt_symplectic(H_prime)

    F1_inv = Matrix(F1.tolist()).inv_mod(2)
    C = (F2 @ np.array(F1_inv.tolist())) % 2

    assert is_symplectic(C), f"{H}\n{H_prime}\n{C}\n{F1}\n{F2}\n{F1_inv}"
    return C

