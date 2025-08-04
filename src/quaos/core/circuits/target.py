"""Codes for finding target Paulis and gates which map a given Pauli to a target Pauli."""
from quaos.core.paulis import PauliString, PauliSum
import numpy as np
from collections import defaultdict
from itertools import product
import galois


def find_map_to_target_pauli_sum(input_pauli: PauliSum, target_pauli: PauliSum) -> tuple[np.ndarray, np.ndarray,
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
        # if input_pauli[:, i] != target_pauli[:, i]:
        qudit_indices.append(i)
    gate_dimension = input_pauli.dimensions[qudit_indices][0]

    if not np.all(input_pauli.dimensions[qudit_indices] == gate_dimension):
        raise ValueError("PauliSum must have the same dimension for all qudits acted upon by the gate.")

    if np.all(input_pauli.symplectic_product_matrix() != target_pauli.symplectic_product_matrix()):
        raise ValueError("Input and target PauliSum must be symplectically equivalent.")

    input_symplectic = input_pauli.symplectic()  #[:, qudit_indices]
    target_symplectic = target_pauli.symplectic()  #[:, qudit_indices]

    F = find_symplectic_map(input_symplectic, target_symplectic)

    # print('IN FUNCTION')
    # print(input_symplectic)
    # print(target_symplectic)
    # print(input_symplectic @ C % 2)
    # print('----------')

    h = get_phase_vector(F, gate_dimension)

    return F, h, qudit_indices, gate_dimension


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

    underdetermined_pauli_indices = [combined_indices[indexes[0]] for indexes in index_dict.values() if len(indexes) > 1]
    underdetermined_pauli_options = []
    for indices in underdetermined_pauli_indices:
        options = []
        for i in range(len(target_pauli_list)):
            if target_pauli_list[i][1] == indices[0] and target_pauli_list[i][2] == indices[1]:
                if target_pauli_list[i][0] not in options:
                    options.append(target_pauli_list[i][0])
        underdetermined_pauli_options.append(options)
    underdetermined_pauli_options = [str_to_int(upo) for upo in underdetermined_pauli_options]

    determined_pauli_indices = [combined_indices[indexes[0]] for indexes in index_dict.values() if len(indexes) == 1]

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


# Symplectic Map Finding utils below #


def symp_inner_product(x, y):
    """Symplectic inner product over GF(2) for single vectors"""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n = x.shape[1] // 2
    J = np.block([[np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
                  [np.eye(n, dtype=np.uint8), np.zeros((n, n), dtype=np.uint8)]])
    return (x @ J @ y.T) % 2


def shift_columns(A):
    A = np.asarray(A, dtype=np.uint8)
    n = A.shape[1]
    shift = (n + 1) // 2
    return np.roll(A, -shift, axis=1)


def solve_gf2_linear_eq(A, b):
    """Solve A x = b over GF(2) using Gaussian elimination."""
    A = A.copy().astype(int) % 2
    b = b.copy().reshape(-1, 1).astype(int) % 2
    m, n = A.shape
    Ab = np.hstack([A, b])
    row = 0
    pivots = []

    for col in range(n):
        pivot_rows = np.where(Ab[row:, col])[0]
        if pivot_rows.size == 0:
            continue
        pivot_row = pivot_rows[0] + row
        if pivot_row != row:
            Ab[[row, pivot_row]] = Ab[[pivot_row, row]]
        pivots.append(col)
        for r in range(m):
            if r != row and Ab[r, col]:
                Ab[r] ^= Ab[row]
        row += 1
        if row == m:
            break

    x = np.zeros(n, dtype=int)
    for i, col in enumerate(pivots):
        x[col] = Ab[i, -1]
    return x


def solve_gf_linear(A, b, p):
    GF = galois.GF(p)
    A_gf = GF(A)
    b_gf = GF(b)

    # Solve A x = b over GF(p)
    try:
        x = np.linalg.solve(A_gf, b_gf)
    except np.linalg.LinAlgError:
        return None  # No solution

    return x


def find_symplectic_map(X, Y):
    m, d2 = X.shape
    n = d2 // 2
    F = np.eye(2 * n, dtype=int)

    def Z_h(h):
        h_shift = shift_columns(h.reshape(1, -1))[0]
        return (np.eye(2 * n, dtype=np.uint8) + np.outer(h_shift, h)) % 2

    def find_w(x, y, Ys):
        if Ys.shape[0] == 0:
            A = shift_columns(np.vstack([x, y]))
            b = np.array([1, 1], dtype=np.uint8)
        else:
            A = shift_columns(np.vstack([x, y, Ys]))
            y_reps = np.tile(y, (Ys.shape[0], 1))
            b = np.concatenate((
                [1, 1],
                symp_inner_product(Ys, y_reps).diagonal()
            ))
        return solve_gf_linear(A, b, 2)

    for i in range(m):
        x_i = X[i]
        y_i = Y[i]
        x_it = (x_i @ F) % 2
        if np.array_equal(x_it, y_i):
            continue
        elif symp_inner_product(x_it, y_i) % 2 == 1:
            h_i = (x_it + y_i) % 2
            F = (F @ Z_h(h_i)) % 2
        else:
            Ys = Y[:i]
            w_i = find_w(x_it, y_i, Ys)
            h1 = (w_i + y_i) % 2
            h2 = (x_it + w_i) % 2
            F = (F @ Z_h(h1) @ Z_h(h2)) % 2

    return F


def make_random_symplectic(n, steps=5, seed=None):

    DeprecationWarning('make_random_symplectic is deprecated. Use quaos.circuits.utils.random_symplectic instead.')

    rng = np.random.default_rng(seed)
    F = np.eye(2 * n, dtype=np.uint8)
    for _ in range(steps):
        h = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
        h_shifted = shift_columns(h.reshape(1, -1))[0]
        Z = (np.eye(2 * n, dtype=np.uint8) + np.outer(h_shifted, h)) % 2
        F = (F @ Z) % 2
    return F


def symp_inn_pdt(X, Y):
    """
    Compute the symplectic inner product over GF(2) between
    corresponding rows of X and Y.
    """
    X = GF(X)
    Y = GF(np.fft.fftshift(Y, axes=1))
    return np.sum(X * Y, axis=1)  # Already in GF(2), no mod needed


def find_symp_mat(X, Y, p=2):
    """
    Find a binary symplectic matrix F such that X @ F = Y over GF(2).
    """
    GF = galois.GF(p)

    X = GF(X)
    Y = GF(Y)
    m, N = X.shape
    n = N // 2
    F = GF.Identity(2 * n)

    def Z_h(h):
        h = h.reshape(1, -1)
        Z = GF(np.eye(2 * n, dtype=int)) + (h.T @ h)
        return Z

    for i in range(m):
        x_i = X[i, :]
        y_i = Y[i, :]
        x_it = x_i @ F

        if np.array_equal(x_it, y_i):
            continue

        if symp_inn_pdt(x_it.reshape(1, -1), y_i.reshape(1, -1))[0] == 1:
            h_i = (x_it + y_i)
            F = (F @ Z_h(h_i))
        else:
            try:
                w_i = find_w(x_it, y_i, Y[:i, :])
            except ValueError:
                w_i = find_w_fallback(x_it, y_i, Y[:i, :])
            h_i1 = (w_i + y_i)
            h_i2 = (x_it + w_i)
            F = ((F @ Z_h(h_i1))) @ Z_h(h_i2)

    return np.array(F)


def find_w(x, y, Ys):
    """
    Find a solution w over GF(2) such that:
        ⟨x, w⟩ = 1
        ⟨w, y⟩ = 1
        ⟨Ys[i], w⟩ = ⟨y, Ys[i]⟩ for all i
    """
    GF2 = galois.GF(2)
    k = Ys.shape[0]

    A = np.vstack([x, y, Ys])
    A = np.fft.fftshift(A, axes=1)
    b = np.concatenate([
        [1],
        [1],
        symp_inn_pdt(Ys, np.tile(y, (k, 1)))
    ])

    A = GF2(A)
    b = GF2(b)

    # Solve using custom Gaussian elimination
    w = solve_gf2_system(A, b)
    if w is None:
        raise ValueError("No solution found for find_w.")

    return w


def solve_gf2_system(A, b):
    """
    Solves A w = b over GF(2) using RREF. Returns one solution if it exists, else None.
    """
    GF2 = galois.GF(2)
    A = GF2(A.copy())
    b = GF2(b.copy())
    m, n = A.shape

    Ab = np.hstack([A, b.reshape(-1, 1)])
    Ab = Ab.copy()

    row = 0
    pivots = []

    for col in range(n):
        # Find a pivot in column `col` at or below row
        pivot_row = None
        for r in range(row, m):
            if Ab[r, col] == 1:
                pivot_row = r
                break

        if pivot_row is None:
            continue  # No pivot in this column

        # Swap rows
        if pivot_row != row:
            Ab[[row, pivot_row], :] = Ab[[pivot_row, row], :]

        # Eliminate below and above
        for r in range(m):
            if r != row and Ab[r, col] == 1:
                Ab[r, :] += Ab[row, :]

        pivots.append(col)
        row += 1

    # Check for inconsistency
    for r in range(m):
        if np.all(Ab[r, :-1] == 0) and Ab[r, -1] != 0:
            return None  # Inconsistent system

    # Back-substitution to extract one solution
    x = np.zeros(n, dtype=int)
    for i, col in enumerate(pivots):
        rhs = Ab[i, -1]
        s = sum((Ab[i, j] * x[j] for j in range(col + 1, n)), start=GF2(0))
        x[col] = int(rhs - s)

    return GF2(x)


def find_w_fallback(x, y, Ys, max_attempts=10000):
    """
    Try to find w satisfying:
        ⟨x, w⟩ = ⟨w, y⟩ = 1
        ⟨w, y_j⟩ = ⟨y, y_j⟩ for all y_j in Ys
    """
    GF2 = galois.GF(2)
    y = GF2(y)
    Ys = GF2(Ys)
    y_inn = symp_inn_pdt(Ys, np.tile(y, (Ys.shape[0], 1)))

    for _ in range(max_attempts):
        w = GF2.Random(x.size)
        if (int(symp_inn_pdt(x.reshape(1, -1), w.reshape(1, -1))[0]) != 1):
            continue
        if (int(symp_inn_pdt(w.reshape(1, -1), y.reshape(1, -1))[0]) != 1):
            continue
        if Ys.shape[0] > 0:
            if not np.array_equal(symp_inn_pdt(Ys, w.reshape(1, -1)), y_inn):
                continue
        return w

    raise ValueError("No solution found for find_w (even by random search).")


def is_symplectic(F):
    """
    Check if F is symplectic over GF(2):
        Fᵀ J F = J
    """
    n = F.shape[0] // 2
    J = np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])
    F = GF(F)
    J = GF(J)
    return np.array_equal(F.T @ J @ F, J)


if __name__ == "__main__":
    from quaos.core.circuits.utils import random_symplectic
    GF = galois.GF(2)

    def test_find_symp_mat(num_tests=10, n=4):
        passed = 0
        for i in range(num_tests):
            F_true = random_symplectic(n, p=2)  # returns 2n x 2n binary symplectic matrix
            F_true = GF(F_true)

            # Sample m random input rows X
            m = np.random.randint(1, 2 * n + 1)
            X = GF(np.random.randint(0, 2, size=(m, 2 * n)))
            Y = X @ F_true

            # Try to recover F
            F_recovered = find_symp_mat(np.array(X), np.array(Y))
            F_recovered = GF(F_recovered)

            # Check that X @ F_recovered == Y
            match = np.array_equal(X @ F_recovered, Y)

            # Optional: Check that F_recovered is symplectic
            symplectic = is_symplectic(F_recovered)

            if match and symplectic:
                passed += 1
            else:
                print(f"Test {i} failed.")
                print(f"X @ F_recovered =\n{X @ F_recovered}")
                print(f"Y =\n{Y}")
                print(f"F_recovered symplectic: {symplectic}")

        print(f"Passed {passed}/{num_tests} tests.")

    test_find_symp_mat()
