from typing import Any
import numpy as np
import re
from .pauli import Pauli
from .pauli_string import PauliString
from .pauli_sum import PauliSum
import networkx as nx


def to_pauli_sum(P: Pauli | PauliString) -> PauliSum:
    if isinstance(P, Pauli):
        x = P.x_exp
        z = P.z_exp
        dims = P.dimension
        ps = PauliString([x], [z], dimensions=[dims])
        return PauliSum([ps])
    elif isinstance(P, PauliString):
        return PauliSum([P])


def to_pauli_string(pauli: Pauli) -> PauliString:
    raise NotImplementedError


def symplectic_product(pauli_string: PauliString, pauli_string2: PauliString) -> bool:
    # Inputs:
    #     pauli_string - (PauliString)
    #     pauli_string2 - (PauliString)
    # Outputs:
    #     (bool) - quditwise inner product of Paulis

    if any(pauli_string.dimensions - pauli_string.dimensions):
        raise Exception("Symplectic inner product only works if Paulis have same dimensions")
    sp = 0
    for i in range(pauli_string.n_qudits()):
        sp += (pauli_string.x_exp[i] * pauli_string2.z_exp[i] - pauli_string.z_exp[i] * pauli_string2.x_exp[i])
    return sp % pauli_string.lcm


def string_to_symplectic(string: str) -> tuple[np.ndarray, int]:
    # split into single qubit paulis by spaces
    substrings = string.split()
    local_symplectics = []
    phases = []
    for s in substrings:
        match = re.match(r'x(\d+)z(\d+)(?:p(\d+))?', s)
        if not match:
            raise ValueError(f"Invalid Pauli string: {s}")
        else:
            x = int(match.group(1))
            z = int(match.group(2))
            p = int(match.group(3)) if match.group(3) is not None else 0
            local_symplectics.append((x, z))
            phases.append(p)

    symplectic = np.array(local_symplectics).T
    return symplectic.flatten(), sum(phases)


def symplectic_to_string(symplectic: np.ndarray, dimension: int) -> str:
    if dimension == 2:
        if symplectic[0] == 0 and symplectic[1] == 0:
            return 'x0z0'
        elif symplectic[0] == 1 and symplectic[1] == 0:
            return 'x1z0'
        elif symplectic[0] == 0 and symplectic[1] == 1:
            return 'x0z1'
        elif symplectic[0] == 1 and symplectic[1] == 1:
            return 'x1z1'
        else:
            raise Exception("Symplectic vector must be of the form (0, 0), (1, 0), (0, 1), or (1, 1)")
    else:
        return f'x{symplectic[0]}z{symplectic[1]}'


def random_pauli_string(dimensions: list[int]) -> PauliString:
    """Generates a random PauliString of n_qudits"""
    n_qudits = len(dimensions)
    x_array = np.zeros(n_qudits, dtype=int)
    z_array = np.zeros(n_qudits, dtype=int)
    for i in range(n_qudits):
        x_array[i] = np.random.randint(0, dimensions[i])
        z_array[i] = np.random.randint(0, dimensions[i])
    p_string = PauliString(x_array, z_array, dimensions=dimensions)

    return p_string


def check_mappable_via_clifford(pauli_sum: PauliSum, target_pauli_sum: PauliSum) -> bool:
    return bool(np.all(pauli_sum.symplectic_product_matrix() == target_pauli_sum.symplectic_product_matrix()))


# TODO: correct type hint error
def concatenate_pauli_sums(pauli_sums: list[PauliSum]) -> PauliSum:
    """
    Concatenate a list of Pauli sums into a single Pauli sum.
    """
    # if len(pauli_sums) == 0:
    #     raise ValueError("List of Pauli sums is empty")
    # if not all(isinstance(p, PauliSum) for p in pauli_sums):
    #     raise ValueError("All elements of the list must be Pauli sums")

    # new_pauli_strings = pauli_sums[0].pauli_strings.copy()
    # new_dimensions = pauli_sums[0].dimensions.copy()
    # new_weights = pauli_sums[0].weights.copy()
    # new_phases = pauli_sums[0].phases.copy()
    # for p in pauli_sums[1:]:
    #     new_dimensions = np.concatenate((new_dimensions, p.dimensions))
    #     new_weights *= p.weights
    #     new_phases += p.phases
    #     for i in range(len(new_pauli_strings)):
    #         new_pauli_strings[i] = new_pauli_strings[i] @ p.pauli_strings[i]

    # concatenated = PauliSum(new_pauli_strings, weights=new_weights, phases=new_phases, dimensions=new_dimensions,
    #                         standardise=False)
    # return concatenated
    raise NotImplementedError()


def are_subsets_equal(pauli_sum_1: PauliSum, pauli_sum_2: PauliSum,
                      subset_1: list[tuple[int, int]], subset_2: list[tuple[int, int]] | None = None):
    """
    Check if two subsets of Pauli sums are equal.
    """
    if subset_2 is None:
        subset_2 = subset_1
    else:
        if len(subset_1) != len(subset_2):
            raise ValueError("Subsets must be of the same length")
        if not all(isinstance(i, tuple) and len(i) == 2 for i in subset_1):
            raise ValueError("Subsets must be lists of tuples of length 2")
        if not all(isinstance(i, tuple) and len(i) == 2 for i in subset_2):
            raise ValueError("Subsets must be lists of tuples of length 2")

    for i in range(len(subset_1)):
        if pauli_sum_1[subset_1[i]] != pauli_sum_2[subset_2[i]]:
            return False
    return True


def commutation_graph(pauli_sum: PauliSum, labels: list[str] | None = None, axis: Any | None = None):
    """
    Plots graph where adjacency matrix is the symplectic product matrix
    """
    adjacency_matrix = pauli_sum.symplectic_product_matrix()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    pos1 = nx.spring_layout(gr)

    nx.draw(gr, pos1, node_size=900, labels=labels, with_labels=True, ax=axis)
    return gr


def modinv(a, d):
    for i in range(1, d):
        if (a * i) % d == 1:
            return i
    raise ValueError(f"No inverse for {a} mod {d}")


def row_reduce_mod_d(A, d):
    A = A.copy() % d
    m, n = A.shape
    rank = 0
    pivots = []
    for col in range(n):
        for row in range(rank, m):
            if A[row, col] != 0:
                break
        else:
            continue
        if row != rank:
            A[[row, rank]] = A[[rank, row]]
        inv = modinv(A[rank, col], d)
        A[rank] = (A[rank] * inv) % d
        for r in range(m):
            if r != rank and A[r, col] != 0:
                A[r] = (A[r] - A[r, col] * A[rank]) % d
        pivots.append(col)
        rank += 1
    return A, pivots, rank


def solve_mod_d(A, b, d, max_solutions=1000):
    from itertools import product
    import sympy as sp

    A = sp.Matrix(A.tolist())
    b = sp.Matrix(b.tolist())
    A_aug = A.row_join(b)
    A_mod = A_aug.applyfunc(lambda x: x % d)

    Ab_rref, pivot_cols = A_mod.rref(iszerofunc=lambda x: x % d == 0, simplify=True)
    n_vars = A.shape[1]

    pivot_cols = [p for p in pivot_cols if p < n_vars]
    free_vars = [i for i in range(n_vars) if i not in pivot_cols]

    solutions = []
    for free_vals in product(range(d), repeat=len(free_vars)):
        sol = [0] * n_vars
        for i, val in zip(free_vars, free_vals):
            sol[i] = val

        for i, pivot in enumerate(pivot_cols):
            rhs = Ab_rref[i, -1]
            lhs = sum(Ab_rref[i, j] * sol[j] for j in range(n_vars)) % d
            sol[pivot] = int((rhs - lhs) % d)

        solutions.append(np.array(sol, dtype=int))
        if len(solutions) >= max_solutions:
            break

    return solutions


def is_symplectic(M, d):
    n = M.shape[0] // 2
    if M.shape[0] != M.shape[1] or M.shape[0] % 2 != 0:
        return False
    J = np.block([[np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
                  [-np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]]) % d
    return np.array_equal((M.T @ J @ M) % d, J % d)


def find_symplectic_maps(H, H_prime, d, max_solutions=1000):
    """
    Find symplectic matrices M such that H M^T = H' mod d.

    This function attempts to find matrices M that satisfy the equation
    H M^T = H' (mod d), where H and H' are given matrices, M^T is the transpose
    of M, and d is the modulus. Only symplectic matrices M are returned.

    Args:
        H (np.ndarray): The left-hand side matrix in the equation, with shape (n, k).
        H_prime (np.ndarray): The right-hand side matrix in the equation, with the same shape as H.
        d (int): The modulus for the equation.
        max_solutions (int, optional): The maximum number of solutions to return. Default is 1000.

    Returns:
        List[np.ndarray]: A list of symplectic matrices M that satisfy the equation.
    """
    n, k = H.shape
    assert H_prime.shape == (n, k)
    num_vars = k * k  # because M is k x k, and we are solving for M^T

    A = np.zeros((n * k, num_vars), dtype=int)
    b = H_prime.flatten()

    for H_row in range(n):       # row index of H, H'
        for H_col in range(k):   # column index of H'
            row_idx = H_row * k + H_col
            for idx in range(k):  # column index of H, row index of M^T
                col_idx = H_col * k + idx  # since M^T[H_col, idx] = M[idx, H_col]
                A[row_idx, col_idx] = H[H_row, idx]

    raw_solutions = solve_mod_d(A, b, d, max_solutions)
    Ms = [sol.reshape((k, k)).T for sol in raw_solutions]  # Transpose back to M
    Ms_symp = [M for M in Ms if is_symplectic(M, d)]
    return Ms_symp
