from typing import Any
import numpy as np
from .pauli import Pauli
from .pauli_string import PauliString
from .pauli_sum import PauliSum
import networkx as nx


def ground_state(P: PauliSum) -> np.ndarray:
    """Returns the ground state of a given Hamiltonian

    Args:
        P: pauli, Paulis for Hamiltonian
        cc: list[int], coefficients for Hamiltonian

    Returns:
        numpy.array: eigenvector corresponding to lowest eigenvalue of Hamiltonian
    """

    m = P.matrix_form()

    m = m.toarray()
    val, vec = np.linalg.eig(m)
    val = np.real(val)
    vec = np.transpose(vec)

    tmp_index = val.argmin(axis=0)

    gs = vec[tmp_index]
    gs = np.transpose(gs)
    gs = gs / np.linalg.norm(gs)

    if abs(min(val) - np.transpose(np.conjugate(gs)) @ m @ gs) > 10**-10:
        print("ERROR with the GS!!!")

    return gs


def to_pauli_sum(P: Pauli | PauliString) -> PauliSum:
    """
    Convert a Pauli or PauliString to a PauliSum.

    Parameters
    ----------
    P : Pauli | PauliString
        The Pauli or PauliString to convert.

    Returns
    -------
    PauliSum
        The resulting PauliSum.
    """
    if isinstance(P, Pauli):
        x = P.x_exp
        z = P.z_exp
        dims = P.dimension
        ps = PauliString([x], [z], dimensions=[dims])
        return PauliSum([ps])
    elif isinstance(P, PauliString):
        return PauliSum([P])


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


def check_mappable_via_clifford(pauli_sum: PauliSum, target_pauli_sum: PauliSum) -> bool:
    return bool(np.all(pauli_sum.symplectic_product_matrix() == target_pauli_sum.symplectic_product_matrix()))


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
