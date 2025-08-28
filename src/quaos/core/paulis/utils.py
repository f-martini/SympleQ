from typing import Any
import numpy as np
import re
from .pauli import Pauli
from .pauli_string import PauliString
from .pauli_sum import PauliSum
import networkx as nx


def ground_state_TMP(P: PauliSum,
                     only_gs: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues/eigenvectors of `P` and (by default) pick
    the ground-state energy (`only_gs` = `True`).

    Parameters
    ----------
    P : PauliSum
        A PauliSum object representing the Hamiltonian/operator.
    only_gs : bool, optional
        If `True` (default), only the ground-state (energy) is kept in `gs` (`en`).
         If `False`, all eigenvectors (eigenvalues) are kept in `gs` (`en`).

    Returns
    -------
    en : float or numpy.ndarray
        If `only_gs` is `True`, the lowest eigenvalue (ground-state energy).
        Otherwise, a 1D array of all eigenvalues sorted ascending.
    gs : numpy.ndarray
        Eigenvectors sorted to match ``en``.

    Raises
    ------
    AssertionError
        If the (internally normalized) eigenvector(s) do not yield real
        expectation values matching the corresponding eigenvalue(s) within
        ``1e-10``.

    Notes
    -----
    - Since :func:`numpy.linalg.eig` is used, the input matrix is treated as
      general (not assumed Hermitian). If the operator is Hermitian, consider
      using :func:`numpy.linalg.eigh` for improved numerical stability.
    """
    # Convert PauliSum to matrix form
    m = P.matrix_form()
    m = m.toarray()
    # Get eigenvalues and eigenvectors
    val, vec = np.linalg.eig(m)
    val = np.real(val)
    vec = np.transpose(vec)
    # Ordering
    tmp_index = np.argsort(val)
    en = val[tmp_index]
    gs = vec[tmp_index]
    # Prepare output
    if only_gs:
        en = en[0]
        gs_out = gs[0]
        gs_out = np.transpose(gs_out)
        gs_out = gs_out / np.linalg.norm(gs_out)
    else:
        gs_out = []
        for el in gs:
            el = np.transpose(el)
            el = el / np.linalg.norm(el)
            gs_out.append(el)
        gs_out = np.array(gs_out)
    # Checks
    exp_en = []
    if not only_gs:
        for el in gs_out:
            exp_en.append(np.transpose(np.conjugate(el)) @ m @ el)
        exp_en = np.array(exp_en)
    else:
        exp_en = np.transpose(np.conjugate(gs_out)) @ m @ gs_out
    assert np.max(abs(en - exp_en)) < 10**-10, \
        "The ground state does not yield a real value <gs | H |gs> = {}".format(exp_en)
    # Return
    return en, gs


def to_pauli_sum(P: Pauli | PauliString
                 ) -> PauliSum:
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


def to_pauli_string(pauli: Pauli
                    ) -> PauliString:
    raise NotImplementedError


def symplectic_product(PauliString_1: PauliString,
                       PauliString_2: PauliString
                       ) -> bool:
    """
    Qudit-wise symplectic product of two Pauli strings.

    Parameters
    ----------
    PauliString_1 : Pauli | PauliString
        The first PauliString for computing the inner product.
    PauliString_2 : Pauli | PauliString
        The second PauliString for computing the inner product.

    Returns
    -------
    int
        The symplectic product of the two PauliStrings objects.
    """
    if any(PauliString_1.dimensions - PauliString_2.dimensions):
        raise Exception("Symplectic inner product only works if Paulis have same dimensions")
    sp = 0
    for i in range(PauliString_1.n_qudits()):
        sp += (PauliString_1.x_exp[i] * PauliString_2.z_exp[i] - PauliString_1.z_exp[i] * PauliString_2.x_exp[i])
    return sp % PauliString_1.lcm


def string_to_symplectic(string: str
                         ) -> tuple[np.ndarray, int]:
    """
    Convert a string representation of a PauliString into its symplectic form.

    Parameters
    ----------
    string : str
        The string representation of the PauliString.

    Returns
    -------
    tuple[np.ndarray, int]
        The symplectic form of the PauliString and its phase.
    """
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


# TODO: The previous and following functions do not seem super useful...
#       I would say one can create a PauliString/Sum object from the string and
#       then use that object for any further processing.
def symplectic_to_string(symplectic: np.ndarray,
                         dimension: int
                         ) -> str:
    """
    Convert a symplectic representation of a PauliString into its string form.

    Parameters
    ----------
    symplectic : np.ndarray
        The symplectic representation of the PauliString.
    dimension : int
        The dimension of the PauliString.

    Returns
    -------
    str
        The string representation of the PauliString.
    """
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


def random_pauli_string(dimensions: list[int]
                        ) -> PauliString:
    """
    Generates a random PauliString with dimensions specified by `dimensions'.
    Each qudit is assigned a random exponent both for its `x' and `z' components,
    that is uniformly distributed between 0 and d-1, d being the dimension of the qudit.

    Parameters
    ----------
    dimensions : list[int]
        The dimensions of the PauliString.

    Returns
    -------
    PauliString
        A random PauliString with the specified dimensions.
    """
    n_qudits = len(dimensions)
    x_array = np.zeros(n_qudits, dtype=int)
    z_array = np.zeros(n_qudits, dtype=int)
    for i in range(n_qudits):
        x_array[i] = np.random.randint(0, dimensions[i])
        z_array[i] = np.random.randint(0, dimensions[i])
    p_string = PauliString(x_array, z_array, dimensions=dimensions)

    return p_string


def check_mappable_via_clifford(PauliSum: PauliSum,
                                target_PauliSum: PauliSum
                                ) -> bool:
    """
    Checks whether the given PauliSum can be mapped to the target PauliSum via Clifford operations.

    Parameters
    ----------
    PauliSum : PauliSum
        The PauliSum to check.
    target_PauliSum : PauliSum
        The target PauliSum to check against.

    Returns
    -------
    bool
        True if the PauliSum can be mapped to the target PauliSum, False otherwise.
    """
    return bool(
        np.all(
            PauliSum.symplectic_product_matrix() == target_PauliSum.symplectic_product_matrix()
        )
    )


# TODO: correct type hint error
def concatenate_pauli_sums(pauli_sums: list[PauliSum]
                           ) -> PauliSum:
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


def are_subsets_equal(PauliSum_1: PauliSum,
                      PauliSum_2: PauliSum,
                      subset_1: list[tuple[int, int]],
                      subset_2: list[tuple[int, int]] | None = None
                      ) -> bool:
    """
    Check if two subsets of two PauliSums objects are equal.
    I.e., the first `subset_1' of `PauliSum_1' must match the second `subset_2' of `PauliSum_2'.

    Parameters
    ----------
    PauliSum_1 : PauliSum
        The first PauliSum to compare.
    PauliSum_2 : PauliSum
        The second PauliSum to compare.
    subset_1 : list[tuple[int, int]]
        The indices of the first PauliSum to compare, given as a list of tuples
    subset_2 : list[tuple[int, int]] | None
        The indices of the second PauliSum to compare, given as a list of tuples

    Returns
    -------
    bool
        True if the subsets are equal, False otherwise.
    """
    if subset_2 is None:
        if subset_1 is None:
            # TODO: maybe set `subset_1' to all indices of `PauliSum_1'
            raise ValueError("At least one subset must be provided")
        else:
            subset_2 = subset_1
    else:
        if len(subset_1) != len(subset_2):
            raise ValueError("Subsets must be of the same length")
        if not all(isinstance(i, tuple) and len(i) == 2 for i in subset_1):
            raise ValueError("Subsets must be lists of tuples of length 2")
        if not all(isinstance(i, tuple) and len(i) == 2 for i in subset_2):
            raise ValueError("Subsets must be lists of tuples of length 2")

    for i in range(len(subset_1)):
        if PauliSum_1[subset_1[i]] != PauliSum_2[subset_2[i]]:
            return False
    return True


def commutation_graph(PauliSum: PauliSum,
                      labels: list[str] | None = None,
                      axis: Any | None = None):
    """
    Plots the commutation graph of a PauliSum, based on the adjacency
    matrix given by the symplectic product matrix.

    Parameters
    ----------
    PauliSum : PauliSum
        The PauliSum to plot as a graph.
    labels : list[str] | None
        The labels for the nodes in the graph.
    axis : Any | None
        The axis to plot the graph on.

    Returns
    -------
    nx.Graph
        The commutation graph of the PauliSum.
    """
    adjacency_matrix = PauliSum.symplectic_product_matrix()
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


def modinv(a : int,
           d: int
           ) -> int:
    """
    Compute the modular multiplicative inverse of an integer.

    Given integers `a` and `d`, this function finds an integer `i` such that
    `(a * i) % d == 1`. If no such integer exists, a `ValueError` is raised.

    Parameters
    ----------
    a : int
        The integer whose modular inverse is to be computed.
    d : int
        The modulus.

    Returns
    -------
    int
        The modular multiplicative inverse of `a` modulo `d`.

    Raises
    ------
    ValueError
        If the modular inverse does not exist (i.e., if `a` and `d` are not coprime).

    Examples
    --------
    >>> modinv(3, 11)
    4
    >>> modinv(10, 17)
    12
    """
    for i in range(1, d):
        if (a * i) % d == 1:
            return i
    raise ValueError(f"No inverse for {a} mod {d}")


def row_reduce_mod_d(A: np.ndarray,
                     d: int
                     ) -> tuple[np.ndarray, list[int], int]:
    """
    Performs row reduction (Gaussian elimination) of a matrix modulo a given integer.

    This function reduces the input matrix `A` to its row-echelon form over the integers modulo `d`.
    It also returns the list of pivot columns and the rank of the matrix modulo `d`.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix to be row reduced. Must be a 2D numpy array of integers (dtype=int).
    d : int
        The modulus for the arithmetic operations.

    Returns
    -------
    A_reduced : numpy.ndarray
        The row-reduced form of the input matrix modulo `d`.
    pivots : list of int
        List of column indices that are pivots in the reduced matrix.
    rank : int
        The rank of the matrix modulo `d`.

    Notes
    -----
    - The input matrix `A` must have integer dtype (e.g., np.int32, np.int64).
    - The function assumes that `modinv` is defined elsewhere and computes modular inverses.
    - The input matrix `A` is not modified; a copy is used internally.
    - All arithmetic is performed modulo `d`.
    """
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
