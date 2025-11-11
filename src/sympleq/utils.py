import numpy as np
from sympleq.core.paulis import PauliSum
import galois
from sympleq.core.finite_field_solvers import solve_modular_linear_system
from collections import defaultdict


def read_luca_test_2(path: str, dims: list[int] | int = 2, spaces: bool = True):
    """Reads a Hamiltonian file and parses the Pauli strings and coefficients.

    Args:
        path (str): Path to the Hamiltonian file.
        dims (Union[int, List[int]]): Dimension(s) of the qudits, default is 2.
        spaces (bool): Whether to expect spaces in the Pauli string format, default is True.

    Returns:
        Tuple: Parsed Pauli operators and corresponding coefficients.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    pauli_strings = []
    coefficients = []

    for line in lines:
        pauli_list = line.split(', {') if spaces else line.split(',{')
        coeff = pauli_list[0][1:].replace(" ", "").replace('*I', 'j')
        coefficients.append(complex(coeff))

        pauli_str = ' '.join(
            f"x{item.count('X')}z{item.count('Z')}" for item in pauli_list[1:])
        pauli_strings.append(pauli_str.strip())

    return PauliSum.from_string(pauli_strings, dimensions=dims if isinstance(dims, list) else [dims],
                                weights=coefficients)


def bases_to_int(base, dimensions) -> int:
    """
    Converts a list of integers (base) given the dimensions to an integer. Base can be thought of as a number
    in basis of the dimensions which is converted to a number in base 10.

    The base is a list of integers, where the i-th element is the value
    in the i-th dimension. The dimensions parameter is a list of integers,
    where the i-th element is the size of the i-th dimension.

    The function returns the integer that corresponds to the input base in
    the given dimensions.

    Parameters
    ----------
    base : list of int
        The base to be converted.
    dimensions : list of int
        The dimensions of the base.

    Returns
    -------
    int
        The integer that corresponds to the input base in the given
        dimensions.
    """
    dimensions = np.flip(dimensions)
    base = np.flip(base)
    number = base[0] + sum([base[qudit] * np.prod(dimensions[:qudit])
                           for qudit in range(1, len(dimensions))])
    dimensions = np.flip(dimensions)
    base = np.flip(base)
    return number


def int_to_bases(number, dimensions) -> np.ndarray:
    """
    Converts an integer to a list of integers given the dimensions. The returned list of integers can be thought of
    as a number in basis of the dimensions which is converted from a number in base 10.

    The function takes two parameters, an integer and a list of integers, where the i-th element of the list is the
    size of the i-th dimension.

    The function returns the list of integers that corresponds to the input number in the given dimensions.

    Parameters
    ----------
    number : int
        The number to be converted.
    dimensions : list of int
        The dimensions of the base.

    Returns
    -------
    np.ndarray
        The list of integers that corresponds to the input number in the given dimensions.
    """
    dimensions = np.flip(dimensions)
    base = [number % dimensions[0]]
    for i in range(1, len(dimensions)):
        s0 = base[0] + sum([base[i1] * dimensions[i1 - 1]
                           for i1 in range(1, i)])
        s1 = np.prod(dimensions[:i])
        base.append(((number - s0) // s1) % dimensions[i])
    dimensions = np.flip(dimensions)
    return np.flip(np.array(base))


# PHYSICS FUNCTIONS


def Hamiltonian_Mean(P: PauliSum, psi: np.ndarray) -> float:
    """Returns the mean of a Hamiltonian with a given state.

    Args:
        P: pauli, Paulis of Hamiltonian
        psi: numpy.array, state for mean

    Returns:
        numpy.float64, mean sum(c*<psi|P|psi>)
    """
    p = P.n_paulis()
    psi_dag = psi.conj().T
    return sum(P.weights[i] * psi_dag @ P.to_hilbert_space(i) @ psi for i in range(p))


def covariance_matrix(P: PauliSum, psi: np.ndarray) -> np.ndarray:
    """
    Computes the covariance matrix for a given set of Pauli operators and a quantum state.

    Args:
        P (PauliSum): The set of Pauli operators, represented as a PauliSum object, with associated weights.
        psi (np.ndarray): The state vector for which the covariance matrix is computed.

    Returns:
        np.ndarray: A 2D numpy array representing the covariance matrix of the Pauli operators with respect to
                    the given state. Each element [i, j] corresponds to the covariance between the i-th and j-th
                    Pauli operators.
    """
    p = P.n_paulis()
    cc = P.weights
    mm = [P.to_hilbert_space(i) for i in range(p)]
    psi_dag = psi.conj().T
    cc1 = [psi_dag @ mm[i] @ psi for i in range(p)]
    cc2 = [psi_dag @ mm[i].conj().T @ psi for i in range(p)]
    return np.array([[np.conj(cc[i0]) * cc[i1] * ((psi_dag @ mm[i0].conj().T @ mm[i1] @ psi) - cc2[i0] * cc1[i1])
                      for i1 in range(p)] for i0 in range(p)])


def commutation_graph(P: PauliSum) -> np.ndarray:
    """
    Computes the commutation graph for a given set of Pauli operators.

    Args:
        P (PauliSum): A set of Pauli operators represented as a PauliSum object.

    Returns:
        np.ndarray: A 2D numpy array representing the commutation graph. Each element [i, j] is 1 if the i-th and
                    j-th Pauli operators commute, otherwise 0.
    """
    p = P.n_paulis()
    return np.array([[int(P[i0, :].commute(P[i1, :])) for i1 in range(p)] for i0 in range(p)])


def complex_phase_value(phase, dimension):
    """
    Computes the a-th eigenvalue of a pauli with dimension d.

    Args:
        a (int): The integer to compute the eigenvalue for.
        d (int): The dimension of the pauli to use.

    Returns:
        complex: The computed eigenvalue.
    """
    return np.exp(2 * np.pi * 1j * phase / dimension)


def rand_state(dimension):
    """
    Generate a random quantum state vector for a system of dimension d.

    Args:
        d (int): Dimension of the quantum system.

    Returns:
        np.ndarray: A normalized random state vector in the complex space of size d.
    """
    gamma_sample = np.random.gamma(1, 1, int(dimension))
    phases = np.random.uniform(0, 2 * np.pi, int(dimension))
    normalized_state = np.sqrt(
        gamma_sample / np.sum(gamma_sample)) * np.exp(1j * phases)
    return normalized_state


def get_linearly_independent_rows(A: np.ndarray, d: int) -> list[int]:
    """
    Returns the pivot column indices for the row-reduced form of matrix A over a Galois field.

    Args:
        A (galois.FieldArray): Input matrix over GF(p).

    Returns:
        List[int]: List of pivot column indices.
    """

    GF = galois.GF(d)
    A = GF(A)
    R = A.row_reduce()
    pivots = []
    for row in R:
        nz_indices = np.nonzero(row)[0]
        if nz_indices.size > 0:
            pivots.append(nz_indices[0])
    return pivots


def get_linear_dependencies(vectors: np.ndarray,
                            p: int | list[int] | np.ndarray
                            ) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    """
    Find linear dependencies among rows of `vectors` over finite fields.

    Parameters
    ----------
    vectors : np.ndarray
        Matrix of shape (m, n). Each row is a vector.
    p : int | list[int] | np.ndarray
        - If int: all columns are over GF(p).
        - If list/array of length n: p[j] is the prime for column j.
        - If list/array of length n//2: per-qudit primes; each value is duplicated for X/Z columns.
        - If list/array of length m: p[i] is the prime field per row (legacy mode).

    Returns
    -------
    pivot_indices : list[int]
        Indices of linearly independent rows.
    dependencies : dict[int, list[tuple[int, int]]]
        Mapping: row index -> list of (pivot_index, coefficient) such that
        row = sum(coeff * pivot).
    """
    m, n = vectors.shape  # m = rows (vectors), n = columns

    # Normalize p into per-column primes when possible.
    p_cols: list[int]
    mode: str
    if isinstance(p, int):
        p_cols = [int(p)] * n
        mode = "per-column"
    elif isinstance(p, (list, np.ndarray)):
        lp = len(p)
        if lp == n:
            p_cols = [int(x) for x in p]
            mode = "per-column"
        elif lp == n // 2 and n % 2 == 0:
            # Per-qudit list provided; duplicate for X/Z halves
            p_cols = [int(x) for x in np.repeat(p, 2)]
            mode = "per-column"
        elif lp == m:
            # Legacy behavior: per-row primes
            ps = [int(x) for x in p]
            mode = "per-row"
        else:
            raise AssertionError(
                f"Length of p must be either rows={m}, cols={n}, or cols/2={n // 2} (for qudits). Got {lp}"
            )
    else:
        raise TypeError(f"p must be int or list/np.ndarray of ints (got {type(p)}).")

    pivot_indices: list[int] = []
    dependencies: dict[int, list[tuple[int, int]]] = {}

    if mode == "per-row":
        # Group rows by prime and process each group independently
        prime_groups = defaultdict(list)
        for i, prime in enumerate(ps):
            prime_groups[prime].append(i)

        for prime, indices in prime_groups.items():
            if len(indices) == 1:
                pivot_indices.append(indices[0])
                continue

            GF = galois.GF(prime)
            V = GF(vectors[indices, :])

            # Identify group pivots incrementally
            group_pivots = []
            seen = GF.Zeros((0, V.shape[1]))
            for idx_in_group, i in enumerate(indices):
                candidate = V[idx_in_group]
                A = GF(np.vstack([seen, candidate]))
                R = A.row_reduce()
                rank = sum(1 for row in R if np.any(row != 0))
                if rank > seen.shape[0]:
                    group_pivots.append(i)
                    seen = A
            pivot_indices.extend(group_pivots)

            # Dependencies within the group
            if len(group_pivots) > 0:
                B = V[[indices.index(j) for j in group_pivots], :]
                for idx_in_group, i in enumerate(indices):
                    if i in group_pivots:
                        continue
                    x = solve_modular_linear_system(B, V[idx_in_group])
                    deps = [(group_pivots[j], int(x[j])) for j in range(len(x)) if x[j] != 0]
                    dependencies[i] = deps

        return pivot_indices, dependencies

    # Per-column primes path
    # Build prime -> column indices map
    prime_cols: dict[int, list[int]] = defaultdict(list)
    for j, prime in enumerate(p_cols):
        prime_cols[int(prime)].append(j)

    unique_primes = list(prime_cols.keys())

    # Track seen rows per-prime on the respective column subsets
    seen_per_prime: dict[int, np.ndarray] = {}
    for q in unique_primes:
        GFq = galois.GF(q)
        seen_per_prime[q] = GFq.Zeros((0, len(prime_cols[q])))

    # Select pivots: a row is independent if it increases rank for any prime on its column subset
    for i in range(m):
        increases_any = False
        updated_seen = {}
        for q in unique_primes:
            cols = prime_cols[q]
            if len(cols) == 0:
                continue
            GFq = galois.GF(q)
            candidate = GFq(vectors[i, cols])
            A_prev = seen_per_prime[q]
            A = GFq(np.vstack([A_prev, candidate]))
            R = A.row_reduce()
            rank_new = sum(1 for row in R if np.any(row != 0))
            if rank_new > A_prev.shape[0]:
                increases_any = True
                updated_seen[q] = A
            else:
                updated_seen[q] = A_prev

        if increases_any:
            pivot_indices.append(i)
            for q in unique_primes:
                seen_per_prime[q] = updated_seen[q]
        else:
            # Mark as dependent; coefficients are computed later as best-effort
            continue

    # Compute dependencies best-effort:
    # If all columns share the same prime, compute exact coefficients as before.
    if len(unique_primes) == 1:
        q = unique_primes[0]
        GFq = galois.GF(q)
        cols = prime_cols[q]
        Vq = GFq(vectors[:, cols])
        Bq = Vq[pivot_indices, :]
        for i in range(m):
            if i in pivot_indices:
                continue
            x = solve_modular_linear_system(Bq, Vq[i])
            deps = [(pivot_indices[j], int(x[j])) for j in range(len(x)) if x[j] != 0]
            dependencies[i] = deps
    else:
        # Mixed primes: return independent set; dependency coefficients are non-unique across fields.
        dependencies = {}

    return pivot_indices, dependencies
