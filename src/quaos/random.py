import numpy as np
from quaos.core.circuits import Gate
from quaos.core.paulis import PauliSum
from quaos.core.circuits.utils import symplectic_form, transvection_matrix


def _isotropic_vector(J):
    """Generate a random vector v such that <v, v> = 0."""
    v = np.random.randint(0, 2, size=(J.shape[0],), dtype=np.uint8)
    return v


def symplectic(n, num_transvections=None):
    """
    Return a random 2n x 2n binary symplectic matrix by composing
    num_transvections random transvections.

    Parameters
    ----------
    n : int
        The number of qudits (i.e. the number of pairs of rows and columns)
    num_transvections : int or None
        The number of transvections to compose.  If None, defaults to 2 * n.

    Returns
    -------
    M : 2n x 2n binary matrix
        A random symplectic matrix, represented as a binary matrix.
    """
    J = symplectic_form(n)
    dim = 2 * n
    M = np.identity(dim, dtype=np.uint8)

    if num_transvections is None:
        num_transvections = 2 * dim  # reasonable default

    for _ in range(num_transvections):
        v = _isotropic_vector(J)
        Mv = transvection_matrix(v, 2)
        M = Mv @ M % 2  # compose

    return M

# Random Clifford Circuit Generation
# so-far only supports qubits (2-dimensional systems)


def clifford(dimensions):
    M = symplectic(len(dimensions))
    phase_vector = np.random.randint(0, 2, size=2 * len(dimensions), dtype=np.uint8)
    M = M.tolist()
    M = [np.array(v) for v in M]
    g = Gate('RandomClifford', list(range(len(dimensions))), np.asarray(M), 2, phase_vector=phase_vector)
    return g


def pauli_hamiltonian(n_qudits, n_paulis, n_redundant=0, n_conditional=0, weight_mode='uniform', phase_mode='zero'):
    # 0: I, 1: X, 2: Z, 3: Y
    n_rest = n_qudits - n_redundant - n_conditional
    # create general structure of the Pauli Hamiltonian before scrambling it with clifford gates
    # redundant qubits
    P = np.zeros((n_paulis, n_qudits), dtype=int)

    # conditional qubits
    for i in range(n_conditional - n_redundant):
        q = np.arange(n_redundant, n_conditional)[i]
        P[2 * n_rest + i, q] = 2  # Z
        for j in range(1, n_paulis - (2 * n_rest + i)):
            P[2 * n_rest + i + j, q] = np.random.choice([0, 2])  # I, Z

    # remaining qubits
    for i in range(n_qudits - (n_redundant + n_conditional)):
        q = np.arange(n_redundant + n_conditional, n_qudits)[i]
        P[i, q] = 1
        P[n_rest + i, q] = 2
        for j in range(1, n_rest - i):
            P[i + j, q] = np.random.choice([0, 1, 2, 3])
            P[n_rest + i + j, q] = np.random.choice([0, 1, 2, 3])

        for j in range(n_paulis - 2 * n_rest):
            P[2 * n_rest + j, q] = np.random.choice([0, 1, 2, 3])

    # Turn P-Matrix into PauliSum
    pauli_strings = ['' for _ in range(n_paulis)]
    for i in range(n_paulis):
        for j in range(n_qudits):
            if P[i, j] == 0:
                pauli_strings[i] += 'x0z0'
            elif P[i, j] == 1:
                pauli_strings[i] += 'x1z0'
            elif P[i, j] == 2:
                pauli_strings[i] += 'x0z1'
            elif P[i, j] == 3:
                pauli_strings[i] += 'x1z1'

            pauli_strings[i] += ' '
        pauli_strings[i] = pauli_strings[i].strip()

    if weight_mode == 'uniform':
        weights = np.ones(n_paulis, dtype=float)
    elif weight_mode == 'random':
        weights = np.random.rand(n_paulis)

    if phase_mode == 'zero':
        phases = np.zeros(n_paulis, dtype=int)
    elif phase_mode == 'random':
        phases = np.random.randint(0, 2, size=n_paulis, dtype=int)

    P = PauliSum(pauli_strings, weights=weights, dimensions=[2] * n_qudits, phases=phases, standardise=False)

    g = clifford([2] * n_qudits)
    P = g.act(P)

    return P
