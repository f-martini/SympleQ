import random
import numpy as np
from quaos.core.paulis import PauliSum, PauliString
from quaos.core.circuits import Circuit, Gate


def random_pauli_hamiltonian(num_paulis, qudit_dims, mode='rand', seed=None):
    """
    Generates a random Pauli Hamiltonian with the given number of Pauli operators plus their Hermitian conjugate pairs.

    Parameters:
        num_paulis (int): Number of Pauli operators to generate.
        qudit_dims (list): List of dimensions for each qudit.
        mode (str): 'rand' or 'uniform' - dictates form of weights in the PauliSum

    Returns:
        tuple: A random PauliSum
    """
    n_qudits = len(qudit_dims)
    pauli_strings = []
    coefficients = []

    for _ in range(num_paulis):
        x_exp = [0 for i in range(n_qudits)]
        z_exp = [0 for i in range(n_qudits)]
        while np.sum(np.array(x_exp) + np.array(z_exp)) == 0:
            # np.random.randint(qudit_dims, size=n_qudits)
            x_exp = [random.randint(0, qudit_dims[i] - 1) for i in range(n_qudits)]
            # np.random.randint(qudit_dims, size=n_qudits)
            z_exp = [random.randint(0, qudit_dims[i] - 1) for i in range(n_qudits)]
        x_exp_H = np.zeros_like(x_exp)
        z_exp_H = np.zeros_like(z_exp)
        phase_factor = 1
        pauli_str = ''
        pauli_str_H = ''

        for j in range(len(qudit_dims)):
            r, s = x_exp[j], z_exp[j]
            pauli_str += f"x{r}z{s} "
            x_exp_H[j] = (-r) % qudit_dims[j]
            z_exp_H[j] = (-s) % qudit_dims[j]
            pauli_str_H += f"x{x_exp_H[j]}z{z_exp_H[j]} "

            omega = np.exp(2 * np.pi * 1j / qudit_dims[j])
            phase_factor *= omega**(r * s)

        pauli_strings.append(PauliString.from_string(pauli_str.strip(), dimensions=qudit_dims))
        if mode == 'rand' or mode == 'random':
            coeff = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
        elif mode == 'uniform' or mode == 'one':
            coeff = 1 + 0 * 1j
        elif mode[0:7] == 'randint':
            # mode is 'randint2', 'randint3', etc.
            d = int(mode[7:])
            coeff = np.random.randint(1, d + 1)

        if (not np.array_equal(x_exp, x_exp_H)) and (not np.array_equal(z_exp, z_exp_H)):
            # random string not Hermitian, add conjugate pair
            coefficients.append(coeff)
            coefficients.append(np.conj(coeff) * phase_factor)
            pauli_strings.append(PauliString.from_string(pauli_str_H.strip(), dimensions=qudit_dims))
        else:
            coefficients.append(coeff.real)

    rand_ham = PauliSum(pauli_strings, weights=coefficients)
    return rand_ham


def pauli_hamiltonian_row_reduced_form(n_qudits: int,
                                       n_paulis: int,
                                       n_redundant: int = 0,
                                       n_conditional: int = 0,
                                       weight_mode: str = 'uniform',
                                       phase_mode: str = 'zero'):
    # 0: I, 1: X, 2: Z, 3: Y
    n_rest = n_qudits - n_redundant - n_conditional
    if n_paulis < 2 * n_rest:
        raise ValueError('Number of Paulis must be at least 2 * (n_qudits - n_redundant - n_conditional)')
    if n_redundant + n_conditional > n_qudits:
        raise ValueError('Number of redundant and conditional qudits exceeds total number of qudits.')
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
        print(n_rest + i, q)
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

    g = Circuit.from_random(n_qudits, 1000, [2] * n_qudits)
    P = g.act(P)

    return P


def random_symmetric_pauli_sum(symmetry: 'Gate', n_qudits: int, n_paulis: int):

    P = pauli_hamiltonian_row_reduced_form(n_qudits, n_paulis, 0, 0)
    G_inv = symmetry.inv()
    P_prime = G_inv.act(P)
    P_sym = P + P_prime
    return P_sym
