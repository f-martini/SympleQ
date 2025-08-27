import random
import numpy as np
from quaos.core.paulis import PauliSum, PauliString


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
