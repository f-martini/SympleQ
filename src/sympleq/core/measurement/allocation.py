import numpy as np
import random
import itertools
from sympleq.core.paulis import PauliSum
from sympleq.core.measurement.covariance_graph import graph
from sympleq.core.circuits import Circuit
from sympleq.core.circuits.gates import Hadamard as H, SUM as CX, PHASE as S


def mcmc_number_initial_samples(shots: int, n_0: int = 500, scaling_factor: float = 1 / 10000):
    return int(shots * scaling_factor) + n_0


def mcmc_number_max_samples(shots: int, n_0: int = 2001, scaling_factor: float = 1 / 10000):
    return 4 * int(shots * scaling_factor) + n_0


def sort_hamiltonian(P: PauliSum):
    # TODO: remove identity (is now not done anymore because of reorder method)
    """
    Sorts the Hamiltonian's Pauli operators based on hermiticity, with hermitian ones first and then pairs of
    Paulis and their hermitian conjugate. !!! Also removes identity !!!

    Parameters:
        P (pauli): A set of Pauli operators.
        coefficients (list): Corresponding coefficients of the Pauli operators.

    Returns:
        tuple: Sorted Pauli operators, coefficients, and the size of Pauli blocks.
    """
    pauli_count = P.n_paulis()
    indices = list(range(pauli_count))

    hermitian_indices = []
    non_hermitian_indices = []
    paired_conjugates = []

    while indices:
        i = indices.pop(0)
        P0 = P[i, :]
        P0_str = str(P0)
        P0_conjugate = P0.hermitian_conjugate()
        P0_conj_str = str(P0_conjugate)

        if P0_str == P0_conj_str:
            if P0.x_exp.any() or P0.z_exp.any():
                hermitian_indices.append(i)
            continue
        else:
            non_hermitian_indices.append(i)

        for j in indices:
            P1 = P[j, :]
            P1_str = str(P1)
            if P0_conj_str == P1_str:
                paired_conjugates.append(j)
                indices.remove(j)
                break

    # Rebuild Pauli set and coefficients
    sorted_indices = []
    pauli_block_sizes = []

    # Handle hermitian indices
    for i in hermitian_indices:
        sorted_indices.append(i)
        pauli_block_sizes.append(1)

    # Handle non-hermitian indices and their conjugates
    for i, j in zip(non_hermitian_indices, paired_conjugates):
        sorted_indices.extend([i, j])
        pauli_block_sizes.append(2)

    # Extract and reorder Pauli strings and coefficients
    P1 = P.copy()
    P1.reorder(sorted_indices)

    # Remove identity
    for i in range(P1.n_paulis()):
        if P1.weights[i] != 0 and P1[i, :].is_identity():
            P1._delete_paulis([i])

    return P1, np.array(pauli_block_sizes)


def choose_measurement(S, V, aaa, allocation_mode):
    p = S.shape[0]
    Ones = [np.ones((i, i), dtype=int) for i in range(p + 1)]
    index_set = set(range(p))
    S1 = S + Ones[p]
    s = 1 / (S.diagonal() | (S.diagonal() == 0))
    s1 = 1 / S1.diagonal()
    factor = p - np.count_nonzero(S.diagonal())
    S1[range(p), range(p)] = [a if a != 1 else -factor for a in S1.diagonal()]
    V1 = V * (S * s * s[:, None] - S1 * s1 * s1[:, None])
    V2 = 2 * V * (S * s * s[:, None] - S * s * s1[:, None])
    aaa, aaa1 = itertools.tee(aaa, 2)
    if allocation_mode == 'set':
        aa = sorted(max(aaa1, key=lambda xx: np.abs(
            V1[xx][:, xx].sum() + V2[xx][:, list(index_set.difference(xx))].sum())))
    elif allocation_mode == 'rand':
        aa = sorted(random.sample(list(set(frozenset(aa1) for aa1 in aaa1)), 1)[0])
    else:
        raise NotImplementedError('Allocation mode not implemented')
    return aa


def construct_circuit_list(P, xxx, D):
    circuit_list = []
    for aa in xxx:
        C, D = construct_diagonalization_circuit(P, aa, D=D)
        if not is_diagonalizing_circuit(P, C, aa):
            raise ValueError('Circuit is not diagonalizing')
        circuit_list.append(C)
    return circuit_list, D


def construct_diagonalization_circuit(P: PauliSum, aa, D={}):
    if str(aa) in D:
        P1, C, k_dict = D[str(aa)]
    else:
        P1 = P.copy()
        P1._delete_paulis([i for i in range(P.n_paulis()) if i not in aa])
        # add products
        k_dict = {str(j0): [(a0, a0, P1.phases[j0])] for j0, a0 in enumerate(aa)}
        for j0, a0 in enumerate(aa):
            for j1, a1 in enumerate(aa):
                if j0 != j1:
                    # isolate a pair of paulis in the clique
                    P_a0 = P1[[j0]]
                    P_a0c = P_a0.H()
                    P_a1 = P1[[j1]]
                    # compute their product pauli
                    P2 = P_a0c * P_a1
                    P2.weights[0] = 1
                    # check if the product is in the original pauli list
                    P1_pauli_string_list = [str(P1[k]) for k in range(P1.n_paulis())]
                    if str(P2[0]) not in P1_pauli_string_list:
                        k_dict[str(P1.n_paulis())] = [(a0, a1, P2.phases[0])]
                        # add the product but make sure to account for possibly different phases
                        P2.phases[0] = 0
                        P1 = P1 + P2
                    else:
                        k_dict[str(P1_pauli_string_list.index(str(P2[0])))].append((a0, a1, P2.phases[0]))

        C = diagonalize(P1)
        P1.set_phases(np.zeros(P1.n_paulis()))
        P1 = C.act(P1)
        D[str(aa)] = (P1, C, k_dict)
    return C, D


# TODO: Make this part of the core function
def diagonalize(P: PauliSum):
    # Inputs:
    #     P - (pauli) - Pauli to be diagonalized
    # Outputs:
    #     (circuit) - circuit which diagonalizes P
    dims = P.dimensions
    q = P.n_qudits()

    if len(set(P.dimensions)) == 1:
        # Currently commutation check is not working for mixed species
        if not P.is_commuting():
            raise Exception("Paulis must be pairwise commuting to be diagonalized")
    P1 = P.copy()
    C = Circuit(dims)

    if P.is_quditwise_commuting():
        # for each dimension, call diagonalize_iter_quditwise_ on the qudits of the same dimension
        for d in sorted(set(dims)):
            aa = [i for i in range(q) if dims[i] == d]
            while aa:
                C = diagonalize_iter_quditwise_(P1, C, aa)
        P1 = C.act(P1)
    else:
        # for each dimension, call diagonalize_iter_ on the qudits of same dimension
        for d in sorted(set(dims)):
            aa = [i for i in range(q) if dims[i] == d]
            while aa:
                C = diagonalize_iter_(P1, C, aa)
        P1 = C.act(P1)

    # if any qudits are X rather than Z, apply H to make them Z
    if [i for i in range(q) if any(P1.x_exp[:, i])]:
        C1 = Circuit(dims)
        for i in range(q):
            if any(P1.x_exp[:, i]):
                g = H(i, dims[i])
                C.add_gate(g)
                C1.add_gate(g)
        P1 = C1.act(P1)
    return C


def diagonalize_iter_(P, C, aa):
    p = P.n_paulis()
    P = C.act(P)
    a = aa.pop(0)

    # if all Paulis have no X-part on qudit a, return C
    if not any(P.x_exp[:, a]):
        return C

    # set a1 to be the index of the minimum Pauli with non-zero X-part of qudit a
    a1 = min(i for i in range(p) if P.x_exp[i, a])

    # add CNOT gates to cancel out all non-zero X-parts on Pauli a1, qudits in aa
    while any(P.x_exp[a1, i] for i in aa):
        C1 = Circuit(P.dimensions)
        for i in aa:
            if P.x_exp[a1, i]:
                g = CX(a, i, P.dimensions[a])
                C.add_gate(g)
                C1.add_gate(g)
        P = C1.act(P)

    # check whether there are any non-zero Z-parts on Pauli a1, qudits in aa
    while any(P.z_exp[a1, i] for i in aa):

        # if Pauli a1, qudit a is X, apply S gate to make it Y
        if not P.z_exp[a1, a]:
            g = S(a, P.dimensions[a])
            C.add_gate(g)
            P = g.act(P)

        # add backwards CNOT gates to cancel out all non-zero Z-parts on Pauli a1, qudits in aa
        gg = [CX(i, a, P.dimensions[a]) for i in aa if P.z_exp[a1, i]]
        C.add_gate(gg)
        for g in gg:
            P = g.act(P)

    # if Pauli a1, qudit a is Y, add S gate to make it X
    while P.z_exp[a1, a]:
        g = S(a, P.dimensions[a])
        C.add_gate(g)
        P = g.act(P)
    return C


def diagonalize_iter_quditwise_(P: PauliSum, C: Circuit, aa: list[int]):
    p = P.n_paulis()
    P = C.act(P)
    a = aa.pop(0)

    # if all Paulis have no X-part on qudit a, return C
    if not any(P.x_exp[:, a]):
        return C

    # set a1 to be the index of the minimum Pauli with non-zero X-part of qudit a
    a1 = min(i for i in range(p) if P.x_exp[i, a])

    # if Pauli a1, qudit a is Y, add S gate to make it X
    while P.z_exp[a1, a]:
        g = S(a, P.dimensions[a])
        C.add_gate(g)
        P = g.act(P)
    return C


def is_diagonalizing_circuit(P: PauliSum, C: Circuit, aa: list[int]):
    P1 = P.copy()
    P1._delete_paulis([i for i in range(P.n_paulis()) if i not in aa])
    P1 = C.act(P1)
    return P1.is_z()


def update_data(xxx, rr, X, D):
    d = len(X[0, 0])
    for i, aa in enumerate(xxx):
        (P1, _, k_dict) = D[str(aa)]
        p1, q1, phases1 = P1.n_paulis(), P1.n_qudits(), P1.phases
        bases_a1 = rr[i]
        ss = [(sum((bases_a1[i1] * P1.z_exp[i0, i1] * P1.lcm) // P1.dimensions[i1]
               for i1 in range(q1))) % P1.lcm for i0 in range(p1)]
        for j0, s0 in enumerate(ss):
            for a0, a1, s1 in k_dict[str(j0)]:
                if (phases1[j0] + s1) % 2 == 1:
                    raise Exception('Odd phase detected for sorting into data matrix')
                X[a0, a1, int(s0 + (phases1[j0] + s1) / 2) % d] += 1
    return X


def update_diagnostic_data(cliques, diagnostic_results, diagnostic_data, mode='zero'):
    if mode == 'zero':
        for i in range(len(diagnostic_results)):
            if np.any(diagnostic_results[i]):
                diagnostic_data[cliques[i], 0] += 1
            else:
                diagnostic_data[cliques[i], 1] += 1
    elif mode == 'random':
        raise Exception('Random diagnostic states not yet implemented')
    else:
        raise Exception('Diagnostic state mode not recognized')
    return diagnostic_data


def scale_variances(A, S):
    # Inputs:
    #     A - (graph)       - variance matrix
    #     S - (numpy.array) - array for tracking number of measurements
    p = A.ord()
    S1 = S.copy()
    S1[range(p), range(p)] = [a if a != 0 else 1 for a in S1.diagonal()]
    s1 = 1 / S1.diagonal()
    return graph(S1 * A.adj * s1 * s1[:, None])


def construct_diagnostic_circuits(circuit_list):
    diagnostic_circuits = []
    for circ in circuit_list:
        C_diag = Circuit(dimensions=circ.dimensions)
        for g in circ.gates:
            C_diag.add_gate(g.copy())
            if g.name == 'H':
                C_diag.add_gate(g.copy())
                C_diag.add_gate(g.copy())
                C_diag.add_gate(g.copy())
        diagnostic_circuits.append(C_diag)
    return diagnostic_circuits


def construct_diagnostic_states(diagnostic_circuits: list[Circuit], mode='zero'):
    if mode == 'zero':
        state = [0] * np.prod(diagnostic_circuits[0].dimensions)
        state[0] = 1
        state_preparation_circuits = [Circuit(diagnostic_circuits[0].dimensions)] * len(diagnostic_circuits)
        return ([state] * len(diagnostic_circuits), state_preparation_circuits)
    elif mode == 'random':
        raise Exception('Random diagnostic states not yet implemented')
    else:
        raise Exception('Diagnostic state mode not recognized')


def standard_noise_probability_function(circuit, p_entangling=0.03, p_local=0.001, p_measurement=0.001):
    n_local = 0
    n_entangling = 0
    for g in circuit.gates:
        if g.name == 'SUM':
            n_entangling += 1
        else:
            n_local += 1
    noise_prob = 1 - ((1 - p_measurement) * (1 - p_entangling)**n_entangling * (1 - p_local)**n_local)
    return noise_prob


def standard_error_function(result, dimensions):
    return np.array([np.random.randint(dimensions[j]) for j in range(len(dimensions))])


def extract_phase(weight, dimension):
    phase = np.floor(dimension * np.angle(weight) / (2 * np.pi))
    remainder = np.angle(weight) - phase * 2 * np.pi / dimension
    return phase, remainder
