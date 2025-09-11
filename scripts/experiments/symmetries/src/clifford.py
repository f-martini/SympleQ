from quaos.core.circuits.target import find_map_to_target_pauli_sum
from quaos.core.paulis import PauliSum
from quaos.utils import get_linear_dependencies
import numpy as np
from quaos.graph_utils import find_one_permutation, permutation_to_swaps, mapping_key, find_swapped_dependent_elements
from quaos.core.circuits.gates import Gate
from quaos.models import Hadamard_Symmetric_PauliSum, SWAP_symmetric_PauliSum
from permutations import find_first_automorphism


def symmetric_symplectic(pauli_sum: PauliSum, max_cycle=20):
    if not np.all(pauli_sum.dimensions == pauli_sum.dimensions[0]):
        raise NotImplementedError('Currently only implemented for qudits of the same dimension')

    d = int(pauli_sum.dimensions[0])
    cs = pauli_sum.weights
    independent_paulis, dependencies = get_linear_dependencies(pauli_sum.tableau(), d)

    graph_dict = make_graph_dictionary(independent_paulis, dependencies, cs)
    permutations, target = _loop_through_permutations(graph_dict, pauli_sum, find_all=False, max_cycle=max_cycle)

    H_indep = pauli_sum[independent_paulis]
    H_t_indep = target[independent_paulis]

    assert np.all(H_indep.symplectic_product_matrix() == H_t_indep.symplectic_product_matrix())
    F, h, _, _ = find_map_to_target_pauli_sum(H_indep, H_t_indep)

    return Gate('Symmetry', [i for i in range(pauli_sum.n_qudits())], F.T, 2, h)


def _loop_through_permutations(graph_dict, pauli_sum, find_all=False, max_cycle=20):
    permutation = ()
    permutations_found = set()

    permutations_attempted = set()
    found = False

    i = 0
    # all_permutations = brute_force_all_permutations(graph_dict[1], np.ones(len(graph_dict[1]), dtype=int))
    # print([permutation_to_swaps(perm) for perm in all_permutations])
    while not found:
        i += 1
        print(i)
        # SHOULD NOT JUST BE graph_dict[1] - this just gets a single weight
        permutation = find_one_permutation(graph_dict[1], pauli_sum.weights, permutations_attempted,
                                           max_cycle_size=max_cycle)
        if permutation is None:
            raise Exception("No valid permutation found")
        print('Checking permutation', permutation_to_swaps(permutation))

        pairs = permutation_to_swaps(permutation)
        swapped_dependents = find_swapped_dependent_elements(pairs, graph_dict[1])
        H_target = pauli_sum.copy()

        ########################################
        # THIS BIT IS POSSIBLY BUGGY
        for p in pairs:
            H_target.swap_paulis(p[0], p[1])
        for p in swapped_dependents:
            if p not in pairs:
                H_target.swap_paulis(p[0], p[1])
        ########################################

        if np.array_equal(H_target.symplectic_product_matrix(), pauli_sum.symplectic_product_matrix()):
            found = True
            if find_all:

                permutations_attempted.add(mapping_key(permutation,
                                                       domain=sorted({x for lst in graph_dict[1] for x in lst})))
                permutations_found.add(permutation_to_swaps(permutation))
            else:
                print(permutation_to_swaps(permutation))
                return permutation_to_swaps(permutation), H_target
        else:
            permutations_attempted.add(mapping_key(permutation,
                                                   domain=sorted({x for lst in graph_dict[1] for x in lst})))
    return list(permutations_found), H_target


def find_swapped_indices(independent_set, dependencies, permutation):
    """
    Find both independent and dependent swaps induced by a permutation.

    Parameters
    ----------
    independent_set : list[int]
        List of indices of independent basis elements.
    dependencies : dict[int, list[tuple[int, int]]]
        Dependent index -> list of (independent_index, coeff).
        (coeff is ignored, only structure matters).
    permutation : list[int]
        Permutation of the independent_set. Must be same length.

    Returns
    -------
    list[tuple[int, int]]
        List of swaps (independent and dependent).
    """
    swaps = []
    visited = set()

    # --- independent swaps ---
    mapping = dict(zip(independent_set, permutation))
    for old, new in mapping.items():
        if old != new:
            pair = tuple(sorted((old, new)))
            if pair not in visited:
                swaps.append(pair)
                visited.add(pair)

    # --- dependent swaps ---
    # normalize dependency sets after applying permutation
    dep_to_normalized = {}
    for dep, terms in dependencies.items():
        mapped_inds = [mapping[i] for (i, _) in terms]
        dep_to_normalized[dep] = tuple(sorted(mapped_inds))

    # group by normalized representation
    seen = {}
    for dep, norm in dep_to_normalized.items():
        if norm in seen:
            pair = tuple(sorted((dep, seen[norm])))
            if pair not in visited:
                swaps.append(pair)
                visited.add(pair)
        else:
            seen[norm] = dep

    return swaps


def find_symmetry(pauli_sum: PauliSum):
    if not np.all(pauli_sum.dimensions == pauli_sum.dimensions[0]):
        raise NotImplementedError('Currently only implemented for qudits of the same dimension')

    d = int(pauli_sum.dimensions[0])
    independent_paulis, dependencies = get_linear_dependencies(pauli_sum.tableau(), d)
    print(independent_paulis, dependencies)
    graph_dict = make_graph_dictionary(independent_paulis, dependencies, pauli_sum.weights)
    n = len(independent_paulis)
    print(n)
    vectors = graph_dict[1]

    def checker(permutation):
        #find_swapped_dependent_elements(pairs, graph_dict[1])
        swaps = find_swapped_indices(independent_paulis, dependencies, permutation)
        H_target = pauli_sum.copy()
        print(swaps)
        for p in swaps:  # THIS ISN'T WORKING
            H_target.swap_paulis(p[0], p[1])
        return np.array_equal(H_target.symplectic_product_matrix(), pauli_sum.symplectic_product_matrix())

    automorphism = find_first_automorphism(vectors, n, checker)
    print(automorphism)

    if automorphism is None:
        raise Exception("No valid automorphism found")

    H_t = pauli_sum.copy()
    H_t = H_t[automorphism]
    H_i = pauli_sum[independent_paulis]

    print(H_i.symplectic_product_matrix() - H_t.symplectic_product_matrix())

    F, h, _, _ = find_map_to_target_pauli_sum(H_i, H_t)
    G = Gate('Symmetry', [i for i in range(pauli_sum.n_qudits())], F.T, 2, h)

    return G


def make_graph_dictionary(independent_paulis, dependencies, weights):

    graph_dict = {}

    for i in independent_paulis:
        key = weights[i]
        if key in graph_dict:
            graph_dict[key].append([i])
        else:
            graph_dict[key] = [[i]]

    for i in dependencies.keys():
        key = weights[i]
        dependency = dependencies[i]
        dependence_indices = [x[0] for x in dependency]
        # dependence_multiplicities = [x[1] for x in dependency]  # this will be needed for qudits! always 1 for now
        if key in graph_dict:
            graph_dict[key].append(dependence_indices)
        else:
            graph_dict[key] = [dependence_indices]

    return graph_dict


def test_hadamard_symmetries():
    correct = 0
    n_tests = 10
    seed = None
    for i in range(n_tests):
        print(f'Running Hadamard test {i + 1}/{n_tests}')
        n_qubits = 5
        n_sym_q = 2
        n_paulis = 12
        H, C = Hadamard_Symmetric_PauliSum(n_paulis, n_qubits, n_sym_q, seed=seed)
        H.combine_equivalent_paulis()
        F = symmetric_symplectic(H, max_cycle=4)
        Fp = F.act(H)
        Fp.standardise()
        H.standardise()
        if np.array_equal(Fp.tableau(), H.tableau()):
            print('Success!')
            correct += 1

    print(f'Correct: {correct}/{n_tests}')


def test_hadamard_symmetries2():
    correct = 0
    n_tests = 10
    seed = None
    for i in range(n_tests):
        print(f'Running Hadamard test {i + 1}/{n_tests}')
        n_qubits = 4
        n_sym_q = 2
        n_paulis = 7
        H, C = Hadamard_Symmetric_PauliSum(n_paulis, n_qubits, n_sym_q, seed=seed)
        H.combine_equivalent_paulis()
        H.remove_trivial_paulis()
        print(H)

        F = find_symmetry(H)
        Fp = F.act(H)
        Fp.standardise()
        H.standardise()
        if np.array_equal(Fp.tableau(), H.tableau()):
            print('Success!')
            correct += 1

    print(f'Correct: {correct}/{n_tests}')


def test_SWAP_symmetries():
    correct = 0
    n_tests = 10
    for i in range(n_tests):
        print(f'Running SWAP test {i+1}/{n_tests}')
        n_qubits = 5
        n_paulis = 12
        H = SWAP_symmetric_PauliSum(n_paulis, n_qubits)
        H.combine_equivalent_paulis()
        F = symmetric_symplectic(H, max_cycle=4)
        if np.all(F.act(H).tableau() == H.tableau()):
            correct += 1
    print(f'Correct: {correct}/{n_tests}')


if __name__ == '__main__':
    # test_hadamard_symmetries()
    test_hadamard_symmetries2()
    # test_SWAP_symmetries()
