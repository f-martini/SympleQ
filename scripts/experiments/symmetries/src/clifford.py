from quaos.core.circuits.target import find_map_to_target_pauli_sum
from quaos.core.paulis import PauliSum
from quaos.utils import get_linear_dependencies
import numpy as np
from .graph_permutations import find_one_permutation, permutation_to_swaps, mapping_key, find_swapped_dependent_elements


def symmetric_symplectic(pauli_sum: PauliSum, find_all=False, max_cycle=20):
    if np.all(pauli_sum.dimensions == pauli_sum.dimensions[0]):
        raise NotImplementedError('Currently only implemented for qudits of the same dimension')

    d = pauli_sum.dimensions[0]
    cs = pauli_sum.weights
    independent_paulis, dependencies = get_linear_dependencies(pauli_sum.tableau(), d)

    graph_dict = make_graph_dictionary(independent_paulis, dependencies, cs)
    permutations = _loop_through_permutations(graph_dict, pauli_sum, find_all=find_all, max_cycle=max_cycle)

    # Now find symplectic
    p_target =
    if find_all:
        for perm in permutations:

    else:
        H_indep = H[independent_paulis]
        H_t_indep = H_target[independent_paulis]

        assert np.all(H_indep.symplectic_product_matrix() == H_t_indep.symplectic_product_matrix())

        # print(H_indep)
        # print(H_t_indep)

        F, _, _, _ = find_map_to_target_pauli_sum(H_indep, H_t_indep)
        return F

def _loop_through_permutations(graph_dict, pauli_sum, find_all=False, max_cycle=20):
    permutation = ()
    permutations_found = set()

    permutations_attempted = set()
    found = False
    targets = []

    i = 0
    # all_permutations = brute_force_all_permutations(graph_dict[1], np.ones(len(graph_dict[1]), dtype=int))
    # print([permutation_to_swaps(perm) for perm in all_permutations])
    while not found:
        i += 1
        ##### SHOULD NOT JUST BE graph_dict[1]
        permutation = find_one_permutation(graph_dict[1], pauli_sum.weights, permutations_attempted,
                                           max_cycle_size=max_cycle)
        if permutation is None:
            raise ValueError("No valid permutation found")

        print(permutation_to_swaps(permutation))
        pairs = permutation_to_swaps(permutation)
        swapped_dependents = find_swapped_dependent_elements(pairs, graph_dict[1])
        print('sd = ', swapped_dependents)
        H_target = pauli_sum.copy()

        ######################################
        # THIS BIT IS BUGGY
        for p in pairs:
            H_target.swap_paulis(p[0], p[1])
        for p in swapped_dependents:
            if p not in pairs:
                H_target.swap_paulis(p[0], p[1])
        ######################################

        if np.array_equal(H_target.symplectic_product_matrix(), pauli_sum.symplectic_product_matrix()):
            found = True
            if find_all:

                permutations_attempted.add(mapping_key(permutation,
                                                       domain=sorted({x for lst in graph_dict[1] for x in lst})))
                permutations_found.add(permutation_to_swaps(permutation))
            else:
                return permutation_to_swaps(permutation), H_target
        else:
            print("Not an automorphism, trying next permutation")
            permutations_attempted.add(mapping_key(
                permutation, domain=sorted({x for lst in graph_dict[1] for x in lst})))

    return permutations_found, H_target


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
        dependence_multiplicities = [x[1] for x in dependency]  # this will be needed for qudits! always 1 for now
        if key in graph_dict:
            graph_dict[key].append(dependence_indices)
        else:
            graph_dict[key] = [dependence_indices]

    return graph_dict


