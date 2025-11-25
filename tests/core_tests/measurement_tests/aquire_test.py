# flake8: noqa
import numpy as np
from sympleq.core.measurement.aquire import Aquire
from sympleq.core.paulis import PauliSum, PauliString
from sympleq.core.paulis.utils import XZ_to_Y
from sympleq.core.measurement.covariance_graph import commutation_graph, weighted_vertex_covering_maximal_cliques
from sympleq.core.measurement.allocation import construct_circuit_list
from sympleq.core.measurement.allocation import sort_hamiltonian
from sympleq.core.measurement.aquire_utils import true_covariance_graph
import json

import pytest

@pytest.mark.integration
class TestAquire:

    def andrew_comparison(self, P, psi, correct_commutation_graph, correct_xxx, correct_circuit_list, correct_pdf_list,
                          simulated_measurement_results, correct_results, true_variance_graph):
        assert P.is_hermitian(), "P not hermitian"
        assert np.abs(np.linalg.norm(psi) - 1) < 10**(-6), "psi not normalized"
        com_graph = commutation_graph(P)
        assert np.all(com_graph.adj == correct_commutation_graph), "commutation graphs do not match"
        clique_covering = weighted_vertex_covering_maximal_cliques(com_graph, cc=P.weights, k=500)
        xxx = sorted(clique_covering)
        assert xxx == correct_xxx, "Clique coverings do not match"
        circuit_list, circuit_dictionary = construct_circuit_list(P, xxx, {})
        for i, c in enumerate(circuit_list):
            comp_c = correct_circuit_list[i]
            for j, g in enumerate(c.gates):
                assert g.name == comp_c[j][0] and np.all(g.qudit_indices == comp_c[j][1]), f"Circuit {i} does not match"

        true_cov_graph = true_covariance_graph(P, psi).adj * com_graph.adj
        assert np.allclose(true_cov_graph, true_variance_graph, atol=10**(-6)), "true covariance graphs do not match"

        for i, aa in enumerate(xxx):
            P1, C, _ = circuit_dictionary[str(aa)]
            psi_diag = C.unitary() @ psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            assert np.allclose(pdf, correct_pdf_list[i]), f"pdf does not match for clique {i}, {aa}"

        d = P.lcm

        for i, aa in enumerate(xxx):
            (P1, C, k_dict) = circuit_dictionary[str(aa)]
            p1, q1, phases1 = P1.n_paulis(), P1.n_qudits(), P1.phases

            bases_a1 = simulated_measurement_results[i]

            ss = [(sum((bases_a1[i1] * P1.z_exp[i0, i1] * P1.lcm) // P1.dimensions[i1]
                       for i1 in range(q1))) % P1.lcm for i0 in range(p1)]

            comp_correct_results = correct_results[i]
            j = 0
            for j0, s0 in enumerate(ss):
                for a0, a1, s1 in k_dict[str(j0)]:
                    assert (phases1[j0] + s1) % 2 != 1, f"Eigenvalue for {i} not even (not able to sort in correctly)"
                    assert (a0 == comp_correct_results[j][0] and a1 == comp_correct_results[j][1])
                    comp = (int(s0 + (phases1[j0] + s1) / 2) % d == comp_correct_results[j][2])
                    assert comp, f"Eigenvalue for clique {i} does not match"
                    j += 1
        pass

    def AEQuO_comparison(self, filename):
        with open(f'./tests/core_tests/measurement_tests/comparison_json/{filename}'+".json", "r") as f:
            comparison_data = json.load(f)

        P = PauliSum.from_string(comparison_data["strings"],
                                 weights=np.array(comparison_data["weights_real"]) + \
                                 1j * np.array(comparison_data["weights_imag"]),
                                 dimensions=comparison_data["dimensions"],
                                 phases=comparison_data["phases"])
        P = XZ_to_Y(P)
        psi = np.array(comparison_data["psi_real"]) + 1j * np.array(comparison_data["psi_imag"])
        correct_commutation_graph = np.array(comparison_data["commutation_graph"])
        correct_xxx = comparison_data["clique_covering"]
        correct_circuit_list = comparison_data["circuit_list"]
        correct_pdf_list = comparison_data["pdf_list"]
        simulated_measurement_results = comparison_data["simulated_measurement_results"]
        correct_results = comparison_data["comparison_results"]
        true_variance_graph = np.array(comparison_data["true_variance_graph_real"]) + \
            1j * np.array(comparison_data["true_variance_graph_imag"])

        self.andrew_comparison(P, psi, correct_commutation_graph,
                               correct_xxx, correct_circuit_list,
                               correct_pdf_list, simulated_measurement_results,
                               correct_results, true_variance_graph)

    def random_comparison_hamiltonian(self, num_paulis, dimensions, mode='rand'):
        paulistrings = []
        weights = []
        phases = []
        selected_paulistring_indexes = []
        while len(paulistrings) < num_paulis:
            ps = PauliString.from_random(dimensions)
            ps_index = ps._to_int()
            if ps_index not in selected_paulistring_indexes:
                # paulistring
                paulistrings.append(ps)

                # weights
                if mode == 'rand':
                    if ps.is_hermitian():
                        weights.append(np.random.normal(0, 1))
                    else:
                        weights.append(np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
                elif mode == 'uniform' or mode == 'one':
                    weights.append(1)
                else:
                    raise ValueError("mode must be 'rand', 'uniform' or 'one'")

                # phases
                if 2 in dimensions:
                    num_Ys = sum([ps.x_exp[i] * ps.z_exp[i] for i in range(len(ps.x_exp)) if dimensions[i] == 2])
                    phases.append(num_Ys * ps.lcm / 2)
                else:
                    phases.append(0)

                selected_paulistring_indexes.append(ps_index)
                if not ps.is_hermitian():
                    ps_conj = PauliSum.from_pauli_strings([ps], weights=[weights[-1]], phases=[phases[-1]]).H()
                    ps_conj.set_phases([(ps_conj.phases[0]-phases[-1])%(2*ps.lcm)])
                    ps_conj.phase_to_weight()

                    weights.append(ps_conj.weights[0])
                    phases.append(phases[-1])
                    ps_conj_str = ps_conj[0].copy()
                    paulistrings.append(ps_conj_str)
                    selected_paulistring_indexes.append(ps_conj_str._to_int())

        return PauliSum.from_pauli_strings(paulistrings, weights=weights, phases=phases)

    def test_aquire_for_AEQuO_mono_qubit_d222_p10(self):
        self.AEQuO_comparison("mono_qubit_d222_p10")
        pass

    def test_aquire_for_AEQuO_mono_qudit_d333_p20(self):
        self.AEQuO_comparison("mono_qudit_d333_p20")
        pass

    def test_aquire_for_AEQuO_mixed_d2233_p20(self):
        self.AEQuO_comparison("mixed_d2233_p20")
        pass

    def test_aquire_for_AEQuO_mixed_d235_p20(self):
        self.AEQuO_comparison("mixed_d235_p20")
        pass

    def test_aquire_for_AEQuO_mono_qubit_d2222_p40(self):
        self.AEQuO_comparison("mono_qubit_d2222_p40")
        pass

    def test_aquire_for_AEQuO_mono_qudit_d555_p16(self):
        self.AEQuO_comparison("mono_qudit_d555_p16")
        pass

    def test_aquire_mean_distance(self):
        update_steps = [6,12,25,50,100,200,400,800,1600,3200,6400,12800]
        dim_list = [[2,2,2], [3,3,3], [5,5,5], [2,3,5], [2,2,3,3]]
        for dims in dim_list:
            P = self.random_comparison_hamiltonian(20, dims, mode='rand')
            psi = np.random.rand(int(np.prod(dims))) + 1j*np.random.rand(int(np.prod(dims)))
            psi = psi/np.linalg.norm(psi)

            model = Aquire(H=P, psi=psi)

            model.config.set_params(commutation_mode='general',
                                    calculate_true_values=True,
                                    save_covariance_graph_checkpoints=False,
                                    auto_update_covariance_graph=True,
                                    auto_update_settings=True,
                                    verbose=True)

            model.simulate_observable(update_steps=update_steps)
            mean_distance = model.true_mean_value - model.estimated_mean[-1]
            distance_in_sigma = mean_distance / np.sqrt(np.abs(model.statistical_variance[-1]))
            assert np.abs(distance_in_sigma) < 5, f"Mean estimate too far from true value for dims {dims}"

    def test_aquire_state_hamiltonian_mismatch_detection(self):
        # check that the function that compares state and hamiltonian dimension raises an error correctly
        P = self.random_comparison_hamiltonian(20, [3,3,3], mode='rand')
        psi = np.random.rand(10) + 1j * np.random.rand(10)
        psi = psi / np.linalg.norm(psi)
        with pytest.raises(ValueError):
            Aquire(H=P, psi=psi)

    def test_aquire_state_normalization_check(self):
        # check that the function that compares state and hamiltonian dimension raises an error correctly
        P = self.random_comparison_hamiltonian(20, [3,3,3], mode='rand')
        psi = np.random.rand(int(np.prod([3, 3, 3]))) + 1j*np.random.rand(int(np.prod([3, 3, 3]))) # not normalized
        with pytest.raises(ValueError):
            Aquire(H=P, psi=psi)

    def test_aquire_commutation_validation(self):
        P = self.random_comparison_hamiltonian(20, [3,3,3], mode='rand')
        commutation_rule = 'generl'
        with pytest.raises(ValueError):
            Aquire(H=P, commutation_mode=commutation_rule)

    def test_aquire_allocation_mode_validation(self):
        P = self.random_comparison_hamiltonian(20, [3,3,3], mode='rand')
        allocation_mode = 'randm'
        with pytest.raises(ValueError):
            Aquire(H=P, allocation_mode=allocation_mode)





