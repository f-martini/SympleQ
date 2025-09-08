import numpy as np
from quaos.core.paulis import PauliSum
from quaos.core.measurement.allocation import (sort_hamiltonian, get_phase_matrix, choose_measurement,
                                               construct_circuit_list, update_X, scale_variances,diagnostic_circuits,
                                               standard_error_function, diagnostic_states)
from quaos.core.measurement.covariance_graph import (quditwise_commutation_graph, commutation_graph,
                                                     weighted_vertex_covering_maximal_cliques, graph)
from quaos.core.measurement.mcmc import bayes_covariance_graph
from quaos.utils import int_to_bases
from typing import Callable


class Aquire:
    def __init__(self,
                 H: PauliSum,
                 psi: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None,
                 general_commutation: bool = False,
                 allocation_mode: str = "set",
                 N_chain: int = 8,
                 N_mcmc: int = 500,
                 N_mcmc_max: int = 2001,
                 mcmc_shot_scale: float = 1/10000):

        H, pauli_block_sizes = sort_hamiltonian(H)

        # supposed to be permanent
        self.H = H
        self.weights = H.weights
        self.pauli_block_sizes = pauli_block_sizes
        self.psi = psi
        self.n_paulis = H.n_paulis()
        self.n_qudits = H.n_qudits()
        self.dimension = int(H.lcm)
        self.k_phases = get_phase_matrix(H)

        # changeable if so desired
        self.N_chain = int(N_chain)
        self.N_mcmc = int(N_mcmc)
        self.N_mcmc_max = int(N_mcmc_max)
        self.mcmc_shot_scale = mcmc_shot_scale
        self.general_commutation = general_commutation
        self.allocation_mode = allocation_mode

        # dependent on changeable parameters
        self.CG = commutation_graph(H) if general_commutation else quditwise_commutation_graph(H)
        self.clique_covering = weighted_vertex_covering_maximal_cliques(self.CG, cc=self.weights, k=3)

        # supposed to change during experiment
        self.data = np.zeros((self.n_paulis, self.n_paulis, self.dimension))
        self.cliques = []
        self.circuits = []
        self.diagnostic_circuits = []
        self.diagnostic_states = []
        self.diagnostic_state_preparation_circuits = []
        self.total_shots = 0
        self.scaling_matrix = np.eye(self.n_paulis, dtype=int)
        self.covariance_graph = graph(np.diag([np.conj(self.weights[_])*self.weights[_] for _ in range(self.n_paulis)]))
        self.circuit_dictionary = {}
        self.measurement_results = []
        self.diagnostic_results = []
        self.update_steps = []

        # variables that track changes since last covariance update
        self.last_update_shots = 0
        self.last_update_cliques = []
        self.last_update_circuits = []
        self.last_update_diagnostic_circuits = []
        self.last_update_diagnostic_states = []
        self.last_update_diagnostic_state_preparation_circuits = []

        # checkpoints (some of the above data collected at specific points)
        self.data_checkpoints = []
        self.scaling_matrix_checkpoints = []
        self.covariance_graph_checkpoints = []

        # results
        self.estimated_mean = []
        self.statistical_variance = []
        self.systematic_variance = []

    def choose_cliques(self, shots):
        self.total_shots += shots
        self.last_update_shots += shots
        Ones = [np.ones((i, i), dtype=int) for i in range(self.n_paulis + 1)]
        new_cliques = []
        for i in range(shots):
            aa = choose_measurement(self.scaling_matrix,self.covariance_graph.adj,
                                    self.clique_covering,self.allocation_mode)
            new_cliques.append(aa)
            self.scaling_matrix[np.ix_(aa, aa)] += Ones[len(aa)]

        self.cliques += new_cliques
        self.last_update_cliques += new_cliques
        self.scaling_matrix[range(self.n_paulis), range(self.n_paulis)] -= np.ones(self.n_paulis, dtype=int)

        # construct list of circuits
        circuit_list, self.circuit_dictionary = construct_circuit_list(self.H,new_cliques,self.circuit_dictionary)
        self.circuits += circuit_list
        self.last_update_circuits += circuit_list

        # construct list of diagnostic circuits
        diagnostic_circuit_list = diagnostic_circuits(circuit_list)
        self.diagnostic_circuits += diagnostic_circuit_list
        self.last_update_diagnostic_circuits += diagnostic_circuit_list
        diagnostic_state_list, dsp_circuits_list = diagnostic_states(diagnostic_circuit_list, mode='Zero')
        self.diagnostic_states += diagnostic_state_list
        self.diagnostic_state_preparation_circuits += dsp_circuits_list

        return new_cliques, circuit_list

    def calculate_mean_estimate(self):
        mean = 0
        for i0 in range(self.n_paulis):
            total_counts = sum(self.data[i0, i0, i1] for i1 in range(self.dimension))
            if total_counts > 0:
                weighted_sum = sum(
                    self.data[i0, i0, i1] * np.exp(2j * np.pi * i1 / self.dimension)
                    for i1 in range(self.dimension)
                )
            mean += self.weights[i0] * (weighted_sum / total_counts)
        mean = mean.real
        return mean

    def calculate_statistical_variance_estimate(self):
        scaled_variance_graph = scale_variances(self.covariance_graph,self.scaling_matrix)
        stat_variance_estimate = np.sum(scaled_variance_graph.adj).real
        return stat_variance_estimate

    def update_covariance_graph(self):
        self.update_steps.append(self.total_shots)
        self.data_checkpoints.append(self.data.copy())
        self.scaling_matrix_checkpoints.append(self.scaling_matrix.copy())

        A = bayes_covariance_graph(self.data,
                                   self.weights,
                                   self.CG.adj,
                                   self.n_paulis,
                                   self.pauli_block_sizes,
                                   self.dimension,
                                   N_chain = self.N_chain,
                                   N = self.N_mcmc + int(self.total_shots * self.mcmc_shot_scale),
                                   N_max = self.N_mcmc_max + 4*int(self.total_shots * self.mcmc_shot_scale))

        self.covariance_graph = graph(A)
        self.covariance_graph_checkpoints.append(self.covariance_graph.copy())

        self.estimated_mean.append(self.calculate_mean_estimate())
        self.statistical_variance.append(self.calculate_statistical_variance_estimate())
        self.last_update_shots = 0
        self.last_update_cliques = []
        self.last_update_circuits = []

    def input_measurement_data(self, measurement_results: list):
        if len(self.data) == len(self.circuits):
            raise Exception(
                "Data already input for all circuits. Please allocate more measurements before inputting more data.")
        if len(measurement_results) < len(self.last_update_cliques):
            raise Exception(
                "Not enough measurement results input. Please input at least as many results as the number of newly allocated measurements.")
        self.data = update_X(self.last_update_cliques,
                             measurement_results,
                             self.data,
                             self.k_phases,
                             self.circuit_dictionary)

        self.measurement_results += measurement_results

    def input_diagnostic_data(self, diagnostic_results: list):
        self.diagnostic_results += diagnostic_results
        if len(diagnostic_results) < len(self.last_update_diagnostic_circuits):
            self.last_update_diagnostic_circuits = self.last_update_diagnostic_circuits[len(diagnostic_results):]
        elif len(diagnostic_results) > len(self.last_update_diagnostic_circuits):
            raise Exception(
                "Too many diagnostic results input. Please input at most as many results as the number of newly allocated diagnostic measurements.")
        else:
            self.last_update_diagnostic_circuits = []

    def simulate_measurement_results(self, noise_probability_function: Callable | None = None, error_function: Callable | None = None):
        simulated_measurement_results = []
        for aa in self.last_update_cliques:
            P1, C, k_dict = self.circuit_dictionary[str(aa)]
            psi_diag = C.unitary() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            p1, q1, phases1, dims1 = P1.paulis(), P1.qudits(), P1.phases, P1.dims
            a1 = np.random.choice(np.prod(dims1), p=pdf)
            result = int_to_bases(a1, dims1)
            if noise_probability_function is not None:
                noise_probability = noise_probability_function(C)
                if np.random.rand() < noise_probability:
                    if error_function is not None:
                        result = error_function(result)
                    else:
                        result = standard_error_function(result, self.H.dimensions)
            simulated_measurement_results.append(result)
        self.data = update_X(self.last_update_cliques,
                             simulated_measurement_results,
                             self.data,
                             self.k_phases,
                             self.circuit_dictionary)

        self.measurement_results += simulated_measurement_results

    def simulate_diagnostic_results(self, noise_probability_function: Callable | None = None, error_function: Callable | None = None):
        simulated_diagnostic_results = []
        for dsp_circuit in self.last_update_diagnostic_state_preparation_circuits:
            psi_diag = dsp_circuit.unitary() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            dims = [self.dimension] * self.n_qudits
            a1 = np.random.choice(np.prod(dims), p=pdf)
            result = int_to_bases(a1, dims)
            if noise_probability_function is not None:
                noise_probability = noise_probability_function(dsp_circuit)
                if np.random.rand() < noise_probability:
                    if error_function is not None:
                        result = error_function(result)
                    else:
                        result = standard_error_function(result, self.H.dimensions)
            simulated_diagnostic_results.append(result)
        self.diagnostic_results += simulated_diagnostic_results
        if len(simulated_diagnostic_results) < len(self.last_update_diagnostic_circuits):
            self.last_update_diagnostic_circuits = self.last_update_diagnostic_circuits[len(
                simulated_diagnostic_results):]
        else:
            self.last_update_diagnostic_circuits = []







