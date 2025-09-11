import numpy as np
from quaos.core.paulis import PauliSum
from quaos.core.measurement.allocation import (sort_hamiltonian, get_phase_matrix, choose_measurement,
                                               construct_circuit_list, update_data, scale_variances, diagnostic_circuits,
                                               standard_error_function, diagnostic_states, update_diagnostic_data)
from quaos.core.measurement.covariance_graph import (quditwise_commutation_graph, commutation_graph,
                                                     weighted_vertex_covering_maximal_cliques, graph)
from quaos.core.measurement.mcmc import bayes_covariance_graph
from quaos.utils import int_to_bases
from typing import Callable
import pickle


class Aquire:
    def __init__(self,
                 H: PauliSum,
                 psi: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None,
                 general_commutation: bool = False,
                 allocation_mode: str = "set",
                 N_chain: int = 8,
                 N_mcmc: int = 500,
                 N_mcmc_max: int = 2001,
                 mcmc_shot_scale: float = 1 / 10000,
                 diagnostic_mode: str = "Zero",
                 noise_probability_function: Callable | None = None,
                 error_function: Callable | None = None,
                 noise_args=[],
                 noise_kwargs={},
                 error_args=[],
                 error_kwargs={}):

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
        self.diagnostic_mode = diagnostic_mode
        self.noise_probability_function = noise_probability_function
        self.error_function = error_function
        self.noise_args = noise_args
        self.noise_kwargs = noise_kwargs
        self.error_args = error_args
        self.error_kwargs = error_kwargs

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
        self.diagnostic_data = np.zeros((self.n_paulis, 2))
        self.total_shots = 0
        self.scaling_matrix = np.eye(self.n_paulis, dtype=int)
        self.covariance_graph = graph(
            np.diag([np.conj(self.weights[_]) * self.weights[_] for _ in range(self.n_paulis)]))
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
        self.diagnostic_data_checkpoints = []

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
        diagnostic_state_list, dsp_circuits_list = diagnostic_states(diagnostic_circuit_list, mode=self.diagnostic_mode)
        self.diagnostic_states += diagnostic_state_list
        self.diagnostic_state_preparation_circuits += dsp_circuits_list

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

    def calculate_systematic_variance_estimate(self):
        # estimate w_i
        w = np.zeros((self.n_paulis,2))
        for i in range(self.n_paulis):
            w[i,0] = (self.diagnostic_data[i,0]+1)/(np.sum(self.diagnostic_data[i,:])+2)
            w[i,1] = (self.diagnostic_data[i,1]+1)/(np.sum(self.diagnostic_data[i,:])+2)

        # estimate theta_i
        theta_est = np.zeros((self.n_paulis,self.dimension))
        for i in range(self.n_paulis):
            if np.sum(self.data[i,i,:]) > 0:
                for j in range(self.dimension):
                    theta_est[i,j] = (self.data[i,i,j]+1)/(np.sum(self.data[i,i,:])+2)
            else:
                theta_est[i,:] = 1/self.dimension

        # eigenvalues
        xis = [np.exp(2*1j*np.pi*beta/self.dimension) for beta in range(self.dimension)]
        error_correction = np.sum([self.weights[i0] * np.sum([xis[beta]*(theta_est[i0,beta] - 1/self.dimension)
                                  for beta in range(self.dimension)])* w[i0,0] for i0 in range(self.n_paulis)])
        return np.abs(error_correction)**2

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
                                   N_chain=self.N_chain,
                                   N=self.N_mcmc + int(self.total_shots * self.mcmc_shot_scale),
                                   N_max=self.N_mcmc_max + 4 * int(self.total_shots * self.mcmc_shot_scale))

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
        self.data = update_data(self.last_update_cliques,
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

        while len(self.diagnostic_data_checkpoints) < len(self.update_steps):
            i = len(self.diagnostic_data_checkpoints)
            if self.update_steps[i] > len(self.diagnostic_results):
                break
            new_cliques = self.cliques[self.update_steps[i - 1]:self.update_steps[i]]
            new_diagnostic_results = self.diagnostic_results[self.update_steps[i - 1]:self.update_steps[i]]
            self.diagnostic_data = update_diagnostic_data(new_cliques,
                                                          new_diagnostic_results,
                                                          self.diagnostic_data,
                                                          mode=self.diagnostic_mode)
            self.diagnostic_data_checkpoints.append(self.diagnostic_data.copy())

    def simulate_measurement_results(self):
        simulated_measurement_results = []
        for aa in self.last_update_cliques:
            P1, C, k_dict = self.circuit_dictionary[str(aa)]
            psi_diag = C.unitary() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            p1, q1, phases1, dims1 = P1.paulis(), P1.qudits(), P1.phases, P1.dims
            a1 = np.random.choice(np.prod(dims1), p=pdf)
            result = int_to_bases(a1, dims1)
            if self.noise_probability_function is not None:
                noise_probability = self.noise_probability_function(C, *self.noise_args, **self.noise_kwargs)
                if np.random.rand() < noise_probability:
                    if self.error_function is not None:
                        result = self.error_function(result, *self.error_args, **self.error_kwargs)
                    else:
                        result = standard_error_function(result, self.H.dimensions)
            simulated_measurement_results.append(result)
        self.data = update_data(self.last_update_cliques,
                                simulated_measurement_results,
                                self.data,
                                self.k_phases,
                                self.circuit_dictionary)

        self.measurement_results += simulated_measurement_results

    def simulate_diagnostic_results(self):
        simulated_diagnostic_results = []
        for dsp_circuit in self.last_update_diagnostic_state_preparation_circuits:
            psi_diag = dsp_circuit.unitary() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            dims = [self.dimension] * self.n_qudits
            a1 = np.random.choice(np.prod(dims), p=pdf)
            result = int_to_bases(a1, dims)
            if self.noise_probability_function is not None:
                noise_probability = self.noise_probability_function(dsp_circuit, *self.noise_args, **self.noise_kwargs)
                if np.random.rand() < noise_probability:
                    if self.error_function is not None:
                        result = self.error_function(result, *self.error_args, **self.error_kwargs)
                    else:
                        result = standard_error_function(result, self.H.dimensions)
            simulated_diagnostic_results.append(result)
        self.diagnostic_results += simulated_diagnostic_results
        if len(simulated_diagnostic_results) < len(self.last_update_diagnostic_circuits):
            self.last_update_diagnostic_circuits = self.last_update_diagnostic_circuits[len(
                simulated_diagnostic_results):]
        else:
            self.last_update_diagnostic_circuits = []

        while len(self.diagnostic_data_checkpoints) < len(self.update_steps):
            i = len(self.diagnostic_data_checkpoints)
            if self.update_steps[i] > len(self.diagnostic_results):
                break
            new_cliques = self.cliques[self.update_steps[i - 1]:self.update_steps[i]]
            new_diagnostic_results = self.diagnostic_results[self.update_steps[i - 1]:self.update_steps[i]]
            self.diagnostic_data = update_diagnostic_data(new_cliques,
                                                          new_diagnostic_results,
                                                          self.diagnostic_data,
                                                          mode=self.diagnostic_mode)
            self.diagnostic_data_checkpoints.append(self.diagnostic_data.copy())

    def save(self,filename:str):
        with open(filename,'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,filename:str):
        with open(filename,'rb') as f:
            return pickle.load(f)

    # TODO: Decide which of these one usually wants to save
    def save_results(self,filename:str):
        results = {
            'estimated_mean':self.estimated_mean,
            'statistical_variance':self.statistical_variance,
            'systematic_variariance':self.systematic_variance,
            'data_checkpoints':self.data_checkpoints,
            'scaling_matrix_checkpoints':self.scaling_matrix_checkpoints,
            'covariance_graph_checkpoints':[cg.adj for cg in self.covariance_graph_checkpoints],
            'update_steps':self.update_steps,
            'total_shots':self.total_shots,
            'cliques':self.cliques,
            'circuits':[circuit.gates for circuit in self.circuits],
            'diagnostic_circuits':[circuit.gates for circuit in self.diagnostic_circuits],
            'diagnostic_states':self.diagnostic_states,
            'diagnostic_state_preparation_circuits':[circuit.gates for circuit in self.diagnostic_state_preparation_circuits],
            'measurement_results':self.measurement_results,
            'diagnostic_results':self.diagnostic_results
        }
        with open(filename,'wb') as f:
            pickle.dump(results,f)

    def simulate_observable(self, update_steps: list[int]):
        initial_shots = update_steps[0]
        rounds = len(update_steps) - 1
        shots_per_round = [update_steps[i + 1] - update_steps[i] for i in range(rounds)]
        self.choose_cliques(initial_shots)
        self.simulate_measurement_results()
        self.update_covariance_graph()
        for i in range(rounds):
            self.choose_cliques(shots_per_round[i])
            self.simulate_measurement_results()
            self.update_covariance_graph()
            self.simulate_diagnostic_results()
