import numpy as np
from quaos.core.paulis import PauliSum
from quaos.core.measurement.allocation import (sort_hamiltonian, get_phase_matrix, choose_measurement,
                                               construct_circuit_list, update_data,
                                               diagnostic_circuits, standard_error_function, diagnostic_states,
                                               update_diagnostic_data)
from quaos.core.measurement.covariance_graph import (quditwise_commutation_graph, commutation_graph,
                                                     weighted_vertex_covering_maximal_cliques, graph)
from quaos.core.measurement.mcmc import bayes_covariance_graph
from quaos.core.measurement.aquire_utils import (calculate_mean_estimate, calculate_statistical_variance_estimate,
                                                 calculate_systematic_variance_estimate, true_mean,
                                                 true_statistical_variance)
from quaos.utils import int_to_bases
from typing import Callable
import pickle
import matplotlib.pyplot as plt


class Aquire:
    def __init__(self,
                 H: PauliSum,
                 psi: list[float | complex] | list[float] | list[complex] | np.ndarray,
                 general_commutation: bool = False,
                 true_values: bool = True,
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
        H.phase_to_weight()

        # supposed to be permanent
        self.H = H
        self.weights = H.weights
        self.pauli_block_sizes = pauli_block_sizes
        self.psi = np.array(psi)
        self.n_paulis = H.n_paulis()
        self.n_qudits = H.n_qudits()
        self.dimension = int(H.lcm)
        self.diagnostic_mode = diagnostic_mode
        self.noise_probability_function = noise_probability_function
        self.error_function = error_function
        self.noise_args = noise_args
        self.noise_kwargs = noise_kwargs
        self.error_args = error_args
        self.error_kwargs = error_kwargs
        self.true_values_flag = true_values

        # changeable if so desired
        self.N_chain = int(N_chain)
        self.N_mcmc = int(N_mcmc)
        self.N_mcmc_max = int(N_mcmc_max)
        self.mcmc_shot_scale = mcmc_shot_scale
        self.general_commutation = general_commutation
        self.allocation_mode = allocation_mode
        if noise_probability_function is not None or error_function is not None:
            self.diagnostic_flag = True
        else:
            self.diagnostic_flag = False

        # dependent on changeable parameters
        self.CG = commutation_graph(H) if general_commutation else quditwise_commutation_graph(H)
        self.clique_covering = weighted_vertex_covering_maximal_cliques(self.CG, cc=self.weights, k=3)
        self.k_phases = get_phase_matrix(H, self.CG)

        # supposed to change during experiment
        self.data = np.zeros((self.n_paulis, self.n_paulis, self.dimension))
        self.cliques = []
        self.circuits = []
        self.total_shots = 0
        self.scaling_matrix = np.eye(self.n_paulis, dtype=int)
        self.covariance_graph = graph(
            np.diag([np.conj(self.weights[_]) * self.weights[_] for _ in range(self.n_paulis)]))
        self.circuit_dictionary = {}
        self.measurement_results = []
        self.update_steps = []

        # variables that track changes since last covariance update
        self.last_update_shots = 0
        self.last_update_cliques = []
        self.last_update_circuits = []

        # checkpoints (some of the above data collected at specific points)
        self.data_checkpoints = []
        self.scaling_matrix_checkpoints = []
        self.covariance_graph_checkpoints = []

        # results
        self.estimated_mean = []
        self.statistical_variance = []

        # Diagnostic variables
        self.diagnostic_circuits = []
        self.diagnostic_states = []
        self.diagnostic_state_preparation_circuits = []
        self.diagnostic_data = np.zeros((self.n_paulis, 2))
        self.diagnostic_results = []
        self.diagnostic_data_checkpoints = []
        self.last_update_diagnostic_circuits = []
        self.last_update_diagnostic_states = []
        self.last_update_diagnostic_state_preparation_circuits = []
        self.systematic_variance = []

        # Comparison values: not used in the algorithm
        # initially set to None, can be set later if desired and H not too large
        if self.true_values_flag:
            self.true_mean_value = true_mean(self.H, self.psi)
            self.true_statistical_variance_value = []

    def choose_cliques(self, shots):
        """
        Choose new cliques to measure according to the allocation mode and
        construct new circuits to measure the cliques.

        Parameters
        ----------
        shots : int
            The number of new cliques to choose.

        Returns
        -------
        None
        """
        if self.total_shots > 0:
            self.scaling_matrix[range(self.n_paulis), range(self.n_paulis)] += np.ones(self.n_paulis, dtype=int)
        self.total_shots += shots
        self.last_update_shots += shots

        Ones = [np.ones((i, i), dtype=int) for i in range(self.n_paulis + 1)]
        new_cliques = []
        for i in range(shots):
            aa = choose_measurement(self.scaling_matrix, self.covariance_graph.adj,
                                    self.clique_covering, self.allocation_mode)
            new_cliques.append(aa)
            self.scaling_matrix[np.ix_(aa, aa)] += Ones[len(aa)]

        self.cliques += new_cliques
        self.last_update_cliques += new_cliques
        self.scaling_matrix[range(self.n_paulis), range(self.n_paulis)] -= np.ones(self.n_paulis, dtype=int)

        # construct list of circuits
        circuit_list, self.circuit_dictionary = construct_circuit_list(self.H, new_cliques, self.circuit_dictionary)
        self.circuits += circuit_list
        self.last_update_circuits += circuit_list

    def choose_diagnostic_circuits(self):
        self.diagnostic_flag = True
        # construct list of diagnostic circuits
        n = len(self.diagnostic_circuits)
        diagnostic_circuit_list = diagnostic_circuits(self.circuits[n:])
        self.diagnostic_circuits += diagnostic_circuit_list
        self.last_update_diagnostic_circuits += diagnostic_circuit_list
        diagnostic_state_list, dsp_circuits_list = diagnostic_states(diagnostic_circuit_list, mode=self.diagnostic_mode)
        self.diagnostic_states += diagnostic_state_list
        self.diagnostic_state_preparation_circuits += dsp_circuits_list

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

        self.estimated_mean.append(calculate_mean_estimate(self.data, self.weights))
        self.statistical_variance.append(calculate_statistical_variance_estimate(
            self.covariance_graph, self.scaling_matrix))
        self.last_update_shots = 0
        self.last_update_cliques = []
        self.last_update_circuits = []
        if self.true_values_flag:
            self.true_statistical_variance_value.append(true_statistical_variance(
                self.H, self.psi, self.scaling_matrix, self.weights))

    def input_measurement_data(self, measurement_results: list):
        if len(self.data) == len(self.circuits):
            raise Exception(
                "Data already input for all circuits. Please allocate more measurements before inputting more data.")
        if len(measurement_results) < len(self.last_update_cliques):
            raise Exception(
                ("Not enough measurement results input. Please input at least as many results as the number of newly "
                 "allocated measurements."))
        self.data = update_data(self.last_update_cliques,
                                measurement_results,
                                self.data,
                                self.k_phases,
                                self.circuit_dictionary)

        self.measurement_results += measurement_results

    def input_diagnostic_data(self, diagnostic_results: list):
        self.diagnostic_flag = True
        self.diagnostic_results += diagnostic_results
        if len(diagnostic_results) < len(self.last_update_diagnostic_circuits):
            self.last_update_diagnostic_circuits = self.last_update_diagnostic_circuits[len(diagnostic_results):]
        elif len(diagnostic_results) > len(self.last_update_diagnostic_circuits):
            raise Exception(
                ("Too many diagnostic results input. Please input at most as many results as the number of newly "
                 "allocated diagnostic measurements."))
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
            self.systematic_variance.append(calculate_systematic_variance_estimate(self.data, self.weights,
                                                                                   self.diagnostic_data))

    def simulate_measurement_results(self):
        simulated_measurement_results = []
        for aa in self.last_update_cliques:
            P1, C, _ = self.circuit_dictionary[str(aa)]
            psi_diag = C.unitary_andrew() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            dims1 = P1.dimensions
            a1 = np.random.choice(np.prod(dims1), p=pdf)
            result = int_to_bases(a1, dims1)
            if self.diagnostic_flag:
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
        self.diagnostic_flag = True
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
            self.systematic_variance.append(calculate_systematic_variance_estimate(self.data, self.weights,
                                                                                   self.diagnostic_data))

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # TODO: Decide which of these one usually wants to save
    def save_results(self, filename: str):
        results = {
            'estimated_mean': self.estimated_mean,
            'statistical_variance': self.statistical_variance
        }
        if self.diagnostic_flag:
            results['systematic_variance'] = self.systematic_variance
        if self.true_values_flag:
            results['true_mean'] = self.true_mean_value
            results['true_statistical_variance'] = self.true_statistical_variance_value
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def simulate_observable(self, update_steps: list[int], hardware_noise: bool = False):
        initial_shots = update_steps[0]
        rounds = len(update_steps) - 1
        shots_per_round = [update_steps[i + 1] - update_steps[i] for i in range(rounds)]
        self.choose_cliques(initial_shots)
        self.simulate_measurement_results()
        self.update_covariance_graph()
        if hardware_noise:
            self.choose_diagnostic_circuits()
            self.simulate_diagnostic_results()
        for i in range(rounds):
            self.choose_cliques(shots_per_round[i])
            self.simulate_measurement_results()
            self.update_covariance_graph()
            if hardware_noise:
                self.choose_diagnostic_circuits()
                self.simulate_diagnostic_results()

    def plot(self, filename: str | None = None):
        # common data stuff
        cm = 1 / 2.54
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(23 * cm, 11 * cm))
        c_stat = '#4FCB8D'
        c_dev = '#D93035'
        cs = 1.75               # capsize
        ct = 1.5                  # capthickness
        ms = 5.5                 # marker size
        ew = 1.5                  # errorbar linewidth
        mew = 1.5               # marker edge width
        ec = c_stat  # bitwise errorbar edgecolor
        mec = '#ffffff'         # marker edge color
        mfc = c_stat  # bitwise marker face color
        fmt = 'o'            # bitwise marker type
        label_fontsize = 14

        M = np.array(self.update_steps)
        est_mean = np.array(self.estimated_mean)
        stat_var = np.array(self.statistical_variance)
        if self.diagnostic_flag:
            sys_var = np.array(self.systematic_variance)

        # Mean Plot
        if self.true_values_flag:
            H_mean = self.true_mean_value
            plot_mean = np.abs(est_mean - H_mean)/np.abs(H_mean)
            ax[0].plot([M[0], M[-1]], [0, 0], 'k--')
            ax[0].set_ylabel(r'$|\widetilde{O} - \langle \hat{O} \rangle|$', fontsize=label_fontsize)
            if self.diagnostic_flag:
                plot_errorbar = np.sqrt(stat_var + sys_var) / np.abs(H_mean)
            else:
                plot_errorbar = np.sqrt(stat_var) / np.abs(H_mean)
        else:
            plot_mean = est_mean
            ax[0].set_ylabel(r'$\widetilde{O}$', fontsize=label_fontsize)
            if self.diagnostic_flag:
                plot_errorbar = np.sqrt(stat_var + sys_var)
            else:
                plot_errorbar = np.sqrt(stat_var)
        print(plot_mean)
        print(plot_errorbar)
        ax[0].errorbar(M, plot_mean, yerr=plot_errorbar,
                       fmt=fmt, ecolor=ec,
                       capsize=cs, capthick=ct, markersize=ms, elinewidth=ew,
                       mec=mec, mew=mew, mfc=mfc)

        ax[0].set_xscale('log')
        ax[0].set_xlabel(r'shots $M$', fontsize=label_fontsize)
        ax[0].legend()

        # Error Plot
        if self.true_values_flag:
            stat_error = stat_var * M / (H_mean)**2
            if self.diagnostic_flag:
                phys_error = sys_var * M / (H_mean)**2
        else:
            stat_error = stat_var * M
            if self.diagnostic_flag:
                phys_error = sys_var * M

        r = 1.25  # ~±22% in log10
        lefts = M / np.sqrt(r)
        rights = M * np.sqrt(r)
        widths = rights - lefts

        ax[1].bar(lefts, stat_error, width=widths, align='edge',
                  label='Statistical Variance', color=c_stat)
        if self.diagnostic_flag:
            ax[1].bar(lefts, phys_error, width=widths, align='edge',
                      bottom=stat_error, label='Systematic Variance',
                      color=c_dev)

        if self.true_values_flag:
            ax[1].plot(M, self.true_statistical_variance_value * M / (H_mean)**2, 'k--', label='True Stat. Variance')

        ax[1].set_xscale('log')
        ax[1].set_ylabel(r'$M \cdot (\widetilde{\Delta O})^2$', fontsize=label_fontsize)
        ax[1].set_xlabel(r'shots $M$', fontsize=label_fontsize)
        ax[1].legend()

        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        if filename is not None:
            plt.savefig(filename, dpi=1200)

        plt.show()

    def simple_plot(self):

        plt.errorbar(self.update_steps, self.estimated_mean, yerr=np.sqrt(self.statistical_variance))
        plt.plot([self.update_steps[0], self.update_steps[-1]], [self.true_mean_value,self.true_mean_value], 'k--')
        plt.xscale('log')
        plt.xlabel('shots M')
        plt.ylabel(r'$\widetilde{O}$')
        plt.show()
