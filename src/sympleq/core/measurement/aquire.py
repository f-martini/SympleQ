import numpy as np
from sympleq.core.paulis import PauliSum
from sympleq.core.measurement.allocation import (sort_hamiltonian, choose_measurement,
                                                 construct_circuit_list, update_data,
                                                 construct_diagnostic_circuits, standard_error_function,
                                                 construct_diagnostic_states, weight_to_phase,
                                                 update_diagnostic_data, standard_noise_probability_function,
                                                 mcmc_number_initial_samples, mcmc_number_max_samples)
from sympleq.core.measurement.covariance_graph import (graph, quditwise_commutation_graph, commutation_graph,
                                                       weighted_vertex_covering_maximal_cliques)
from sympleq.core.measurement.mcmc import bayes_covariance_graph
from sympleq.core.measurement.aquire_utils import (calculate_mean_estimate, calculate_statistical_variance_estimate,
                                                   calculate_systematic_variance_estimate, true_mean,
                                                   true_statistical_variance)
from sympleq.core.circuits import Circuit
from sympleq.utils import int_to_bases
from typing import Callable
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import warnings


@dataclass
class AquireConfig:
    H: PauliSum
    psi: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None

    # different settings indicated by strings
    commutation_mode: str = "general"  # "general" or "qudit-wise"/"bitwise"/"local"
    allocation_mode: str = "set"  # "set" or "random"
    diagnostic_mode: str = "Zero"  # None, "zero", "random (change checks to all lowercase)

    # different settings indicated by booleans
    calculate_true_values: bool = True
    enable_diagnostics: bool = False
    enable_simulated_hardware_noise: bool = False
    auto_update_covariance_graph: bool = True
    verbose: bool = False
    auto_update_settings: bool = True
    enable_debug_checks: bool = False

    # MCMC settings
    mcmc_number_of_chains: int = 8
    mcmc_initial_samples_per_chain: int | Callable = mcmc_number_initial_samples
    mcmc_initial_samples_per_chain_kwargs: dict = {'n_0': 500, 'scaling_factor': 1 / 10000}
    mcmc_max_samples_per_chain: int | Callable = mcmc_number_max_samples
    mcmc_max_samples_per_chain_kwargs: dict = {'n_0': 2001, 'scaling_factor': 1 / 10000}

    # noise and error functions
    noise_probability_function: Callable = standard_noise_probability_function
    noise_args: tuple = ()
    noise_kwargs: dict = {'p_entangling': 0.03, 'p_local': 0.001, 'p_measurement': 0.001}
    error_function: Callable = standard_error_function
    error_args: tuple = ()
    error_kwargs: dict = {}

    def __post_init__(self):
        if self.psi is not None:
            self.psi = np.array(self.psi)

        if self.commutation_mode == "general":
            self.commutation_graph = commutation_graph(self.H).adj
        else:
            self.commutation_graph = quditwise_commutation_graph(self.H).adj

        self.clique_covering = weighted_vertex_covering_maximal_cliques(graph(self.commutation_graph),
                                                                        cc=self.H.weights,
                                                                        k=3)

        if self.error_function is standard_error_function:
            self.error_args = (self.H.dimensions,)

        self.test_mcmc_settings()

    def update_all(self):
        if self.verbose:
            warnings.warn("Updating all dependent parameters ...", UserWarning)
        if self.commutation_mode == "general":
            self.commutation_graph = commutation_graph(self.H).adj
        else:
            self.commutation_graph = quditwise_commutation_graph(self.H).adj

        self.clique_covering = weighted_vertex_covering_maximal_cliques(graph(self.commutation_graph),
                                                                        cc=self.H.weights,
                                                                        k=3)

    def test_mcmc_settings(self):
        shot_tests = [100, 1000, 10000, 100000, 1000000]
        for shots in shot_tests:
            if isinstance(self.mcmc_initial_samples_per_chain, int):
                N = self.mcmc_initial_samples_per_chain
            else:
                N = self.mcmc_initial_samples_per_chain(shots, **self.mcmc_initial_samples_per_chain_kwargs)

            if isinstance(self.mcmc_max_samples_per_chain, int):
                N_max = self.mcmc_max_samples_per_chain
            else:
                N_max = self.mcmc_max_samples_per_chain(shots, **self.mcmc_max_samples_per_chain_kwargs)

            if N > N_max:
                warnings.warn(f"Warning: For {shots} shots, initial MCMC samples per chain ({N}) "
                              f"exceeds maximum MCMC samples per chain ({N_max}).", UserWarning)

    # TODO: Check whether this works
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    # TODO: Check whether this works
    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        # only recalculate if a dependent parameter changed
        if self.auto_update_settings:
            if name == "commutation_mode" or name == "H":
                self.update_all()
        else:
            if name == "commutation_mode":
                warnings.warn("Changing commutation mode may cause commutation graph and clique covering to "
                              "be deprecated. Run update_all() to update them.", UserWarning)
                pass

    def __str__(self):
        config_string = ''
        config_string += 'Configuration class for AQUIRE experiment on observable: \n'
        config_string += str(self.H)
        if self.psi is None:
            config_string += 'No state vector has been provided for the experiment.\n '
        else:
            config_string += 'State vector provided for the experiment: \n'
            config_string += str(self.psi)

        config_string += 'Commutation mode: ' + self.commutation_mode + '\n'
        config_string += 'Allocation mode: ' + self.allocation_mode + '\n'
        config_string += 'Diagnostic mode: ' + self.diagnostic_mode + '\n'
        config_string += 'Enable simulated hardware noise: ' + str(self.enable_simulated_hardware_noise) + '\n'
        config_string += 'Calculate true values: ' + str(self.calculate_true_values) + '\n'
        config_string += 'Enable diagnostics: ' + str(self.enable_diagnostics) + '\n'
        config_string += 'Auto update covariance graph: ' + str(self.auto_update_covariance_graph) + '\n'
        config_string += 'Verbose: ' + str(self.verbose) + '\n'
        config_string += 'Auto update settings: ' + str(self.auto_update_settings) + '\n'
        config_string += 'Enable debug checks: ' + str(self.enable_debug_checks) + '\n'
        config_string += 'MCMC number of chains: ' + str(self.mcmc_number_of_chains) + '\n'
        if isinstance(self.mcmc_initial_samples_per_chain, int):
            config_string += 'MCMC initial samples per chain: ' + str(self.mcmc_initial_samples_per_chain) + '\n'
        else:
            config_string += 'MCMC initial samples per chain: ' + \
                str(self.mcmc_initial_samples_per_chain.__name__) + '\n'
            config_string += str(self.mcmc_initial_samples_per_chain_kwargs) + '\n'

        if isinstance(self.mcmc_max_samples_per_chain, int):
            config_string += 'MCMC max samples per chain: ' + str(self.mcmc_max_samples_per_chain) + '\n'
        else:
            config_string += 'MCMC max samples per chain: ' + str(self.mcmc_max_samples_per_chain.__name__) + '\n'
            config_string += str(self.mcmc_max_samples_per_chain_kwargs) + '\n'

        config_string += 'Noise probability function: ' + str(self.noise_probability_function.__name__) + '\n'
        config_string += str(self.noise_args) + '\n'
        config_string += str(self.noise_kwargs) + '\n'

        config_string += 'Error function: ' + str(self.error_function.__name__) + '\n'
        config_string += str(self.error_args) + '\n'
        config_string += str(self.error_kwargs) + '\n'

        return config_string


class Aquire:
    def __init__(self,
                 H: PauliSum,
                 psi: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None,
                 config: AquireConfig | None = None):
        """
        Constructor for the Aquire class.

        Parameters:
            H (PauliSum): Hamiltonian of the system
            psi (list[float | complex] | list[float] | list[complex] | np.ndarray): Initial state of the system
            config (AquireConfig | None): Configuration class for the Aquire experiment

        Attributes:
            # Permanent attributes
            H (PauliSum): Hamiltonian of the system
            pauli_block_sizes (list): Sizes of the Pauli blocks in the Hamiltonian

            # Supposed to change during experiment
            data (np.ndarray): Measurement outcome data collected so far for each Paulistring
            cliques (list): List of cliques measured so far
            circuits (list): List of circuits used for measurements of cliques
            total_shots (int): Total number of measurements allocated so far
            scaling_matrix (np.ndarray): Number of measurements allocated for each pair of Paulistrings
            covariance_graph (graph): Covariance graph of the Hamiltonian given the measurement data collected so far
            circuit_dictionary (dict): Dictionary mapping cliques to circuits used for their measurement
            measurement_results (list): List of measurement results collected so far
            update_steps (list): List of total shots at which covariance graph updates were performed
            diagnostic_circuits (list): List of diagnostic circuits constructed so far (only if diagnostics enabled)
            diagnostic_states (list): List of diagnostic states constructed so far (only if diagnostics enabled)
            diagnostic_state_preparation_circuits (list): List of circuits used to prepare diagnostic states (only if
                                                          diagnostics enabled)
            diagnostic_data (np.ndarray): Measurement outcome data collected so far for each Paulistring in
                                          diagnostic measurements (only if diagnostics enabled)
            diagnostic_results (list): List of diagnostic measurement results collected so far (only if diagnostics
                                       enabled)

            # Checkpoints (some of the above data collected at specific points)
            covariance_graph_checkpoints (list): Covariance graph at each covariance graph update

            # Results
            estimated_mean (list): Estimated mean of the Hamiltonian at each covariance graph update
            statistical_variance (list): Estimated statistical variance of the Hamiltonian at each covariance graph
                                         update
            systematic_variance (list): Estimated systematic variance of the Hamiltonian at each covariance graph update
                                        (only if diagnostics enabled)

            # Comparison Values: not used in the algorithm
            true_mean_value (float): True mean of the Hamiltonian with respect to the state psi (only if
                                     true_values_flag is True)
            true_statistical_variance_value (list): True statistical variance of the Hamiltonian with respect to the
                                                    state psi at each covariance graph update (only if true_values_flag
                                                    is True)

        """
        # TODO: implement all the steps needed to prepare the Hamiltonian
        P, pauli_block_sizes = sort_hamiltonian(H)
        P = weight_to_phase(P)
        if P.n_paulis() < H.n_paulis():
            for i in range(H.n_paulis()):
                if H.weights[i] != 0 and H[i, :].is_identity():
                    warnings.warn("Identity term with weight" + str(H.weights[i].real) + "ignored.", UserWarning)

        if not P.is_hermitian():
            raise Exception("Hamiltonian is not Hermitian.")

        # supposed to be permanent
        self._H = P
        self._pauli_block_sizes = pauli_block_sizes  # might be able to write a function that can calculate this quickly

        # Config for the experiment
        if config is None:
            self.config = AquireConfig(self.H, psi=np.array(psi) if psi is not None else None)
        else:
            self.config = config
            self.config.H = self.H

        # supposed to change during experiment
        self.cliques = []
        self.circuits = []
        _, self.circuit_dictionary = construct_circuit_list(self.H, self.config.clique_covering, {})
        self.measurement_results = []
        self.scaling_matrix = np.eye(self.H.n_paulis(), dtype=int)
        self.data = np.zeros((self.H.n_paulis(), self.H.n_paulis(), int(self.H.lcm)))
        self.covariance_graph = graph(
            np.diag([np.conj(self.H.weights[_]) * self.H.weights[_] for _ in range(self.H.n_paulis())]))
        self.update_steps = []
        self.diagnostic_circuits = []
        self.diagnostic_state_preparation_circuits = []
        self.diagnostic_states = []
        self.diagnostic_results = []
        self.diagnostic_data = np.zeros((self.H.n_paulis(), 2))

        # checkpoints (some of the above data collected at specific points)
        self.covariance_graph_checkpoints = []

        # results
        self.estimated_mean = []
        self.statistical_variance = []
        self.systematic_variance = []

        # Comparison values: not used in the algorithm
        # initially set to None, can be set later if desired and H not too large
        if self.config.calculate_true_values and self.config.psi is not None:
            self.true_mean_value = true_mean(self.H, self.config.psi)
            self.true_statistical_variance_value = []
        elif self.config.calculate_true_values and self.config.psi is None:
            warnings.warn("Warning: true values not available without state psi, calculate_true_values set to False.",
                          UserWarning)
            self.config.calculate_true_values = False
        else:
            pass

    @property
    def H(self):
        return self._H

    @property
    def pauli_block_sizes(self):
        return self._pauli_block_sizes

    ###################################################################
    # Attribute-like Methods ##########################################
    ###################################################################

    def total_shots(self):
        """
        Get the total number of shots allocated so far.

        Returns
        -------
        int
            The total number of shots allocated so far.
        """
        return len(self.cliques)

    def shots_since_last_update(self):
        """
        Get the number of shots allocated since the last covariance graph update.

        Returns
        -------
        int
            The number of shots allocated since the last covariance graph update.
        """
        return self.total_shots() - self.update_steps[-1] if self.update_steps else self.total_shots()

    def cliques_since_last_update(self):
        """
        Get the list of cliques measured since the last covariance graph update.

        Returns
        -------
        list
            The list of cliques measured since the last covariance graph update.
        """
        return self.cliques[self.update_steps[-1]:] if self.update_steps else self.cliques

    def measurement_circuits_since_last_update(self):
        """
        Get the list of measurement circuits used since the last covariance graph update.

        Returns
        -------
        list
            The list of measurement circuits used since the last covariance graph update.
        """
        return self.circuits[self.update_steps[-1]:] if self.update_steps else self.circuits

    def diagnostic_circuits_since_last_update(self):
        """
        Get the list of diagnostic circuits constructed since the last covariance graph update.

        Returns
        -------
        list
            The list of diagnostic circuits constructed since the last covariance graph update.
        """
        return (self.diagnostic_circuits[len(self.diagnostic_results):]
                if self.update_steps else self.diagnostic_circuits)

    def diagnostic_states_since_last_update(self):
        """
        Get the list of diagnostic states constructed since the last covariance graph update.

        Returns
        -------
        list
            The list of diagnostic states constructed since the last covariance graph update.
        """
        return self.diagnostic_states[len(self.diagnostic_results):] if self.update_steps else self.diagnostic_states

    def diagnostic_state_preparation_circuits_since_last_update(self):
        """
        Get the list of diagnostic state preparation circuits constructed since the last covariance graph update.

        Returns
        -------
        list
            The list of diagnostic state preparation circuits constructed since the last covariance graph update.
        """
        return (self.diagnostic_state_preparation_circuits[len(self.diagnostic_results):]
                if self.update_steps else self.diagnostic_state_preparation_circuits)

    def data_at_shot(self, shot: int | list[int]):
        data = np.zeros((self.H.n_paulis(), self.H.n_paulis(), int(self.H.lcm)))
        if isinstance(shot, int):
            data = update_data(self.cliques[:shot], self.measurement_results[:shot], data, self.circuit_dictionary)
            return data
        else:
            # technically one can use that data to build up the next one, but this is simpler
            data_list = []
            for s in shot:
                data = update_data(self.cliques[:s], self.measurement_results[:s], data, self.circuit_dictionary)
                data_list.append(data)
            return data_list

    def scaling_matrix_at_shot(self, shot: int | list[int]):
        if isinstance(shot, int):
            shot = [shot]
        scaling_matrices = []
        for s in shot:
            # technically one can use that data to build up the next one, but this is simpler right now
            scaling_matrix = np.eye(self.H.n_paulis(), dtype=int)
            for aa in self.cliques[:s]:
                scaling_matrix[np.ix_(aa, aa)] += np.ones((len(aa), len(aa)), dtype=int)
            scaling_matrix[range(self.H.n_paulis()), range(self.H.n_paulis())] -= np.ones(self.H.n_paulis(), dtype=int)
            scaling_matrices.append(scaling_matrix)
        if len(shot) == 1:
            return scaling_matrices[0]
        else:
            return scaling_matrices

    def diagnostic_data_at_shot(self, shot: int | list[int]):
        if self.config.diagnostic_mode is None:
            return
        data = np.zeros((self.H.n_paulis(), 2))
        if isinstance(shot, int):
            shot = [shot]
        for s in shot:
            # technically one can use that data to build up the next one, but this is simpler right now
            data = update_diagnostic_data(self.cliques[:s],
                                          self.diagnostic_results[:s],
                                          data,
                                          mode=self.config.diagnostic_mode)
        if len(shot) == 1:
            return data
        else:
            return data

    def set_mcmc_parameters(self, shots: int):
        if isinstance(self.config.mcmc_initial_samples_per_chain, int):
            N = self.config.mcmc_initial_samples_per_chain
        else:
            N = self.config.mcmc_initial_samples_per_chain(shots, **self.config.mcmc_initial_samples_per_chain_kwargs)

        if isinstance(self.config.mcmc_max_samples_per_chain, int):
            N_max = self.config.mcmc_max_samples_per_chain
        else:
            N_max = self.config.mcmc_max_samples_per_chain(shots, **self.config.mcmc_max_samples_per_chain_kwargs)
        if N > N_max:
            raise ValueError("Initial MCMC samples per chain cannot be greater than maximum MCMC samples per chain.")
        return N, N_max

    def covariance_graph_at_shot(self, shot: int | list[int]):
        if isinstance(shot, int):
            shot = [shot]
        graphs = []
        for s in shot:
            data = self.data_at_shot(shot)
            N, N_max = self.set_mcmc_parameters(s)
            A = bayes_covariance_graph(data,
                                       self.H.weights,
                                       self.config.commutation_graph,
                                       self.H.n_paulis(),
                                       self.pauli_block_sizes,
                                       int(self.H.lcm),
                                       N_chain=self.config.mcmc_number_of_chains,
                                       N=N,
                                       N_max=N_max)
            graphs.append(graph(A))
        if len(shot) == 1:
            return graphs[0]
        else:
            return graphs

    ###################################################################
    # Main experiment methods #########################################
    ###################################################################

    def allocate_measurements(self, shots):
        """
        Choose new cliques to measure according to the allocation mode and
        construct new circuits to measure the cliques.

        Parameters
        ----------
        shots : int
            The number of new cliques to measure.

        Changes
        -------
        self.scaling_matrix : np.ndarray
            Incremented on the diagonal by 1 for each Pauli before allocation,
            incremented on sub-matrices corresponding to new cliques during allocation,
            and decremented on the diagonal by 1 for each Pauli after allocation.
        self.cliques : list
            Extended with the newly chosen cliques.
        self.circuit_dictionary : dict
            Updated with new circuits constructed for the new cliques.
        self.circuits : list
            Extended with the new circuits.
        """
        if self.total_shots() > 0:
            self.scaling_matrix[range(self.H.n_paulis()), range(self.H.n_paulis())
                                ] += np.ones(self.H.n_paulis(), dtype=int)
        Ones = [np.ones((i, i), dtype=int) for i in range(self.H.n_paulis() + 1)]
        new_cliques = []
        for i in range(shots):
            aa = choose_measurement(self.scaling_matrix, self.covariance_graph.adj,
                                    self.config.clique_covering, self.config.allocation_mode)
            new_cliques.append(aa)
            self.scaling_matrix[np.ix_(aa, aa)] += Ones[len(aa)]

        self.cliques += new_cliques
        self.scaling_matrix[range(self.H.n_paulis()), range(self.H.n_paulis())] -= np.ones(self.H.n_paulis(), dtype=int)
        circuit_list, self.circuit_dictionary = construct_circuit_list(self.H, new_cliques, self.circuit_dictionary)
        self.circuits += circuit_list
        return circuit_list

    def construct_diagnostic_circuits(self):
        """
        Construct diagnostic circuits based on the last allocated circuits.

        Parameters
        ----------
        None

        Changes
        -------
        self.diagnostic_circuits : list
            Extended with the newly constructed diagnostic circuits.
        self.diagnostic_states : list
            Extended with the newly constructed diagnostic states.
        self.diagnostic_state_preparation_circuits : list
            Extended with the newly constructed diagnostic state preparation-circuits.
        """
        self.config.enable_diagnostics = True
        if self.config.diagnostic_mode is None:
            self.config.diagnostic_mode = 'Zero'
        n = len(self.diagnostic_circuits)
        diagnostic_circuit_list = construct_diagnostic_circuits(self.circuits[n:])
        self.diagnostic_circuits += diagnostic_circuit_list
        diagnostic_state_list, dsp_circuits_list = construct_diagnostic_states(diagnostic_circuit_list,
                                                                               mode=self.config.diagnostic_mode)
        self.diagnostic_states += diagnostic_state_list
        self.diagnostic_state_preparation_circuits += dsp_circuits_list
        return diagnostic_circuit_list, dsp_circuits_list

    def update_covariance_graph(self):
        """
        Update the covariance graph.

        Parameters
        ----------
        None

        Changes
        -------
        self.update_steps : list
            Extended with the current total number of shots.
        self.covariance_graph : graph
            Updated with the new covariance graph.
        self.covariance_graph_checkpoints : list
            Extended with a copy of the current covariance graph.
        self.estimated_mean : list
            Extended with the estimated mean of the current data.
        self.statistical_variance : list
            Extended with the estimated statistical variance of the current data.
        self.true_statistical_variance_value : list
            Extended with the true statistical variance of the current data if true_values_flag is True.
        """
        if self.total_shots() == self.update_steps[-1]:
            warnings.warn("No new data to update covariance graph.", UserWarning)
            return
        self.update_steps.append(self.total_shots())
        N, N_max = self.set_mcmc_parameters(self.total_shots())
        A = bayes_covariance_graph(self.data,
                                   self.H.weights,
                                   self.config.commutation_graph,
                                   self.H.n_paulis(),
                                   self.pauli_block_sizes,
                                   int(self.H.lcm),
                                   N_chain=self.config.mcmc_number_of_chains,
                                   N=N,
                                   N_max=N_max)

        self.covariance_graph = graph(A)
        self.covariance_graph_checkpoints.append(self.covariance_graph.copy())

        self.estimated_mean.append(calculate_mean_estimate(self.data, self.H.weights))
        self.statistical_variance.append(calculate_statistical_variance_estimate(
            self.covariance_graph, self.scaling_matrix))
        if self.config.enable_debug_checks:
            self.check_results()
        if self.config.calculate_true_values:
            self.true_statistical_variance_value.append(true_statistical_variance(
                self.H, self.config.psi, self.scaling_matrix, self.H.weights))

    def input_measurement_data(self, measurement_results: list):
        """
        Input new measurement data.

        Parameters
        ----------
        measurement_results : list
            List of measurement results for newly allocated measurements.

        Changes
        -------
        self.data : np.ndarray
            Updated with the input measurement results.
        self.measurement_results : list
            Extended with the input measurement results.

        Raises
        ------
        Exception
            If data already input for all circuits or not enough measurement results input.
        """
        if len(self.measurement_results) == len(self.circuits):
            warnings.warn("Data already input for all circuits.", UserWarning)
            return
        if len(measurement_results) < len(self.cliques_since_last_update()):
            warnings.warn("Not enough measurement results input. Please input at least as many results as the "
                          "number of newly allocated measurements.", UserWarning)
            return

        self.data = update_data(self.cliques_since_last_update(),
                                measurement_results,
                                self.data,
                                self.circuit_dictionary)

        self.measurement_results += measurement_results

        if self.config.enable_debug_checks:
            self.check_measurement_data()

        if self.config.auto_update_covariance_graph:
            self.update_covariance_graph()

    def input_diagnostic_data(self, diagnostic_results: list):
        """
        Input new diagnostic data.

        Parameters
        ----------
        diagnostic_results : list
            List of diagnostic results for newly allocated diagnostic measurements.

        Changes
        -------
        self.diagnostic_flag : bool
            Set to True.
        self.diagnostic_results : list
            Extended with the input diagnostic results.
        self.diagnostic_data_checkpoints : list
            Updated with the new diagnostic data.
        self.systematic_variance : list
            Updated with the new systematic variance estimate.

        Raises
        ------
        Exception
            If too many diagnostic results input.
        """
        if len(diagnostic_results) > len(self.diagnostic_circuits_since_last_update()):
            warnings.warn(
                "Too many diagnostic results input. Please input at most as many results as the number of newly "
                "allocated diagnostic measurements.", UserWarning)
        self.config.enable_diagnostics = True
        if self.config.diagnostic_mode is None:
            self.config.diagnostic_mode = 'Zero'
        self.diagnostic_results += diagnostic_results

        while len(self.systematic_variance) < len(self.update_steps):
            i = len(self.systematic_variance)
            if self.update_steps[i] > len(self.diagnostic_results):
                break
            new_cliques = self.cliques[self.update_steps[i - 1]:self.update_steps[i]]
            new_diagnostic_results = self.diagnostic_results[self.update_steps[i - 1]:self.update_steps[i]]
            self.diagnostic_data = update_diagnostic_data(new_cliques,
                                                          new_diagnostic_results,
                                                          self.diagnostic_data,
                                                          mode=self.config.diagnostic_mode)
            if self.config.enable_debug_checks:
                self.check_diagnostic_data()
            self.systematic_variance.append(calculate_systematic_variance_estimate(self.data, self.H.weights,
                                                                                   self.diagnostic_data))

    def simulate_measurement_results(self):
        """
        Simulate measurement results for newly allocated measurements.

        Returns
        -------
        None

        Changes
        -------
        self.data : np.ndarray
            Updated with the simulated measurement results.
        self.measurement_results : list
            Extended with the simulated measurement results.
        """
        if self.config.psi is None:
            raise Exception("State psi not provided, cannot simulate measurement results.")

        simulated_measurement_results = []
        for aa in self.cliques_since_last_update():
            P1, C, _ = self.circuit_dictionary[str(aa)]
            result = simulate_measurement(P1, self.config.psi, C)
            if self.config.enable_simulated_hardware_noise:
                noise_probability = self.config.noise_probability_function(C, *self.config.noise_args,
                                                                           **self.config.noise_kwargs)
                if np.random.rand() < noise_probability:
                    result = self.config.error_function(result, *self.config.error_args, **self.config.error_kwargs)
            simulated_measurement_results.append(result)

        self.data = update_data(self.cliques_since_last_update(),
                                simulated_measurement_results,
                                self.data,
                                self.circuit_dictionary)

        self.measurement_results += simulated_measurement_results

        if self.config.enable_debug_checks:
            self.check_measurement_data()

        if self.config.auto_update_covariance_graph:
            self.update_covariance_graph()

    def simulate_diagnostic_results(self):
        """
        Simulate diagnostic results for newly allocated diagnostic measurements.

        Returns
        -------
        None

        Changes
        -------
        self.diagnostic_results : list
            Extended with the simulated diagnostic results.
        self.last_update_diagnostic_circuits : list
            Trimmed to the length of the simulated diagnostic results if the length of the simulated
            diagnostic results is less than the length of self.last_update_diagnostic_circuits.
        self.diagnostic_data_checkpoints : list
            Extended with the diagnostic data at each measurement step.
        self.systematic_variance : list
            Extended with the estimated systematic variance at each measurement step.
        """
        if self.config.psi is None:
            raise Exception("State psi not provided, cannot simulate measurement results.")
        self.config.enable_diagnostics = True
        simulated_diagnostic_results = []
        for i, dsp_circuit in enumerate(self.diagnostic_circuits_since_last_update()):
            result = simulate_measurement(self.H, self.config.psi, dsp_circuit)
            noise_probability = self.config.noise_probability_function(dsp_circuit, *self.config.noise_args,
                                                                       **self.config.noise_kwargs)

            if np.random.rand() < noise_probability:
                result = self.config.error_function(result, *self.config.error_args,
                                                    **self.config.error_kwargs)

            simulated_diagnostic_results.append(result)
        self.diagnostic_results += simulated_diagnostic_results

        while len(self.systematic_variance) < len(self.update_steps):
            i = len(self.systematic_variance)
            if self.update_steps[i] > len(self.diagnostic_results):
                break
            new_cliques = self.cliques[self.update_steps[i - 1]:self.update_steps[i]]
            new_diagnostic_results = self.diagnostic_results[self.update_steps[i - 1]:self.update_steps[i]]
            self.diagnostic_data = update_diagnostic_data(new_cliques,
                                                          new_diagnostic_results,
                                                          self.diagnostic_data,
                                                          mode=self.config.diagnostic_mode)
            if self.config.enable_debug_checks:
                self.check_diagnostic_data()

            self.systematic_variance.append(calculate_systematic_variance_estimate(self.data, self.H.weights,
                                                                                   self.diagnostic_data))

    def simulate_observable(self, update_steps: list[int]):
        """
        Simulate the measurement of an observable adaptively changing the underlying covariance graph at each value in
        update_steps.

        Parameters
        ----------
        update_steps : list[int]
            A list of integers representing the total number of shots to take at each measurement step.
        hardware_noise : bool
            A boolean indicating whether to include hardware noise in the simulation.

        Returns
        -------
        None

        Notes
        -----
        The simulate_observable method allocates measurements, simulates measurement results, updates the covariance
        graph, and optionally constructs and simulates diagnostic circuits. It repeats this process for the number of
        rounds specified by the update_steps list. The estimated mean, statistical variance, and optionally the
        systematic variance are recorded at each measurement step.

        Examples
        --------
        >>> acquire.simulate_observable([1000, 2000, 3000])
        """
        if self.config.psi is None:
            raise Exception("State psi not provided, cannot simulate observable.")

        shots_per_round = [update_steps[0]] + [update_steps[i + 1] - update_steps[i]
                                               for i in range(len(update_steps) - 1)]
        for i in range(len(shots_per_round)):
            self.allocate_measurements(shots_per_round[i])
            self.simulate_measurement_results()
            if not self.config.auto_update_covariance_graph:
                self.update_covariance_graph()
            if self.config.enable_diagnostics:
                self.construct_diagnostic_circuits()
                self.simulate_diagnostic_results()

    ###################################################################
    # Meta methods ####################################################
    ###################################################################

    # TODO: construct working save function
    def save(self, filename: str):
        """
        Save the current state of the Aquire object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the Aquire object to.

        Returns
        -------
        None
        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # TODO: construct working load function
    @classmethod
    def load(cls, filename: str):
        """
        Load an Aquire object from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the Aquire object from.

        Returns
        -------
        Aquire
            The loaded Aquire object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save_results(self, filename: str):
        """
        Save the results of the Aquire object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the results to.

        Returns
        -------
        None

        Notes
        -----
        The saved results include the estimated mean, statistical variance, and
        optionally the systematic variance, true mean, and true statistical
        variance.

        Examples
        --------
        >>> acquire.save_results('results.pkl')
        """
        results = {
            'estimated_mean': self.estimated_mean,
            'statistical_variance': self.statistical_variance
        }
        if self.config.enable_diagnostics:
            results['systematic_variance'] = self.systematic_variance
        if self.config.calculate_true_values:
            results['true_mean'] = self.true_mean_value
            results['true_statistical_variance'] = self.true_statistical_variance_value
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def __str__(self) -> str:
        aquire_str = "AQUIRE experiment for observable:\n"
        aquire_str += str(self.H) + "\n"
        if self.config.psi is not None:
            aquire_str += "State vector psi:\n"
            aquire_str += str(self.config.psi) + "\n"
        aquire_str += " \n"
        aquire_str += "General settings:\n"
        aquire_str += 'Commutation mode: ' + self.config.commutation_mode + '\n'
        aquire_str += 'Allocation mode: ' + self.config.allocation_mode + '\n'
        if self.config.enable_diagnostics:
            aquire_str += 'Diagnostic mode: ' + self.config.diagnostic_mode + '\n'
        aquire_str += " \n"
        aquire_str += "MCMC settings:\n"
        aquire_str += 'MCMC number of chains: ' + str(self.config.mcmc_number_of_chains) + '\n'
        if isinstance(self.config.mcmc_initial_samples_per_chain, int):
            aquire_str += 'MCMC initial samples per chain: ' + str(self.config.mcmc_initial_samples_per_chain) + '\n'
        else:
            aquire_str += 'MCMC initial samples per chain: ' + \
                str(self.config.mcmc_initial_samples_per_chain.__name__) + '\n'
            aquire_str += str(self.config.mcmc_initial_samples_per_chain_kwargs) + '\n'

        if isinstance(self.config.mcmc_max_samples_per_chain, int):
            aquire_str += 'MCMC max samples per chain: ' + str(self.config.mcmc_max_samples_per_chain) + '\n'
        else:
            aquire_str += 'MCMC max samples per chain: ' + str(self.config.mcmc_max_samples_per_chain.__name__) + '\n'
            aquire_str += str(self.config.mcmc_max_samples_per_chain_kwargs) + '\n'

        if self.config.enable_simulated_hardware_noise:
            aquire_str += " \n"
            aquire_str += 'Noise probability function: ' + str(self.config.noise_probability_function.__name__) + '\n'
            aquire_str += str(self.config.noise_args) + '\n'
            aquire_str += str(self.config.noise_kwargs) + '\n'

            aquire_str += 'Error function: ' + str(self.config.error_function.__name__) + '\n'
            aquire_str += str(self.config.error_args) + '\n'
            aquire_str += str(self.config.error_kwargs) + '\n'

        aquire_str += " \n"
        aquire_str += "Results:\n"
        aquire_str += "Covariance graph updated at steps: " + str(self.update_steps) + '\n'
        aquire_str += "Estimated means: " + str(self.estimated_mean) + '\n'
        aquire_str += "Statistical variances: " + str(self.statistical_variance) + '\n'
        if self.config.enable_diagnostics:
            aquire_str += "Systematic variances: " + str(self.systematic_variance) + '\n'
        if self.config.calculate_true_values:
            aquire_str += "True means: " + str(self.true_mean_value) + '\n'
            aquire_str += "True statistical variances: " + str(self.true_statistical_variance_value)
        return aquire_str

    ###################################################################
    # Validation Check methods ########################################
    ###################################################################

    def check_measurement_data(self):
        for i in range(self.H.n_paulis()):
            for j in range(self.H.n_paulis()):
                if self.data[i, j, :] < 0:
                    warnings.warn("Negative value in data.", UserWarning)

    def check_diagnostic_data(self):
        for j in range(self.H.n_paulis()):
            if self.diagnostic_data[j, 0] < 0:
                warnings.warn("Negative value in diagnostic data.", UserWarning)
            if self.diagnostic_data[j, 1] < 0:
                warnings.warn("Negative value in diagnostic data.", UserWarning)
            if np.sum(self.diagnostic_data[j, :]) > 10:
                if self.diagnostic_data[j, 0] / np.sum(self.diagnostic_data[j, :]) > 0.9:
                    warnings.warn("Abnormally high error rate.", UserWarning)

    def check_results(self):
        if self.statistical_variance[-1] < 0:
            warnings.warn("Negative statistical variance estimate.", UserWarning)


def simulate_measurement(PauliSum: PauliSum, psi: list[float | complex] | list[float] | list[complex] | np.ndarray,
                         circuit: Circuit):
    # Simulate measurement
    psi_diag = circuit.unitary() @ psi
    pdf = np.abs(psi_diag * psi_diag.conj())
    dims1 = PauliSum.dimensions
    a1 = np.random.choice(np.prod(dims1), p=pdf)
    result = int_to_bases(a1, dims1)
    return result


def plot_aquire(model: Aquire, filename: str | None = None):
    """
    Plot the estimated mean value and statistical/systematic variance as a function of the number of shots taken.

    Parameters
    ----------
    filename : str | None
        The filename to save the plot to. If None, the plot will not be saved.

    Returns
    -------
    None

    Notes
    -----
    The plot shows the estimated mean value and statistical/systematic variance as a function of the number of
    shots taken. The estimated mean value is plotted with error bars representing the statistical/systematic
    variance. The true mean value and statistical/systematic variance can be plotted as dashed lines if the
    true_values_flag is True. The plot is saved to the specified filename if it is not None.

    Examples
    --------
    >>> acquire.plot('plot.png')
    """
    cm = 1 / 2.54
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(23 * cm, 11 * cm))
    c_stat = '#4FCB8D'
    c_dev = '#D93035'
    cs = 1.75               # capsize
    ct = 1.5                  # cap-thickness
    ms = 5.5                 # marker size
    ew = 1.5                  # errorbar linewidth
    mew = 1.5               # marker edge width
    ec = c_stat  # bitwise errorbar edge-color
    mec = '#ffffff'         # marker edge color
    mfc = c_stat  # bitwise marker face color
    fmt = 'o'            # bitwise marker type
    label_fontsize = 14

    M = np.array(model.update_steps)
    est_mean = np.array(model.estimated_mean)
    stat_var = np.array(model.statistical_variance)
    if model.config.enable_diagnostics:
        sys_var = np.array(model.systematic_variance)

    # Mean Plot
    if model.config.calculate_true_values:
        H_mean = model.true_mean_value
        plot_mean = np.abs(est_mean - H_mean) / np.abs(H_mean)
        ax[0].plot([M[0], M[-1]], [0, 0], 'k--')
        ax[0].set_ylabel(r'$|\widetilde{O} - \langle \hat{O} \rangle|$', fontsize=label_fontsize)
        if model.config.enable_diagnostics:
            plot_errorbar = np.sqrt(stat_var + sys_var) / np.abs(H_mean)
        else:
            plot_errorbar = np.sqrt(stat_var) / np.abs(H_mean)
    else:
        plot_mean = est_mean
        ax[0].set_ylabel(r'$\widetilde{O}$', fontsize=label_fontsize)
        if model.config.enable_diagnostics:
            plot_errorbar = np.sqrt(stat_var + sys_var)
        else:
            plot_errorbar = np.sqrt(stat_var)
    ax[0].errorbar(M, plot_mean, yerr=plot_errorbar,
                   fmt=fmt, ecolor=ec,
                   capsize=cs, capthick=ct, markersize=ms, elinewidth=ew,
                   mec=mec, mew=mew, mfc=mfc)

    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'shots $M$', fontsize=label_fontsize)

    # Error Plot
    if model.config.calculate_true_values:
        stat_error = stat_var * M / (H_mean)**2
        if model.config.enable_diagnostics:
            phys_error = sys_var * M / (H_mean)**2
    else:
        stat_error = stat_var * M
        if model.config.enable_diagnostics:
            phys_error = sys_var * M

    r = 1.25  # ~±22% in log10
    lefts = M / np.sqrt(r)
    rights = M * np.sqrt(r)
    widths = rights - lefts

    ax[1].bar(lefts, stat_error, width=widths, align='edge',
              label='Statistical Variance', color=c_stat)
    if model.config.enable_diagnostics:
        ax[1].bar(lefts, phys_error, width=widths, align='edge',
                  bottom=stat_error, label='Systematic Variance',
                  color=c_dev)

    if model.config.calculate_true_values:
        ax[1].plot(M, model.true_statistical_variance_value * M / (H_mean)**2, 'k--', label='True Stat. Variance')

    ax[1].set_xscale('log')
    ax[1].set_ylabel(r'$M \cdot (\widetilde{\Delta O})^2$', fontsize=label_fontsize)
    ax[1].set_xlabel(r'shots $M$', fontsize=label_fontsize)
    ax[1].legend()

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    if filename is not None:
        plt.savefig(filename, dpi=1200)

    plt.show()
