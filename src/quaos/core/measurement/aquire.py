import numpy as np
from quaos.core.paulis import PauliSum
from quaos.core.measurement.allocation import (sort_hamiltonian, get_phase_matrix, choose_measurement,
                                               construct_circuit_list, update_data,
                                               diagnostic_circuits, standard_error_function, diagnostic_states,
                                               update_diagnostic_data, standard_noise_probability_function)
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
                 psi: list[float | complex] | list[float] | list[complex] | np.ndarray | None = None,
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
        """
        Constructor for the Aquire class.

        Parameters:
            H (PauliSum): Hamiltonian of the system
            psi (list[float | complex] | list[float] | list[complex] | np.ndarray): Initial state of the system
            general_commutation (bool): Whether to use general commutation or qudit-wise commutation
            true_values (bool): Whether to calculate true mean and statistical variance
            allocation_mode (str): Allocation mode for measurements
            N_chain (int): Number of measurement chains
            N_mcmc (int): Number of MCMC steps per chain
            N_mcmc_max (int): Maximum number of MCMC steps
            mcmc_shot_scale (float): Scale for MCMC steps
            diagnostic_mode (str): Diagnostic mode for the experiment
            noise_probability_function (Callable): Function to calculate the probability of a noisy measurement
            error_function (Callable): Function to calculate the error correction
            noise_args (list): Arguments for the noise probability function
            noise_kwargs (dict): Keyword arguments for the noise probability function
            error_args (list): Arguments for the error correction function
            error_kwargs (dict): Keyword arguments for the error correction function

        Attributes:
            # Permanent attributes
            H (PauliSum): Hamiltonian of the system
            weights (np.ndarray): Weights of the Pauli terms in the Hamiltonian
            pauli_block_sizes (list): Sizes of the Pauli blocks in the Hamiltonian
            psi (np.ndarray): State to calculate estimates for
            n_paulis (int): Number of Pauli terms in the Hamiltonian
            n_qudits (int): Number of qudits in the Hamiltonian
            dimension (int): Dimension of the system
            diagnostic_mode (str): Diagnostic mode for the experiment
            noise_probability_function (Callable): Function to calculate the probability of a noisy measurement
            error_function (Callable): Function to calculate the error applied to a measurement result
            noise_args (list): Arguments for the noise probability function
            noise_kwargs (dict): Keyword arguments for the noise probability function
            error_args (list): Arguments for the error correction function
            error_kwargs (dict): Keyword arguments for the error correction function
            true_values_flag (bool): Whether to calculate true mean and statistical variance

            # Changeable attributes
            N_chain (int): Number of measurement chains
            N_mcmc (int): Number of MCMC steps per chain
            N_mcmc_max (int): Maximum number of MCMC steps
            mcmc_shot_scale (float): Scale for MCMC steps
            general_commutation (bool): Whether to use general commutation or qudit-wise commutation
            allocation_mode (str): Allocation mode for measurements
            diagnostic_flag (bool): Whether diagnostics for systematic errors are enabled (automatically set to True if
                                    noise or error functions are provided or if diagnostic circuits are constructed)

            # Dependent on changeable parameters
            CG (graph): Commutation graph of the Hamiltonian
            clique_covering (list): List of cliques covering the commutation graph
            k_phases (np.ndarray): Matrix of phase of the products of Paulistrings in the Hamiltonian (could probably
                                   be improved with new PauliSum and PauliString methods)

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

            # Variables that track changes since last covariance update
            last_update_shots (int): Number of measurements allocated since last covariance graph update
            last_update_cliques (list): List of cliques measured since last covariance graph update
            last_update_circuits (list): List of circuits used for measurements since last covariance graph update

            # Checkpoints (some of the above data collected at specific points)
            data_checkpoints (list): Measurement outcome data at each covariance graph update
            scaling_matrix_checkpoints (list): Scaling matrix at each covariance graph update
            covariance_graph_checkpoints (list): Covariance graph at each covariance graph update
            diagnostic_data_checkpoints (list): Diagnostic measurement outcome data at each covariance graph update
                                                (only if diagnostics enabled)
            last_update_diagnostic_circuits (list): List of diagnostic circuits constructed since last covariance graph
                                                    update (only if diagnostics enabled)
            last_update_diagnostic_states (list): List of diagnostic states constructed since last covariance graph
                                                  update (only if diagnostics enabled)
            last_update_diagnostic_state_preparation_circuits (list): List of circuits used to prepare diagnostic states
                                                                      since last covariance graph update (only if
                                                                      diagnostics enabled)

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
        P, pauli_block_sizes = sort_hamiltonian(H)
        P.phase_to_weight()
        if P.n_paulis() < H.n_paulis():
            for i in range(H.n_paulis()):
                if H.weights[i] != 0 and H[i,:].is_identity():
                    print("Identity term with weight", H.weights[i].real, "ignored.")

        # supposed to be permanent
        self.H = P
        self.weights = P.weights
        self.pauli_block_sizes = pauli_block_sizes
        self.psi = np.array(psi)
        self.n_paulis = P.n_paulis()
        self.n_qudits = P.n_qudits()
        self.dimension = int(P.lcm)
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
        self.CG = commutation_graph(self.H) if general_commutation else quditwise_commutation_graph(self.H)
        self.clique_covering = weighted_vertex_covering_maximal_cliques(self.CG, cc=self.weights, k=3)
        self.k_phases = get_phase_matrix(self.H, self.CG)

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
        self.diagnostic_circuits = []
        self.diagnostic_states = []
        self.diagnostic_state_preparation_circuits = []
        self.diagnostic_data = np.zeros((self.n_paulis, 2))
        self.diagnostic_results = []

        # variables that track changes since last covariance update
        self.last_update_shots = 0
        self.last_update_cliques = []
        self.last_update_circuits = []

        # checkpoints (some of the above data collected at specific points)
        self.data_checkpoints = []
        self.scaling_matrix_checkpoints = []
        self.covariance_graph_checkpoints = []
        self.diagnostic_data_checkpoints = []
        self.last_update_diagnostic_circuits = []
        self.last_update_diagnostic_states = []
        self.last_update_diagnostic_state_preparation_circuits = []

        # results
        self.estimated_mean = []
        self.statistical_variance = []
        self.systematic_variance = []

        # Comparison values: not used in the algorithm
        # initially set to None, can be set later if desired and H not too large
        if self.true_values_flag and self.psi is not None:
            self.true_mean_value = true_mean(self.H, self.psi)
            self.true_statistical_variance_value = []
        elif self.true_values_flag and self.psi is None:
            print("Warning: true values not available without state psi, true_values_flag set to False.")
            self.true_values_flag = False
        else:
            pass

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
            incremented on submatrices corresponding to new cliques during allocation,
            and decremented on the diagonal by 1 for each Pauli after allocation.
        self.total_shots : int
            Increased by the value of `shots`.
        self.last_update_shots : int
            Increased by the value of `shots`.
        self.cliques : list
            Extended with the newly chosen cliques.
        self.last_update_cliques : list
            Extended with the newly chosen cliques.
        self.circuit_dictionary : dict
            Updated with new circuits constructed for the new cliques.
        self.circuits : list
            Extended with the new circuits.
        self.last_update_circuits : list
            Extended with the new circuits.
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
        circuit_list, self.circuit_dictionary = construct_circuit_list(self.H, new_cliques, self.circuit_dictionary)
        self.circuits += circuit_list
        self.last_update_circuits += circuit_list

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
        self.last_update_diagnostic_circuits : list
            Extended with the newly constructed diagnostic circuits.
        self.diagnostic_states : list
            Extended with the newly constructed diagnostic states.
        self.diagnostic_state_preparation_circuits : list
            Extended with the newly constructed diagnostic state preparation-circuits.
        """
        self.diagnostic_flag = True
        n = len(self.diagnostic_circuits)
        diagnostic_circuit_list = diagnostic_circuits(self.circuits[n:])
        self.diagnostic_circuits += diagnostic_circuit_list
        self.last_update_diagnostic_circuits += diagnostic_circuit_list
        diagnostic_state_list, dsp_circuits_list = diagnostic_states(diagnostic_circuit_list, mode=self.diagnostic_mode)
        self.diagnostic_states += diagnostic_state_list
        self.diagnostic_state_preparation_circuits += dsp_circuits_list
        self.last_update_diagnostic_states += diagnostic_state_list
        self.last_update_diagnostic_state_preparation_circuits += dsp_circuits_list

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
        self.data_checkpoints : list
            Extended with a copy of the current data.
        self.scaling_matrix_checkpoints : list
            Extended with a copy of the current scaling matrix.
        self.covariance_graph : graph
            Updated with the new covariance graph.
        self.covariance_graph_checkpoints : list
            Extended with a copy of the current covariance graph.
        self.estimated_mean : list
            Extended with the estimated mean of the current data.
        self.statistical_variance : list
            Extended with the estimated statistical variance of the current data.
        self.last_update_shots : int
            Reset to 0.
        self.last_update_cliques : list
            Reset to an empty list.
        self.last_update_circuits : list
            Reset to an empty list.
        self.true_statistical_variance_value : list
            Extended with the true statistical variance of the current data if true_values_flag is True.
        """
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
        self.last_update_diagnostic_circuits : list
            Updated with the newly allocated diagnostic circuits.
        self.diagnostic_data_checkpoints : list
            Updated with the new diagnostic data.
        self.systematic_variance : list
            Updated with the new systematic variance estimate.

        Raises
        ------
        Exception
            If too many diagnostic results input.
        """
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
        if self.psi is None:
            raise Exception("State psi not provided, cannot simulate measurement results.")
        simulated_measurement_results = []
        for aa in self.last_update_cliques:
            P1, C, _ = self.circuit_dictionary[str(aa)]
            psi_diag = C.unitary() @ self.psi
            pdf = np.abs(psi_diag * psi_diag.conj())
            dims1 = P1.dimensions
            a1 = np.random.choice(np.prod(dims1), p=pdf)
            result = int_to_bases(a1, dims1)
            if self.diagnostic_flag:
                if self.noise_probability_function is not None:
                    noise_probability = self.noise_probability_function(C, *self.noise_args, **self.noise_kwargs)
                else:
                    noise_probability = standard_noise_probability_function(C)

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
        if self.psi is None:
            raise Exception("State psi not provided, cannot simulate measurement results.")
        self.diagnostic_flag = True
        simulated_diagnostic_results = []
        for i, dsp_circuit in enumerate(self.last_update_diagnostic_circuits):
            psi_diag = dsp_circuit.unitary() @ self.diagnostic_states[i]
            pdf = np.abs(psi_diag * psi_diag.conj())
            dims = dsp_circuit.dimensions
            a1 = np.random.choice(np.prod(dims), p=pdf)
            result = int_to_bases(a1, dims)
            if self.noise_probability_function is not None:
                noise_probability = self.noise_probability_function(dsp_circuit, *self.noise_args, **self.noise_kwargs)
            else:
                noise_probability = standard_noise_probability_function(dsp_circuit)

            if np.random.rand() < noise_probability:
                if self.error_function is not None:
                    result = self.error_function(result, *self.error_args, **self.error_kwargs)
                else:
                    result = standard_error_function(result, self.H.dimensions)
            else:
                pass

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
        if self.diagnostic_flag:
            results['systematic_variance'] = self.systematic_variance
        if self.true_values_flag:
            results['true_mean'] = self.true_mean_value
            results['true_statistical_variance'] = self.true_statistical_variance_value
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def simulate_observable(self, update_steps: list[int], hardware_noise: bool = False):
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
        if self.psi is None:
            raise Exception("State psi not provided, cannot simulate observable.")
        if hardware_noise:
            self.diagnostic_flag = True
        initial_shots = update_steps[0]
        rounds = len(update_steps) - 1
        shots_per_round = [update_steps[i + 1] - update_steps[i] for i in range(rounds)]
        self.allocate_measurements(initial_shots)
        self.simulate_measurement_results()
        self.update_covariance_graph()
        if hardware_noise:
            self.construct_diagnostic_circuits()
            self.simulate_diagnostic_results()
        for i in range(rounds):
            self.allocate_measurements(shots_per_round[i])
            self.simulate_measurement_results()
            self.update_covariance_graph()
            if hardware_noise:
                self.construct_diagnostic_circuits()
                self.simulate_diagnostic_results()

    def plot(self, filename: str | None = None):
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
            plot_mean = np.abs(est_mean - H_mean) / np.abs(H_mean)
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
        ax[0].errorbar(M, plot_mean, yerr=plot_errorbar,
                       fmt=fmt, ecolor=ec,
                       capsize=cs, capthick=ct, markersize=ms, elinewidth=ew,
                       mec=mec, mew=mew, mfc=mfc)

        ax[0].set_xscale('log')
        ax[0].set_xlabel(r'shots $M$', fontsize=label_fontsize)

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
