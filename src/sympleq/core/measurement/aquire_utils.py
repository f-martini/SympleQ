import numpy as np
import matplotlib.pyplot as plt
from sympleq.core.measurement.allocation import scale_variances
from sympleq.core.measurement.covariance_graph import graph
from sympleq.core.measurement import Aquire
from sympleq.core.paulis import PauliSum


def calculate_mean_estimate(data: np.ndarray, weights: np.ndarray):
    p = data.shape[0]
    d = data.shape[2]
    mean = 0
    for i0 in range(p):
        total_counts = sum(data[i0, i0, i1] for i1 in range(d))
        if total_counts > 0:
            weighted_sum = sum(
                data[i0, i0, i1] * np.exp(2j * np.pi * i1 / d)
                for i1 in range(d))
            mean += weights[i0] * (weighted_sum / total_counts)
        else:
            mean += 0
    mean = mean.real
    return mean


def calculate_statistical_variance_estimate(covariance_graph: graph, scaling_matrix: np.ndarray):
    scaled_variance_graph = scale_variances(covariance_graph, scaling_matrix)
    stat_variance_estimate = np.sum(scaled_variance_graph.adj).real
    return stat_variance_estimate


def calculate_systematic_variance_estimate(data: np.ndarray, weights: np.ndarray, diagnostic_data: np.ndarray):
    p = data.shape[0]
    d = data.shape[2]
    # estimate w_i
    w = np.zeros((p, 2))
    for i in range(p):
        w[i, 0] = (diagnostic_data[i, 0] + 1) / (np.sum(diagnostic_data[i, :]) + 2)
        w[i, 1] = (diagnostic_data[i, 1] + 1) / (np.sum(diagnostic_data[i, :]) + 2)

    # estimate theta_i
    theta_est = np.zeros((p, d))
    for i in range(p):
        if np.sum(data[i, i, :]) > 0:
            for j in range(d):
                theta_est[i, j] = (data[i, i, j] + 1) / (np.sum(data[i, i, :]) + 2)
        else:
            theta_est[i, :] = 1 / d

    # eigenvalues
    xis = [np.exp(2 * 1j * np.pi * beta / d) for beta in range(d)]
    error_correction = np.sum([weights[i0] * np.sum([xis[beta] * (theta_est[i0, beta] - 1 / d)
                                                     for beta in range(d)]) * w[i0, 0] for i0 in range(p)])
    return np.abs(error_correction)**2


def true_mean(H: PauliSum, psi):
    mu = np.real(np.transpose(np.conjugate(psi)) @ H.to_hilbert_space() @ psi)
    return mu


def true_covariance_graph(H: PauliSum, psi):
    p = H.n_paulis()
    mm = [H.to_hilbert_space(i) for i in range(p)]
    psi_dag = psi.conj().T
    cc1 = [psi_dag @ mm[i] @ psi for i in range(p)]
    cc2 = [psi_dag @ mm[i].conj().T @ psi for i in range(p)]
    cm = np.zeros((p, p), dtype=complex)
    for i0 in range(p):
        for i1 in range(p):
            cov = (psi_dag @ mm[i0].conj().T @ mm[i1] @ psi) - cc2[i0] * cc1[i1]
            cm[i0, i1] = cov
    return graph(cm)


def true_statistical_variance(H, psi, S):
    sigma = np.sum(scale_variances(true_covariance_graph(H, psi), S).adj).real
    return sigma


def config_params():
    param_info = {
        "Hamiltonian": {
            "type": "PauliSum",
            "optional": False,
            "default": None,
            "description": "Hamiltonian/Observable to be simulated.",
            "accepted": "Any hermitian PauliSum.",
            "example": "PauliSum(['x1z1 x1z0 x0z0', 'x0z0 x1z0 x1z0', 'x0z1 x0z0 x0z1'], dimensions=[2, 2, 2])",
        },
        "psi": {
            "type": "np.ndarray | None",
            "optional": True,
            "default": None,
            "description": ("State vector to simulate the average of the observable. If None, comparison of "
                            "experimentally estimated values to true values is not possible."),
            "accepted": "list[float | complex] | list[float] | list[complex] | np.ndarray | None",
            "example": "np.ones(np.prod(Hamiltonian.dimensions)) / np.sqrt(np.prod(Hamiltonian.dimensions))",
        },
        "commutation_mode": {
            "type": "str",
            "optional": False,
            "default": "general",
            "description": ("Wether to use a general commutation graph or a bitwise commutation graph. 'bitwise', "
                            "'local' or 'quditwise' all indicate a bitwise commutation graph."),
            "accepted": "general | bitwise | local | quditwise"
        },
        "allocation_mode": {
            "type": "str",
            "optional": False,
            "default": "set",
            "description": ("Wether to allocate measurement shots to the clique with the highest (potential) decrease "
                            "in error (set) or wether to build a random probability distribution proportional to the "
                            "error decrease for each clique and distribute shots randomly according to that (random)"),
            "accepted": "random | set"
        },
        "diagnostic_mode": {
            "type": "str",
            "optional": False,
            "default": "Zero",
            "description": ("Wether to use the zero state for hardware error diagnostics or a random (computational) "
                            "stabilizer state of the circuit"),
            "accepted": "Zero | Random | informed"
        },
        "calculate_true_values": {
            "type": "bool",
            "optional": False,
            "default": True,
            "description": ("Wether to calculate the true mean and true statistical variance of the observable.")
        },
        "enable_diagnostics": {
            "type": "bool",
            "optional": False,
            "default": False,
            "description": ("Wether to enable hardware error diagnostics.")
        },
        "auto_update_covariance_graph": {
            "type": "bool",
            "optional": False,
            "default": True,
            "description": ("Wether to automatically update the covariance graph after simulating or inputting "
                            "measurement results.")
        },
        "enable_simulated_hardware_noise": {
            "type": "bool",
            "optional": False,
            "default": False,
            "description": ("Wether to enable simulated hardware noise (if measurements are simulated classically).")
        },
        "save_covariance_graph_checkpoints": {
            "type": "bool",
            "optional": False,
            "default": False,
            "description": ("Wether to save the covariance graph whenever it is updated.")
        },
        "verbose": {
            "type": "bool",
            "optional": False,
            "default": False,
            "description": ("Wether to print verbose output whenever settings are updated.")
        },
        "auto_update_settings": {
            "type": "bool",
            "optional": False,
            "default": True,
            "description": ("Wether to automatically update dependent settings after manual change of settings.")
        },
        "enable_debug_checks": {
            "type": "bool",
            "optional": False,
            "default": False,
            "description": ("Wether to checks that might slow down simulation but make sure that all values are "
                            "correct.")
        },
        "mcmc_number_of_chains": {
            "type": "int",
            "optional": False,
            "default": 8,
            "description": ("Number of MCMC chains to use for covariance calculation.")
        },
        "mcmc_initial_samples_per_chain": {
            "type": "int | Callable",
            "optional": False,
            "default": "function mcmc_number_initial_samples",
            "description": ("Number of initial samples to use for each MCMC chain. Can be integer if the same number "
                            "should be used for all shots, but normally for high shot numbers an increased precision "
                            "in the covariance estimate is desired. In that case this can be a function that takes the "
                            "current measurement shot number as well as other arguments as input to calculate the "
                            "number of initial mcmc samples. The function should return an integer. Additional "
                            "arguments can be parsed via mcmc_initial_samples_per_chain_kwargs."),
            "accepted": "int | Callable"
        },
        "mcmc_max_samples_per_chain": {
            "type": "int | Callable",
            "optional": False,
            "default": "function mcmc_number_max_samples",
            "description": ("Maximum number of samples to use for each MCMC chain. MCMC calculation will increase "
                            "number of samples either until convergence (according to geweke and rubin criterion) or "
                            "until the max number of samples is reached. Can be integer if the same number "
                            "should be used for all shots, but normally for high shot numbers an increased precision "
                            "in the covariance estimate is desired. In that case this can be a function that takes the "
                            "current measurement shot number as well as other arguments as input to calculate the "
                            "number of mcmc samples. The function should return an integer. Additional arguments can "
                            "be parsed via mcmc_max_samples_per_chain_kwargs."),
            "accepted": "int | Callable"
        },
        "noise_probability_function": {
            "type": "Callable",
            "optional": False,
            "default": "function standard_noise_probability_function",
            "description": ("Function that takes a circuit and additional noise arguments as input and returns the "
                            "noise probability of the circuit. Used to classically simulate noise during measurement. "
                            "Additional arguments can be parsed via noise_probability_args and "
                            "noise_probability_function_kwargs."),
            "accepted": "Callable"
        },
        "error_function": {
            "type": "Callable",
            "optional": False,
            "default": "function standard_error_function",
            "description": ("Function that takes a measurement result and additional error arguments as input and "
                            "returns a noisy measurement result. Used to classically simulate noise during measurement."
                            " Additional arguments can be parsed via error_function_args and error_function_kwargs."),
            "accepted": "Callable"
        },
        "commutation_graph": {
            "type": "graph",
            "description": ("Graph encoding the commutation relation between the paulistrings in the PauliSum. Here 1 "
                            "indicates commutation and 0 indicates non-commutation. Will be calculated based on the "
                            "input PauliSum and commutation_mode. Dependent setting that will automatically be updated "
                            "if either Hamiltonian or commutation_mode are changed and auto_update_settings is True."),
        },
        "clique_covering": {
            "type": "list[list[int]]",
            "description": ("Clique covering of the commutation graph. Will be calculated based on the input PauliSum "
                            "and commutation mode. Consists of a list of lists of integers. The integers refer to the "
                            "index of the paulistrings in the maximally connected clique")
        },
        "validate_parameters": {
            "type": "method",
            "input": None,
            "description": ("Checks that all parameters fulfill some loose validity criteria. Example: Normalization "
                            "of the state vector psi")
        },
        "update_all": {
            "type": "method",
            "input": None,
            "description": ("Updates all dependent parameters. Example: Commutation graph and clique covering.")
        },
        "from_json": {
            "type": "class method",
            "input": None,
            "description": ("Loads config from a json file.")
        },
        "to_json": {
            "type": "method",
            "input": None,
            "description": ("Saves config to a json file.")
        },
        "set_params": {
            "type": "method",
            "input": "kwargs",
            "description": ("Set multiple parameters at once."),
            "example": "AquireConfig.set_params(**{'commutation_mode': 'quditwise', 'auto_update_settings': False})"
        }
    }
    return param_info


def aquire_params():
    param_info = {
        "config": {
            "type": "AquireConfig",
            "optional": True,
            "default": None,
            "description": ("Configuration class for Aquire. Used to set various parameters that influence the "
                            "experiment. For more information call AquireConfig.info().")
        },
        "cliques": {
            "type": "list[list[int]]",
            "description": ("List of cliques selected for measurement. Each clique is a list of indices, "
                            "representing the paulistrings in the given observable.")
        },
        "circuits": {
            "type": "list[Circuit]",
            "description": "List of circuits to measure the selected cliques. Each circuit is a Circuit object."
        },
        "circuit_dictionary": {
            "type": "dict",
            "description": ("Dictionary mapping cliques to sub-PauliSums, circuits and a dictionary that explains, "
                            "which of the paulistrings in the sub-PauliSum are the product of two paulistrings in the "
                            "original observable.")
        },
        "measurement_results": {
            "type": "list",
            "description": ("List of (measured or simulated) measurement results for the selected cliques. Each "
                            "measurement result is a list of integers representing the outcome for the respective qudit"
                            " after applying the corresponding circuit in circuits")
        },
        "scaling_matrix": {
            "type": "numpy.ndarray",
            "description": ("Numpy array that keeps track of the number of times each paulistring has been measured. "
                            "Off-diagonal elements count how often two different paulistrings have been measured"
                            " together. Used for allocation of new cliques.")
        }

    }
    return param_info


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

    return fig, ax