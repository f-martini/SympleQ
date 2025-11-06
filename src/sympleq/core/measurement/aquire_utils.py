import numpy as np
from sympleq.core.measurement.allocation import scale_variances
from sympleq.core.measurement.covariance_graph import graph
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
    mu = np.real(np.transpose(np.conjugate(psi)) @ H.matrix_form() @ psi)
    return mu


def true_covariance_graph(H: PauliSum, psi):
    weights = H.weights
    p = H.n_paulis()
    mm = [np.exp(2 * np.pi * 1j * H.phases[i] / (2 * H.lcm)) * H.matrix_form(i) for i in range(p)]
    psi_dag = psi.conj().T
    cc1 = [psi_dag @ mm[i] @ psi for i in range(p)]
    cc2 = [psi_dag @ mm[i].conj().T @ psi for i in range(p)]
    cm = np.zeros((p, p), dtype=complex)
    for i0 in range(p):
        for i1 in range(p):
            pre_factor = np.conj(weights[i0]) * weights[i1]
            cov = (psi_dag @ mm[i0].conj().T @ mm[i1] @ psi) - cc2[i0] * cc1[i1]
            cm[i0, i1] = pre_factor * cov
    return graph(cm)


def true_statistical_variance(H, psi, S, weights):
    sigma = np.sum(scale_variances(true_covariance_graph(H, psi), S).adj).real
    return sigma
