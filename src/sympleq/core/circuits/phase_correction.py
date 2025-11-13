import numpy as np
import galois
from sympleq.core.circuits.utils import symplectic_form
from sympleq.core.paulis import PauliSum, PauliString
from sympleq.core.circuits.gates import PauliGate, Gate
from sympleq.core.finite_field_solvers import solve_linear_system_over_gf


def pauli_phase_correction(H: np.ndarray, delta_phi_2p: np.ndarray, p: int, dimensions: list[int] | None = None):
    """
    Given tableau H (k x 2n, rows [x|z]) and a target Δφ (mod 2L),
    attempt to construct a Clifford whose action adjusts the phases by Δφ.

    Preference order:
        1. Solve for a pure phase-vector gate (identity symplectic, arbitrary h).
        2. Fallback to a Pauli gate when Δφ is an even multiple and the above fails.

    Returns a Gate (or PauliGate) or None if no solution exists.
    """
    H = np.asarray(H, dtype=int)
    k, two_n = H.shape
    n = two_n // 2

    if dimensions is None:
        dimensions = [p] * n
    dims = np.asarray(dimensions, dtype=int)
    L = int(np.lcm.reduce(dims))
    modulus = 2 * L

    delta_phi_2p = np.asarray(delta_phi_2p, dtype=int) % modulus

    # 1) Try solving for a phase-only Clifford (identity symplectic, unknown h)
    h_vec = solve_linear_system_over_gf(H, delta_phi_2p % modulus, modulus)
    if h_vec is not None:
        symplectic = np.eye(2 * n, dtype=int)
        return Gate("PhaseFix", list(range(n)), symplectic, dims, h_vec % modulus)

    # 2) Fallback: Pauli correction requires Δφ even (since phases change in steps of 2)
    if np.any(delta_phi_2p % 2 != 0):
        return None

    rhs = ((delta_phi_2p // 2) % p).astype(int)
    GF = galois.GF(int(p))
    Omega = symplectic_form(n, p)
    A = (GF(H) @ GF(Omega)).view(np.ndarray).astype(int)

    try:
        P = solve_linear_system_over_gf(A, rhs % p, GF)  # (2n,1)
    except ValueError:
        return None

    if not np.all((GF(A) @ GF(P)).reshape(-1) == GF(rhs % p)):
        return None

    P = np.asarray(P.reshape(-1), dtype=int) % p
    pauli = PauliString.from_exponents(P[:n], P[n:], dimensions=dims)
    return PauliGate(pauli)


if __name__ == "__main__":
    p = 2                     # works for qubits and odd primes
    GF = galois.GF(p)
    n_qudits = 3
    n_paulis = 5

    # Random H over GF(p)
    H = GF.Random((n_paulis, 2 * n_qudits)).view(np.ndarray).astype(int)

    # Random Δφ in Z_{2p}^k (may be unachievable)
    rng = np.random.default_rng()
    delta_phi_2p = rng.integers(0, p, size=n_paulis, dtype=int) * 2
    print("Random Δφ (mod 2p):", delta_phi_2p.tolist())

    try:
        G_p = pauli_phase_correction(H, delta_phi_2p, p, [p] * n_qudits)
        # Verify using your PauliGate (which sets h = 2 Ω P mod 2p and C = I for Paulis)
        ps = PauliSum.from_tableau(H, dimensions=[p] * n_qudits)

        if G_p is None:
            print("No solution found.")
            exit(0)
        ps_after = G_p.act(ps)

        got = (np.asarray(ps_after.phases) - np.asarray(ps.phases)) % (2 * p)
        want = delta_phi_2p % (2 * p)
        # If the sign convention in Ω vs gate were flipped, sgn would be -1 and this still passes:

        assert np.array_equal(got, want), f"Mismatch: got {got}, want {want}"
        print("Success: phase correction matches the random target.")

    except ValueError as e:
        print("ERROR:", e)
