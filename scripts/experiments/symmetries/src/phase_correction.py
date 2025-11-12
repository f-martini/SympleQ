import numpy as np
import galois
from sympleq.core.circuits.utils import symplectic_form
from sympleq.core.paulis import PauliSum, PauliString
from sympleq.core.circuits.gates import PauliGate
from sympleq.core.finite_field_solvers import gf_solve


def pauli_phase_correction(H, delta_phi_2p, p):
    """
    Given tableau H (k x 2n, rows [x|z] over GF(p)) and target Δφ in Z_{2p}^k,
    find P in GF(p)^{2n} such that 2*(H Ω P) ≡ Δφ (mod 2p).
    Raises ValueError if Δφ is unachievable by Pauli conjugation.
    """
    H = np.asarray(H, dtype=int)
    delta_phi_2p = np.asarray(delta_phi_2p, dtype=int) % (2*p)
    k, two_n = H.shape
    n = two_n // 2

    if np.any(delta_phi_2p % 2 != 0):
        raise ValueError("Unachievable target: Δφ has odd entries mod 2p.")

    rhs = ((delta_phi_2p // 2) % p).astype(int)

    # Build A = H Omega (over GF(p)) and solve A P = rhs (mod p)
    GF = galois.GF(p)
    Omega = symplectic_form(n, p)
    A = (GF(H) @ GF(Omega)).view(np.ndarray).astype(int)

    P = gf_solve(A, rhs % p, GF)  # (2n,1)

    if not np.all((GF(A) @ GF(P)).reshape(-1) == GF((rhs) % p)):
        raise ValueError("Internal GF(p) verification failed.")

    P = P.reshape(-1)
    pauli = PauliString.from_exponents(P[:n_qudits].reshape(-1), P[n_qudits:].reshape(-1), dimensions=[p] * n_qudits)

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
        G_p = pauli_phase_correction(H, delta_phi_2p, p)
        # Verify using your PauliGate (which sets h = 2 Ω P mod 2p and C = I for Paulis)
        ps = PauliSum.from_tableau(H, dimensions=[p] * n_qudits)

        ps_after = G_p.act(ps)

        got = (np.asarray(ps_after.phases) - np.asarray(ps.phases)) % (2 * p)
        want = delta_phi_2p % (2 * p)
        # If the sign convention in Ω vs gate were flipped, sgn would be -1 and this still passes:

        assert np.array_equal(got, want), f"Mismatch: got {got}, want {want}"
        print("Success: phase correction matches the random target.")

    except ValueError as e:
        print("ERROR:", e)
