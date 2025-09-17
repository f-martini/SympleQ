import numpy as np
import galois
from quaos.core.circuits.utils import symplectic_form  # not used in calibration path, but kept if you want
from quaos.core.paulis import PauliSum, PauliString
from quaos.core.circuits.gates import PauliGate

# ---------- finite-field helpers ----------

def gf_rref(A, GF):
    A = GF(A)
    m, n = A.shape
    R = A.copy()
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for rr in range(r, m):
            if R[rr, c] != GF(0):
                piv = rr; break
        if piv is None:
            continue
        if piv != r:
            R[[r, piv], :] = R[[piv, r], :]
        inv = GF(1) / R[r, c]
        R[r, :] *= inv
        for rr in range(m):
            if rr != r and R[rr, c] != GF(0):
                R[rr, :] -= R[rr, c] * R[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return R, pivots

def gf_solve(A, b, GF):
    A = GF(A)
    b = GF(b).reshape(-1, 1)
    m, n = A.shape
    M = np.hstack((A, b))
    R, _ = gf_rref(M, GF)
    for i in range(m):
        if np.all(R[i, :n] == GF(0)) and R[i, n] != GF(0):
            raise ValueError("Inconsistent linear system over GF(p).")
    x = GF.Zeros(n)
    row = 0
    for c in range(n):
        if row < m and R[row, c] == GF(1) and np.all(R[row, :c] == GF(0)):
            s = GF(0)
            for j in range(c+1, n):
                s += R[row, j] * x[j]
            x[c] = R[row, n] - s
            row += 1
    return np.asarray(x).reshape(n, 1)

# ---------- convention-agnostic calibration ----------

def calibrate_A_impl(H_int, p):
    """
    Given an integer tableau H (k x 2n) over GF(p), empirically build A_impl (k x 2n) over GF(p)
    such that for any Pauli vector P over GF(p), the phase kick observed from PauliGate.act is
    Δφ (mod 2p) = 2 * (A_impl @ P) (mod 2p).
    """
    H_int = np.asarray(H_int, dtype=int)
    k, two_n = H_int.shape
    n = two_n // 2
    GF = galois.GF(p)

    # Build a PauliSum with zero phases so we can read pure increments
    ps0 = PauliSum.from_tableau(H_int.copy(), dimensions=[p]*n)

    Acols = []
    for j in range(two_n):
        # Standard basis Pauli: P = e_j
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        if j < n:
            x[j] = 1
        else:
            z[j - n] = 1
        pauli = PauliString(x % p, z % p, dimensions=[p]*n)
        pg = PauliGate(pauli)

        # Apply and measure phase increment
        ps_after = pg.act(ps0)
        # Δφ_j (mod 2p)
        delta = (np.asarray(ps_after.phases) - np.asarray(ps0.phases)) % (2*p)

        # Must be even; divide by 2 to get A_impl column over GF(p)
        if np.any(delta % 2 != 0):
            raise RuntimeError("Internal: observed odd phase increment from a basis Pauli?!")
        col = ((delta // 2) % p).astype(int)  # k-vector in GF(p)
        Acols.append(col)

    A_impl = np.stack(Acols, axis=1) % p  # shape (k, 2n)
    return A_impl

def pauli_phase_correction_calibrated(H, delta_phi_2p, p, A_impl=None):
    """
    Solve for P over GF(p) using the empirically calibrated A_impl so that:
        2 * (A_impl @ P) ≡ delta_phi_2p (mod 2p)
    Raises ValueError if the target is unachievable.
    """
    H = np.asarray(H, dtype=int)
    delta_phi_2p = (np.asarray(delta_phi_2p).reshape(-1)) % (2*p)

    # Necessary condition: evenness mod 2p
    if np.any(delta_phi_2p % 2 != 0):
        raise ValueError("Unachievable target: Δφ has odd entries mod 2p.")

    # Build A_impl if not provided (cost: one pass of 2n Pauli conjugations)
    if A_impl is None:
        A_impl = calibrate_A_impl(H, p)

    rhs = ((delta_phi_2p // 2) % p).astype(int)  # GF(p) RHS

    GF = galois.GF(p)
    P = gf_solve(A_impl, rhs, GF)  # (2n,1) over GF(p)
    return np.asarray(P, dtype=int), A_impl

# ---------- demo that matches your environment ----------

if __name__ == "__main__":
    p = 2  # works for qubits and odd primes
    GF = galois.GF(p)

    n_qudits = 3
    n_paulis = 5

    H = GF.Random((n_paulis, 2*n_qudits)).view(np.ndarray).astype(int)

    # Random Δφ in Z_{2p}^k (may be unachievable; we error out cleanly)
    rng = np.random.default_rng()
    delta_phi_2p = rng.integers(0, p, size=n_paulis, dtype=int) * 2

    print("Target phases:", delta_phi_2p.tolist())

    try:
        P, A_impl = pauli_phase_correction_calibrated(H, delta_phi_2p, p)

        # Verify against your actual PauliGate.act()
        ps = PauliSum.from_tableau(H, dimensions=[p]*n_qudits)
        pauli = PauliString(P[:n_qudits].reshape(-1), P[n_qudits:].reshape(-1), dimensions=[p]*n_qudits)
        ps_after = PauliGate(pauli).act(ps)

        print(ps_after)

        got = (np.asarray(ps_after.phases) - np.asarray(ps.phases)) % (2*p)
        assert np.array_equal(got, delta_phi_2p), f"Mismatch: got {got}, want {delta_phi_2p}"
        print("Success: solver matches random target.")
    except ValueError as e:
        print("ERROR:", e)
