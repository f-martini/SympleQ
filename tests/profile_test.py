import numpy as np
import cProfile
import pstats
import io

from quaos.core.paulis.pauli_sum import PauliSum


def profile_paulisum_multiplication():
    n_paulis = 20
    n_qudits = 100
    dimensions = np.random.randint(2, 8, size=n_qudits)

    ps1 = PauliSum.from_random(n_paulis, n_qudits, dimensions, rand_weights=True)
    ps2 = PauliSum.from_random(n_paulis, n_qudits, dimensions, rand_weights=True)

    def multiply():
        _ = ps1 * ps2

    N = 50
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N):
        multiply()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(f"\n===== PROFILE REPORT - {N} runs =====")
    print(s.getvalue())


if __name__ == "__main__":
    profile_paulisum_multiplication()
