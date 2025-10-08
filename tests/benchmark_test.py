import pytest
import numpy as np

from quaos.core.paulis.pauli_string import PauliString
from quaos.core.paulis.pauli_sum import PauliSum


@pytest.mark.benchmark
def test_main_function(benchmark):
    # TODO: Add meaningful benchmark when a stable version of the project will be available
    pass


@pytest.mark.benchmark
def test_paulisum_multiplication(benchmark):
    n_paulis = 20
    n_qudits = 10
    dimensions = np.random.randint(2, 8, size=n_qudits)
    ps1 = PauliSum.from_random(n_paulis, n_qudits, dimensions, rand_weights=True)
    ps2 = PauliSum.from_random(n_paulis, n_qudits, dimensions, rand_weights=True)

    def multiply():
        _ = ps1 * ps2

    benchmark(multiply)


@pytest.mark.benchmark
def test_paulistring_amend(benchmark):
    n_qudits = 20
    dimensions = np.random.randint(2, 8, size=n_qudits)
    ps = PauliString.from_random(n_qudits, dimensions)

    def amend_all():
        for i in range(n_qudits):
            new_x = np.random.randint(0, dimensions[i])
            new_z = np.random.randint(0, dimensions[i])
            ps.amend(i, new_x, new_z)

    benchmark(amend_all)
