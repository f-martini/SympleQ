import numpy as np
from sympleq.core.paulis import PauliSum
from sympleq.core.paulis.utils import make_hermitian


class TestPauliUtils:

    def test_make_hermitian(self):
        available_dimensions = [2, 3, 5, 7]
        # with random phases
        for i in range(50):
            n_qubits = np.random.randint(1, 10)
            dims = [np.random.choice(available_dimensions) for _ in range(n_qubits)]
            n_paulis = np.random.randint(1, np.min([4**n_qubits - 1, 10]))
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=dims,
                                     rand_weights=True)
            P.set_phases(np.random.randint(0, 2 * P.lcm, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        # without random phases
        for i in range(50):
            n_qubits = np.random.randint(1, 10)
            dims = [np.random.choice(available_dimensions) for _ in range(n_qubits)]
            n_paulis = np.random.randint(1, np.min([4**n_qubits - 1, 10]))
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=dims,
                                     rand_weights=True)
            P1 = make_hermitian(P)
            assert P1.is_hermitian()
