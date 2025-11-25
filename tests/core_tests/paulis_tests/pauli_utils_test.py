import numpy as np
import pytest
from sympleq.core.paulis import PauliSum
from sympleq.core.paulis.utils import make_hermitian
from sympleq.core.measurement.allocation import weight_to_phase


class TestPauliUtils:

    def test_make_hermitian(self):
        available_dimensions = [2, 3, 5, 7]
        # with random phases
        for i in range(200):
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
        for i in range(200):
            n_qubits = np.random.randint(1, 10)
            dims = [np.random.choice(available_dimensions) for _ in range(n_qubits)]
            n_paulis = np.random.randint(1, np.min([4**n_qubits - 1, 10]))
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=dims,
                                     rand_weights=True)
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

    def test_weight_to_phase(self):
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1j], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[1])
        assert P == P1

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[1j], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[1], phases=[3])
        assert P == P1

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[3, 3], weights=[np.exp(1 * 2 * np.pi * 1j / 3)], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[3, 3], weights=[1], phases=[2])
        assert P.is_close(P1)

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[np.exp(1 * 2 * np.pi * 1j / 3)], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[1], phases=[4])
        assert P == P1

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[5, 5], weights=[np.exp(1 * 2 * np.pi * 1j / 5)], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[5, 5], weights=[1], phases=[2])
        assert P.is_close(P1)
