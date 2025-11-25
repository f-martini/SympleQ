import numpy as np
import pytest
from sympleq.core.paulis import PauliSum
from sympleq.core.paulis.utils import make_hermitian
from sympleq.core.measurement.allocation import weight_to_phase


class TestPauliUtils:

    def test_make_hermitian(self):
        # random qubit PauliSums
        for i in range(50):
            n_qubits = np.random.randint(1, 10)
            n_paulis = np.random.randint(1, np.min([4**n_qubits - 1, 10]))
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[2] * n_qubits,
                                     rand_weights=True)
            P.set_phases(np.random.randint(0, 4, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian(), "make_hermitian failed for random qubit PauliSum"

        # random pure qutrit PauliSums
        for i in range(50):
            n_qubits = np.random.randint(1, 6)
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[3] * n_qubits,
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_phases(np.random.randint(0, 6, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        # random pure ququint PauliSums
        for i in range(50):
            n_qubits = np.random.randint(1, 5)
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[5] * n_qubits,
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_phases(np.random.randint(0, 10, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        # random pure qusept PauliSums
        for i in range(50):
            n_qubits = np.random.randint(1, 4)
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[7] * n_qubits,
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_phases(np.random.randint(0, 14, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        # random mixed dimension PauliSums
        for i in range(50):
            n_qubits = 3
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[2, 2, 3],
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_phases(np.random.randint(0, 12, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        for i in range(50):
            n_qubits = 3
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[2, 3, 5],
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_weights(np.random.randint(0, 60, n_paulis))
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
