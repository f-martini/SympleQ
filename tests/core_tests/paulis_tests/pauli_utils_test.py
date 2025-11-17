import numpy as np
import pytest
from sympleq.core.paulis import PauliSum
from sympleq.core.paulis.utils import make_hermitian, isclose
from sympleq.core.measurement.allocation import weight_to_phase


class TestPauliUtils:

    @pytest.mark.skip()
    def test_make_hermitian(self):
        #       Check for various qubit PauliSums
        # Standard (need i for Y)
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[1])
        assert P1 == P2

        # Already hermitian
        P = PauliSum.from_string(['x1z0 x0z1'], dimensions=[2, 2], weights=[1], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z0 x0z1'], dimensions=[2, 2], weights=[1], phases=[0])
        assert P1 == P2

        # 2 Y but since i*i = -1 this is already hermitian
        P = PauliSum.from_string(['x1z1 x1z1 x1z0'], dimensions=[2, 2, 2], weights=[1], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x1z1 x1z0'], dimensions=[2, 2, 2], weights=[1], phases=[0])
        assert P1 == P2

        # 4 Y that cancel each other out
        P = PauliSum.from_string(['x1z1 x1z1 x1z1 x1z1'], dimensions=[2, 2, 2, 2], weights=[1], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x1z1 x1z1 x1z1'], dimensions=[2, 2, 2, 2], weights=[1], phases=[0])
        assert P1 == P2

        # Non 1 weights
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1 + 1j], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[np.sqrt(2)], phases=[1])
        assert P1 == P2

        # Already correct phases in original
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[1])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[1])
        assert P1 == P2

        # Wrong additional phase in original
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[2])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 2], weights=[1], phases=[1])
        assert P1 == P2

        P = PauliSum.from_string(['x1z0 x0z1'], dimensions=[2, 2], weights=[1], phases=[1])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z0 x0z1'], dimensions=[2, 2], weights=[1], phases=[0])
        assert P1 == P2

        # random qubit PauliSums
        for i in range(100):
            n_qubits = np.random.randint(1, 10)
            n_paulis = np.random.randint(1, np.min([4**n_qubits - 1, 10]))
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[2] * n_qubits,
                                     rand_weights=True)
            P.set_phases(np.random.randint(0, 4, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian(), "make_hermitian failed for random qubit PauliSum"

        #   Check for pure qudit PauliSums
        # Missing hermitian counter part
        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[3, 3], weights=[1], phases=[0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0', 'x2z2 x0z0'], dimensions=[3, 3], weights=[1, 1], phases=[0, 2])
        assert P1 == P2

        # Already hermitian
        P = PauliSum.from_string(['x1z1 x0z0', 'x2z2 x0z0'], dimensions=[3, 3], weights=[1, 1], phases=[0, 2])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0', 'x2z2 x0z0'], dimensions=[3, 3], weights=[1, 1], phases=[0, 2])
        assert P1 == P2

        # Hermitian PauliString is there but with wrong phase
        P = PauliSum.from_string(['x1z1 x0z0', 'x2z2 x0z0'], dimensions=[3, 3], weights=[1, 1], phases=[0, 0])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x0z0', 'x2z2 x0z0'], dimensions=[3, 3], weights=[1, 1], phases=[0, 2])
        assert P1 == P2

        # more complex phases that cancel each other out
        P = PauliSum.from_string(['x1z1 x2z1', 'x2z2 x1z2'], dimensions=[3, 3], weights=[1, 1], phases=[0, 2])
        P1 = make_hermitian(P)
        P2 = PauliSum.from_string(['x1z1 x2z1', 'x2z2 x1z2'], dimensions=[3, 3], weights=[1, 1], phases=[0, 0])
        assert P1 == P2

        # random pure qutrit PauliSums
        for i in range(100):
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
        for i in range(100):
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
        for i in range(100):
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
        for i in range(100):
            n_qubits = 3
            n_paulis = np.random.randint(1, 8)
            P = PauliSum.from_random(n_paulis=n_paulis,
                                     dimensions=[2, 2, 3],
                                     rand_weights=True)
            P.set_weights(np.random.random(n_paulis) + 1j * np.random.random(n_paulis))
            P.set_phases(np.random.randint(0, 12, n_paulis))
            P1 = make_hermitian(P)
            assert P1.is_hermitian()

        for i in range(100):
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
        assert P == P1

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[np.exp(1 * 2 * np.pi * 1j / 3)], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[2, 3], weights=[1], phases=[4])
        assert P == P1

        P = PauliSum.from_string(['x1z1 x0z0'], dimensions=[5, 5], weights=[np.exp(1 * 2 * np.pi * 1j / 5)], phases=[0])
        P = weight_to_phase(P)
        P1 = PauliSum.from_string(['x1z1 x0z0'], dimensions=[5, 5], weights=[1], phases=[2])
        assert isclose(P, P1)
