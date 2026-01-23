import numpy as np
import random
import pytest
from sympleq.core.circuits import GATES, Gate, Circuit, PauliGate
from sympleq.core.circuits.utils import is_symplectic
from sympleq.core.paulis import PauliSum, PauliString
from sympleq.core.circuits.random_symplectic import (symplectic_gf2, symplectic_group_size,
                                                     symplectic_random_transvection)


class TestGates():

    @staticmethod
    def states(dim):
        ps1 = PauliString.from_string('x1z0 x0z0', dimensions=[dim, dim])
        ps2 = PauliString.from_string('x0z0 x0z1', dimensions=[dim, dim])
        ps3 = PauliString.from_string('x1z0 x1z0', dimensions=[dim, dim])
        ps4 = PauliString.from_string('x0z1 x0z1', dimensions=[dim, dim])
        ps5 = PauliString.from_string('x1z1 x0z0', dimensions=[dim, dim])
        p_sum = PauliSum.from_string(['x1z0 x0z0 x1z1', 'x0z0 x0z1 x1z0',
                                      'x1z1 x1z0 x0z0'], dimensions=[dim, dim, dim])
        return ps1, ps2, ps3, ps4, ps5, p_sum

    def random_pauli_sum(self, dim, n_paulis=10):
        ps_list = []
        element_list = [(0, 0, 0, 0)]
        for _ in range(n_paulis):
            ps, r1, r2, s1, s2 = self.random_pauli_string(dim)
            element_list.append((r1, r2, s1, s2))
            while (r1, r2, s1, s2) in element_list:
                ps, r1, r2, s1, s2 = self.random_pauli_string(dim)
            ps_list.append(ps)
        return PauliSum.from_pauli_strings(ps_list)

    def random_pauli_string(self, dim):
        r1 = np.random.randint(0, dim)
        r2 = np.random.randint(0, dim)
        s1 = np.random.randint(0, dim)
        s2 = np.random.randint(0, dim)
        return PauliString.from_string(f'x{r1}z{s1} x{r2}z{s2}', dimensions=[dim, dim]), r1, r2, s1, s2

    def test_SUM(self):
        # acts the SUM gate on a bunch of random pauli strings and pauli sums
        for d in [2, 5, 11]:
            # test pauli_strings
            for i in range(1000):
                r1 = np.random.randint(0, d)
                r2 = np.random.randint(0, d)
                s1 = np.random.randint(0, d)
                s2 = np.random.randint(0, d)

                input_str = f"x{r1}z{s1} x{r2}z{s2}"
                output_str_correct = f"x{r1}z{(s1 - s2) % d} x{(r2 + r1) % d}z{s2}"

                input_ps = PauliString.from_string(input_str, dimensions=[d, d])
                output_ps = GATES.SUM.act(input_ps, (0, 1))
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in SUM gate'

            # test pauli_sums
            for i in range(10):
                ps_list_in = []
                ps_list_out_correct = []
                ps_phase_out_correct = []
                for j in range(10):
                    r1 = np.random.randint(0, d)
                    r2 = np.random.randint(0, d)
                    s1 = np.random.randint(0, d)
                    s2 = np.random.randint(0, d)

                    input_str = f"x{r1}z{s1} x{r2}z{s2}"
                    output_str_correct = f"x{r1}z{(s1 - s2) % d} x{(r2 + r1) % d}z{s2}"

                    input_ps = PauliString.from_string(input_str, dimensions=[d, d])
                    output_ps_correct = PauliString.from_string(output_str_correct, dimensions=[d, d])
                    ps_list_in.append(input_ps)
                    ps_list_out_correct.append(output_ps_correct)
                    ps_phase_out_correct.append(0)

                input_psum = PauliSum.from_pauli_strings(ps_list_in)
                output_psum_correct = PauliSum.from_pauli_strings(ps_list_out_correct, phases=ps_phase_out_correct)

                output_psum = GATES.SUM.act(input_psum, (0, 1))
                assert output_psum == output_psum_correct, (
                    'Error in SUM gate: \n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_SWAP(self):
        for d in [2, 5, 11]:
            # test pauli_strings
            for i in range(100):
                r1 = np.random.randint(0, d)
                r2 = np.random.randint(0, d)
                s1 = np.random.randint(0, d)
                s2 = np.random.randint(0, d)

                input_str = f"x{r1}z{s1} x{r2}z{s2}"
                output_str_correct = f"x{r2}z{(s2) % d} x{r1}z{s1}"

                input_ps = PauliString.from_string(input_str, dimensions=[d, d])
                output_ps = GATES.SWAP.act(input_ps, (0, 1))
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in SWAP gate'

            # test pauli_sums
            for i in range(100):
                ps_list_in = []
                ps_list_out_correct = []
                ps_phase_out_correct = []
                for j in range(10):
                    r1 = np.random.randint(0, d)
                    r2 = np.random.randint(0, d)
                    s1 = np.random.randint(0, d)
                    s2 = np.random.randint(0, d)

                    input_str = f"x{r1}z{s1} x{r2}z{s2}"
                    output_str_correct = f"x{r2}z{s2} x{r1}z{s1}"

                    input_ps = PauliString.from_string(input_str, dimensions=[d, d])
                    output_ps_correct = PauliString.from_string(output_str_correct, dimensions=[d, d])
                    ps_list_in.append(input_ps)
                    ps_list_out_correct.append(output_ps_correct)
                    ps_phase_out_correct.append(0)

                input_psum = PauliSum.from_pauli_strings(ps_list_in)
                output_psum_correct = PauliSum.from_pauli_strings(ps_list_out_correct,
                                                                  phases=ps_phase_out_correct)

                output_psum = GATES.SWAP.act(input_psum, (0, 1))
                assert output_psum == output_psum_correct, (
                    'Error in SWAP gate: \n' +
                    input_psum.__str__() + '\n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_Hadamard(self):
        for d in [2, 5, 11]:
            # test pauli_strings on qudit 0
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{(-s1) % d}z{r1} x{r2}z{s2}"

                output_ps = GATES.H.act(input_ps, 0)
                assert output_ps.has_equal_tableau(PauliString.from_string(
                    output_str_correct, dimensions=[d, d]
                )), 'Error in Hadamard gate 0'

            # test on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{(-s2) % d}z{r2}"

                output_ps = GATES.H.act(input_ps, 1)
                assert output_ps.has_equal_tableau(PauliString.from_string(
                    output_str_correct, dimensions=[d, d]
                )), 'Error in Hadamard gate 1'

            # test pauli_sums
            for i in range(100):
                ps_list_in = []
                ps_list_out_correct = []
                ps_phase_out_correct = []
                for j in range(10):
                    input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                    output_str_correct = f"x{(-s1) % d}z{r1} x{r2}z{s2}"

                    ps_list_in.append(input_ps)
                    ps_list_out_correct.append(PauliString.from_string(output_str_correct, dimensions=[d, d]))
                    ps_phase_out_correct.append(-2 * r1 * s1)

                input_psum = PauliSum.from_pauli_strings(ps_list_in)
                output_psum = GATES.H.act(input_psum, 0)
                output_psum_correct = PauliSum.from_pauli_strings(
                    ps_list_out_correct, phases=ps_phase_out_correct)
                assert output_psum.has_equal_tableau(output_psum_correct), (
                    'Error in Hadamard gate: \n' +
                    input_psum.__str__() + '\n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_PHASE(self):
        for d in [2, 5, 11]:
            # test on qudit 0
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{(r1 + s1) % d} x{r2}z{s2}"

                output_ps = GATES.S.act(input_ps, 0)
                assert output_ps.has_equal_tableau(PauliString.from_string(
                    output_str_correct, dimensions=[d, d])), 'Error in PHASE gate 0'

            # test on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{r2}z{(r2 + s2) % d}"

                output_ps = GATES.S.act(input_ps, 1)
                assert output_ps.has_equal_tableau(PauliString.from_string(
                    output_str_correct, dimensions=[d, d])), 'Error in PHASE gate 1'

    def test_group_homomorphism(self):
        # Tests all gates and all combinations of two pauli strings for the group homomorphism property:
        # gate.act(p1) * gate.act(p2) == gate.act(p1 * p2)

        # Test for dim = 2 with all combinations
        gates_and_qudits = [
            (GATES.SUM, (0, 1)), (GATES.SUM, (1, 0)),
            (GATES.SWAP, (0, 1)),
            (GATES.H, 0), (GATES.H, 1),
            (GATES.S, 0), (GATES.S, 1)
        ]

        for gate, qudits in gates_and_qudits:
            for x0 in range(2):
                for z0 in range(2):
                    for x1 in range(2):
                        for z1 in range(2):
                            for x0p in range(2):
                                for z0p in range(2):
                                    for x1p in range(2):
                                        for z1p in range(2):
                                            p1 = PauliSum.from_string([f'x{x0}z{z0} x{x1}z{z1}'], dimensions=[2, 2])
                                            p2 = PauliSum.from_string([f'x{x0p}z{z0p} x{x1p}z{z1p}'], dimensions=[2, 2])
                                            lhs = gate.act(p1, qudits) * gate.act(p2, qudits)
                                            rhs = gate.act(p1 * p2, qudits)
                                            assert lhs == rhs, f"Failed for {gate.name} on {qudits}"

        # Test for larger dimensions with random samples
        for dim in [3, 5, 7, 15]:
            for gate, qudits in gates_and_qudits:
                for _ in range(100):
                    p1 = self.random_pauli_sum(dim, n_paulis=1)
                    p2 = self.random_pauli_sum(dim, n_paulis=1)
                    lhs = gate.act(p1, qudits) * gate.act(p2, qudits)
                    rhs = gate.act(p1 * p2, qudits)
                    assert lhs == rhs, f"Failed for {gate.name} on {qudits} dim={dim}"

    def test_is_symplectic(self):
        # Tests if the symplectic matrix of the gate is symplectic
        gates = [GATES.SUM, GATES.SWAP, GATES.H, GATES.S]
        for gate in gates:
            assert is_symplectic(gate.symplectic, 2), (
                f"Gate {gate.name} is not symplectic. \n" +
                gate.symplectic.__str__()
            )

    def test_random_symplectic(self, num_tests=20, max_n=10, primes=[2, 3, 5, 7]):
        """
        Test random_symplectic() across several n, p values using assertions only.
        """
        for n in range(2, max_n + 1):
            for d in primes:
                for i in range(num_tests):
                    if n < 6 and d == 2:
                        index = random.randint(0, symplectic_group_size(n) - 1)
                        F = symplectic_gf2(index, n)
                    else:
                        F = symplectic_random_transvection(n, dimension=d)
                    assert is_symplectic(F, d), f"Failed symplectic check: n={n}, test {i}"

    @pytest.mark.skip(reason="Gate.solve_from_target not in new API")
    def test_gate_from_target(self):
        pass

    @pytest.mark.skip(reason="Gate.from_random not in new API")
    def test_gate_transvection_symplecticity(self):
        pass

    @pytest.mark.skip(reason="Gate.from_random not in new API")
    def test_gate_inverse(self):
        pass

    @pytest.mark.skip(reason="unitary() not implemented in new API yet")
    def test_phase_table(self):
        pass

    @pytest.mark.skip(reason="unitary() not implemented in new API yet")
    def test_SUM_unitary(self):
        pass

    @pytest.mark.skip(reason="unitary() not implemented in new API yet")
    def test_two_qudit_unitary(self):
        pass

    @pytest.mark.skip(reason="unitary() not implemented in new API yet")
    def test_one_qudit_unitary(self):
        pass

    @pytest.mark.skip(reason="unitary() not implemented in new API yet")
    def test_pauli_gate_unitary(self):
        pass
