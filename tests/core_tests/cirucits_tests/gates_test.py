from sympleq.core.circuits import SUM, SWAP, Hadamard, PHASE, Gate, Circuit
from sympleq.core.circuits.utils import is_symplectic
from sympleq.core.circuits.target import find_map_to_target_pauli_sum, map_pauli_sum_to_target_tableau
from sympleq.core.paulis import PauliSum, PauliString
import numpy as np
from sympleq.core.circuits.random_symplectic import symplectic_gf2, symplectic_group_size, symplectic_random_transvection
import random


class TestGates():

    @staticmethod
    def states(dim):
        ps1 = PauliString.from_string('x1z0 x0z0', dimensions=[dim, dim])
        ps2 = PauliString.from_string('x0z0 x0z1', dimensions=[dim, dim])
        ps3 = PauliString.from_string('x1z0 x1z0', dimensions=[dim, dim])
        ps4 = PauliString.from_string('x0z1 x0z1', dimensions=[dim, dim])
        ps5 = PauliString.from_string('x1z1 x0z0', dimensions=[dim, dim])
        p_sum = PauliSum(['x1z0 x0z0 x1z1', 'x0z0 x0z1 x1z0', 'x1z1 x1z0 x0z0'], dimensions=[dim, dim, dim])
        return ps1, ps2, ps3, ps4, ps5, p_sum

    def random_pauli_sum(self, dim, n_paulis=10):
        # Generates a random PauliSum with n_paulis random PauliStrings of dimension dim
        #
        ps_list = []
        element_list = [(0, 0, 0, 0)]  # to keep track of already generated PauliStrings. Avoids identity and duplicates
        for _ in range(n_paulis):
            ps, r1, r2, s1, s2 = self.random_pauli_string(dim)
            element_list.append((r1, r2, s1, s2))
            while (r1, r2, s1, s2) in element_list:
                ps, r1, r2, s1, s2 = self.random_pauli_string(dim)
            ps_list.append(ps)
        return PauliSum(ps_list, dimensions=[dim, dim], standardise=True)

    def random_pauli_string(self, dim):
        # Generates a random PauliString of dimension dim
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
                output_ps = SUM(0, 1, d).act(input_ps)
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

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d],
                                               standardise=False)

                output_psum = SUM(0, 1, d).act(input_psum)
                assert output_psum == output_psum_correct, (
                    'Error in SUM gate: \n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_SWAP(self):
        # acts the SUM gate on a bunch of random pauli strings and pauli sums
        # C = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        # U = np.zeros((4, 4), dtype=int)
        # U[2:, :2] = np.eye(2, dtype=int)
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
                output_ps = SWAP(0, 1, d).act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in SUM gate'

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
                    # a = np.array([r1, r2, s1, s2])

                    # phase_out = np.diag(C.T @ U @ C) @ a
                    # + 2 * a @ np.triu(C.T @ U @ C) @ a - a @ np.diag(np.diag(C.T @ U @ C)) @ a
                    ps_phase_out_correct.append(0)

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d],
                                               standardise=False)

                output_psum = SWAP(0, 1, d).act(input_psum)
                assert output_psum == output_psum_correct, (
                    'Error in SUM gate: \n' +
                    input_psum.__str__() + '\n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_Hadamard(self):
        # TODO: Be certain of inverse convention - ultimately arbitrary but should match prevalent literature
        for d in [2, 5, 11]:
            # test pauli_strings
            gate = Hadamard(0, d, inverse=False)  # Hadamard on qubit 0

            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{(-s1) % d}z{r1} x{r2}z{s2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(
                    output_str_correct, dimensions=[d, d]
                ), 'Error in Hadamard gate 0'

            gate = Hadamard(1, d, inverse=False)  # Hadamard on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{(-s2) % d}z{r2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(
                    output_str_correct, dimensions=[d, d]
                ), 'Error in Hadamard gate 1'
            # test pauli_sums
            gate = Hadamard(0, d, inverse=False)  # Hadamard on qubit 0

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

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum = gate.act(input_psum)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d],
                                               standardise=False)
                assert output_psum == output_psum_correct, (
                    'Error in Hadamard gate: \n' +
                    input_psum.__str__() + '\n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_PHASE(self):
        # TODO: Better approach - is there another approach that we can test the first one with like Hadamard?
        for d in [2, 5, 11]:
            gate = PHASE(0, d)  # PHASE on qubit 0

            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{(r1 + s1) % d} x{r2}z{s2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in H gate 0'

            gate = PHASE(1, d)  # PHASE on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{r2}z{(r2 + s2) % d}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in H gate 1'
            # test pauli_sums
            gate = PHASE(0, d)  # PHASE on qubit 0
            h = gate.phase_vector
            C = gate.symplectic
            for i in range(100):
                ps_list_in = []
                ps_list_out_correct = []
                ps_phase_out_correct = []
                for j in range(10):
                    input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                    output_str_correct = f"x{r1}z{(r1 + s1) % d} x{r2}z{s2}"

                    ps_list_in.append(input_ps)
                    ps_list_out_correct.append(PauliString.from_string(output_str_correct, dimensions=[d, d]))
                    a1 = np.array([r1, s1])
                    U = np.zeros((len(input_ps.dimensions), len(input_ps.dimensions)), dtype=int)
                    U[1, 0] = 1

                    p = (
                        np.dot(h, a1) -
                        np.diag(C.T @ U @ C) @ a1 +
                        2 * a1 @ np.triu(C.T @ U @ C) @ a1 -
                        a1 @ np.diag(np.diag(C.T @ U @ C)) @ a1
                    )
                    ps_phase_out_correct.append(p)

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum = gate.act(input_psum)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d],
                                               standardise=False)
                assert output_psum == output_psum_correct, (
                    'Error in Hadamard gate: \n' +
                    input_psum.__str__() + '\n' +
                    output_psum.__str__() + '\n' +
                    output_psum_correct.__str__()
                )

    def test_group_homomorphism(self):
        # Tests all gates and all combinations of two pauli strings for the group homomorphism property:
        # gate.act(p1) * gate.act(p2) == gate.act(p1 * p2)

        gates = [SUM(0, 1, 2), SUM(1, 0, 2), SWAP(0, 1, 2), Hadamard(0, 2), Hadamard(1, 2), PHASE(0, 2), PHASE(1, 2)]  #
        i = 0

        # test all for dim = 2
        for gate in gates:
            for x0 in range(2):
                for z0 in range(2):
                    for x1 in range(2):
                        for z1 in range(2):
                            for x0p in range(2):
                                for z0p in range(2):
                                    for x1p in range(2):
                                        for z1p in range(2):
                                            i += 1
                                            p1 = PauliSum([f'x{x0}z{z0} x{x1}z{z1}'], dimensions=[2, 2])
                                            p2 = PauliSum([f'x{x0p}z{z0p} x{x1p}z{z1p}'], dimensions=[2, 2])
                                            err0 = 'In: \n' + p1.__str__() + '\n' + p2.__str__()
                                            err = (
                                                'Out: \n' +
                                                (gate.act(p1) * gate.act(p2)).__str__() +
                                                '\n' +
                                                gate.act(p1 * p2).__str__()
                                            )
                                            assert gate.act(p1) * gate.act(p2) == gate.act(p1 * p2), err0 + err + '\n'

        for dim in [3, 5, 7, 15]:
            gates = [SUM(0, 1, dim), SUM(1, 0, dim), SWAP(0, 1, dim), Hadamard(0, dim),
                     Hadamard(1, dim), PHASE(0, dim), PHASE(1, dim)]
            for gate in gates:

                for _ in range(100):
                    p1 = self.random_pauli_sum(dim, n_paulis=1)
                    p2 = self.random_pauli_sum(dim, n_paulis=1)
                    err0 = 'In: \n' + p1.__str__() + '\n' + p2.__str__()
                    err = 'Out: \n' + (gate.act(p1) * gate.act(p2)).__str__() + '\n' + gate.act(p1 * p2).__str__()
                    assert gate.act(p1) * gate.act(p2) == gate.act(p1 * p2), err0 + err + '\n'

    def test_is_symplectic(self):

        # Tests if the symplectic matrix of the gate is symplectic
        gates = [SUM(0, 1, 2), SWAP(0, 1, 2), Hadamard(0, 2), Hadamard(1, 2), PHASE(0, 2), PHASE(1, 2)]
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
                        index = random.randint(0, symplectic_group_size(n))
                        F = symplectic_gf2(index, n)
                    else:
                        F = symplectic_random_transvection(n, dimension=d)
                    assert is_symplectic(F, d), f"Failed symplectic check: n={n}, test {i}"

    # def test_find_symplectic_map(self):
    #     # this just tests the underlying solver, not the Gate or Pauli... implementation
    #     # see test below for implementation

    #     for n in [5, 10, 20, 50]:  # n_qudits
    #         for n_p in range(1, 10):
    #             for _ in range(100):
    #                 index = random.randint(0, symplectic_group_size(n))
    #                 F_true = symplectic_gf2(index, n)
    #                 X = np.random.randint(0, 2, size=(n_p, 2 * n), dtype=np.uint8)
    #                 Y = (X @ F_true) % 2
    #                 F_found = map_pauli_sum_to_target_tableau(X, Y)
    #                 assert np.array_equal((X @ F_found) % 2, Y)

    def test_gate_from_target(self):
        n_qudits = 4
        n_paulis = 2
        dim = 2
        dimensions = [dim] * n_qudits

        input_ps = PauliSum.from_random(n_paulis, n_qudits, dimensions=dimensions)
        input_ps.remove_trivial_paulis()
        input_ps.combine_equivalent_paulis()
        print(input_ps)
        circuit = Circuit.from_random(n_qudits, 100, dimensions)
        target_ps = circuit.act(input_ps)
        target_ps.phases = np.zeros(n_qudits)

        gate_from_solver = Gate.solve_from_target('ArbGate', input_ps, target_ps, dimensions)
        output_ps = gate_from_solver.act(input_ps)
        output_ps.phases = np.zeros(n_qudits)
        assert output_ps == target_ps, (
            'Error in gate_from_solver\n input:\n' + input_ps.__str__() +
            '\n target:\n' + target_ps.__str__() + '\n output:\n' + output_ps.__str__())

        for d in [2]:  # only solves on GF(2) for now...
            for i in range(10):
                input_ps = self.random_pauli_sum(d, n_paulis=6)
                target_ps = self.random_pauli_sum(d, n_paulis=6)

                if np.all(input_ps.symplectic_product_matrix() == target_ps.symplectic_product_matrix()):
                    print(i)
                    print('input')
                    print(input_ps.tableau())
                    print('target')
                    print(target_ps.tableau())
                    gate = Gate.solve_from_target('ArbGate', input_ps, target_ps, dimensions)
                    output_ps = gate.act(input_ps)
                    output_ps.phases = input_ps.phases  # So far it does not solve for phases as well
                    print('output')
                    print(output_ps.tableau())
                    print('gate')
                    print(gate.symplectic)
                    print('check')
                    print(input_ps.tableau() @ gate.symplectic % 2)

                    assert output_ps == target_ps, (
                        f'Error test {i} \n In: \n' +
                        input_ps.__str__() +
                        '\n Out: \n' +
                        output_ps.__str__() +
                        '\n Target: \n' +
                        target_ps.__str__()
                    )

    def test_gate_transvection(self):

        for _ in range(100):
            g = Gate.from_random(5, 2)
            gt = g.transvection(np.random.randint(0, 1, size=10))
            assert is_symplectic(gt.symplectic, 2), 'Error in transvection'

    # def test_gate_inverse(self):
    #     n_qudits = 2
    #     n_paulis = 3
    #     dimension = 3
    #     for _ in range(1):
    #         g = Gate.from_random(n_qudits, dimension)
    #         gt = g.inv()
    #         rps = PauliSum.from_random(n_paulis, n_qudits, [dimension] * n_qudits, False, seed=1)
    #         print(rps)
    #         assert rps == g.act(gt.act(rps)), 'Inversion Error:\n' + rps.__str__() + '\n' + g.act(gt.act(rps)).__str__()

    def phase_table_local(self, G):
        d = G.dimensions[0]
        phase_table_unitary = np.zeros((d, d))
        phase_table_symplectic = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                pauli_string = 'x' + str(i) + 'z' + str(j)
                ps = PauliSum([pauli_string],
                              dimensions=[d],
                              weights=[1], phases=[0])
                ps_m = ps.matrix_form()

                ps_res = G.act(ps)
                phase_table_symplectic[i, j] = ps_res.phases[0]
                ps_res.phases = [0]
                ps_res_m = ps_res.matrix_form()

                ps_m_res = G.unitary() @ ps_m @ G.unitary().conj().T

                mask = (ps_res_m.toarray() != 0)
                factors = np.around(ps_m_res.toarray()[mask] / ps_res_m.toarray()[mask], 14)
                factor = factors[0]
                phase_table_unitary[i, j] = np.around(d * np.angle(factor) / (np.pi), 6) % (2 * d)
        return phase_table_unitary, phase_table_symplectic

    def phase_table_entangling(self, G):
        d = G.dimensions[0]
        phase_table_unitary = np.zeros((d**2, d**2))
        phase_table_symplectic = np.zeros((d**2, d**2))
        U = G.unitary()
        for i1 in range(d):
            for j1 in range(d):
                for i2 in range(d):
                    for j2 in range(d):
                        pauli_string = 'x' + str(i1) + 'z' + str(j1) + ' ' + 'x' + str(i2) + 'z' + str(j2)
                        ps = PauliSum([pauli_string],
                                      dimensions=[d, d],
                                      weights=[1], phases=[0])
                        ps_m = ps.matrix_form()

                        ps_res = G.act(ps)
                        phase_table_symplectic[i1 * d + i2, j1 * d + j2] = ps_res.phases[0]
                        ps_res.phases = [0]
                        ps_res_m = ps_res.matrix_form()

                        ps_m_res = U @ ps_m @ U.conj().T

                        mask = (ps_res_m.toarray() != 0)
                        factors = np.around(ps_m_res.toarray()[mask] / ps_res_m.toarray()[mask], 14)
                        factor = factors[0]
                        phase_table_unitary[i1 * d + i2, j1 * d +
                                            j2] = np.around(d * np.angle(factor) / (np.pi), 6) % (2 * d)

        return phase_table_unitary, phase_table_symplectic

    def test_phase_table(self):
        # d = 2
        for d in [2, 3, 5, 7, 11]:
            G = Hadamard(0, d)
            phase_table_unitary, phase_table_symplectic = self.phase_table_local(G)
            diff_m = np.around(phase_table_unitary - phase_table_symplectic, 10)
            assert not np.any(diff_m), 'Symplectic phase table does not match unitary phase table for Hadamard gate'

            G = PHASE(0, d)
            phase_table_unitary, phase_table_symplectic = self.phase_table_local(G)
            diff_m = np.around(phase_table_unitary - phase_table_symplectic, 10)
            assert not np.any(diff_m), 'Symplectic phase table does not match unitary phase table for Phase gate'

            G = SUM(0, 1, d)
            phase_table_unitary, phase_table_symplectic = self.phase_table_entangling(G)
            diff_m = np.around(phase_table_unitary - phase_table_symplectic, 10)
            assert not np.any(diff_m), 'Symplectic phase table does not match unitary phase table for SUM[0,1] gate'

            G = SUM(1, 0, d)
            phase_table_unitary, phase_table_symplectic = self.phase_table_entangling(G)
            diff_m = np.around(phase_table_unitary - phase_table_symplectic, 10)
            assert not np.any(diff_m), 'Symplectic phase table does not match unitary phase table for SUM[1,0] gate'




