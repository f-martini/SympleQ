from quaos.circuits import SUM, SWAP, Hadamard, PHASE, ArbitraryGate
from quaos.circuits.utils import is_symplectic
from quaos.paulis import PauliSum, PauliString
import numpy as np


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
        ps_list = []
        for _ in range(n_paulis):
            r1 = np.random.randint(0, dim)
            r2 = np.random.randint(0, dim)
            s1 = np.random.randint(0, dim)
            s2 = np.random.randint(0, dim)
            ps_list.append(PauliString.from_string(f'x{r1}z{s1} x{r2}z{s2}', dimensions=[dim, dim]))
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
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d], standardise=False)

                output_psum = SUM(0, 1, d).act(input_psum)
                assert output_psum == output_psum_correct, 'Error in SUM gate: \n' + output_psum.__str__() + '\n' + output_psum_correct.__str__()

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

                    # phase_out = np.diag(C.T @ U @ C) @ a + 2 * a @ np.triu(C.T @ U @ C) @ a - a @ np.diag(np.diag(C.T @ U @ C)) @ a
                    ps_phase_out_correct.append(0)

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d], standardise=False)

                output_psum = SWAP(0, 1, d).act(input_psum)
                assert output_psum == output_psum_correct, 'Error in SUM gate: \n' + input_psum.__str__() + '\n' + output_psum.__str__() + '\n' + output_psum_correct.__str__()

    def test_Hadamard(self):
        # TODO: Be certain of inverse convention - ultimately arbitrary but should match prevalent literature
        for d in [2, 5, 11]:
            # test pauli_strings
            gate = Hadamard(0, d, inverse=True)  # Hadamard on qubit 0

            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{(-s1)%d}z{r1} x{r2}z{s2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in Hadamard gate 0'

            gate = Hadamard(1, d, inverse=True)  # Hadamard on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{(-s2)%d}z{r2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in Hadamard gate 1'
            # test pauli_sums
            gate = Hadamard(0, d, inverse=True)  # Hadamard on qubit 0

            for i in range(100):
                ps_list_in = []
                ps_list_out_correct = []
                ps_phase_out_correct = []
                for j in range(10):
                    input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                    output_str_correct = f"x{(-s1)%d}z{r1} x{r2}z{s2}"

                    ps_list_in.append(input_ps)
                    ps_list_out_correct.append(PauliString.from_string(output_str_correct, dimensions=[d, d]))
                    ps_phase_out_correct.append(-2 * r1 * s1)

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum = gate.act(input_psum)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d], standardise=False)
                assert output_psum == output_psum_correct, 'Error in Hadamard gate: \n' + input_psum.__str__() + '\n' + output_psum.__str__() + '\n' + output_psum_correct.__str__()

    def test_PHASE(self):
        # TODO: Better approach to phases here - is there another approach that we can test the first one with like Hadamard?
        for d in [2, 5, 11]:
            gate = PHASE(0, d)  # PHASE on qubit 0

            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{(r1 + s1) % d} x{r2}z{s2}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in Hadamard gate 0'

            gate = PHASE(1, d)  # PHASE on qudit 1
            for i in range(100):
                input_ps, r1, r2, s1, s2 = self.random_pauli_string(d)
                output_str_correct = f"x{r1}z{s1} x{r2}z{(r2 + s2) % d}"

                output_ps = gate.act(input_ps)
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[d, d]), 'Error in Hadamard gate 1'
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

                    p = np.dot(h, a1) - np.diag(C.T @ U @ C) @ a1 + 2 * a1 @ np.triu(C.T @ U @ C) @ a1 - a1 @ np.diag(np.diag(C.T @ U @ C)) @ a1
                    ps_phase_out_correct.append(p)

                input_psum = PauliSum(ps_list_in, dimensions=[d, d], standardise=False)
                output_psum = gate.act(input_psum)
                output_psum_correct = PauliSum(ps_list_out_correct, phases=ps_phase_out_correct, dimensions=[d, d], standardise=False)
                assert output_psum == output_psum_correct, 'Error in Hadamard gate: \n' + input_psum.__str__() + '\n' + output_psum.__str__() + '\n' + output_psum_correct.__str__()

    # def test_arbitrary_gate(self):

    #     for d in [2, 5, 11]:
    #         for i in range(1000):
    #             input_ps = self.random_pauli_sum(d, n_paulis=2)
    #             target_ps = self.random_pauli_sum(d, n_paulis=2)
    #             if np.all(input_ps.symplectic_product_matrix() == target_ps.symplectic_product_matrix()):
    #                 gate = ArbitraryGate('ArbGate', input_ps, target_ps)
    #                 output_ps = gate.act(input_ps)
    #                 assert output_ps == target_ps, 'Error in Arbitrary gate: \n' + input_ps.__str__() + '\n' + output_ps.__str__() + '\n' + target_ps.__str__()


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
                                            err = 'Out: \n' + (gate.act(p1) * gate.act(p2)).__str__() + '\n' + gate.act(p1 * p2).__str__()
                                            assert gate.act(p1) * gate.act(p2) == gate.act(p1 * p2), err0 + err + '\n'

        for dim in [3, 5, 7, 15]:
            gates = [SUM(0, 1, dim), SUM(1, 0, dim), SWAP(0, 1, dim), Hadamard(0, dim), Hadamard(1, dim), PHASE(0, dim), PHASE(1, dim)]
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
            assert is_symplectic(gate.symplectic), f"Gate {gate.name} is not symplectic"


if __name__ == "__main__":

    # dim = 2
    # phase_mod = 2 * dim
    # gate = SWAP(0, 1, dim)  # SUM(0, 1, dim)  #
    # i = 0
    # for x0 in range(dim):
    #     for z0 in range(dim):
    #         for x1 in range(dim):
    #             for z1 in range(dim):
    #                 for x0p in range(dim):
    #                     for z0p in range(dim):
    #                         for x1p in range(dim):
    #                             for z1p in range(dim):

    #                                 i += 1
    #                                 p1 = PauliString.from_string(f'x{x0}z{z0} x{x1}z{z1}', dimensions=[dim, dim])
    #                                 p2 = PauliString.from_string(f'x{x0p}z{z0p} x{x1p}z{z1p}', dimensions=[dim, dim])

    #                                 p3 = p1 * p2

    #                                 C = gate.symplectic

    #                                 nq = 2
    #                                 U = np.zeros((2 * nq, 2 * nq), dtype=int)
    #                                 U[nq:, :nq] = np.eye(nq, dtype=int)

    #                                 a1 = p1.symplectic()
    #                                 a2 = p2.symplectic()
    #                                 a3 = p3.symplectic()

    #                                 multiplicative_phase = (2 * (C @ a1) @ U @ (C @ a2)) % phase_mod
    #                                 p3_phase = (2 * a1 @ U @  a2) % phase_mod

    #                                 # check actions of gates on symplectic
    #                                 assert np.all(C @ a1 % dim == gate.act(p1).symplectic())
    #                                 assert np.all(C @ a2 % dim == gate.act(p2).symplectic())
    #                                 assert np.all(C @ a3 % dim == gate.act(p3).symplectic())
    #                                 assert np.all((gate.act(p1) * gate.act(p2)).symplectic() == gate.act(p3).symplectic())
    #                                 assert np.all((C @ a1 + C @ a2) % dim == C @ a3 % dim)
    #                                 assert np.all((C @ a1 + C @ a2) % dim == gate.act(p3).symplectic())

    #                                 # check acquired phases

    #                                 phase1 = - np.diag(C.T @ U @ C) @ a1  + 2 * a1 @ np.triu(C.T @ U @ C) @ a1 - a1 @ np.diag(np.diag(C.T @ U @ C)) @ a1
    #                                 phase2 = - np.diag(C.T @ U @ C) @ a2  + 2 * a2 @ np.triu(C.T @ U @ C) @ a2 - a2 @ np.diag(np.diag(C.T @ U @ C)) @ a2
    #                                 phase3 = p3_phase - np.diag(C.T @ U @ C) @ a3  # + 2 * a3 @ np.triu(C.T @ U @ C) @ a3 - a3 @ np.diag(np.diag(C.T @ U @ C)) @ a3
    #                                 # phase1 += 0 * gate.phase_function(p1)
    #                                 # phase2 += 0 * gate.phase_function(p2)
    #                                 # phase3 += 0 * gate.phase_function(p3)

    #                                 # print(i, phase1, phase2, multiplicative_phase, phase3)

    #                                 # print(gate.phase_function(p1) + gate.phase_function(p2), gate.phase_function(p3))
    #                                 assert p3_phase == multiplicative_phase
    #                                 assert ((phase1 + phase2) + multiplicative_phase) % phase_mod == phase3 % phase_mod, (p1.__str__(), p2.__str__(), p3.__str__())
    # print('Done')

    # dim = 2
    # phase_mod = 2 * dim
    # gate = Hadamard(0, dim)
    # i = 0
    # for x0 in range(dim):
    #     for z0 in range(dim):
    #         for x1 in range(dim):
    #             for z1 in range(dim):

    #                 i += 1
    #                 p1 = PauliString([x0], [z0], dimensions=[dim])
    #                 p2 = PauliString([x0], [z0], dimensions=[dim])

    #                 p3 = p1 * p2

    #                 C = gate.symplectic
    #                 h = gate.phase_vector

    #                 nq = 1
    #                 U = np.zeros((2 * nq, 2 * nq), dtype=int)
    #                 U[nq:, :nq] = np.eye(nq, dtype=int)

    #                 a1 = p1.symplectic()
    #                 a2 = p2.symplectic()
    #                 a3 = p3.symplectic()

    #                 multiplicative_phase = (2 * (C @ a1).T @ U @ (C @ a2)) % phase_mod
    #                 p3_phase = (2 * a1.T @ U @  a2) % phase_mod

    #                 # check actions of gates on symplectic
    #                 # print(C @ a1 % dim, gate.act(p1).symplectic())
    #                 assert np.all(C @ a1 % dim == gate.act(p1).symplectic()), (C @ a1 % dim, gate.act(p1).symplectic())
    #                 assert np.all(C @ a2 % dim == gate.act(p2).symplectic()), (p2.__str__(), (C @ a2) % dim, gate.act(p2).symplectic())
    #                 assert np.all(C @ a3 % dim == gate.act(p3).symplectic()), (p3, a3, gate.act(p3).symplectic())
    #                 assert np.all((gate.act(p1) * gate.act(p2)).symplectic() == gate.act(p3).symplectic())
    #                 assert np.all((C @ a1 + C @ a2) % dim == C @ a3 % dim)
    #                 assert np.all((C @ a1 + C @ a2) % dim == gate.act(p3).symplectic())

    #                 phase1 = np.dot(h, a1) - np.diag(C.T @ U @ C) @ a1 + 2 * a1 @ np.triu(C.T @ U @ C) @ a1 - a1 @ np.diag(np.diag(C.T @ U @ C)) @ a1
    #                 phase2 = np.dot(h, a2) - np.diag(C.T @ U @ C) @ a2 + 2 * a2 @ np.triu(C.T @ U @ C) @ a2 - a2 @ np.diag(np.diag(C.T @ U @ C)) @ a2
    #                 phase3 = np.dot(h, a3) + p3_phase - np.diag(C.T @ U @ C) @ a3 + 2 * a3 @ np.triu(C.T @ U @ C) @ a3 - a3 @ np.diag(np.diag(C.T @ U @ C)) @ a3

    #                 print(i, phase1, phase2, multiplicative_phase, p3_phase, phase3)
    #                 assert ((phase1 + phase2) + multiplicative_phase) % phase_mod == phase3 % phase_mod, (p1.__str__(), p2.__str__(), p3.__str__())


    # print('Done')
    tst = TestGates()
    d = 2
    input_ps = tst.random_pauli_sum(d, n_paulis=4)

    for i in range(100):
        target_ps = tst.random_pauli_sum(d, n_paulis=4)
        if np.all(input_ps.symplectic_product_matrix() == target_ps.symplectic_product_matrix()) and input_ps != target_ps:
            gate = ArbitraryGate('ArbGate', input_ps, target_ps)
            output_ps = gate.act(input_ps)
            print(target_ps)
            print(output_ps)

    print('done')
