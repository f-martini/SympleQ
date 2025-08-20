from quaos.core.paulis import random_pauli_string
from quaos.core.circuits.known_circuits import to_x, to_ix
from quaos.core.circuits import Circuit, SUM, SWAP, Hadamard, PHASE
from quaos.core.circuits.utils import embed_symplectic
from quaos.core.paulis import PauliSum, PauliString
import numpy as np


class TestKnownCircuits():

    def test_to_x(self):
        target_x = 0
        list_of_failures = []
        for _ in range(20000):
            ps = random_pauli_string([3, 3, 3, 3])
            if ps.n_identities() == 4:
                continue
            c = to_x(ps, 0)
            if c.act(ps).x_exp[target_x] == 0 or c.act(ps).z_exp[target_x] != 0:
                print(f"Failed: {ps} -> {c.act(ps)}")
                list_of_failures.append(ps)

        return list_of_failures

    def test_to_ix(self):
        target_x = 0
        list_of_failures = []
        for _ in range(2000):
            ps = random_pauli_string([3, 3, 3, 3])
            if ps.n_identities() == 4:
                continue
            c = to_ix(ps, 0)
            if c is None:
                print(f"Failed: {ps} -> {c}")
                list_of_failures.append(ps)
                continue
            failed = False
            for i in range(ps.n_qudits()):
                if i == target_x and failed is False:
                    if c.act(ps).x_exp[target_x] == 0 or c.act(ps).z_exp[target_x] != 0:
                        print(f"Failed target x: {ps} -> {c.act(ps)}")
                        list_of_failures.append(ps)
                        failed = True
                elif failed is False:
                    if c.act(ps).x_exp[i] != 0 or c.act(ps).z_exp[i] != 0:
                        print(f"Failed identity: {ps} -> {c.act(ps)}")
                        list_of_failures.append(ps)
                        failed = True

        return list_of_failures


class TestCircuits():

    def random_pauli_sum(self, dim, n_qudits, n_paulis=10):
        # Generates a random PauliSum with n_paulis random PauliStrings of dimension dim
        #
        ps_list = []
        element_list = [[0, 0] * n_qudits]  # to keep track of already generated PauliStrings. Avoids identity and duplicates
        for _ in range(n_paulis):
            ps, elements = self.random_pauli_string(dim, n_qudits)
            element_list.append(elements)
            while elements in element_list:
                ps, elements = self.random_pauli_string(dim, n_qudits)
            ps_list.append(ps)
        return PauliSum(ps_list, dimensions=[dim] * n_qudits, standardise=True)

    def random_pauli_string(self, dim, n_qudits):
        # Generates a random PauliString of dimension dim
        string = ''
        elements = []
        for i in range(n_qudits):
            r = np.random.randint(0, dim)
            s = np.random.randint(0, dim)
            string += f'x{r}z{s} '
            elements.append(r)
            elements.append(s)
        return PauliString.from_string(string, dimensions=[dim] * n_qudits), elements

    def make_random_circuit(self, n_gates, n_qudits, dimension):

        dimensions = [dimension] * n_qudits
        gates_list = []
        for _ in range(n_gates):
            gate_int = np.random.randint(0, 4)
            if gate_int == 0:
                gate = Hadamard(np.random.randint(0, n_qudits), dimension)
            elif gate_int == 1:
                gate = PHASE(np.random.randint(0, n_qudits), dimension)
            elif gate_int == 2:
                gate = SUM(np.random.randint(0, n_qudits), np.random.randint(0, n_qudits), dimension)
            else:
                gate = SWAP(np.random.randint(0, n_qudits), np.random.randint(0, n_qudits), dimension)
            if gate == SUM or gate == SWAP:
                gates_list.append(gate)
            else:
                gates_list.append(gate)

        return Circuit(dimensions, gates_list)

    # def test_circuit_composition(self):
    #     n_qudits = 2
    #     dimension = 2
    #     n_gates = 4
    #     n_paulis = 2
    #     # make a random circuit
    #     circuit = self.make_random_circuit(n_gates, n_qudits, dimension)

    #     # make a random pauli sum
    #     pauli_sum = self.random_pauli_sum(dimension, n_qudits, n_paulis=n_paulis)

    #     # compose the circuit and pauli sum
    #     composed_gate = circuit.composite_gate()
    #     print(composed_gate.symplectic)
    #     output_composite = composed_gate.act(pauli_sum)
    #     output_sequential = circuit.act(pauli_sum)

    #     # show that the composed gate returns the same thing as the circuit when acting on the pauli sum
    #     assert output_composite == output_sequential, f'\n Composed gate:\n{output_composite} \n Sequential gate:\n{output_sequential}'

    # def test_hadamard_composition(self):
    #     # simple case of two Hadamards on different qubits. Known symplectic in this case.

    #     n_qudits = 1
    #     dimension = 2
    #     n_paulis = 2
    #     # make a random circuit
    #     circuit = Circuit([dimension] * n_qudits, [Hadamard(0, dimension), PHASE(0, dimension)])

    #     # make a random pauli sum
    #     pauli_sum = self.random_pauli_sum(dimension, n_qudits, n_paulis=n_paulis)

    #     # compose the circuit and pauli sum
    #     composed_gate = circuit.composite_gate()
    #     output_composite = composed_gate.act(pauli_sum)
    #     output_sequential = circuit.act(pauli_sum)

    #     print(composed_gate.symplectic)

    #     print((Hadamard(0, dimension).symplectic.T @ PHASE(0, dimension).symplectic.T).T)

    #     print(output_composite)
    #     print(output_sequential)

    #     # show that the composed gate returns the same thing as the circuit when acting on the pauli sum
    #     assert output_composite == output_sequential

    def test_symplectic_embedding(self):

        # hadamard on a two qubit symplectic
        gate = Hadamard(0, 2)

        symplectic = gate.symplectic

        embedded_symplectic_correct = np.zeros([4, 4], dtype=int)
        embedded_symplectic_correct[0, 2] = 1  # image of x0 is z0
        embedded_symplectic_correct[2, 0] = 1  # image of z0 is x0

        embedded_symplectic_correct[1, 1] = 1  # image of x1 is x1
        embedded_symplectic_correct[3, 3] = 1  # image of z1 is z1

        embedded_symplectic, embedded_h = embed_symplectic(symplectic, gate.phase_vector, [0], 2, 2)

        print(embedded_symplectic_correct)
        print(embedded_symplectic)

        assert np.array_equal(embedded_symplectic, embedded_symplectic_correct)

        # test that the symplectic is a valid symplectic


if __name__ == "__main__":
    # TestCircuits().test_circuit_composition()
    # TestCircuits().test_hadamard_composition()
    TestCircuits().test_symplectic_embedding()
