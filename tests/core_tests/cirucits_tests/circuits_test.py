# from sympleq.core.paulis import random_pauli_string
from sympleq.core.circuits.known_circuits import to_x, to_ix
from sympleq.core.circuits import Circuit, SUM, SWAP, Hadamard, PHASE, PauliGate
from sympleq.core.circuits.utils import embed_symplectic
from sympleq.core.paulis import PauliSum, PauliString
import numpy as np
import scipy.sparse as sp


class TestCircuits():

    def test_to_x(self, n_tests=1000):
        target_x = 0
        list_of_failures = []
        for _ in range(n_tests):
            ps = PauliString.from_random([3, 3, 3, 3])
            if ps.n_identities() == 4:
                continue
            c = to_x(ps, 0)
            if c.act(ps).x_exp[target_x] == 0 or c.act(ps).z_exp[target_x] != 0:
                print(f"Failed: {ps} -> {c.act(ps)}")
                list_of_failures.append(ps)

        assert len(list_of_failures) == 0, f"Failures: {list_of_failures}"

    def test_to_ix(self, n_tests=1000):
        target_x = 0
        list_of_failures = []
        for _ in range(n_tests):
            ps = PauliString.from_random([2, 2, 2, 2])
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
        print(list_of_failures)
        assert len(list_of_failures) == 0

    def random_pauli_sum(self, dim, n_qudits, n_paulis=10):
        # Generates a random PauliSum with n_paulis random PauliStrings of dimension dim
        #
        ps_list = []
        element_list = [[0, 0] * n_qudits]  # to keep track of already generated PauliStrings.
        # Avoids identity and duplicates
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
                # two random non-equal numbers
                q1, q2 = np.random.randint(0, n_qudits, 2)
                while q1 == q2:
                    q1, q2 = np.random.randint(0, n_qudits, 2)
                gate = SUM(q1, q2, dimension)
            else:
                q1, q2 = np.random.randint(0, n_qudits, 2)
                while q1 == q2:
                    q1, q2 = np.random.randint(0, n_qudits, 2)
                gate = SWAP(q1, q2, dimension)
            if gate == SUM or gate == SWAP:
                gates_list.append(gate)
            else:
                gates_list.append(gate)

        return Circuit(dimensions, gates_list)

    def test_circuit_composition(self):
        # TODO: Full test for mixed dimensions
        for _ in range(100):
            n_qudits = 3
            dimension = 2
            n_gates = 15
            n_paulis = 5
            # make a random circuit
            circuit = self.make_random_circuit(n_gates, n_qudits, dimension)
            print(circuit)

            # make a random pauli sum
            pauli_sum = self.random_pauli_sum(dimension, n_qudits, n_paulis=n_paulis)
            print(dimension)
            # compose the circuit and pauli sum
            composed_gate = circuit.composite_gate()
            print(composed_gate.symplectic)
            output_composite = composed_gate.act(pauli_sum)
            output_sequential = circuit.act(pauli_sum)

            # show that the composed gate returns the same thing as the circuit when acting on the pauli sum
            assert output_composite == output_sequential, (
                f'Input: \n {pauli_sum}\n'
                f'Composed gate:\n{output_composite} \n'
                f'Sequential gate:\n{output_sequential}'
            )

    def test_hadamard_composition(self):
        # simple case of two Hadamards on different qubits. Known symplectic in this case.

        n_qudits = 1
        dimension = 2
        n_paulis = 2
        # make a random circuit
        circuit = Circuit([dimension] * n_qudits, [Hadamard(0, dimension), PHASE(0, dimension)])

        # make a random pauli sum
        pauli_sum = self.random_pauli_sum(dimension, n_qudits, n_paulis=n_paulis)

        # compose the circuit and pauli sum
        composed_gate = circuit.composite_gate()
        output_composite = composed_gate.act(pauli_sum)
        output_sequential = circuit.act(pauli_sum)

        # print(composed_gate.symplectic)

        # print((Hadamard(0, dimension).symplectic.T @ PHASE(0, dimension).symplectic.T).T)

        print(output_composite)
        print(output_sequential)

        # show that the composed gate returns the same thing as the circuit when acting on the pauli sum
        assert output_composite == output_sequential

    def test_random_circuit(self):
        # test that a random circuit can be generated with the correct dimensions on mixed qudits
        for _ in range(1000):
            n_qudits = np.random.randint(2, 10)
            dimensions = np.random.randint(2, 5, size=n_qudits)
            C = Circuit.from_random(n_gates=10, dimensions=dimensions)
            ps = PauliSum.from_random(10, n_qudits, dimensions)
            out = C.act(ps)
            assert np.all(out.dimensions == dimensions)

    def test_single_hadamard_unitary(self):
        # For a single-qudit circuit with one Hadamard, the circuit unitary
        # should equal the gate's local unitary.
        for d in [2, 3, 5, 11]:
            gate = Hadamard(0, d)
            circuit = Circuit([d], [gate])
            U_circ = circuit.unitary()
            assert sp.issparse(U_circ)
            U_circ = U_circ.toarray()
            U_gate = gate.unitary()
            assert U_circ.shape == U_gate.shape
            print()
            assert np.allclose(U_circ, U_gate.toarray()), f"Failed for d={d}\n{U_circ}\n{U_gate}"

    def test_mixed_qudits_phase_with_unitary(self):
        N = 1000
        dimensions = [2, 3, 5]
        n_paulis = 1
        n_qudits = 3
        for _ in range(N):
            P = PauliSum.from_random(n_paulis, n_qudits, dimensions, rand_weights=False)
            C = Circuit.from_random(n_gates=np.random.randint(1, 6), dimensions=dimensions)
            U = C.unitary()

            ps_m = P.matrix_form()
            ps_res = C.act(P)
            ps_res_m = ps_res.matrix_form()
            phase_symplectic = ps_res.phases[0]
            ps_res.phases = [0]
            ps_res_m = ps_res.matrix_form()
            ps_m_res = U @ ps_m @ U.conj().T
            mask = (ps_res_m.toarray() != 0)
            factors = np.unique(np.around(ps_m_res.toarray()[mask] / ps_res_m.toarray()[mask], 10))
            assert len(factors) == 1
            factor = factors[0]
            d = int(P.lcm)
            phase_unitary = int(np.around((d * np.angle(factor) / (np.pi)) % (2 * d), 1))
            assert phase_symplectic == phase_unitary

    def test_phase_mixed_species(self):
        def debug_steps(C: Circuit, P: PauliSum):
            print(f"Initial phases: {P.phases} -- exponents: {P.tableau()}")
            for i, partial_p in enumerate(C.act_iter(P)):
                gate = C.gates[i]
                print(f"Phases after {gate.name}: {partial_p.phases} -- exponents: {partial_p.tableau()}")

        # Test multiplication
        P1 = PauliSum(['x1z1 x0z0'],
                      dimensions=[3, 2],
                      weights=[1], phases=[0])

        P2 = PauliSum(['x2z2 x0z0'],
                      dimensions=[3, 2],
                      weights=[1], phases=[0])

        product = P1.H() * P2
        assert product.phases == [8]

        # Test 1: Simple qutrit + qubit
        P = PauliSum(['x2z0 x0z0'],
                     dimensions=[3, 2],
                     weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 4

        # Test 2: Simple ququint + qubit
        P = PauliSum(['x2z0 x0z0'],
                     dimensions=[5, 2],
                     weights=[1], phases=[0])

        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 4

        # Test 3: More complex ququint + qubit
        P = PauliSum(['x3z0 x0z0'],
                     dimensions=[5, 2],
                     weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 12

        # Test 4: Simple ququint + qutrit
        P = PauliSum(['x2z0 x0z0'],
                     dimensions=[5, 3],
                     weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 6

        # Test 5: Simple qutritt + qubit but action on qubit
        P = PauliSum(['x0z0 x1z0'],
                     dimensions=[3, 2],
                     weights=[1], phases=[0])
        idx = 1
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 3

        # Test 6: Simple ququint + qutrit + qubit
        P = PauliSum(['x2z0 x0z0 x0z0'],
                     dimensions=[5, 3, 2],
                     weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[PHASE(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 12

        # Test 7: composite circuit
        P = PauliSum(['x2z2 x0z0'],
                     dimensions=[3, 2],
                     weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[
            PHASE(idx, P.dimensions[idx]),
            PHASE(idx, P.dimensions[idx]),
            Hadamard(idx, P.dimensions[idx])])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 8

    @staticmethod
    def _linear_index(dims, idxs):
        # Row-major: idx = sum_k idxs[k] * prod_{l>k} dims[l]
        strides = [1] * len(dims)
        for k in range(len(dims) - 2, -1, -1):
            strides[k] = strides[k + 1] * dims[k + 1]
        return sum(idxs[k] * strides[k] for k in range(len(dims)))

    def test_swap_embedding_on_equal_dims(self):
        # Verify SWAP on qudits (0,1) within a 2-qudit system with equal dimensions.
        dims = [3, 3]
        c = Circuit(dims, [SWAP(0, 1, 3)])
        U = c.unitary().toarray()

        # Start in |i,j> with i=1, j=2
        i, j = 1, 2
        D = np.prod(dims)
        psi = np.zeros(D, dtype=complex)
        psi[self._linear_index(dims, [i, j])] = 1.0

        # Expected after SWAP: |j,i>
        phi = U @ psi
        expected = np.zeros(D, dtype=complex)
        expected[self._linear_index(dims, [j, i])] = 1.0

        assert np.allclose(phi, expected), f"Expected:\n{expected}\nGot:\n{phi}"

    def test_sum_embedding_on_three_qudits(self):
        # Verify SUM on qudits (1,2) inside a 3-qudit system.
        d = 5
        dims = [d, d, d]
        c = Circuit(dims, [SUM(1, 2, d)])
        U = c.unitary().toarray()

        # Start in |i,j,k> = |3,1,4>
        i, j, k = 3, 1, 4
        D = np.prod(dims)
        psi = np.zeros(D, dtype=complex)
        psi[self._linear_index(dims, [i, j, k])] = 1.0

        # After SUM(1->2): |i, j, k+j mod d>
        phi = U @ psi
        expected = np.zeros(D, dtype=complex)
        expected[self._linear_index(dims, [i, j, (k + j) % d])] = 1.0

        assert np.allclose(phi, expected)

    def test_phase_embedding_on_middle_qudit(self):
        # Verify PHASE acting on middle qudit multiplies amplitude appropriately.
        d0, d1, d2 = 3, 5, 2
        dims = [d0, d1, d2]
        c = Circuit(dims, [PHASE(1, d1)])
        U = c.unitary().toarray()

        # Basis |i,j,k> = |2,3,1>
        i, j, k = 2, 3, 1
        D = np.prod(dims)
        psi = np.zeros(D, dtype=complex)
        psi[self._linear_index(dims, [i, j, k])] = 1.0

        phi = U @ psi

        # PHASE unitary is diag(zeta^{j^2}) on that qudit, with zeta = exp(2πi/(2d)).
        zeta = np.exp(1j * 2 * np.pi / (2 * d1))
        factor = zeta ** (j * j)
        expected = np.zeros(D, dtype=complex)
        expected[self._linear_index(dims, [i, j, k])] = factor

        assert np.allclose(phi, expected)

    def test_circuit_unitary(self):
        n_tests = 100
        n_paulis = 10
        n_qudits = 3
        c_depth = 1
        dims = [2, 3, 5]
        gate_list = [Hadamard, SWAP, SUM, PHASE]
        for dim in dims:
            dimensions = [dim] * n_qudits
            for _ in range(n_tests):
                ps = PauliSum.from_random(n_paulis, n_qudits, dimensions, False)
                c = Circuit.from_random(c_depth, dimensions, gate_list=gate_list)
                U_c = c.unitary()
                P_from_conjugation = U_c @ ps.matrix_form() @ U_c.conj().T
                P_from_act = c.act(ps).matrix_form()
                diff_m = np.around(P_from_conjugation.toarray() - P_from_act.toarray(), 10)
                assert not np.any(diff_m), f'failed for dim {_, dim, c.__str__()}'

    def test_single_gate_circuit_unitary(self):
        gate_list = [Hadamard, SWAP, SUM, PHASE, PauliGate]
        n_qudits = 5
        dims = [2, 3]   # much bigger and the Hilbert space gets too large
        indices = [0, 3]
        for gate in gate_list:
            for dim in dims:
                dimensions = [dim] * n_qudits
                if gate in [SUM, SWAP]:
                    c = Circuit(dimensions, [gate(indices[0], indices[1], dim)])
                    g = gate(indices[0], indices[1], dim)
                elif gate == PauliGate:
                    g = gate(PauliString.from_random(dimensions))
                    c = Circuit(dimensions, [g])
                else:
                    c = Circuit(dimensions, [gate(indices[0], dim)])
                    g = gate(indices[0], dim)

                U_c = c.unitary().toarray()
                U_g = g.unitary(dims=dimensions).toarray()
                assert np.allclose(U_c, U_g)


if __name__ == '__main__':
    TestCircuits().test_circuit_unitary()
