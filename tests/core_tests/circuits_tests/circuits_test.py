import numpy as np
from scipy.sparse import issparse
from sympleq.core.circuits.known_circuits import to_x, to_ix
from sympleq.core.circuits import Circuit, GATES
from sympleq.core.paulis import PauliSum, PauliString


class TestCircuits():

    # TODO: I have tried to generalise this to mixed dimensions, but it seems that the to_x
    # function does not yet support this properly. If this is OK ignore this
    # (I added an error when mixed dimensions are inputted for the time being)
    def test_to_x(self, n_tests=500):
        max_qudits = 25
        list_of_failures = []
        for _ in range(n_tests):
            dimension_chosen = np.random.choice([2, 3, 5, 7, 11, 13, 17], size=1)[0]
            dims = [dimension_chosen for _ in range(np.random.randint(1, max_qudits))]
            target_x = np.random.randint(0, len(dims))
            ps = PauliString.from_random(dims)
            if ps.n_identities() == len(dims):
                continue
            c = to_x(ps, target_x)
            if c.act(ps).x_exp[target_x] == 0 or c.act(ps).z_exp[target_x] != 0:
                print(f"Failed: {ps} -> {c.act(ps)}")
                list_of_failures.append(ps)

        assert len(list_of_failures) == 0, f"Failures: {list_of_failures}"

    def test_to_ix(self, n_tests=500):
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

    def random_pauli_sum(self, dim: int, n_qudits: int, n_paulis: int = 10) -> PauliSum:
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
        return PauliSum.from_pauli_strings(ps_list)

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

    def make_random_circuit(self, n_gates, n_qudits, dimension) -> Circuit:
        dimensions = [dimension] * n_qudits
        gates_list = []
        qudits_list = []
        for _ in range(n_gates):
            gate_int = np.random.randint(0, 4)
            if gate_int == 0:
                gates_list.append(GATES.H)
                qudits_list.append((np.random.randint(0, n_qudits),))
            elif gate_int == 1:
                gates_list.append(GATES.S)
                qudits_list.append((np.random.randint(0, n_qudits),))
            elif gate_int == 2:
                # two random non-equal numbers
                q1, q2 = np.random.randint(0, n_qudits, 2)
                while q1 == q2:
                    q1, q2 = np.random.randint(0, n_qudits, 2)
                gates_list.append(GATES.SUM)
                qudits_list.append((int(q1), int(q2)))
            else:
                q1, q2 = np.random.randint(0, n_qudits, 2)
                while q1 == q2:
                    q1, q2 = np.random.randint(0, n_qudits, 2)
                gates_list.append(GATES.SWAP)
                qudits_list.append((int(q1), int(q2)))

        return Circuit(dimensions, gates_list, qudits_list)

    def test_circuit_composition(self):
        # TODO: Full test for mixed dimensions
        for _ in range(10):
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
            # For composite gate, act on all qudits
            output_composite = composed_gate.act(pauli_sum, tuple(range(n_qudits)))
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
        circuit = Circuit([dimension] * n_qudits, [GATES.H, GATES.S], [(0,), (0,)])

        # make a random pauli sum
        pauli_sum = self.random_pauli_sum(dimension, n_qudits, n_paulis=n_paulis)

        # compose the circuit and pauli sum
        composed_gate = circuit.composite_gate()
        output_composite = composed_gate.act(pauli_sum, tuple(range(n_qudits)))
        output_sequential = circuit.act(pauli_sum)

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
            ps = PauliSum.from_random(10, dimensions)
            out = C.act(ps)
            assert np.all(out.dimensions == dimensions)

    def test_single_hadamard_unitary(self):
        # For a single-qudit circuit with one Hadamard, the circuit unitary
        # should equal the gate's local unitary.
        for d in [2, 3, 5, 11]:
            circuit = Circuit([d], [GATES.H], [(0,)])
            U_circ = circuit.unitary()
            assert issparse(U_circ)
            U_gate = GATES.H.unitary(d)
            assert U_circ.shape == U_gate.shape
            assert np.allclose(U_circ.toarray(), U_gate.toarray())

    def test_mixed_qudits_phase_with_unitary(self):
        N = 100
        dimensions = [2, 3, 5]
        n_paulis = 1
        for _ in range(N):
            P = PauliSum.from_random(n_paulis, dimensions, rand_weights=False)
            C = Circuit.from_random(n_gates=np.random.randint(1, 6), dimensions=dimensions)
            U = C.unitary()

            ps_m = P.to_hilbert_space()
            ps_res = C.act(P)
            ps_res_m = ps_res.to_hilbert_space()
            phase_symplectic = ps_res.phases[0]

            ps_res.reset_phases()
            ps_res_m = ps_res.to_hilbert_space().toarray()
            ps_m_res = (U @ ps_m @ U.conj().T)
            mask = (ps_res_m != 0)
            factors = np.unique(np.around(ps_m_res[mask] / ps_res_m[mask], 10))
            assert len(factors) == 1
            factor = factors[0]
            d = P.lcm
            phase_unitary = int(np.around((d * np.angle(factor) / (np.pi)) % (2 * d), 1))
            assert np.array_equal(phase_symplectic, phase_unitary)

    def test_phase_mixed_species(self):
        def debug_steps(C: Circuit, P: PauliSum):
            print(f"Initial phases: {P.phases} -- exponents: {P.tableau}")
            for i, partial_p in enumerate(C.act_iter(P)):
                gate = C.gates[i]
                print(f"Phases after {gate.name}: {partial_p.phases} -- exponents: {partial_p.tableau}")

        # Test 1: Simple qutrit + qubit
        P = PauliSum.from_string(['x2z0 x0z0'],
                                 dimensions=[3, 2],
                                 weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 4

        # Test 2: Simple ququint + qubit
        P = PauliSum.from_string(['x2z0 x0z0'],
                                 dimensions=[5, 2],
                                 weights=[1], phases=[0])

        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 4

        # Test 3: More complex ququint + qubit
        P = PauliSum.from_string(['x3z0 x0z0'],
                                 dimensions=[5, 2],
                                 weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 12

        # Test 4: Simple ququint + qutrit
        P = PauliSum.from_string(['x2z0 x0z0'],
                                 dimensions=[5, 3],
                                 weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 6

        # Test 5: Simple qutrit + qubit but action on qubit
        P = PauliSum.from_string(['x0z0 x1z0'],
                                 dimensions=[3, 2],
                                 weights=[1], phases=[0])
        idx = 1
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 3

        # Test 6: Simple ququint + qutrit + qubit
        P = PauliSum.from_string(['x2z0 x0z0 x0z0'],
                                 dimensions=[5, 3, 2],
                                 weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions, gates=[GATES.S], qudits=[(idx,)])
        debug_steps(C, P)
        P = C.act(P)
        assert P.phases[0] == 12

        # Test 7: composite circuit
        P = PauliSum.from_string(['x2z2 x0z0'],
                                 dimensions=[3, 2],
                                 weights=[1], phases=[0])
        idx = 0
        C = Circuit(dimensions=P.dimensions,
                    gates=[GATES.S, GATES.S, GATES.H],
                    qudits=[(idx,), (idx,), (idx,)])
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
        c = Circuit(dims, [GATES.SWAP], [(0, 1)])
        U = c.unitary()

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
        c = Circuit(dims, [GATES.SUM], [(1, 2)])
        U = c.unitary()

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
        c = Circuit.from_tuples(dims, (GATES.S, 1))
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
        for dim in dims:
            dimensions = [dim] * n_qudits
            for _ in range(n_tests):
                ps = PauliSum.from_random(n_paulis, dimensions, False)
                c = Circuit.from_random(c_depth, dimensions)
                U_c = c.unitary()
                P_from_conjugation = U_c @ ps.to_hilbert_space() @ U_c.conj().T
                P_from_act = c.act(ps).to_hilbert_space()
                diff_m = np.around(P_from_conjugation - P_from_act.toarray(), 10)
                assert not np.any(diff_m), f'failed for dim {_, dim, c.__str__()}'

    def test_single_gate_circuit_unitary(self):
        pass


if __name__ == '__main__':
    TestCircuits().test_circuit_composition()
