from sympleq.core.circuits.gates import GATES
from sympleq.models.random_hamiltonian import random_gate_symmetric_hamiltonian
from sympleq.core.symmetries.clifford import find_clifford_symmetries  # , qudit_cost, min_qudit_clifford_symmetry
from sympleq.core.circuits import Circuit
import numpy as np


class TestSymmetryFinder:

    def test_random_SWAP_symmetry(self):
        n_tests = 30
        dimension = 2
        n_qudits = 2
        n_paulis = 2
        for _ in range(n_tests):
            qudit_indices = (0, 1)
            all_qudit_indices = tuple(range(n_qudits))
            # unscrambled H
            H = random_gate_symmetric_hamiltonian(
                GATES.SWAP, dimension, qudit_indices, n_qudits, n_paulis, scrambled=False)

            print("Hamiltonian")
            print(H)
            C = Circuit.from_random(4, H.dimensions)
            print(C.gates_layout())
            C_gate = C.composite_gate()  # scrambling circuit
            print("Gate from circuti")
            print(C_gate.symplectic)
            print(C_gate.phase_vector(dimension))
            print(C_gate.inverse().symplectic)
            print(C_gate.inverse().phase_vector(dimension))

            H = C_gate.act(H, all_qudit_indices)
            H.weight_to_phase()
            print("Hamiltonian")
            print(H)

            scrambled_sym = Circuit.from_gates_and_qudits(
                H.dimensions,
                [C_gate.inverse(), GATES.SWAP, C_gate],
                [all_qudit_indices, qudit_indices, all_qudit_indices]).composite_gate()

            check = scrambled_sym.act(H, all_qudit_indices)
            assert H.is_close(check, literal=False), f"\n{H}\n{check}"

            known_F = scrambled_sym.symplectic
            symmetries = find_clifford_symmetries(H)

            assert len(symmetries) != 0

            for sym_gate in symmetries:
                print(np.all(sym_gate.symplectic == known_F) and np.all(
                    sym_gate.phase_vector() == scrambled_sym.phase_vector()))
                H_s = H.to_standard_form()
                H_out = sym_gate.act(H, all_qudit_indices).to_standard_form()
                H_s.weight_to_phase()
                H_out.weight_to_phase()
                assert np.all(H_s.tableau == H_out.tableau)
                assert np.all(H_s.phases == H_out.phases)
                assert np.all(H_s.weights == H_out.weights)

                assert sym_gate.act(H, all_qudit_indices).is_close(H, literal=False)

    def test_random_multi_SWAP_symmetry(self):

        n_tests = 100
        p = 2
        n_qudits = 3
        n_paulis = 7
        for _ in range(n_tests):
            sym = Circuit([p] * n_qudits, [SWAP(0, 1, p), SWAP(1, 2, p)])  #
            sym = sym.composite_gate()
            # unscrambled H
            H = random_gate_symmetric_hamiltonian(sym, n_qudits, n_paulis, scrambled=False)
            C = Circuit.from_random(100, H.dimensions).composite_gate()  # scrambling circuit
            H = C.act(H)
            H.weight_to_phase()
            scrambled_sym = Circuit(H.dimensions, [C.inv(), sym, C]).composite_gate()
            assert H.to_standard_form() == scrambled_sym.act(H).to_standard_form(
            ), f"\n{H.to_standard_form().__str__()}\n{sym.act(H).to_standard_form().__str__()}"

            known_F = scrambled_sym.symplectic
            circ = find_clifford_symmetries(H)

            assert len(circ) != 0

            for c in circ:
                print(np.all(c.symplectic == known_F) and np.all(
                    c.phase_vector == scrambled_sym.phase_vector))
                H_s = H.to_standard_form()
                H_out = c.act(H).to_standard_form()
                H_s.weight_to_phase()
                H_out.weight_to_phase()
                assert np.all(H_s.tableau == H_out.tableau)
                assert np.all(H_s.phases == H_out.phases)
                assert np.all(H_s.weights == H_out.weights)

                assert c.act(H).to_standard_form() == H.to_standard_form()

    def test_generate_symmetric_hamiltonian(self):
        n_qudits = 5
        n_paulis = 12
        dimension = 2
        all_qudit_indices = tuple(range(n_qudits))
        n_tests = 100
        for _ in range(n_tests):
            C1 = Circuit.from_random(10, [dimension] * n_qudits)
            C1_gate = C1.composite_gate()

            # sym = SWAP(0, 1, p)
            # unscrambled H
            H = random_gate_symmetric_hamiltonian(C1_gate, dimension, all_qudit_indices,
                                                  n_qudits, n_paulis, scrambled=False)
            # print(H.tableau)
            # H = H.to_standard_form()
            # print(H.to_standard_form().tableau)
            check = C1_gate.act(H, all_qudit_indices)
            assert H.is_close(check, literal=False), "Hamiltonian not symmetric. \n H: \n" + \
                H.to_standard_form().tableau + "\n sym: \n" + \
                check.to_standard_form().tableau

            C2 = Circuit.from_random(100, H.dimensions)
            C2_gate = C2.composite_gate()  # scrambling circuit
            H = C2_gate.act(H, all_qudit_indices)
            H.weight_to_phase()
            H.weights = np.round(H.weights, 2)
            scrambled_C = Circuit.from_gates_and_qudits(H.dimensions,
                                                        [C2_gate.inverse(), C1_gate, C2_gate],
                                                        [all_qudit_indices, all_qudit_indices, all_qudit_indices])
            scrambled_C_gate = scrambled_C.composite_gate()
            assert H.is_close(scrambled_C_gate.act(H, all_qudit_indices),
                              literal=False), "Scrambled Hamiltonian not symmetric."

    def test_random_arbitrary_symmetry(self):
        n_tests = 20
        p = 2
        n_qudits = 15
        n_paulis = 40

        for _ in range(n_tests):
            sym = Circuit.from_random(10, [p] * n_qudits)  #
            sym = sym.composite_gate()
            # unscrambled H
            H = random_gate_symmetric_hamiltonian(sym, n_qudits, n_paulis, scrambled=False)
            C = Circuit.from_random(100, H.dimensions).composite_gate()  # scrambling circuit
            H = C.act(H)
            H.weight_to_phase()
            H.weights = np.round(H.weights, 2)
            scrambled_sym = Circuit(H.dimensions, [C.inv(), sym, C]).composite_gate()
            # , f"\n{H.to_standard_form().__str__()}\n{sym.act(H).to_standard_form().__str__()}"
            assert H.is_close(scrambled_sym.act(H), literal=False), "Scrambled Hamiltonian not symmetric."

            known_F = scrambled_sym.symplectic
            if np.array_equal(known_F, np.eye(known_F.shape[0], dtype=known_F.dtype)) or H.n_paulis() <= 2 * n_qudits:
                # Trivial symmetry, or incomplete basis, skipping test
                continue
            else:
                circ = find_clifford_symmetries(H)

                assert len(
                    circ) != 0, (f"No symmetries found for run {_} \n F:{known_F}"
                                 " \n H:\n{H.tableau}\nHF:\n{scrambled_sym.act(H).tableau}")

                for c in circ:

                    H_s = H.to_standard_form()
                    H_out = c.act(H).to_standard_form()
                    H_s.weight_to_phase()
                    H_out.weight_to_phase()
                    assert np.all(H_s.tableau == H_out.tableau)
                    assert np.all(H_s.phases == H_out.phases)
                    assert np.all(H_s.weights == H_out.weights)

                    assert c.act(H).to_standard_form() == H.to_standard_form()
