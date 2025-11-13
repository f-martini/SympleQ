from sympleq.core.circuits import Gate, Circuit, CNOT, Hadamard, PHASE, SUM, SWAP
from sympleq.core.circuits.decomposition_GFp import decompose_symplectic_to_circuit, ensure_invertible_A
import numpy as np
from sympleq.core.circuits.utils import is_symplectic


class TestDecomposition:
    def test_decompose_symplectic_to_circuit(self):
        n_qidits_list = [4, 6, 8]
        dimension_list = [3, 5]

        for n_qidits in n_qidits_list:
            for dimension in dimension_list:
                test_gate = Gate.from_random(n_qidits, dimension)
                test_circuit = decompose_symplectic_to_circuit(test_gate.symplectic, dimension)

                gate_out = test_circuit.composite_gate()

                # correction_gates, F_new = ensure_invertible_A(gate_out.symplectic, dimension)

                assert is_symplectic(gate_out.symplectic, dimension), "Symplectic decomposition failed."
                print(test_gate.symplectic)
                print(gate_out.symplectic)
                assert np.array_equal(gate_out.symplectic, test_gate.symplectic), ("Symplectic decomposition failed "
                                                                                   f"for {n_qidits} qubits and "
                                                                                   f"{dimension} dimension.")

    # def test_decomposition_with_phase(self):
    #     pass

    # def test_A_invertible(self):
    #     n_qidits_list = [4, 6, 8]
    #     dimension_list = [2, 3, 5]

    #     for n_qidits in n_qidits_list:
    #         for dimension in dimension_list:
    #             test_gate = Gate.from_random(n_qidits, dimension)
    #             test_circuit = decompose_symplectic_to_circuit(test_gate.symplectic, dimension)

    #             gate_out = test_circuit.composite_gate()

    #             correction_gates, F_new = ensure_invertible_A(gate_out.symplectic, dimension)
