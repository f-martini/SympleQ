"""
Tests of Pauli operations such as multiplication, addition and tensor product
"""
import sys
sys.path.append("./")
from quaos.circuits import Gate, SUM, SWAP, Hadamard, PHASE, CNOT
from quaos.circuits.utils import is_symplectic
from quaos.paulis import PauliSum, PauliString, Pauli
import numpy as np


class TestOperations():

    def random_pauli_string(self, dim):
        r1 = np.random.randint(0, dim)
        r2 = np.random.randint(0, dim)
        s1 = np.random.randint(0, dim)
        s2 = np.random.randint(0, dim)
        return PauliString.from_string(f"x{r1}z{s1} x{r2}z{s2}", dimensions=[dim, dim]), r1, r2, s1, s2

    def test_pauli_multiplication(self):
        for dim in [2]:
            x1 = Pauli(1, 0, dim)
            y1 = Pauli(1, 1, dim)
            z1 = Pauli(0, 1, dim)
            id = Pauli(0, 0, dim)

            assert x1 * z1 == y1, 'Error in Pauli multiplication (x * z = y) ' + (x1 * z1).__str__()
            assert x1**dim == id, 'Error in Pauli exponentiation (x**dim = id) ' + (x1**dim).__str__()
            assert y1**dim == id, 'Error in Pauli exponentiation (y**dim = id) ' + (y1**dim).__str__()
            assert z1**dim == id, 'Error in Pauli exponentiation (z**dim = id)  ' + (z1**dim).__str__()
            assert x1 * y1 == z1, 'Error in Pauli multiplication (x * y = z) ' + (x1 * y1).__str__()
            assert y1 * z1 == x1, 'Error in Pauli multiplication (y * z = x) ' + (y1 * z1).__str__()
            assert z1 * x1 == y1, 'Error in Pauli multiplication (z * x = y) ' + (z1 * x1).__str__()
            assert x1 * id == x1, 'Error in Pauli multiplication (x * id = x) ' + (x1 * id).__str__()
            assert y1 * id == y1, 'Error in Pauli multiplication (y * id = y) ' + (y1 * id).__str__()
            assert z1 * id == z1, 'Error in Pauli multiplication (z * id = z) ' + (z1 * id).__str__()

        for dim in [3, 5, 11]:
            for i in range(100):
                s = np.random.randint(0, dim)
                r = np.random.randint(0, dim)
                s2 = np.random.randint(0, dim)
                r2 = np.random.randint(0, dim)
                p1 = Pauli(r, s, dim)
                p2 = Pauli(r2, s2, dim)
                p3 = p1 * p2
                assert p3.x_exp == (p1.x_exp + p2.x_exp) % dim, 'Error in Pauli multiplication (x_exp)'
                assert p3.z_exp == (p1.z_exp + p2.z_exp) % dim, 'Error in Pauli multiplication (z_exp)'
                assert p3.dimension == dim, 'Error in Pauli multiplication (dimension)'

    def test_pauli_string_multiplication(self):
        for dim in [2, 3, 5, 11]:
            for i in range(100):
                r1 = np.random.randint(0, dim)
                r2 = np.random.randint(0, dim)
                s1 = np.random.randint(0, dim)
                s2 = np.random.randint(0, dim)

                r12 = np.random.randint(0, dim)
                r22 = np.random.randint(0, dim)
                s12 = np.random.randint(0, dim)
                s22 = np.random.randint(0, dim)

                input_str1 = f"x{r1}z{s1} x{r2}z{s2}"
                input_str2 = f"x{r12}z{s12} x{r22}z{s22}"
                output_str_correct = f"x{(r1 + r12) % dim}z{(s1 + s12) % dim} x{(r2 + r22) % dim}z{(s2 + s22) % dim}"
                input_ps1 = PauliString.from_string(input_str1, dimensions=[dim, dim])
                input_ps2 = PauliString.from_string(input_str2, dimensions=[dim, dim])
                output_ps = input_ps1 * input_ps2
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[dim, dim]), 'Error in PauliString multiplication'

    def test_pauli_string_tensor_product(self):
        for dim in [2, 3, 5, 11]:
            for i in range(100):
                r1 = np.random.randint(0, dim)
                r2 = np.random.randint(0, dim)
                s1 = np.random.randint(0, dim)
                s2 = np.random.randint(0, dim)

                r12 = np.random.randint(0, dim)
                r22 = np.random.randint(0, dim)
                s12 = np.random.randint(0, dim)
                s22 = np.random.randint(0, dim)

                input_str1 = f"x{r1}z{s1} x{r2}z{s2}"
                input_str2 = f"x{r12}z{s12} x{r22}z{s22}"
                output_str_correct = f"x{r1}z{s1} x{r2}z{s2} x{r12}z{s12} x{r22}z{s22}"
                input_ps1 = PauliString.from_string(input_str1, dimensions=[dim, dim])
                input_ps2 = PauliString.from_string(input_str2, dimensions=[dim, dim])
                output_ps = input_ps1 @ input_ps2
                assert output_ps == PauliString.from_string(output_str_correct, dimensions=[dim] * 4), 'Error in PauliString tensor product'

    def test_pauli_string_indexing(self):
        for dim in [2, 3, 5]:
            for i in range(100):
                p_string1, r1, r2, s1, s2 = self.random_pauli_string(dim)
                ps1 = Pauli.from_string(f"x{r1}z{s1}", dimension=dim)
                ps2 = Pauli.from_string(f"x{r2}z{s2}", dimension=dim)

                assert p_string1[0] == ps1, 'Error in PauliString indexing (first PauliString)'
                assert p_string1[1] == ps2, 'Error in PauliString indexing'

    def test_pauli_sum_multiplication(self):
        for dim in [2, 3, 5]:

            for i in range(100):
                p_string1, r1, r2, s1, s2 = self.random_pauli_string(dim)
                p_string2, r12, r22, s12, s22 = self.random_pauli_string(dim)

                random_pauli_sum = PauliSum([p_string1, p_string2], standardise=False)

                # Test multiplication of PauliSum with PauliString
                input_ps1, r13, r23, s13, s23 = self.random_pauli_string(dim)

                output_str_correct = f"x{(r1 + r13) % dim}z{(s1 + s13) % dim} x{(r2 + r23) % dim}z{(s2 + s23) % dim}"
                output_str2_correct = f"x{(r12 + r13) % dim}z{(s12 + s13) % dim} x{(r22 + r23) % dim}z{(s22 + s23) % dim}"
                output_ps = random_pauli_sum * input_ps1

                phase1 = (r1 * s13 + r13 * s1 + r2 * s23 + r23 * s2) % dim
                phase2 = (r12 * s13 + r13 * s12 + r22 * s23 + r23 * s22) % dim
                output_phases = [phase1, phase2]
                output_ps_correct = PauliSum([PauliString.from_string(output_str_correct, dimensions=[dim, dim]),
                                              PauliString.from_string(output_str2_correct, dimensions=[dim, dim])],
                                             phases=output_phases,
                                             standardise=False)

                assert output_ps == output_ps_correct, 'Error in PauliSum multiplication with PauliString'

                # Test multiplication of PauliSum with PauliSum
                random_pauli_sum2 = PauliSum([input_ps1], standardise=False)
                output_ps2 = random_pauli_sum * random_pauli_sum2
                print(f"Output PauliSum: \n {output_ps2}")
                print(f"Expected PauliSum: \n {output_ps_correct}")
                assert output_ps2 == output_ps_correct, 'Error in PauliSum multiplication with PauliSum'

    def test_pauli_sum_addition(self):

        for dim in [2, 3, 5]:

            for i in range(100):
                p_string1, r1, r2, s1, s2 = self.random_pauli_string(dim)
                p_string2, r12, r22, s12, s22 = self.random_pauli_string(dim)
                p_string3, r13, r23, s13, s23 = self.random_pauli_string(dim)

                random_pauli_sum = PauliSum([p_string1, p_string2], standardise=False)
                random_pauli_sum2 = PauliSum([p_string3], standardise=False)
                ps_out = random_pauli_sum + random_pauli_sum2
                ps_out_correct = PauliSum([p_string1, p_string2, p_string3], standardise=False)
                assert ps_out == ps_out_correct, 'Error in PauliSum addition'

    def test_pauli_sum_tensor_product(self):

        for dim in [2, 3, 5]:
            for i in range(100):
                p_string1, r1, r2, s1, s2 = self.random_pauli_string(dim)
                p_string2, r12, r22, s12, s22 = self.random_pauli_string(dim)
                p_string3, r13, r23, s13, s23 = self.random_pauli_string(dim)

                random_pauli_sum = PauliSum([p_string1, p_string2], standardise=False)
                random_pauli_sum2 = PauliSum([p_string3], standardise=False)
                ps_out = random_pauli_sum @ random_pauli_sum2
                ps_out_correct = PauliSum([p_string1 @ p_string3, p_string2 @ p_string3], standardise=False)
                assert ps_out == ps_out_correct, 'Error in PauliSum tensor product'

    def test_pauli_sum_indexing(self):

        for dim in [2, 3, 5]:
            for i in range(100):
                p_string1, r1, r2, s1, s2 = self.random_pauli_string(dim)
                p_string2, r12, r22, s12, s22 = self.random_pauli_string(dim)

                random_pauli_sum = PauliSum([p_string1, p_string2], standardise=False)
                assert random_pauli_sum[0] == p_string1, 'Error in PauliSum indexing (first PauliString)'
                assert random_pauli_sum[1] == p_string2, 'Error in PauliSum indexing (second PauliString)'
                assert random_pauli_sum[0, 0] == p_string1[0], 'Error in PauliSum indexing (first PauliString, first Pauli)'
                assert random_pauli_sum[0, 1] == p_string1[1], 'Error in PauliSum indexing (first PauliString, second Pauli)'


