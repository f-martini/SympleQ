import numpy as np
from quaos.core.paulis import PauliSum, PauliString, Pauli, Xnd, Ynd, Znd, Id


class TestSymplectic:

    def test_symplectic_matrix_single_pauli(self):
        pauli_list = ['x1z0']
        weights = np.array([1.])
        sp = PauliSum(pauli_list, weights, dimensions=[2])
        expected_matrix = np.array([[1, 0]])
        np.testing.assert_array_equal(sp.symplectic(), expected_matrix)

    def test_symplectic_matrix_multiple_paulis(self):
        pauli_list = ['x1z0', 'x0z1', 'x1z1']
        weights = np.array([1, 2, 3])
        sp = PauliSum(pauli_list, weights, dimensions=[2], standardise=False)
        expected_matrix = np.array([
            [1, 0],
            [0, 1],
            [1, 1]
        ])

        np.testing.assert_array_equal(sp.symplectic(), expected_matrix)

    def test_basic_pauli_relations(self):
        dims = 3
        x1 = Xnd(1, dims)
        y1 = Ynd(1, dims)
        z1 = Znd(1, dims)
        id = Id(dims)

        assert x1 * z1 == y1
        assert x1 * x1 * x1 == id

    def test_pauli_equality(self):
        dims = 3
        p1 = Pauli.from_string('x1z0', dimension=dims)
        p2 = Pauli.from_string('x0z1', dimension=dims)
        p3 = Pauli.from_string('x1z1', dimension=dims)

        assert Xnd(1, dims) == p1
        assert Znd(1, dims) == p2
        assert Ynd(1, dims) == p3

    def test_pauli_multiplication(self):
        dims = 3
        p1 = Pauli.from_string('x1z0', dimension=dims)
        p2 = Pauli.from_string('x0z1', dimension=dims)
        p3 = Pauli.from_string('x1z1', dimension=dims)

        assert isinstance(p1 * p2, Pauli)
        assert p1 * p1 == Pauli.from_string('x2z0', dimension=dims)
        assert p1 * p2 * p3 == Pauli.from_string('x2z2', dimension=dims)

    def test_pauli_string_construction(self):
        dims = [3, 3]
        x1x1 = PauliString.from_string('x1z0 x1z0', dimensions=dims)
        x1x1_2 = PauliString([1, 1], [0, 0], dims)

        assert x1x1 == x1x1_2

        x1y1 = PauliString([1, 1], [0, 1], dimensions=dims)
        x1y1_2 = PauliString.from_string('x1z0 x1z1', dimensions=dims)

        assert x1y1 == x1y1_2

    def test_pauli_sum_addition(self):
        dims = [3, 3]
        x1x1 = PauliSum(PauliString.from_string('x1z0 x1z0', dimensions=dims))
        x1y1 = PauliSum(PauliString.from_string('x1z0 x1z1', dimensions=dims))

        psum = x1x1 + x1y1
        expected = PauliSum([PauliString.from_string('x1z0 x1z0', dimensions=dims),
                             PauliString.from_string('x1z0 x1z1', dimensions=dims)], weights=[1, 1], phases=[0, 0])

        assert psum == expected

    def test_phase_and_dot_product(self):
        d = 7
        x = PauliString.from_string('x1z0', dimensions=[d])
        z = PauliString.from_string('x0z1', dimensions=[d])

        assert x.acquired_phase(z) == 1.0

        dims = [3, 3]
        x1x1 = PauliSum(PauliString.from_string('x1z0 x1z0', dimensions=dims))
        x1y1 = PauliSum(PauliString.from_string('x1z0 x1z1', dimensions=dims))

        s1 = x1x1 + x1y1 * 0.5
        s2 = x1x1 + x1x1

        s3 = PauliSum(['x2z0 x2z0', 'x2z0 x2z0', 'x2z0 x2z1', 'x2z0 x2z1'],
                      weights=[1, 1, 0.5, 0.5],
                      phases=[0, 0, 1, 1],
                      dimensions=dims, standardise=False)

        assert s1 * s2 == s3

    def test_tensor_product_distributivity(self):
        dims = [3, 3]
        x1x1 = PauliSum(PauliString.from_string('x1z0 x1z0', dimensions=dims))
        x1y1 = PauliSum(PauliString.from_string('x1z0 x1z1', dimensions=dims))

        s1 = x1x1 + x1y1 * 0.5
        s2 = x1x1 + x1x1

        left = (s1 + s2) @ s2
        right = s1 @ s2 + s2 @ s2

        assert left == right

    def test_pauli_sum_indexing(self):
        dims = [3, 3, 3]
        ps = PauliSum(['x2z0 x2z0 x1z1', 'x2z0 x2z0 x0z0', 'x2z0 x2z1 x2z0', 'x2z0 x2z1 x1z1'],
                      weights=[1, 1, 0.5, 0.5],
                      phases=[0, 0, 1, 1],
                      dimensions=dims, standardise=False)

        assert ps[0] == PauliString.from_string('x2z0 x2z0 x1z1', dimensions=dims)
        assert ps[1] == PauliString.from_string('x2z0 x2z0 x0z0', dimensions=dims)
        assert ps[2] == PauliString.from_string('x2z0 x2z1 x2z0', dimensions=dims)
        assert ps[3] == PauliString.from_string('x2z0 x2z1 x1z1', dimensions=dims)
        assert ps[0:2] == PauliSum(['x2z0 x2z0 x1z1', 'x2z0 x2z0 x0z0'], dimensions=dims, standardise=False)
        assert ps[[0, 3]] == PauliSum(['x2z0 x2z0 x1z1', 'x2z0 x2z1 x1z1'], weights=[1, 0.5], phases=[0, 1], dimensions=dims,
                                      standardise=False)
        assert ps[[0, 2], 1] == PauliSum(['x2z0', 'x2z1'], weights=[1, 0.5], phases=[0, 1], dimensions=[3],
                                         standardise=False)
        assert ps[[0, 2], [0, 2]] == PauliSum(['x2z0 x1z1', 'x2z0 x2z0'], weights=[1, 0.5], phases=[0, 1], dimensions=[3, 3],
                                              standardise=False)
