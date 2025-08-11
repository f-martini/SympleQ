from quaos.core.circuits.find_symplectic import (
    map_single_pauli_string_to_target,
    check_mappable_via_clifford,
    find_symplectic_solution,
    find_symplectic_solution_extended,
    symplectic_product,
    transvection_matrix,
    transvection,
    solve_gf2,
    map_pauli_sum_to_target
)
import numpy as np


class TestSymplecticSolver:

    def verify_solution_extended(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                 t_vectors: list | None = None) -> bool:
        """
        Verify that w satisfies all the extended symplectic conditions.

        Args:
            u, v, w: Binary vectors
            t_vectors: List of additional constraint vectors

        Returns:
            True if all conditions are satisfied
        """
        if t_vectors is None:
            t_vectors = []

        # Check primary conditions
        if not (symplectic_product(u, w) == 1 and symplectic_product(v, w) == 1):
            return False

        # Check additional conditions
        for t in t_vectors:
            if symplectic_product(t, w) != symplectic_product(t, v):
                return False

        return True

    def verify_solution(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> bool:
        """
        Verify that w satisfies the symplectic conditions.

        Args:
            u, v, w: Binary vectors

        Returns:
            True if <u,w> = <v,w> = 1
        """
        return (symplectic_product(u, w) == 1 and symplectic_product(v, w) == 1)

    def assert_transvection_property(self, x, h):
        """
        For x, h in GF(2)^2n

        Z_h(x) = x if <x, h> = 0, and h if <x, h> = 1
        """
        if symplectic_product(x, h) == 0:
            assert np.all(transvection(h, x) == x)
            assert np.all((x.T @ transvection_matrix(h)) % 2 == x), f"\n{x}\n{h}\n{(x @ transvection_matrix(h)) % 2}"
        elif symplectic_product(x, h) == 1:
            assert np.all(transvection(h, x) == (h + x) % 2)
            assert np.all((x.T @ transvection_matrix(h)) % 2 == (h + x) % 2), f"\n{(x + h) % 2}\n{(x @ transvection_matrix(h)) % 2}"
        else:
            raise Exception('This test only works in GF(2)')

    def test_transvection_property(self, h):
        n = len(h) // 2
        for _ in range(100):
            x = np.random.randint(2, size=2 * n)
            self.assert_transvection_property(x, h)

    def test_transvection_and_transvection_matrix(self, h):
        n = len(h) // 2
        for _ in range(100):
            x = np.random.randint(2, size=2 * n)

            assert np.all(transvection(h, x) == (x @ transvection_matrix(h)) % 2)

    def test_map_single_pauli_string_to_target(self):
        n = 14
        p = 2
        for _ in range(1000):
            input_ps = np.random.randint(p, size=2 * n)
            target_ps = np.random.randint(p, size=2 * n)
            if np.array_equal(input_ps, np.zeros(2 * n)):
                input_ps[np.random.randint(2 * n)] = 1  # Ensure non-zero input
            if np.array_equal(target_ps, np.zeros(2 * n)):
                target_ps[np.random.randint(2 * n)] = 1  # Ensure non-zero target

            F_map = map_single_pauli_string_to_target(input_ps, target_ps)
            assert np.all((input_ps @ F_map) % p == target_ps), f"\n{F_map}\n{input_ps}\n{(input_ps @ F_map) % p}\n{target_ps}"

    def test_find_w(self):
        """Test function that properly handles cases where no solution exists."""
        n = 10
        p = 2

        for i in range(1000):
            u = np.random.randint(p, size=2 * n)
            v = np.random.randint(p, size=2 * n)

            w = find_symplectic_solution(u, v)

            if w is not None:
                # If a solution is found, it must be valid
                assert self.verify_solution(u, v, w), (f"Solution verification failed for u={u},'\
                                                ' v={v}, w={w}: <u,w>={symplectic_product(u, w)},'\
                                                ' <v,w>={symplectic_product(v, w)}")
            else:
                # If no solution found, verify this is correct by checking if either vector is zero
                # or by attempting to solve and confirming inconsistency
                is_zero_case = np.array_equal(u, np.zeros(2 * n)) or np.array_equal(v, np.zeros(2 * n))
                if is_zero_case:
                    # Zero vector case - correctly identified as impossible
                    continue
                else:
                    # Non-zero case - verify the system is actually inconsistent
                    # by checking that the linear system has no solution
                    A = np.zeros((2, 2 * n), dtype=int)
                    A[0, :n] = u[n:]
                    A[0, n:] = u[:n]
                    A[1, :n] = v[n:]
                    A[1, n:] = v[:n]
                    b = np.array([1, 1])

                    # The solve_gf2 function should return None for inconsistent systems
                    result = solve_gf2(A, b)
                    assert result is None, f"System should be inconsistent but solver found solution for u={u}, v={v}"

    def test_extended_constraints(self):

        """Test the extended constraint functionality."""
        n = 2  # 2 qubits for simpler testing

        # Test case 1: Simple extended constraint
        u = np.array([1, 0, 0, 0])  # X_1
        v = np.array([0, 1, 0, 0])  # X_2
        t1 = np.array([0, 0, 1, 0])  # Z_1

        # Find w with additional constraint <t1, w> = <t1, v>
        w = find_symplectic_solution_extended(u, v, [t1])
        if w is not None:
            assert self.verify_solution_extended(u, v, w, [t1]), "Extended solution should be valid"

        # Test case 2: Multiple additional constraints
        u = np.array([1, 0, 0, 0])  # X_1
        v = np.array([0, 1, 0, 0])  # X_2
        t1 = np.array([0, 0, 1, 0])  # Z_1
        t2 = np.array([0, 0, 0, 1])  # Z_2

        w = find_symplectic_solution_extended(u, v, [t1, t2])
        if w is not None:
            assert self.verify_solution_extended(u, v, w, [t1, t2]), "Multi-constraint solution should be valid"
        # # Test case 3: Overconstrained system (should potentially fail)
        # n = 1  # 1 qubit system has only 2 degrees of freedom
        # u_small = np.array([1, 0])  # X_1
        # v_small = np.array([0, 1])  # Z_1
        # t1_small = np.array([1, 1])  # X_1 + Z_1 (Y_1)

        # # This system has 3 constraints on 2 variables - likely overconstrained
        # w_small = find_symplectic_solution_extended(u_small, v_small, [t1_small])
        # Don't assert anything here - this may or may not have a solution
        # print(w_small)
        # Test case 4: Random extended constraints
        n = 5
        for _ in range(200):
            u = np.random.randint(2, size=2 * n)
            v = np.random.randint(2, size=2 * n)

            # Skip zero vectors
            if np.array_equal(u, np.zeros(2 * n)) or np.array_equal(v, np.zeros(2 * n)):
                continue

            # Add 1-2 random additional constraints
            num_constraints = np.random.randint(1, 3)
            t_vectors = [np.random.randint(2, size=2 * n) for _ in range(num_constraints)]

            w = find_symplectic_solution_extended(u, v, t_vectors)
            if w is not None:
                assert self.verify_solution_extended(u, v, w, t_vectors), "Random extended solution failed verification"

    def test_map_pauli_sum_to_target(self):
        n = 4  # Number of qubits
        p = 2  # Field size
        m = 3  # Number of Pauli strings

        for _ in range(30):
            # Generate random Pauli sums
            pauli_sum = np.random.randint(p, size=(m, 2 * n))
            target_pauli_sum = np.random.randint(p, size=(m, 2 * n))

            if np.any([np.array_equal(pauli_sum[i], np.zeros(2 * n)) or np.array_equal(target_pauli_sum[i], np.zeros(2 * n)) for i in range(m)]) is False:
                continue  # Skip zero vectors

            if not check_mappable_via_clifford(pauli_sum, target_pauli_sum):
                continue  # Skip if not mappable

            F = map_pauli_sum_to_target(pauli_sum, target_pauli_sum)

            # Verify the mapping
            mapped_pauli_sum = (pauli_sum @ F) % p
            print("~~~~~~~~~~~~")
            print(np.array_equal(mapped_pauli_sum, target_pauli_sum))
            assert np.array_equal(mapped_pauli_sum, target_pauli_sum), f"Mapping failed:\n{mapped_pauli_sum}\n{target_pauli_sum}"
