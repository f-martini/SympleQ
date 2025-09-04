import numpy as np
import galois
from quaos.core.circuits.utils import symplectic_product
from quaos.core.circuits.find_symplectic_qudits import *

class TestSymplecticSolverQudits:

    def verify_intermediate_ps(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, p) -> bool:
        """
        Verify that w satisfies the symplectic conditions.

        Args:
            u: Input PauliSum
            v: Output PauliSum
            w: Intermediate PauliSum
            p: dimension; prime

        Returns:
            True if <u,w> != 0 and  <w,v> != 0
        """
        if not galois.is_prime(p):
            raise NotImplementedError(f"Prime dimension expected, got p={p}")

        return (symplectic_product(u, w, p) != 0 and symplectic_product(w, v, p) != 0)

    def test_find_w(self):
        """Test function that checks for intermediate solution."""
        n = 10
        p=19
        if not galois.is_prime(p):
            raise NotImplementedError(f"Prime dimension expected, got p={p}")

        for _ in range(3000):
            u = np.random.randint(p, size=2 * n)
            v = np.random.randint(p, size=2 * n)
            is_zero_case = np.array_equal(u, np.zeros(2 * n)) or np.array_equal(v, np.zeros(2 * n))
            if is_zero_case:
                # Zero vector case - correctly identified as impossible
                continue

            elif symplectic_product(u, v, p) !=0:
                continue

            else:
                w = build_symplectic_for_transvection(u, v, p)

                if w is not None:
                    # If a solution is found, it must be valid
                    assert self.verify_intermediate_ps(u, v, w, p), (f"Solution verification failed for u={u},'\
                                                    ' v={v}, w={w}: <u,w>={symplectic_product(u, w, p)},'\
                                                    ' <v,w>={symplectic_product(w,v,p)}")

    def test_transvection_matrix(self):
        n=10
        p=13
        u = np.random.randint(p, size=2 * n)
        v= np.random.randint(p, size=2 * n)

        if not galois.is_prime(p):
            raise NotImplementedError(f"Prime dimension expected, got p={p}")

        for _ in range(10000):
            is_zero_case = np.array_equal(u, np.zeros(2 * n)) or np.array_equal(v, np.zeros(2 * n))

            if is_zero_case:
                # Zero vector case - correctly identified as impossible
                continue

            else:
                F_h= Find_transvection_map(u, v, p)

                assert (u @ F_h %p == v).all(), (f"Mapping failed for u={u},'\
                                                    ' v={v}, w={w}: ")
