from __future__ import annotations
from typing import Optional, Callable
import numpy as np
from algebra import gauss_solve_mod_prime

# You can inject your transvection-based mapper here (p=2)
# signature: map_sum_to_target(src: np.ndarray, tgt: np.ndarray) -> np.ndarray (F)
TransvectionBuilder = Callable[[np.ndarray, np.ndarray], np.ndarray]

def P_phase(n_qud: int, p: int) -> np.ndarray:
    P_upper = np.hstack([np.zeros((n_qud, n_qud), dtype=int), np.eye(n_qud, dtype=int)])
    P_lower = np.hstack([(-np.eye(n_qud, dtype=int)) % (2*p), np.zeros((n_qud, n_qud), dtype=int)])
    return (np.vstack([P_upper, P_lower]) % (2*p)).astype(int)

def left_inverse_tableau(tableau: np.ndarray, p: int) -> Optional[np.ndarray]:
    """
    Return L_left (2n×N) with L_left @ tableau = I_{2n} (mod p), else None.
    """
    N, two_n = tableau.shape
    A = (tableau.T % p).astype(int)  # 2n × N
    L = np.zeros((two_n, N), dtype=int)
    for j in range(two_n):
        e = np.zeros(two_n, dtype=int); e[j] = 1
        x = gauss_solve_mod_prime(A, e, p)
        if x is None:
            return None
        L[j, :] = x % p
    return L % p

def is_symplectic(F: np.ndarray, p: int) -> bool:
    two_n = F.shape[0]
    n = two_n // 2
    Pp = P_phase(n, p) % p
    lhs = (F.T % p) @ Pp @ (F % p) % p
    return np.array_equal(lhs % p, Pp % p)

def build_F_right_rowperm(
    tableau: np.ndarray,
    pi: np.ndarray,
    p: int,
    *,
    transvection_builder: Optional[TransvectionBuilder] = None,
    L_left: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    N, two_n = tableau.shape
    Pi = np.eye(N, dtype=int)[pi, :]

    # NEW: trivial identity-permutation fast-path (no rank assumptions)
    if np.array_equal(pi, np.arange(N, dtype=pi.dtype)):
        F = np.eye(two_n, dtype=int) % p
        # sanity (these always hold for identity)
        if not is_symplectic(F, p):
            return None
        if not np.array_equal((tableau % p) @ F % p, (Pi @ (tableau % p)) % p):
            return None
        return F

    if p == 2 and transvection_builder is not None:
        A_src = (tableau % 2).astype(int)
        A_tgt = (Pi @ A_src) % 2
        F = transvection_builder(A_src, A_tgt) % 2
        if not np.array_equal((tableau @ F) % 2, (Pi @ tableau) % 2):
            return None
        if not is_symplectic(F, 2):
            return None
        return F

    # fallback: left-inverse (needs full row rank)
    if L_left is None:
        return None
    F = (L_left @ (Pi @ (tableau % p))) % p
    if not np.array_equal((tableau % p) @ F % p, (Pi @ (tableau % p)) % p):
        return None
    if not is_symplectic(F, p):
        return None
    return F % p

def check_S_invariance(S_mod: np.ndarray, pi: np.ndarray) -> bool:
    return np.array_equal(S_mod[np.ix_(pi, pi)], S_mod)


if __name__ == "__main__":
    # Case 1: identity permutation always returns identity F (works even if N < 2n).
    T = np.array([[1,0, 0,1],
                  [0,1, 1,0]], dtype=int)  # N=2, 2n=4 (just a sanity case)
    pi_id = np.array([0,1], dtype=int)
    F_id = build_F_right_rowperm(T, pi_id, 2, transvection_builder=None, L_left=None)
    assert F_id is not None and np.array_equal(F_id % 2, np.eye(4, dtype=int) % 2)
    assert is_symplectic(F_id, 2)

    # Case 2: non-identity permutation that IS symplectic.
    # Build a tableau listing the canonical unit rows for n=2 qubits:
    # columns are [x1, x2 | z1, z2]; rows are X1, Z1, X2, Z2
    T2 = np.array([
        [1,0, 0,0],  # X1
        [0,0, 1,0],  # Z1
        [0,1, 0,0],  # X2
        [0,0, 0,1],  # Z2
    ], dtype=int)  # N=4, 2n=4, full column rank

    # Hadamard on qubit 1 swaps X1 <-> Z1, so the row permutation is (0 1)
    pi_h = np.array([1,0,2,3], dtype=int)

    # Left-inverse exists here (N==2n and rows are independent)
    L2 = left_inverse_tableau(T2, 2)
    assert L2 is not None

    F_h = build_F_right_rowperm(T2, pi_h, 2, transvection_builder=None, L_left=L2)
    assert F_h is not None, "Failed to build F for a symplectic permutation"
    assert is_symplectic(F_h, 2), "F is not symplectic"

    # Verify tableau @ F == Pi @ tableau (mod 2)
    Pi_h = np.eye(T2.shape[0], dtype=int)[pi_h, :]
    assert np.array_equal((T2 @ F_h) % 2, (Pi_h @ T2) % 2)

    print("[symplectic] ok")
