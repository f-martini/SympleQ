from symplectic_to_circuit import (inv_gfp, mod_p, ensure_invertible_A_circuit, blocks,
                                   zeros, identity, is_symplectic_gfp, _is_invertible_mod,
                                   synth_linear_A_to_gates, _emit_local_ops_for_D, synth_lower_from_symmetric, 
                                   synth_upper_from_symmetric_via_H, _as_int_mod, decompose_symplectic_to_circuit,
                                   _compose_symp, _full_from_upper, _full_from_lower)
from sympleq.core.circuits import Circuit, SUM
import numpy as np


# ---------- simple random symplectic for testing ----------
def random_invertible(n: int, rng: np.random.Generator, p: int) -> np.ndarray:
    while True:
        A = rng.integers(0, p, size=(n, n), dtype=int)
        try:
            _ = inv_gfp(A, p)
            return mod_p(A, p)
        except ValueError:
            pass

def random_symmetric(n: int, rng: np.random.Generator, p: int) -> np.ndarray:
    M = rng.integers(0, p, size=(n, n), dtype=int)
    return mod_p(M + M.T, p)

def build_F_from_LMR(n: int, A: np.ndarray, Bsym: np.ndarray, Csym: np.ndarray, p: int) -> np.ndarray:
    """F = L M R with R=[I B;0 I], M=diag(A, (A^T)^{-1}), L=[I 0; C I]."""
    A = mod_p(A, p)
    Bsym = mod_p(Bsym, p)
    Csym = mod_p(Csym, p)
    A_invT = inv_gfp(A.T, p)
    Z = zeros((n, n)); I = identity(n)
    L = np.block([[I, Z], [Csym, I]])
    M = np.block([[A, Z], [Z, A_invT]])
    R = np.block([[I, Bsym], [Z, I]])
    return mod_p(L @ M @ R, p)

def random_symplectic(n: int, p: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    A = random_invertible(n, rng, p)
    B = random_symmetric(n, rng, p)
    C = random_symmetric(n, rng, p)
    F = build_F_from_LMR(n, A, B, C, p)
    assert is_symplectic_gfp(F, p)
    return F


# ---------- preconditioner tests----------

def test_preconditioner_noop_when_A_invertible(F, p):
    n = F.shape[0] // 2
    from copy import deepcopy
    F0 = deepcopy(F)
    C_pre = ensure_invertible_A_circuit(F, p)
    if np.linalg.det(F[:n,:n]) % p != 0:  # or inv_gfp try/except
        # already invertible: either empty or something that keeps A invertible
        assert len(C_pre.gates) == 0
        F_prime = (C_pre.full_symplectic() @ F) % p
        assert np.array_equal(F_prime, F0 % p)

def test_preconditioner_preserves_symplectic(F, p):
    n = F.shape[0] // 2
    C_pre = ensure_invertible_A_circuit(F, p)
    F_prime = (C_pre.full_symplectic() @ F) % p
    assert is_symplectic_gfp(F_prime, p)


def test_A_invertible_after_precondition(F, p):
    n = F.shape[0] // 2
    C_pre = ensure_invertible_A_circuit(F, p)
    F_prime = (C_pre.full_symplectic() @ F) % p
    A = F_prime[:n,:n]
    _ = inv_gfp(A, p)  # should not raise

def test_preconditioner_determinism(F, p):
    C1 = ensure_invertible_A_circuit(F, p)
    C2 = ensure_invertible_A_circuit(F, p)
    assert [g.name for g in C1.gates] == [g.name for g in C2.gates]
    assert [tuple(g.qudit_indices) for g in C1.gates] == [tuple(g.qudit_indices) for g in C2.gates]

def test_preconditioner_respects_depth(F, p):
    C_pre = ensure_invertible_A_circuit(F, p, max_depth=4)
    assert len(C_pre.gates) <= 4


def test_preconditioner_after_random_conjugation(n=4, p=3, depth=10, seed=1):
    rng = np.random.default_rng(seed)
    F = random_symplectic(n, p, rng)
    # random circuit (uses only valid gates for dimension p)
    C = Circuit.from_random(n_qudits=n, depth=depth, dimensions=[p]*n)
    F_conj = (C.full_symplectic() @ F) % p
    assert is_symplectic_gfp(F_conj, p), "Conjugated F must remain symplectic"
    C_pre = ensure_invertible_A_circuit(F_conj, p)
    F_prime = (C_pre.full_symplectic() @ F_conj) % p
    A = F_prime[:n, :n]
    A_inv_ok = False
    try:
        _ = inv_gfp(A, p)
        A_inv_ok = True
    except ValueError:
        A_inv_ok = False

    # --- assertions ---
    assert A_inv_ok, "A-block is not invertible after preconditioning"
    assert is_symplectic_gfp(F_prime, p), "Preconditioning broke symplecticity"

def test_preconditioner(num_trials: int = 10, n: int = 3, p: int = 3, seed: int = 2025) -> None:
    """
    Generate random symplectic F, build C_pre, and verify that
    A((C_pre)·F) is invertible (mod p). Also sanity-check symplecticity.
    """
    test_preconditioner_after_random_conjugation(n, p, depth=num_trials, seed=seed)
    rng = np.random.default_rng(seed)
    for t in range(num_trials):
        F = random_symplectic(n, p, rng)
        C_pre = ensure_invertible_A_circuit(F, p)
        F_prime = (C_pre.full_symplectic() @ F) % p  # left multiply for the matrix factorization step
        A, _, _, _ = blocks(F_prime, p)
        okA = _is_invertible_mod(A, p)
        okSp = is_symplectic_gfp(F_prime, p)
        if not okA or not okSp:
            raise AssertionError(
                f"[trial {t}] Preconditioner failed:\n"
                f"  A invertible? {okA}\n"
                f"  F' symplectic? {okSp}\n"
                f"F:\n{F}\nF':\n{F_prime}\nA(F'):\n{A}"
            )
        test_preconditioner_noop_when_A_invertible(F, p)
        test_A_invertible_after_precondition(F, p)
        test_preconditioner_preserves_symplectic(F, p)
        test_preconditioner_determinism(F, p)
        test_preconditioner_respects_depth(F, p)

    print(f"Preconditioner smoke test passed for n={n}, p={p}, trials={num_trials}.")


# M block tests
def symp_of_gates(n, p, gates):
    return Circuit([p]*n, list(gates)).composite_gate().symplectic % p


def check_linear_from_A(n, p, A):
    A = mod_p(A, p)
    D = mod_p(inv_gfp(A, p).T, p)
    Z = np.zeros((n,n), dtype=int)
    return mod_p(np.block([[A, Z], [Z, D]]), p)


# --- Test 1: Whole M block matches diag(A, (A^T)^{-1}) ---
def test_M_block_only(n, p, rng=np.random.default_rng(42)):
    A = random_invertible(n, rng, p)
    ops_M = synth_linear_A_to_gates(n, A, p)
    FM = symp_of_gates(n, p, ops_M)
    FM_exp = check_linear_from_A(n, p, A)
    assert np.array_equal(FM, FM_exp), f"\nA=\n{A}\nGot=\n{FM}\nExp=\n{FM_exp}"


# --- Test 2: Local D(u) synthesis on a single wire (via remap) ---
def test_local_Du_synthesis(p):
    for u in range(1, p):  # skip 0
        ops = _emit_local_ops_for_D(0, u, p)
        F = Circuit([p], ops).composite_gate().symplectic % p
        target = np.array([[u, 0],
                           [0, pow(int(u), -1, p)]], dtype=int) % p
        assert np.array_equal(F, target), f"Failed D({u}) over GF({p})"


# --- Test 3: SUM direction micro-test (pins the column-add mapping) ---
def test_SUM_direction_in_M(n: int, p: int, rng: np.random.Generator = np.random.default_rng(42)):
    # Try a few random (i, c, f) with i != c
    for _ in range(20):
        i = int(rng.integers(0, n))
        c = int(rng.integers(0, n))
        while i == c:
            c = int(rng.integers(0, n))
        f = int(rng.integers(1, p))   # 1..p-1

        # Build the A we expect AFTER applying: col_i <- col_i - f * col_c (RIGHT)
        A_exp = np.eye(n, dtype=int) % p
        A_exp[c, i] = (A_exp[c, i] + (-f) % p) % p  # NOTE: (c, i), not (i, c)

        # Synthesize M from this A and compare to the theoretical block diag
        ops = synth_linear_A_to_gates(n, A_exp, p)
        FM = symp_of_gates(n, p, ops)
        FM_exp = check_linear_from_A(n, p, A_exp)

        assert np.array_equal(FM, FM_exp), (
            f"Wrong SUM mapping for (i={i}, c={c}, f={f}) over GF({p})\n"
            f"A_exp=\n{A_exp}\nGot=\n{FM}\nExp=\n{FM_exp}"
        )

def test_local_Du_includes_identity_and_nontrivial(p):
    # identity case
    ops = _emit_local_ops_for_D(0, 1, p)
    M = Circuit([p], ops).composite_gate().symplectic % p
    assert np.array_equal(M, np.eye(2, dtype=int) % p)

    # all nontrivial u
    for u in range(2, p):
        ops = _emit_local_ops_for_D(0, u, p)
        M = Circuit([p], ops).composite_gate().symplectic % p
        invu = pow(u, -1, p)
        target = np.array([[u, 0], [0, invu]], dtype=int) % p
        assert np.array_equal(M, target), f"Failed D({u}) over GF({p})"

def test_single_wire_scaling_M(p: int):
    n = 3
    for u in range(2, p):  # skip 0 and 1
        A = np.eye(n, dtype=int)
        A[1,1] = u % p
        ops = synth_linear_A_to_gates(n, A, p)
        FM = symp_of_gates(n, p, ops)
        FM_exp = check_linear_from_A(n, p, A)
        assert np.array_equal(FM, FM_exp), f"Failed D({u}) on wire 1 over GF({p})"


def A_block_of(gate, p):
    F = Circuit([p, p], [gate]).composite_gate().symplectic % p
    return F[:2, :2]


def test_one_col_add_in_M(n=3, p=5, c=0, i=2, f=1):
    A = np.eye(n, dtype=int) % p
    A_exp = A.copy()
    A_exp[:, i] = (A_exp[:, i] + f * A_exp[:, c]) % p # col_i += f col_c

    # Build M ops for just that step:
    ops = [SUM(i, c, p)] * (f % p)
    FM  = Circuit([p]*n, ops).composite_gate().symplectic % p
    assert np.array_equal(FM[:n,:n], A_exp)


def test_m_block(num_trials, n, p, rng=np.random.default_rng(42)):
    test_single_wire_scaling_M(p)
    test_local_Du_includes_identity_and_nontrivial(p)
    test_local_Du_synthesis(p)
    # test_SUM_direction_and_sign(n, p)

    # Expect A = [[1,1],[0,1]] for SUM(0,1,p)
    A = A_block_of(SUM(0, 1, p), p)
    assert np.array_equal(A, np.array([[1,0],[1,1]])), A

    for _ in range(num_trials):
        test_SUM_direction_in_M(n, p, rng)
        test_M_block_only(n, p, rng)
        test_one_col_add_in_M(n, p)
        
    print(f"M block smoke test passed for n={n}, p={p}, trials={num_trials}.")
#  ############################################################################  #
#                                 L-R Block Tests                                        #
#  ############################################################################  #
def _rand_symm(n, p, rng):
    M = rng.integers(0, p, size=(n, n), dtype=int)
    return _as_int_mod(M + M.T, p)

def _check_equal(Fgot, Fexp, p, msg=""):
    G = _as_int_mod(Fgot, p)
    E = _as_int_mod(Fexp, p)
    if not np.array_equal(G, E):
        raise AssertionError(f"{msg}\nGot=\n{G}\nExp=\n{E}")

def test_L_block_unit(num_trials, n, ps, rng):
    """Test L(C) = [[I,0],[C,I]] synthesis correctness."""
    for _ in range(num_trials):
        Csym = _rand_symm(n, ps, rng)
        ops = synth_lower_from_symmetric(n, Csym, ps)
        F = _compose_symp(n, ps, ops)
        Fexp = _full_from_lower(n, Csym, ps)
        _check_equal(F, Fexp, ps, f"[L] mismatch over GF({ps})")

    print(f"L-block unit test passed for GF({ps})")


def test_R_block_unit(num_trials, n, ps, rng):
    """Test R(S) = [[I,S],[0,I]] synthesis correctness."""
    for _ in range(num_trials):
        Ssym = _rand_symm(n, ps, rng)
        ops = synth_upper_from_symmetric_via_H(n, Ssym, ps)
        F = _compose_symp(n, ps, ops)
        Fexp = _full_from_upper(n, Ssym, ps)
        _check_equal(F, Fexp, ps, f"[R] mismatch over GF({ps})")

    print(f"R-block unit test passed for GF({ps})")


def test_elementary_pairs(num_trials, n, ps, rng):
    """Test elementary diagonal/off-diagonal increments for both L and R."""
    for _ in range(num_trials):
        # Diagonal (L)
        for i in range(n):
            C = np.zeros((n, n), dtype=int)
            C[i, i] = rng.integers(0, ps)
            Fl = _compose_symp(n, ps, synth_lower_from_symmetric(n, C, ps))
            _check_equal(Fl, _full_from_lower(n, C, ps), p, f"[L diag] GF({ps})")

        # Off-diagonal (L)
        for i in range(n):
            for j in range(i + 1, n):
                C = np.zeros((n, n), dtype=int)
                t = rng.integers(0, ps)
                C[i, j] = C[j, i] = t % ps
                Fl = _compose_symp(n, ps, synth_lower_from_symmetric(n, C, ps))
                _check_equal(Fl, _full_from_lower(n, C, ps), p, f"[L off] GF({ps})")

        # Diagonal (R)
        for i in range(n):
            S = np.zeros((n, n), dtype=int)
            S[i, i] = rng.integers(0, ps)
            Fr = _compose_symp(n, ps, synth_upper_from_symmetric_via_H(n, S, ps))
            _check_equal(Fr, _full_from_upper(n, S, ps), p, f"[R diag] GF({ps})")

    print(f"Elementary L/R tests passed for GF({ps})")


def test_L_R_block(num_trials, n, p, seed=42):
    rng = np.random.default_rng(seed)
    test_L_block_unit(num_trials=num_trials, n=n, ps=p, rng=rng)
    test_R_block_unit(num_trials=num_trials, n=n, ps=p, rng=rng)
    test_elementary_pairs(num_trials=num_trials, n=n, ps=p, rng=rng)
    print(f"All L/R block tests passed for GF({p})")


#  ############################################################################  #
#                                 All together tests                             #
#  ############################################################################  #


def test_full_decomposition_roundtrip(num_trials: int, n: int, p: int, seed: int = 2025):
    rng = np.random.default_rng(seed)
    for t in range(num_trials):
        F = random_symplectic(n, p, rng=rng)
        C = decompose_symplectic_to_circuit(F, p, check=True)
        F_rec = C.composite_gate().symplectic % p
        _check_equal(F_rec, F, p, f"[roundtrip] GF({p}) trial={t}")

def test_full_decomposition_on_factored_instances(num_trials: int, n: int, p: int, seed: int = 2025):
    """
    Build F explicitly as L(A^{-1}B)·M(A)·R(CA^{-1}) (i.e., from random A, Bsym, Csym),
    then check we recover it exactly via the decomposition.
    """
    rng = np.random.default_rng(seed)
    for t in range(num_trials):
        # random invertible A and symmetric B,C
        A = random_invertible(n, rng, p)
        Bsym = random_symmetric(n, rng, p)
        Csym = random_symmetric(n, rng, p)

        # compose a ground-truth F = L·M·R
        F_true = build_F_from_LMR(n, A, Bsym, Csym, p)      # uses the same block builders you already have

        # Run decomposition
        C = decompose_symplectic_to_circuit(F_true, p, check=True)
        F_rec = C.composite_gate().symplectic % p

        _check_equal(F_rec, F_true, p, f"[factored-instance] GF({p}) trial={t}")


def decomposition_tests(num_trials, n, p):
    test_full_decomposition_roundtrip(num_trials=num_trials, n=n, p=p)
    test_full_decomposition_on_factored_instances(num_trials=num_trials, n=n, p=p)
    print(f"Decomposition tests passed for GF({p})")


if __name__ == "__main__":
    # quick try for p=2 and p=3
    n = 3
    n_trials = 10
    for p in (2, 3, 5, 7):
        test_preconditioner(num_trials=n_trials, n=n, p=p)
        test_m_block(num_trials=n_trials, n=n, p=p)
        test_L_R_block(num_trials=n_trials, n=n, p=p)
        decomposition_tests(num_trials=n_trials, n=n, p=p)
