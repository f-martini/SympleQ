import numpy as np
from sympleq.core.circuits.target import find_map_to_target_pauli_sum
from sympleq.core.circuits import Gate, Circuit
from sympleq.core.paulis import PauliSum
from scripts.experiments.symmetries.src.matroid_w_spm import find_clifford_symmetries
from scripts.experiments.symmetries.src.block_decomposition import block_decompose, ordered_block_sizes


def clifford_symmetry(pauli_sum: PauliSum,
                      check_symmetry: bool = True,
                      ) -> tuple[Gate, Gate, Gate]:

    G = find_clifford_symmetries(pauli_sum, num_symmetries=1,
                                 dynamic_refine_every=0)
    g = G[0]
    if check_symmetry:
        lhs = g.act(pauli_sum).standard_form()
        rhs = pauli_sum.standard_form()
        assert lhs == rhs, f'Symmetry finder failed\n{lhs.__str__()}\n{rhs.__str__()}'

    S, T = block_decompose(g.symplectic, int(pauli_sum.lcm))
    h_S, h_T = clifford_phase_decomposition(g.symplectic, g.phase_vector, S, T, int(pauli_sum.lcm))
    S_gate = Gate('S', g.qudit_indices, S, g.dimensions, h_S)
    T_gate = Gate('T', g.qudit_indices, T, g.dimensions, h_T)

    if check_symmetry:
        lhs = T_gate.act(S_gate.act(T_gate.inv().act(pauli_sum))).standard_form()
        rhs = pauli_sum.standard_form()
        ts = T_gate.symplectic
        tis = T_gate.inv().symplectic
        ss = S_gate.symplectic
        fs = g.symplectic
        C = Circuit(pauli_sum.dimensions, [T_gate.inv(), S_gate, T_gate])
        F = C.composite_gate()

        assert np.all(F.symplectic == g.symplectic), f'symplectic failed\n{F.symplectic}\n{g.symplectic}'
        assert np.all(F.phase_vector == g.phase_vector), f'phase vector failed\n{F.phase_vector}\n{g.phase_vector}'

        assert np.array_equal(fs, (ts @ ss @ tis) % 2), f'symplectic failed\n{fs}\n{ts @ ss @ ts.T}'
        assert lhs == rhs, f'Localisation failed\n{lhs.__str__()}\n{rhs.__str__()}'
    return g, S_gate, T_gate


def multiple_clifford_symmetries(pauli_sum: PauliSum,
                                 n_symmetries: int = 1,
                                 check_symmetry: bool = True,
                                 ) -> tuple[list[Gate], list[Gate], list[Gate]]:

    G = find_clifford_symmetries(pauli_sum, num_symmetries=n_symmetries,
                                 dynamic_refine_every=0)

    if check_symmetry:
        for g in G:
            assert g.act(pauli_sum).standard_form() == pauli_sum.standard_form()

    Ss = []
    Ts = []
    for i, g in enumerate(G):
        S, T = block_decompose(g.symplectic, pauli_sum.lcm)
        h_S, h_T = clifford_phase_decomposition(g.symplectic, g.phase_vector, S, T, int(pauli_sum.lcm))
        S_gate = Gate(f'S{i}', g.qudit_indices, S, g.dimensions, h_S)
        T_gate = Gate(f'T{i}', g.qudit_indices, T, g.dimensions, h_T)
        Ss.append(S_gate)
        Ts.append(T_gate)

    return G, Ss, Ts


def get_coupled_qudits_by_gate(gate: Gate):
    symp = gate.symplectic
    sizes = np.asarray(ordered_block_sizes(symp, int(gate.lcm)), dtype=int) / 2
    return sizes


def clifford_phase_decomposition(F: np.ndarray, h_F: np.ndarray,
                                 S: np.ndarray, T: np.ndarray, d: int,
                                 l_T: np.ndarray | None = None):
    """
    Inputs:
      F,h_F : composite Clifford (symplectic F, phase vector h_F) with phases mod 2d
      S,T   : symplectics satisfying F = T^{-1} S T
      d     : qudit dimension
      l_T   : optional gauge vector (same shape as h_F) giving l_T; default is 0

    Outputs:
      h_S, h_T : phase vectors of S and T (mod 2d)

    Conventions:
      - Pauli exponent rows update as a' = a @ F.T.
    """
    mod = 2 * d
    n2 = F.shape[0]
    Id = np.eye(n2, dtype=int)

    def m_wrap(x):           # wrap to [0,mod)
        return np.mod(x, mod)

    def m_mul(A, B):         # modular matrix multiply
        return m_wrap(A @ B)

    def U_matrix(n2):       # standard symplectic form [[0,I],[-I,0]]
        n = n2 // 2
        I_n = np.eye(n, dtype=int)
        Z = np.zeros((n, n), dtype=int)
        return np.block([[Z, I_n], [-I_n, Z]])

    def V_diag(M):           # vector of diagonal entries
        return np.diag(M) % mod

    U = U_matrix(n2)
    U_inv = m_wrap(-U)       # since U^{-1} = -U

    # S_C = C^T U C (independent of row/column convention as long as consistent)
    S_F = m_mul(F.T, m_mul(U, F))
    S_S = m_mul(S.T, m_mul(U, S))
    S_T = m_mul(T.T, m_mul(U, T))

    # ℓ_F = h_F - V_diag(S_F)
    l_F = m_wrap(h_F - V_diag(S_F))

    # Gauge choice for T: default ℓ_T = 0  ⇒  h_T = V_diag(S_T)
    if l_T is None:
        l_T = np.zeros_like(h_F)

    h_T = m_wrap(l_T + V_diag(S_T))

    # Use the symplectic identity to avoid numeric inversion:
    # T^{-T} = U T U^{-1}
    T_inv_T = m_mul(U, m_mul(T, U_inv))

    # ℓ_S = T^{-T} [ ℓ_F + (F^T - I) ℓ_T ]
    l_S = m_mul(T_inv_T, m_wrap(l_F + m_mul(F.T - Id, l_T)))

    # h_S = ℓ_S + V_diag(S_S)
    h_S = m_wrap(l_S + V_diag(S_S))

    return h_S, h_T
