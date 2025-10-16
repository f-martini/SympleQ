import numpy as np
from quaos.core.circuits.target import find_map_to_target_pauli_sum
from quaos.core.circuits import Gate, Circuit
from quaos.core.paulis import PauliSum
from quaos.core.finite_field_solvers import get_linear_dependencies
from scripts.experiments.symmetries.src.matroid_w_spm import find_k_automorphisms_symplectic
from .phase_correction import pauli_phase_correction


def clifford_symmetry(pauli_sum: PauliSum,
                      n_symmetries: int = 1,
                      check_symmetry: bool = True,
                      phase_correction: bool = True
                      ) -> Gate:

    d = pauli_sum.dimensions
    lcm = pauli_sum.lcm
    independent_paulis, dependencies = get_linear_dependencies(pauli_sum.tableau(), d)

    S = pauli_sum.symplectic_product_matrix()
    perms = find_k_automorphisms_symplectic(independent_paulis, dependencies,
                                            S=S, p=2, k=n_symmetries,
                                            require_nontrivial=True,
                                            basis_first="any",
                                            dynamic_refine_every=0,           # or a small number like 8 if helpful
                                            extra_column_invariants="none",   # or "hist" if k is modest
                                            )
    automorphism = []
    for i in independent_paulis:
        automorphism.append(perms[0][i])

    H_t = pauli_sum.copy()
    H_t = H_t[automorphism]
    H_i = pauli_sum[independent_paulis]

    F, h, _, _ = find_map_to_target_pauli_sum(H_i, H_t)

    G = Gate('Symmetry', [i for i in range(pauli_sum.n_qudits())], F.T, pauli_sum.dimensions, h)

    # phase correction - add Pauli to make circuit, collapse to single Gate
    if phase_correction:
        output_phase = G.act(pauli_sum).phases
        delta_phi = output_phase - pauli_sum.phases
        if np.any(delta_phi):
            print(delta_phi % (2 * lcm))
            G_p = pauli_phase_correction(pauli_sum.tableau(), delta_phi % (2 * lcm), 2, pauli_sum.dimensions)
            C = Circuit(G.dimensions, [G, G_p])
            G = C.composite_gate()

    if check_symmetry:
        if phase_correction:
            assert G.act(pauli_sum).standard_form() == pauli_sum.standard_form()
        else:
            assert np.array_equal(G.act(pauli_sum).standard_form().tableau(), pauli_sum.standard_form().tableau())

    return G
