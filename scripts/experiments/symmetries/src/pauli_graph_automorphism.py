from quaos.core.paulis import PauliSum
from quaos.core.circuits import Gate, Circuit
from graph_colour_utils import wl_coloring, build_color_invariants
from search_utils import (smallest_wl_block_first_search, selective_wl2_split,
                          )
from typing import Dict, List
import numpy as np
from quaos.core.circuits.target import find_map_to_target_pauli_sum
from phase_correction import pauli_phase_correction


def find_graph_automorphism(pauli_sum: PauliSum,
                            max_wl_rounds: int = 10,
                            small_circuit_size: int = 0,
                            use_selective_wl2: bool = True,
                            wl2_size_threshold: int = 64,
                            ir_rounds: int = 2,
                            ir_every_steps: int = 0,
                            k_wanted: int = 1,
                            ac3_symmetric: bool = True,
                            wl2_rounds: int = 1,
                            F_known_debug: np.ndarray | None = None,
                            ) -> List[Circuit]:
    # Build a graph representation of the PauliSum where nodes are Pauli terms
    # edges are coloured based on commutation relations from the symplectic product matrix
    n = pauli_sum.n_paulis()
    labels = list(range(n))
    # setup the Clifford invariants for WL seeding
    p = int(pauli_sum.lcm)
    generator_matroid, basis_order = pauli_sum.matroid()  # encodes linear dependencies
    basis_mask = pauli_sum._basis_mask()  # boolean mask of which Paulis are in the basis

    S = np.mod(pauli_sum.symplectic_product_matrix(), p).astype(np.int64, copy=False)
    coeffs = np.unique(pauli_sum.weights, return_inverse=True)[1]  # stable int IDs

    # WL seeding
    wl_seed = build_color_invariants(G=generator_matroid, basis_mask=basis_mask,
                                     coeffs=coeffs,
                                     small_circuit_size=small_circuit_size)
    base_colors = wl_coloring(S, p, seed=wl_seed, max_rounds=max_wl_rounds)

    base_classes: Dict[int, List[int]] = {}
    for i, c in enumerate(base_colors):
        base_classes.setdefault(int(c), []).append(i)
    for c in base_classes:
        base_classes[c].sort()

    # ---- Optional pre-split (safe for either strategy) ----
    if use_selective_wl2:
        split = selective_wl2_split(S, p, base_colors,
                                    size_threshold=wl2_size_threshold,
                                    rounds=wl2_rounds)
        if split is not None:
            base_colors, base_classes = split  # refine feasibility safely

    independent_labels = [i for i, v in enumerate(basis_mask) if v]
    Cs = smallest_wl_block_first_search(
        pauli_sum,
        independent_labels=independent_labels,
        S_mod=S, p=p,
        base_colors=base_colors,
        base_classes=base_classes,
        G=generator_matroid, basis_order=basis_order, labels=labels,
        # col_invariants=wl_seed,
        coeffs=coeffs,
        k_wanted=k_wanted,
        # ir_rounds=ir_rounds,
        # ir_every_steps=ir_every_steps,
        # wl2_size_threshold=wl2_size_threshold,
        # wl2_rounds=wl2_rounds,
        # ac3_symmetric=ac3_symmetric,
        F_known_debug=F_known_debug
    )

    for C in Cs:
        assert C.act(pauli_sum).standard_form() == pauli_sum.standard_form(), f"\n{pauli_sum.standard_form().__str__()}\n{C.act(pauli_sum).standard_form().__str__()}"

    return Cs


if __name__ == "__main__":
    # from quaos.core.finite_field_solvers import get_linear_dependencies
    from quaos.models.random_hamiltonian import random_gate_symmetric_hamiltonian
    from quaos.core.circuits import SWAP, SUM, PHASE, Hadamard, Circuit

    failed = 0
    for _ in range(3):
        sym = SWAP(0, 1, 2)
        # symC = Circuit.from_random(2, 10, [2, 2])
        # print(symC)
        # sym = symC.composite_gate()
        H = random_gate_symmetric_hamiltonian(sym, 5, 10, scrambled=False)
        C = Circuit.from_random(H.n_qudits(), 100, H.dimensions).composite_gate()
        # C = Circuit(H.dimensions, [Hadamard(i, 2) for i in range(H.n_qudits())])
        H = C.act(H)
        H.weight_to_phase()
        scrambled_sym = Circuit(H.dimensions, [C.inv(), sym, C]).composite_gate()
        assert H.standard_form() == scrambled_sym.act(H).standard_form(), f"\n{H.standard_form().__str__()}\n{sym.act(H).standard_form().__str__()}"

        gate = find_graph_automorphism(H, k_wanted=10, ir_rounds=2,
                                        ir_every_steps=0, ac3_symmetric=False, max_wl_rounds=10,
                                        small_circuit_size=0,
                                        use_selective_wl2=False, wl2_size_threshold=64, wl2_rounds=1,
                                        F_known_debug=scrambled_sym.symplectic)
        # perms = find_graph_automorphism_equiv(H, k_wanted=1, use_basis_first=False)
        if len(gate) == 0:
            print('failed automorphism')
            # print(H)
            failed += 1
        else:
            print('found automorphism ------------------------ ')
        # else:
            # print("Found automorphism:", H.phases)
    # print('permutations = ', perms)
    print('Done')
    print(f'Failed: {failed}')
