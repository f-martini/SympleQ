from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import galois
from algebra import gauss_inverse_mod_prime

def build_generator_matrix(
    independent: List[int],
    dependencies: Dict[int, List[Tuple[int,int]]],
    labels: List[int],
    p: int,
) -> tuple[galois.FieldArray, List[int], np.ndarray]:
    GF = galois.GF(p)
    basis_order = sorted(independent)
    k, n = len(basis_order), len(labels)
    lab_to_col = {lab: j for j, lab in enumerate(labels)}
    basis_index = {b: i for i, b in enumerate(basis_order)}
    G_int = np.zeros((k, n), dtype=int)
    for b in basis_order:
        G_int[basis_index[b], lab_to_col[b]] = 1
    for d, pairs in dependencies.items():
        j = lab_to_col[d]
        for b, m in pairs:
            G_int[basis_index[b], j] = (G_int[basis_index[b], j] + int(m)) % p
    G = GF(G_int)
    basis_mask = np.zeros(n, dtype=bool)
    for b in basis_order:
        basis_mask[lab_to_col[b]] = True
    return G, basis_order, basis_mask

def compute_U_from_basis(G: galois.FieldArray, pi: np.ndarray, basis_order_idx: List[int], p: int):
    Bcols = np.array(basis_order_idx, dtype=int)
    PBcols = pi[Bcols]
    C = G[:, PBcols]
    U_int = gauss_inverse_mod_prime(C.view(np.ndarray) % p, p)
    if U_int is None:
        return None
    GF = galois.GF(p)
    return GF(U_int)

def check_code_automorphism(G: galois.FieldArray, pi: np.ndarray, basis_order: List[int], labels: List[int], p: int) -> bool:
    U = compute_U_from_basis(G, pi, basis_order, p)
    if U is None:
        return False
    return np.array_equal(U @ G[:, pi], G)

if __name__ == "__main__":
    # smoke
    import numpy as np
    GF = galois.GF(2)
    G = GF([[1,0,1,0],[0,1,1,1]])
    # permutation swapping col 2<->3 while keeping basis cols 0,1 in place fails code autom if not consistent
    pi = np.array([0,1,3,2])
    # basis: [0,1]
    ok = check_code_automorphism(G, pi, [0,1], [0,1,2,3], 2)
    print("[code] ok (result =", ok, ")")
