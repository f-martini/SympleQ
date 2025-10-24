"""Codes for finding target Paulis and gates which map a given Pauli to a target Pauli."""
from quaos.core.paulis import PauliString, PauliSum
import numpy as np
from collections import defaultdict
from itertools import product
from quaos.core.circuits.find_symplectic import map_pauli_sum_to_target_tableau


def find_map_to_target_pauli_sum(input_pauli: PauliSum, target_pauli: PauliSum) -> tuple[np.ndarray, np.ndarray,
                                                                                         list[int], int]:
    """
    TODO: For efficiency improvement act only on target qudits

    Find a gate that maps Pauli P to target Pauli.

    Args:
        P (Pauli): The Pauli to be mapped.
        target (Pauli): The target Pauli.
        dimension (int): The dimension of the qudit.

    Returns:
        images (list[np.ndarray]): The images of the gate.
        h (np.ndarray): The phase vector of the gate.
        qudit_indices (list[int]): The indices of the qudits acted upon by the gate.
        gate_dimension (int): The dimension of the gate.
        """
    if np.all(input_pauli.dimensions != target_pauli.dimensions):
        raise ValueError("PauliSum and gate must have the same dimension.")

    n_qudits = input_pauli.n_qudits()
    if n_qudits != target_pauli.n_qudits():
        raise ValueError("PauliSum and target must have the same number of qudits.")

    # get list of qudits where input and target differ
    qudit_indices = []
    for i in range(n_qudits):
        # if input_pauli[:, i] != target_pauli[:, i]:
        qudit_indices.append(i)
    gate_dimension = input_pauli.dimensions[qudit_indices][0]

    if not np.all(input_pauli.dimensions[qudit_indices] == gate_dimension):
        raise ValueError("PauliSum must have the same dimension for all qudits acted upon by the gate.")

    if np.all(input_pauli.symplectic_product_matrix() != target_pauli.symplectic_product_matrix()):
        raise ValueError("Input and target PauliSum must be symplectically equivalent.")

    input_tableau = input_pauli.tableau()  # [:, qudit_indices]
    target_tableau = target_pauli.tableau()  # [:, qudit_indices]

    F = map_pauli_sum_to_target_tableau(input_tableau, target_tableau)

    # print('IN FUNCTION')
    # # print(input_symplectic)
    # # print()
    # print(target_symplectic - input_symplectic @ F % 2)
    # print('----------')

    h = get_phase_vector(F, gate_dimension)

    return F, h, qudit_indices, gate_dimension


def find_allowed_target(pauli_sum, target_pauli_list):
    pauli_list = [target_pauli_list[i][0] for i in range(len(target_pauli_list))]
    pauli_list = str_to_int(pauli_list)
    string_indices = [target_pauli_list[i][1] for i in range(len(target_pauli_list))]
    qudit_indices = [target_pauli_list[i][2] for i in range(len(target_pauli_list))]

    dims = pauli_sum.dimensions
    combined_indices = list(zip(string_indices, qudit_indices))
    index_dict = defaultdict(list)

    for idx, tup in enumerate(combined_indices):
        index_dict[tup].append(idx)

    underdetermined_pauli_indices = [combined_indices[indexes[0]]
                                     for indexes in index_dict.values() if len(indexes) > 1]
    underdetermined_pauli_options = []
    for indices in underdetermined_pauli_indices:
        options = []
        for i in range(len(target_pauli_list)):
            if target_pauli_list[i][1] == indices[0] and target_pauli_list[i][2] == indices[1]:
                if target_pauli_list[i][0] not in options:
                    options.append(target_pauli_list[i][0])
        underdetermined_pauli_options.append(options)
    underdetermined_pauli_options = [str_to_int(upo) for upo in underdetermined_pauli_options]

    determined_pauli_indices = [combined_indices[indexes[0]] for indexes in index_dict.values() if len(indexes) == 1]

    options_matrix = np.empty([pauli_sum.n_paulis(), pauli_sum.n_qudits()], dtype=object)
    for i in range(pauli_sum.n_paulis()):
        for j in range(pauli_sum.n_qudits()):
            if (i, j) in underdetermined_pauli_indices:
                options_matrix[i, j] = underdetermined_pauli_options[underdetermined_pauli_indices.index((i, j))]
            elif (i, j) in determined_pauli_indices:
                options_matrix[i, j] = [pauli_list[determined_pauli_indices.index((i, j))]]
            else:
                options_matrix[i, j] = [0, 1, 2, 3]

    # flag matrix to track which indices are determined
    flag_matrix = np.zeros([pauli_sum.n_paulis(), pauli_sum.n_qudits()], dtype=int)
    # 0 for not determined, 1 for underdetermined, -1 for determined
    for idx, tup in enumerate(combined_indices):
        if tup in underdetermined_pauli_indices:
            flag_matrix[tup[0], tup[1]] = -1  # -1 for underdetermined
        else:
            flag_matrix[tup[0], tup[1]] = 1  # 1 for determined

    spm = pauli_sum.symplectic_product_matrix()

    possible_targets = []
    for combo in product(*options_matrix.flatten()):
        combo = np.reshape(combo, (pauli_sum.n_paulis(), pauli_sum.n_qudits()))
        pauli_sum_candidate = matrix_to_pauli_sum(combo, pauli_sum.weights, pauli_sum.phases, dims)
        candidate_spm = pauli_sum_candidate.symplectic_product_matrix()

        if np.all(spm == candidate_spm):
            possible_targets.append(pauli_sum_candidate)
    return possible_targets


# def get_phase_vector(
#     gate_symplectic: np.ndarray,
#     dimension: int,
#     *,
#     order: str = "AB_CD",                # "AB_CD" => h = [diag(A B^T); diag(C D^T)]
#                                          # "CD_AB" => h = [diag(C D^T); diag(A B^T)]
#     tableau_after_F: np.ndarray | None = None,  # optional: a few rows after applying F (mod p)
# ) -> np.ndarray:
#     """
#     Lift choice for U = D(h) μ(F) under row/right-action convention: T' = T @ F^T.

#     Parameters
#     ----------
#     gate_symplectic : (2n, 2n) int ndarray
#         Symplectic matrix F over Z_p (give as ints; we'll reduce mod p).
#     dimension : int
#         Local dimension p (prime).
#     order : {"AB_CD","CD_AB"}
#         Whether to place diag(A B^T) in h_x (then diag(C D^T) in h_z), or swapped.
#         For most row/right implementations, "AB_CD" is the canonical choice.
#     tableau_after_F : (N, 2n) int ndarray, optional
#         If provided (reduced mod p), and p==2, we will auto-validate parity evenness
#         and, if needed, retry with the swapped order.

#     Returns
#     -------
#     h : (2n,) int ndarray
#         Phase (displacement) vector in Z_p giving a consistent lift of μ(F).
#         For p=2, entries are in {0,1}. For odd p, entries are in {0,...,p-1}.

#     Notes
#     -----
#     - For p=2, the odd-mod-4 phase part produced by μ(F) is captured by this h.
#       If your residuals are still odd after using this h, swap `order` or pass a few
#       post-F rows via `tableau_after_F` so the function can auto-swap.
#     - For odd p, the factor (p-1)//2 is the inverse of 2 in Z_p (i.e., -2^{-1}).
#     """
#     p = int(dimension)
#     F = np.asarray(gate_symplectic, dtype=int) % p
#     m = F.shape[0]
#     assert m % 2 == 0, "F must be 2n x 2n"
#     n = m // 2

#     A = F[:n, :n] % p
#     B = F[:n, n:] % p
#     C = F[n:, :n] % p
#     D = F[n:, n:] % p

#     # base diagonals
#     dAB = (np.einsum("ik,jk->ij", A, B) if False else (A @ B.T))  # clarity
#     dCD = (np.einsum("ik,jk->ij", C, D) if False else (C @ D.T))
#     hx0 = np.diag(dAB) % p
#     hz0 = np.diag(dCD) % p

#     if p == 2:
#         if order == "AB_CD":
#             h = np.concatenate([hx0, hz0], axis=0) % 2
#         else:
#             h = np.concatenate([hz0, hx0], axis=0) % 2

#         # Optional auto-validate (qubits): ensure even residual parity possible
#         if tableau_after_F is not None:
#             T2 = (np.asarray(tableau_after_F, dtype=int) % 2)
#             # Right-action displacement phase uses u' J h; parity (mod 2) is (u' J h)/2 mod 2,
#             # which for qubits reduces to (u' J h) mod 2 since only parity matters.
#             # Build J mod 2
#             I = np.eye(n, dtype=int)
#             J2 = np.block([[np.zeros((n, n), dtype=int),  I],
#                            [I,                              np.zeros((n, n), dtype=int)]]) % 2  # (since -I == I mod 2)
#             par = (T2 @ (J2 @ (h % 2))) % 2  # shape (N,)
#             # If some rows are odd while the problem expects even parity after lift,
#             # try swapped order automatically.
#             if np.any(par == 1):
#                 h_alt = np.concatenate([hz0, hx0], axis=0) % 2 if order == "AB_CD" \
#                         else np.concatenate([hx0, hz0], axis=0) % 2
#                 par_alt = (T2 @ (J2 @ (h_alt % 2))) % 2
#                 # If the alternate order reduces odd parity, use it
#                 if par_alt.sum() < par.sum():
#                     h = h_alt
#         return h.astype(int, copy=False)

#     # odd p: include the inverse of 2 (i.e., (p-1)//2)
#     half = (p + 1) // 2  # equals 2^{-1} mod p (since 2*half ≡ 1 mod p)
#     # Many references write -(1/2); in mod p arithmetic that equals (p-1)/2.
#     # If your stack expects the negative, swap to ((p-1)//2) instead of half.
#     # Here we use +1/2 which is common for row/right convention.
#     hx = (half * hx0) % p
#     hz = (half * hz0) % p

#     if order == "AB_CD":
#         h = np.concatenate([hx, hz], axis=0) % p
#     else:
#         h = np.concatenate([hz, hx], axis=0) % p

#     return h.astype(int, copy=False)


def get_phase_vector(gate_symplectic: np.ndarray, dimension: int) -> np.ndarray:
    """
    Calculate the phase vector for a gate given its symplectic matrix.

    See PRA 71, 042315 (2005) Eq. (10).
    Solves for h

    Args:
        gate_symplectic (np.ndarray): The symplectic matrix of the gate.
        dimension (int): The dimension of the qudit.

    Returns:
        np.ndarray: The phase vector of the gate.
    """
    n_qudits = gate_symplectic.shape[0] // 2

    U = np.zeros((2 * n_qudits, 2 * n_qudits), dtype=int)
    U[n_qudits:, :n_qudits] = np.eye(n_qudits, dtype=int)
    lhs = (dimension - 1) * np.diag(gate_symplectic.T @ U @ gate_symplectic) % 2  
    return lhs



def str_to_int(string):
    output = []

    for s in string:
        if s == 'I':
            output.append(0)
        elif s == 'X':
            output.append(1)
        elif s == 'Y':
            output.append(2)
        elif s == 'Z':
            output.append(3)
    return output


def int_to_pauli(integer):
    if integer == 0:
        return [0, 0]
    elif integer == 1:
        return [1, 0]
    elif integer == 2:
        return [1, 1]
    elif integer == 3:
        return [0, 1]
    else:
        raise ValueError(f"Invalid integer for Pauli representation: {integer}. Must be in [0, 3].")


def matrix_to_pauli_sum(matrix, weights, phases, dimensions):

    n_paulis = matrix.shape[0]
    n_qudits = matrix.shape[1]

    pauli_strings = []
    for i in range(n_paulis):
        x_exp = np.zeros(n_qudits, dtype=int)
        z_exp = np.zeros(n_qudits, dtype=int)
        for j in range(n_qudits):
            x_exp[j] = int_to_pauli(matrix[i, j])[0]
            z_exp[j] = int_to_pauli(matrix[i, j])[1]
        ps = PauliString(x_exp, z_exp, dimensions)
        pauli_strings.append(ps)

    return PauliSum(pauli_strings, weights, phases, dimensions, standardise=False)
