import numpy as np
from pyscipopt import Model, quicksum
from collections import defaultdict


def solve_HT_equals_PH_blockwise(H, c, d):
    n, k = H.shape
    assert len(c) == n
    groups = defaultdict(list)
    for i, coeff in enumerate(c):
        groups[coeff].append(i)

    group_indices = list(groups.values())
    model = Model()
    model.setPresolve(0)

    # Integer M ∈ ℤ_d^{k×k}
    M = [[model.addVar(vtype="I", lb=0, ub=d - 1, name=f"M_{i}_{j}") for j in range(k)] for i in range(k)]

    # Blockwise permutation matrices P_g
    Ps = []
    for g_idx, group in enumerate(group_indices):
        m = len(group)
        P = [[model.addVar(vtype="B", name=f"P_{g_idx}_{i}_{j}") for j in range(m)] for i in range(m)]
        for i in range(m):
            model.addCons(quicksum(P[i][j] for j in range(m)) == 1)
        for j in range(m):
            model.addCons(quicksum(P[i][j] for i in range(m)) == 1)
        Ps.append((P, group))

    # Constraint: For each group, H_g @ M^T = P_g @ H_g (mod d)
    for (P, indices) in Ps:
        m = len(indices)
        H_block = H[indices, :]
        for i in range(m):
            for l in range(k):
                lhs = quicksum(H_block[i, j] * M[l][j] for j in range(k))
                rhs = quicksum(P[i][r] * int(H_block[r, l]) for r in range(m))
                z = model.addVar(vtype="I", name=f"Z_g{i}_{l}")
                model.addCons(lhs - rhs - d * z == 0)

    # Avoid trivial identity solution
    # Avoid identity: at most k-1 diagonal entries can be 1
    identity_diag = [model.addVar(vtype="B", name=f"IdCheck_{i}") for i in range(k)]
    for i in range(k):
        model.addCons(M[i][i] == identity_diag[i])  # identity_diag[i] is 1 iff M[i][i] == 1

    model.addCons(quicksum(identity_diag[i] for i in range(k)) <= k - 1)
    model.setObjective(quicksum(M[i][j] for i in range(k) for j in range(k)), "minimize")
    model.optimize()

    if model.getStatus() != "optimal":
        return None, None

    M_val = np.array([[int(round(model.getVal(M[i][j]))) % d for j in range(k)] for i in range(k)])
    P_val = np.zeros((n, n), dtype=int)

    for g_idx, (P, indices) in enumerate(Ps):
        m = len(indices)
        for i in range(m):
            for j in range(m):
                if round(model.getVal(P[i][j])) > 0.5:
                    P_val[indices[i], indices[j]] = 1

    return M_val, P_val


########################################################################################################################



if __name__ == "__main__":
    import sys
    sys.path.append('./')  # Adjust path to import quaos
    from quaos.hamiltonian import random_pauli_hamiltonian
    from time import time

    if True:

        failures = []
        successes = []
        failures2 = []
        successes2 = []

        time_solve1 = 0
        time_solve2 = 0
        print("Starting tests...")
        n_tests = 10
        for i in range(n_tests):
            d = 2
            n_paulis = 5
            n_qudits = 5
            ham = random_pauli_hamiltonian(n_paulis, [d] * n_qudits, mode='randint2')
            H = ham.symplectic_matrix()
            coeffs = ham.weights
            print(coeffs)

            print('Running test', i+1, 'of', n_tests)

            time1 = time()
            M, P = solve_HT_equals_PH_blockwise(H, coeffs, d)
            time_solve1 += time() - time1
            if M is not None:
                left = (H @ M.T) % d
                right = (P @ H) % d
                print(np.array_equal(left, right))
                if not np.array_equal(left, right) or np.all(M - np.eye(M.shape[0], dtype=int) == 0):
                    failures.append((H, M, P))
                else:
                    successes.append((H, M, P))
            # time2 = time()
            # M, P = solve_HT_equals_PH_by_intersection(H, coeffs, d)
            # print("Global solver result:")
            # print(M)
            # time_solve2 += time() - time2
            # if M is not None:
            #     left = (H @ M.T) % d
            #     right = (P @ H) % d
            #     print(np.array_equal(left, right))
            #     if not np.array_equal(left, right) or np.all(M - np.eye(M.shape[0], dtype=int) == 0):
            #         failures2.append((H, M, P))
            #     else:
            #         successes2.append((H, M, P))

        print(f"Total time for SCIP solver: {time_solve1:.2f} seconds, average {time_solve1 / n_tests:.2f} seconds per test.")
        # print(f'Total time for global solver: {time_solve2:.2f} seconds, average {time_solve2 / n_tests:.2f} seconds per test.')
        print(len(successes), "successful tests for SCIP solver.")
        # print(len(successes2), "successful tests for global solver.")
        # for H, M, P in successes:
        #     print("Success!")
        #     print("H =\n", H)
        #     print("M =\n", M)
        #     print("P =\n", P)



    if False:
        # Example usage
        d = 5
        H = np.array([
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 1]
        ], dtype=int)
        c = [1, 1, 1, 2]  # coefficients per row

        M, P = solve_HT_equals_PH_global(H, c, d)
        if M is not None:
            print("Found consistent M:")
            print(M)
            print("Permutation P:")
            print(P)
            print("Check H @ M.T % d == P @ H % d:")
            print(np.all((H @ M.T) % d == (P @ H) % d))
            print((M) % d)
        else:
            print("No consistent M found.")

        M, P = solve_HT_equals_PH_blockwise(H, c, d)
        if M is not None:
            print("Found consistent M:")
            print(M)
            print("Permutation P:")
            print(P)
            print("Check H @ M.T % d == P @ H % d:")
            print(np.all((H @ M.T) % d == (P @ H) % d))
            print((M) % d)
        else:
            print("No consistent M found.")