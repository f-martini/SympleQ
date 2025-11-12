import numpy as np
import galois
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from sympy.combinatorics import Permutation
from itertools import permutations, combinations
import itertools

# -------------------- permutation utilities --------------------
def permutation_to_swaps(perm: Permutation):
    swaps = []
    for cycle in perm.cyclic_form:
        if len(cycle) > 1:
            a0 = cycle[0]
            for aj in reversed(cycle[1:]):
                swaps.append((a0, aj))
    return swaps


def apply_permutation(A, perm: Permutation):
    idx = list(perm.array_form)
    return A[np.ix_(idx, idx)]


def find_automorphisms(M: np.ndarray) -> list[Permutation]:
    """
    Returns a list of sympy Permutations corresponding to all row/column
    permutations leaving M invariant.
    """
    n = M.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0:
                G.add_edge(i, j, label=int(M[i, j]))
    matcher = GraphMatcher(G, G, edge_match=lambda e1, e2: e1['label'] == e2['label'])
    perms = []
    for iso in matcher.isomorphisms_iter():
        mapping = [iso[i] for i in range(n)]
        perms.append(Permutation(mapping))
    return perms


# -------------------- 3. verification --------------------
def verify_automorphisms(M, perms, forced_perms=None):
    """
    Check that each permutation is a valid automorphism of M, print swaps,
    and verify that all forced permutations are present.
    """
    nontrivial_count = 0
    perms_set = set([tuple(g.array_form) for g in perms])

    # 1. Verify each non-trivial automorphism
    for g in perms:
        if g != Permutation(list(range(M.shape[0]))):
            M_perm = apply_permutation(M, g)
            assert np.all(M == M_perm), f"Automorphism failed for {g}"
            swaps = permutation_to_swaps(g)
            print(f"Non-trivial automorphism {nontrivial_count + 1}: {g}, swaps: {swaps}")
            nontrivial_count += 1

    assert nontrivial_count > 0, "No non-trivial automorphisms found!"
    print(f"Total non-trivial automorphisms verified: {nontrivial_count}")

    # 2. Verify forced permutations are present
    if forced_perms is not None:
        for fpi in forced_perms:
            if tuple(fpi.array_form) not in perms_set:
                raise AssertionError(f"Forced permutation {fpi} not found in automorphisms!")
            else:
                print(f"Forced permutation {fpi} verified present.")


# ---------------------------------------PRELIMINARY -------------------------------------------------------------------------------

# ---------------- Helper functions ----------------

def list_to_bitvec(lst, size=None):
    """Convert a list of integer indices to a bit vector of given size."""
    if size is None:
        size = max(lst) + 1 if lst else 0
    v = np.zeros(size, dtype=int)
    for i in lst:
        v[i] = 1
    return v

def basis_matrix_from_vectors(vecs, size=None):
    """Stack list of integer vectors as columns of a bit matrix."""
    if size is None:
        size = max(max(v) for v in vecs) + 1
    cols = [list_to_bitvec(v, size) for v in vecs]
    return np.stack(cols, axis=1)

def is_full_rank_gf2(mat):
    """Check if a matrix over GF(2) has full rank (rank = min(rows, cols))."""
    m = mat.copy() % 2
    n_rows, n_cols = m.shape
    rank = 0
    for i in range(n_rows):
        for j in range(rank, n_cols):
            if m[i,j] == 1:
                m[:, [rank,j]] = m[:, [j,rank]]
                break
        else:
            continue
        for k in range(n_rows):
            if k != i and m[k, rank] == 1:
                m[k] ^= m[i]
        rank += 1
    return rank == min(n_rows, n_cols)

def invert_gf2(mat):
    """
    Compute left-inverse over GF(2) for full-rank (rectangular) matrix.
    Returns X such that X @ mat = I (size n x n if mat is dim x n)
    """
    from galois import GF2
    GF = GF2
    n_rows, n_cols = mat.shape
    if n_rows < n_cols:
        raise ValueError("Cannot invert: rows < cols")
    # Augment with identity
    aug = np.hstack([mat, np.eye(n_rows, dtype=int)])
    aug = GF(aug)
    # Gaussian elimination
    for i in range(n_cols):
        pivot_rows = np.where(aug[i:, i])[0]
        if len(pivot_rows) == 0:
            raise ValueError("Matrix not full rank")
        pivot = pivot_rows[0] + i
        if pivot != i:
            aug[[i,pivot]] = aug[[pivot,i]]
        for j in range(n_rows):
            if j != i and aug[j,i]:
                aug[j] += aug[i]
    X = aug[:n_cols, n_cols:]  # left-inverse
    return X.astype(np.uint8)   


def matmul_gf2(A, B):
    return galois.GF2(A) @ galois.GF2(B)

def vecs_to_frozenset(vecs):
    """Convert list of bit vectors to a hashable frozenset of tuples."""
    return frozenset(tuple(v.tolist()) for v in vecs)


# ---------------- Automorphism Enumerator ----------------
class AutomorphismEnumerator:
    def __init__(self, independent, dependencies, n, checker=None, max_cycle_length=4):
        """
        Enumerates automorphisms of a vector set with independent and dependent indices.

        Parameters
        ----------
        independent : list[int]
            List of independent basis indices.
        dependencies : dict[int, list[tuple[int, int]]]
            Mapping from dependent indices to independent combinations.
            Example: {8: [(2,1), (4,1)]}
        checker : callable | None
            Function (perm, independent, dependencies) -> bool
            Returns True if perm is an automorphism.
        max_cycle_length : int
            Maximum cycle length to explore (start from 2).
        """
        self.independent = independent
        self.dependencies = dependencies
        self.checker = checker
        self.max_cycle_length = max_cycle_length
        self.n = len(independent) + len(dependencies)

        # bookkeeping
        self._seen = set()
        self._generator = self._enumerate()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)

    def _enumerate(self):
        identity = tuple(range(self.n))
        if self.checker is None or self.checker(identity, self.independent, self.dependencies):
            yield []

        # try cycle lengths 2 up to max_cycle_length
        for cycle_len in range(2, self.max_cycle_length + 1):
            for cycle in itertools.permutations(range(self.n), cycle_len):
                perm = list(range(self.n))
                # rotate the cycle
                for i in range(cycle_len):
                    perm[cycle[i]] = cycle[(i + 1) % cycle_len]
                perm = tuple(perm)

                if perm in self._seen:
                    continue
                self._seen.add(perm)

                if self.checker is None or self.checker(perm, self.independent, self.dependencies):
                    yield self._perm_to_swaps(perm)

    @staticmethod
    def _perm_to_swaps(perm):
        """
        Convert a permutation into a list of swaps (transpositions).
        """
        swaps = []
        seen = set()
        for i in range(len(perm)):
            if i in seen or perm[i] == i:
                continue
            cycle = []
            j = i
            while j not in seen:
                seen.add(j)
                cycle.append(j)
                j = perm[j]
            if len(cycle) == 2:
                swaps.append((cycle[0], cycle[1]))
            elif len(cycle) > 2:
                # break into swaps
                for k in range(len(cycle) - 1):
                    swaps.append((cycle[k], cycle[k+1]))
        return swaps
                    

def find_first_automorphism(independent, dependencies, n, check_fn) -> list[int] | None:
    """
    Iterate over automorphisms of 'vectors' (basis + dependents)
    and return the first one that passes the custom check.

    Parameters
    ----------
    vectors : list[list[int]]
        Original set of vectors (first n are the basis).
    n : int
        Number of basis elements.
    check_fn : callable
        Function to run on each automorphism.
        Signature: check_fn(index_vector) -> bool
        Return True if it passes your custom check, False otherwise.

    Returns
    -------
    index_vector : list[int] or None
        The first automorphism (compact index-vector) that passes the check,
        or None if no automorphism passes.
    """
    enumerator = AutomorphismEnumerator(independent, dependencies, n)

    for idx_vec in enumerator:
        # idx_vec is the compact index-vector: length-n list of indices into 'vectors'
        if check_fn(idx_vec):
            return idx_vec  # stop at the first valid automorphism

    return None  # no automorphism passed
    
# ---- Tests ----
def test_enumerator_example():
    vectors = [[0], [1], [2], [3], [0, 1], [0, 2], [1, 3], [2, 3]]
    n = 4
    enumerator = AutomorphismEnumerator(vectors, n)

    results = []
    for i, idx_vec in zip(range(5), enumerator):  # take first 5
        results.append(idx_vec)

    # Each result must be a valid length-n index-vector
    for idx_vec in results:
        assert len(idx_vec) == n
        assert all(isinstance(x, int) for x in idx_vec)
        # Ensure indices are within range
        assert all(0 <= x < len(vectors) for x in idx_vec)

    # At least one nontrivial automorphism should exist
    assert any(idx_vec != list(range(n)) for idx_vec in results)

    print("Test passed. Example automorphism index-vectors:")
    for r in results:
        print(r)



# -------------------- TESTING --------------------
def generate_test_matrix(p=2, n=8, num_forced=2, rng=None):
    """
    Generate a symmetric GF(p) matrix with zero diagonals and known non-trivial automorphisms.
    Returns the matrix and the list of forced permutations.
    """
    if rng is None:
        rng = np.random.default_rng()
    GF = galois.GF(p)
    M = GF.Zeros((n, n))

    # generate random upper-triangular part
    for i in range(n):
        for j in range(i + 1, n):
            val = GF(rng.integers(0, p))
            M[i, j] = val
            M[j, i] = val

    # define multiple forced automorphisms on disjoint blocks
    forced_perms = []
    block_size = n // num_forced
    for b in range(num_forced):
        start = b * block_size
        end = min(start + block_size, n)
        if end - start >= 2:
            block_perm = list(range(start, end))
            block_perm[0], block_perm[1] = block_perm[1], block_perm[0]  # swap first two
            perm_list = list(range(n))
            perm_list[start:end] = block_perm
            pi = Permutation(perm_list)
            forced_perms.append(pi)

    # impose all forced automorphisms
    for pi in forced_perms:
        for i in range(n):
            for j in range(n):
                M[i, j] = M[pi(i), pi(j)]

    return M, forced_perms


def run_multiple_tests(trials=5, p=2, n=8, num_forced=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    for t in range(trials):
        print(f"\n=== Trial {t + 1}/{trials} ===")
        # 1. generate test matrix
        M, forced_perms = generate_test_matrix(p=p, n=n, num_forced=num_forced, rng=rng)
        print("Generated matrix:\n", M)
        print("Forced permutations:", forced_perms)

        # 2. find automorphisms
        perms = find_automorphisms(M)
        print(f"Found {len(perms)} automorphisms in total")

        # 3. verify all non-trivial automorphisms and forced ones
        verify_automorphisms(M, perms, forced_perms=forced_perms)

    print(f"\nAll {trials} trials passed successfully.")


# ----------------------------------------------------------------------
# Generate symmetric matrix with zero diagonals and guaranteed automorphisms
# ----------------------------------------------------------------------
def generate_test_matrix_with_aut(p=2, n=6, num_forced=1, rng=None):
    """
    Generate symmetric zero-diagonal matrix over GF(p) with at least
    a subgroup of automorphisms generated by `num_forced` random swaps.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pick random disjoint swaps
    indices = list(range(n))
    rng.shuffle(indices)
    swaps = [(indices[2 * i], indices[2 * i + 1]) for i in range(num_forced) if 2 * i + 1 < n]

    # Build permutations as sympy objects
    forced_perms = []
    for (i, j) in swaps:
        perm = list(range(n))
        perm[i], perm[j] = perm[j], perm[i]
        forced_perms.append(Permutation(perm))

    # Partition indices into orbits under these swaps
    # Start with each index in its own orbit
    orbits = [{i} for i in range(n)]
    for (i, j) in swaps:
        for orb in orbits:
            if i in orb or j in orb:
                orb.update({i, j})
    # merge overlapping orbits
    merged = []
    while orbits:
        o = orbits.pop()
        for other in list(orbits):
            if o & other:
                o |= other
                orbits.remove(other)
        merged.append(o)
    orbits = merged

    # Assign a representative index for each orbit
    reps = [list(o)[0] for o in orbits]

    # Now build symmetric matrix
    M = np.zeros((n, n), dtype=int)
    for a, rep_a in enumerate(reps):
        for b, rep_b in enumerate(reps):
            if a <= b:
                val = rng.integers(0, p)
                for i in orbits[a]:
                    for j in orbits[b]:
                        if i != j:
                            M[i, j] = val
                            M[j, i] = val
    np.fill_diagonal(M, 0)

    return M % p, forced_perms


# ----------------------------------------------------------------------
# Brute force check of all automorphisms
# ----------------------------------------------------------------------
def brute_force_automorphisms(M):
    n = M.shape[0]
    auts = []
    for perm in permutations(range(n)):
        P = np.eye(n, dtype=int)[list(perm)]
        M_perm = P @ M @ P.T
        if np.all(M_perm == M):
            auts.append(Permutation(perm))
    return auts


# ----------------------------------------------------------------------
# Full test: generate matrix, brute-force group, and compare
# ----------------------------------------------------------------------
def test_against_bruteforce(p=2, n=6, trials=3, num_forced=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    for t in range(trials):
        print(f"\n=== Trial {t + 1}/{trials}, n={n}, p={p} ===")

        # 1. matrix with forced automorphisms
        M, forced = generate_test_matrix_with_aut(p=p, n=n, num_forced=num_forced, rng=rng)
        # print("Matrix:\n", M)
        print("Forced automorphisms:", forced)

        # 2. brute force ground truth
        ground_truth = brute_force_automorphisms(M)
        ground_truth_set = {tuple(g.array_form) for g in ground_truth}
        print(f"Brute force found {len(ground_truth)} automorphisms")

        # 3. algorithm under test
        found = find_automorphisms(M)
        found_set = {tuple(g.array_form) for g in found}
        print(f"find_automorphisms found {len(found)} automorphisms")

        # 4. check equality
        assert ground_truth_set == found_set, \
            f"Mismatch! Missing: {ground_truth_set - found_set}, Extra: {found_set - ground_truth_set}"

        # 5. check that all forced ones are present
        for f in forced:
            assert tuple(f.array_form) in found_set, \
                f"Forced automorphism {f} not found in result!"

        print("✅ Match confirmed, forced automorphisms recovered")

    print(f"\nAll {trials} tests passed successfully (n={n}, p={p}).")

if __name__ == "__main__":
    run_multiple_tests(trials=5, p=3, n=8, num_forced=1)
    test_against_bruteforce(p=4, n=5, trials=50, num_forced=3)
    test_against_bruteforce(p=5, n=6, trials=30, num_forced=3)
