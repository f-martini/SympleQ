import numpy as np
import galois
from typing import Dict, List, Optional, Tuple


def wl_coloring(
    S_mod: np.ndarray,
    p: int,
    seed: Optional[np.ndarray] = None,
    max_rounds: int = 10
) -> np.ndarray:
    """
    1-WL (colour refinement) for a complete edge-coloured digraph given by S_mod (entries in 0..p-1).

    Parameters
    ----------
    S_mod : (n, n) int64
        Edge colours S[i,j] ∈ {0, ..., p-1} (mod p).
    p : int
        Prime or small integer; used only to size the histogram buckets.
    seed : (n, t) int or None
        Optional per-vertex integer features to include in the initial seed
        (parallel IDs, circuit counts, coeff buckets, etc.).
        If None, only the S-row histogram is used.
    max_rounds : int
        Max refinement rounds (stop early if stable).

    Returns
    -------
    color : (n,) int64
        Stable colour classes (compacted to 0..C-1).
    """
    S_mod = np.asarray(S_mod, dtype=np.int64, order="C")
    n = S_mod.shape[0]
    assert S_mod.shape == (n, n)
    assert p >= 2

    # --- Build S-row histograms (safe invariant) ---
    # hist[i,a] = |{ j : S[i,j] = a }|
    hist = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts = np.bincount(S_mod[i], minlength=p)
        hist[i, :p] = counts[:p]

    # --- Initial seed key: (seed_row, hist_row) ---
    # Pack row-wise integers into tuples for hashing to palette ids
    def row_tuple(i: int):
        seed_part = () if seed is None else tuple(int(x) for x in np.atleast_1d(seed[i]))
        return (seed_part, tuple(int(x) for x in hist[i]))

    palette = {}
    color = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = row_tuple(i)
        color[i] = palette.setdefault(key, len(palette))

    # --- Refinement rounds ---
    # For each i, count occurrences of (neighbor_color, edge_value)
    for _ in range(max_rounds):
        changed = False
        new_keys = []
        for i in range(n):
            d = {}
            row = S_mod[i]
            # Count with a small dict; robust and simple
            for j in range(n):
                k = (int(color[j]), int(row[j]))
                d[k] = d.get(k, 0) + 1
            # Signature = (own_color, sorted multiset of neighbour (color, edge_value) counts)
            new_keys.append((int(color[i]), tuple(sorted(d.items()))))

        palette2 = {}
        new_color = np.empty(n, dtype=np.int64)
        for i, key in enumerate(new_keys):
            c = palette2.setdefault(key, len(palette2))
            new_color[i] = c
            if c != color[i]:
                changed = True
        color = new_color
        if not changed:
            break

    # Compact to 0..C-1 deterministically
    uniq, inv = np.unique(color, return_inverse=True)
    return inv.astype(np.int64)


def _parallel_class_ids(G) -> np.ndarray:
    GF = type(G)
    p = GF.characteristic
    k, N = G.shape
    ids = -np.ones(N, dtype=int)
    rep = {}
    next_id = 0

    for j in range(N):
        col = G[:, j]
        if np.all(col == 0):
            key = ("ZERO",)
        else:
            nz_idx = int(np.nonzero(col)[0][0])
            inv = pow(int(col[nz_idx]), -1, p)
            invF = GF(inv)
            norm = col * invF
            key = tuple(int(v) for v in np.asarray(norm))
        if key not in rep:
            rep[key] = next_id
            next_id += 1
        ids[j] = rep[key]
    return ids


def _fundamental_circuit_counts(G: galois.FieldArray, basis_mask: np.ndarray) -> np.ndarray:
    k, N = G.shape
    counts = np.zeros(N, dtype=int)
    basis_cols = np.where(basis_mask)[0]
    non_basis_cols = np.where(~basis_mask)[0]
    if non_basis_cols.size == 0:
        return counts
    D = G[:, non_basis_cols]  # k x (N-k)
    for t, j_col in enumerate(non_basis_cols):
        nz = np.nonzero(D[:, t])[0]
        if nz.size == 0:  # zero dependent column (rare)
            continue
        counts[j_col] += 1
        counts[basis_cols[nz]] += 1
    return counts


def _small_circuit_counts_optional(G: galois.FieldArray, max_size: int) -> Optional[np.ndarray]:
    if max_size < 3:
        return None
    A = np.asarray(G, dtype=int)
    k, N = A.shape
    counts = np.zeros(N, dtype=int)
    import itertools
    for sz in range(3, max_size + 1):
        for subset in itertools.combinations(range(N), sz):
            sub = A[:, subset]
            if np.linalg.matrix_rank(sub) < sz:
                # minimal check
                minimal = True
                for r in range(sz):
                    if np.linalg.matrix_rank(np.delete(sub, r, axis=1)) < sz - 1:
                        minimal = False
                        break
                if minimal:
                    for j in subset:
                        counts[j] += 1
    return counts


def _bucket_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """
    Map coefficients to stable integer buckets.
    - If they're already exact ints, just cast.
    - If complex/float, you can replace this with project-specific bucketing (e.g., value bins or equality).
    """
    try:
        return np.asarray(coeffs, dtype=int)
    except Exception:
        # Fallback: treat as a single bucket to avoid accidental over-splitting
        return -np.ones_like(np.asarray(coeffs)).astype(int)


def build_color_invariants(
    *,
    G: Optional[galois.FieldArray] = None,
    basis_mask: Optional[np.ndarray] = None,
    coeffs: Optional[np.ndarray] = None,
    small_circuit_size: int = 0
) -> Optional[np.ndarray]:
    """
    Build the (n, t) integer matrix used as WL seed `seed`:
      cols may include [parallel_id, fundamental_circuit_count, small_circuit_counts, coeff_bucket]

    Any of G/basis_mask/coeffs may be omitted if you don't want that invariant.

    Returns None if no columns are requested.
    """
    cols: List[np.ndarray] = []

    if G is not None and small_circuit_size >= 3:
        sc = _small_circuit_counts_optional(G, small_circuit_size)
        if sc is not None:
            cols.append(sc)

    if coeffs is not None:
        cols.append(_bucket_coeffs(coeffs))

    if not cols:
        return None
    # Stack to (n, t)
    col_invariants = np.stack(cols, axis=1).astype(int, copy=False)
    return col_invariants


def wl_colors_from_S(
    S_mod: np.ndarray,
    p: int,
    *,
    G: Optional[galois.FieldArray] = None,
    basis_mask: Optional[np.ndarray] = None,
    coeffs: Optional[np.ndarray] = None,
    include_parallel: bool = True,
    include_fundamental_circuits: bool = True,
    small_circuit_size: int = 0,
    max_rounds: int = 10
) -> np.ndarray:
    """
    Build WL seed from (optional) invariants and run generic WL-1 colouring.

    Typical usage:
        seed = build_color_invariants(G=G, basis_mask=basis_mask, coeffs=coeffs,
                                      include_parallel=True, include_fundamental_circuits=True,
                                      small_circuit_size=0)
        colors = wl_coloring(S_mod, p, seed=seed, max_rounds=10)

    This wrapper does both steps for convenience.
    """
    seed = build_color_invariants(
        G=G, basis_mask=basis_mask, coeffs=coeffs,
        include_parallel=include_parallel,
        include_fundamental_circuits=include_fundamental_circuits,
        small_circuit_size=small_circuit_size
    )
    return wl_coloring(S_mod, p, seed=seed, max_rounds=max_rounds)


# ---------------------------------------------------------------------
# 0) Utility: rebuild {color -> sorted [indices]} from a color vector
# ---------------------------------------------------------------------

def _rebuild_classes(colors: np.ndarray) -> Dict[int, List[int]]:
    classes: Dict[int, List[int]] = {}
    for i, c in enumerate(colors):
        c = int(c)
        classes.setdefault(c, []).append(i)
    for c in classes:
        classes[c].sort()
    return classes


# ---------------------------------------------------------------------
# 1) A) Individualize-and-Refine (IR) dynamic WL on a partial mapping
#    Uses your generic wl_coloring() engine.
# ---------------------------------------------------------------------

def dynamic_refine_colors_IR(
    S_mod: np.ndarray,
    p: int,
    base_colors: np.ndarray,            # original base WL colors (for reference/tying)
    phi: np.ndarray,                    # length-n, -1 for unmapped, else image index
    coeffs: Optional[np.ndarray],       # None if you don't want weight-preserving autos
    col_invariants: Optional[np.ndarray],  # (n, t) ints; e.g., parallel ids + circuit counts
    rounds: int = 2
) -> np.ndarray:
    """
    Re-seed WL with info from the current partial mapping.
    Mapped vertices i get (mapped=1, image_base_color=base_colors[phi[i]]).
    Unmapped vertices get (mapped=0, image_base_color=-1).
    We also include the original base_colors[i] and optional col_invariants[i,*].
    """
    n = S_mod.shape[0]
    mapped_flag = (phi >= 0).astype(int)
    image_color = np.full(n, -1, dtype=int)
    mapped_idx = np.where(phi >= 0)[0]
    if mapped_idx.size:
        image_color[mapped_idx] = base_colors[phi[mapped_idx]]

    # Compose seed columns (all integers)
    pieces = [
        mapped_flag.reshape(-1, 1),
        image_color.reshape(-1, 1),
        base_colors.reshape(-1, 1)
    ]
    if col_invariants is not None:
        pieces.append(col_invariants.astype(int, copy=False))

    # Optional coefficients (only if you truly enforce weight-preserving autos)
    if coeffs is not None:
        # Expect pre-bucketed ints (recommended). If not, put all in one bucket to avoid over-splitting.
        try:
            coeff_bucket = np.asarray(coeffs, dtype=int).reshape(-1, 1)
        except Exception:
            coeff_bucket = -np.ones((n, 1), dtype=int)
        pieces.append(coeff_bucket)

    seed = np.concatenate(pieces, axis=1)

    # Call your generic WL-1 engine. We assume wl_coloring(...) is defined elsewhere.
    colors = wl_coloring(S_mod, p, seed=seed, max_rounds=rounds)
    return colors


# ---------------------------------------------------------------------
# 2) C) AC-3 style domain filtering (arc consistency on candidate sets)
# ---------------------------------------------------------------------
def init_candidates_from_base(
    n: int,
    base_classes: Dict[int, List[int]],
    base_colors: np.ndarray,
    coeffs: Optional[np.ndarray] = None
) -> Dict[int, List[int]]:
    """
    Initial domains: D[i] = indices in the SAME base WL class (and same coeff if provided).
    """
    D: Dict[int, List[int]] = {}
    for i in range(n):
        cls = base_classes[int(base_colors[i])]
        if coeffs is None:
            D[i] = list(cls)
        else:
            D[i] = [y for y in cls if coeffs[y] == coeffs[i]]
    return D


def ac3_filter_domains(
    S_mod: np.ndarray,
    p: int,                               # kept for signature; not used directly here
    D: Dict[int, List[int]],
    symmetric: bool = True
) -> bool:
    """
    Enforce arc-consistency: for all i != j, each y in D[i] must have a 'support' z in D[j]
    with S[y,z] == S[i,j] (and if symmetric=True also S[z,y] == S[j,i]).
    Returns False if any domain becomes empty; otherwise True.
    """
    from collections import deque

    n = S_mod.shape[0]
    # Precompute required edge colors for each (i,j)
    need = {(i, j): (int(S_mod[i, j]), int(S_mod[j, i])) for i in range(n) for j in range(n) if i != j}

    # Initialize queue with all ordered pairs (i, j), i != j
    Q = deque((i, j) for i in range(n) for j in range(n) if i != j)

    while Q:
        i, j = Q.popleft()
        need_ij, need_ji = need[(i, j)]
        Dj = D[j]
        if not Dj:
            return False  # already empty elsewhere

        changed = False
        new_Di: List[int] = []
        # Keep y in D[i] if it has some support z in D[j]
        for y in D[i]:
            ok = False
            # Early exit on first supporting z
            for z in Dj:
                if S_mod[y, z] == need_ij and (not symmetric or S_mod[z, y] == need_ji):
                    ok = True
                    break
            if ok:
                new_Di.append(y)
            else:
                changed = True

        if changed:
            if not new_Di:
                D[i] = []
                return False
            # Update and re-enqueue neighbors of i
            D[i] = new_Di
            for k in range(n):
                if k != i and k != j:
                    Q.append((k, i))
    return True


# ---------------------------------------------------------------------
# 3) B) Selective WL-2 split on the fattest color class
#    Lightweight, restricted to the largest class; returns refined colors/classes.
# ---------------------------------------------------------------------
def selective_wl2_split(
    S_mod: np.ndarray,
    p: int,
    colors: np.ndarray,
    size_threshold: int = 64,
    rounds: int = 1
) -> Optional[Tuple[np.ndarray, Dict[int, List[int]]]]:
    """
    If the largest color class has size > size_threshold, run a tiny WL-2 refinement
    restricted to that class and lift the split back to vertex colors.

    Returns:
        (new_colors, new_classes) on success, or None if no change or no large class.
    """
    # Find fattest class
    classes = _rebuild_classes(colors)
    fat_c, fat_idxs = max(classes.items(), key=lambda kv: len(kv[1]))
    m = len(fat_idxs)
    if m <= size_threshold:
        return None

    idx = np.array(fat_idxs, dtype=int)
    # Initialize pair colors for the induced subgraph
    # Start from (vertex_color, vertex_color) pairs encoded as a small int
    C_base = int(colors.max()) + 3
    pair = (colors[idx][:, None] * C_base + colors[idx][None, :]).astype(np.int64)

    # A very small number of WL-2 rounds
    for _ in range(rounds):
        # Build a signature for each ordered pair (i,j) using middle vertex k in the same class
        # Simple, fast hash over k: combine pair[i,k] and pair[k,j]
        mod = (1 << 61) - 1
        P1, P2 = 1315423911, 2654435761  # two large-ish mix constants
        new_pair = np.empty_like(pair)
        for a in range(m):
            # vectorized combine over k for this (a, :)
            # sig(a,b) = sum_k [ (pair[a,k]*P1 + pair[k,b]*P2) mod mod ] mod mod
            row_a = pair[a, :]            # (m,)
            col_b = pair[:, :]            # (m, m)
            # Broadcasting: (m,) -> (m,m) and combine, then reduce over k
            sig = ((row_a[None, :] * P1 + col_b * P2) % mod).sum(axis=1) % mod
            new_pair[a, :] = sig
        # Reindex to compact integers for stability
        uniq, inv = np.unique(new_pair, return_inverse=True)
        pair = inv.reshape(m, m)

    # Lift back to vertex colors: color(i) := hash( pair[i,*], pair[* ,i] )
    lifted = colors.copy()
    for loc_i, v in enumerate(idx):
        row_key = tuple(pair[loc_i, :])
        col_key = tuple(pair[:, loc_i])
        lifted[v] = (hash((row_key, col_key)) & 0x7fffffff)

    # Canonicalize to 0..C-1 deterministically
    uniq, inv = np.unique(lifted, return_inverse=True)
    new_colors = inv.astype(np.int64)

    # Rebuild classes
    new_classes = _rebuild_classes(new_colors)

    # If no effective change on the big class, return None
    if len(new_classes) == len(classes):
        # Could still be a reindex; check whether fat class actually split
        new_fat_parts = [c for c, vs in new_classes.items() if set(vs).issubset(set(fat_idxs))]
        if len(new_fat_parts) <= 1:
            return None

    return new_colors, new_classes
