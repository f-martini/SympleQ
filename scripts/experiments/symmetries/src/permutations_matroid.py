from collections import defaultdict, deque, Counter
from itertools import combinations
from typing import Dict, List, Tuple, Callable, Optional, Set
import random

Label = int
DepPairs = Dict[Label, List[Tuple[Label, int]]]  # external API format


# ----------------------------
# Utilities: multiplicity dicts
# ----------------------------
def _to_multidict(pairs: List[Tuple[Label, int]], p: int) -> Dict[Label, int]:
    acc = defaultdict(int)
    for j, m in pairs:
        acc[j] = (acc[j] + m) % p
    return {j: m for j, m in acc.items() if m % p != 0}


def _signature(md: Dict[Label, int], p: int) -> Tuple[Tuple[Label, int], ...]:
    return tuple(sorted((j, m % p) for j, m in md.items() if m % p != 0))


def _identity_perm(labels: List[Label]) -> Dict[Label, Label]:
    return {x: x for x in labels}


def _compose_perm(a_then_b: Dict[Label, Label], g: Dict[Label, Label]) -> Dict[Label, Label]:
    # return g ∘ a_then_b : apply a_then_b first, then g
    return {k: g[a_then_b[k]] for k in a_then_b}


def _perm_key(perm: Dict[Label, Label], labels: List[Label]) -> Tuple[Label, ...]:
    return tuple(perm[x] for x in labels)


def perm_to_cycles(perm: Dict[Label, Label], domain: Optional[Set[Label]] = None) -> List[Tuple[Label, ...]]:
    if domain is None:
        domain = set(perm.keys())
    seen, out = set(), []
    for s in sorted(domain):
        if s in seen:
            continue
        cyc, cur = [], s
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = perm[cur]
        if len(cyc) > 1:
            out.append(tuple(cyc))
    return out


def _is_automorphism_simple(independent: List[int],
                            dependencies: Dict[int, List[Tuple[int, int]]],
                            perm: Dict[int, int],
                            p: int) -> bool:
    """
    Compare the multiset of vectors before/after applying 'perm',
    with both expressed in the ORIGINAL basis coordinates.
    """
    from collections import Counter
    orig = _build_vectors_original_basis(independent, dependencies, p)
    indep2, deps2 = _apply_perm_to_presentation(independent, dependencies, perm)

    k = len(next(iter(orig.values())))
    zero = tuple(0 for _ in range(k))

    def v_add(a, b):
        return tuple((x + y) % p for x, y in zip(a, b))

    def v_scale(a, s):
        return tuple((s * x) % p for x in a)

    vecs2: Dict[int, Tuple[int, ...]] = {}
    indep2_set = set(indep2)
    # new independents keep their original-basis vectors
    for b in indep2:
        vecs2[b] = orig[b]
    # new dependents recomposed from those originals
    for d2, pairs in deps2.items():
        if d2 in indep2_set:
            continue
        v = zero
        for j, m in pairs:
            v = v_add(v, v_scale(orig[j], m % p))
        vecs2[d2] = v

    return Counter(orig.values()) == Counter(vecs2.values())

# ----------------------------
# Presentation & checks
# ----------------------------
def _check_inputs(independent: List[Label], dependencies: DepPairs, p: int):
    basis = set(independent)
    dep_keys = set(dependencies.keys())
    if basis & dep_keys:
        raise ValueError("Independent and dependent label sets must be disjoint.")
    deps_md = {}
    for d, pairs in dependencies.items():
        md = _to_multidict(pairs, p)
        if not set(md.keys()).issubset(basis):
            bad = set(md.keys()) - basis
            raise ValueError(f"Dependency {d} references non-independent labels: {bad}")
        deps_md[d] = md
    return basis, deps_md


def _all_labels(basis: Set[Label], deps_md: Dict[Label, Dict[Label, int]]) -> List[Label]:
    return sorted(list(basis | set(deps_md.keys())))


# ----------------------------
# Independent-only automorphisms (with multiplicities)
# ----------------------------
def _deps_multiset(deps_md: Dict[Label, Dict[Label, int]], p: int) -> Counter:
    return Counter(_signature(md, p) for md in deps_md.values())


def _implied_dep_map_for_indep_perm(
    basis: Set[Label], deps_md: Dict[Label, Dict[Label, int]],
    indep_map: Dict[Label, Label], p: int
) -> Optional[Dict[Label, Label]]:
    src = _deps_multiset(deps_md, p)

    def mapped_sig(md: Dict[Label, int]) -> Tuple[Tuple[Label, int], ...]:
        acc = defaultdict(int)
        for j, m in md.items():
            acc[indep_map[j]] = (acc[indep_map[j]] + m) % p
        return tuple(sorted((j, m) for j, m in acc.items() if m % p != 0))

    mapped = Counter(mapped_sig(md) for md in deps_md.values())
    if mapped != src:
        return None

    # Build buckets by original signatures
    from_sig = defaultdict(deque)
    for d, md in sorted(deps_md.items()):
        from_sig[_signature(md, p)].append(d)

    dep_map = {}
    for d in sorted(deps_md.keys()):
        sig_tgt = mapped_sig(deps_md[d])
        if not from_sig[sig_tgt]:
            return None
        dep_map[d] = from_sig[sig_tgt].popleft()
    return dep_map


# ----------------------------
# Pivot algebra over GF(p)
# ----------------------------
def _allowed_pivots(basis: Set[Label], deps_md: Dict[Label, Dict[Label, int]], p: int) -> List[Tuple[Label, Label]]:
    piv = []
    for d, md in deps_md.items():
        for i, m in md.items():
            if i in basis and (m % p) != 0:
                piv.append((i, d))
    return sorted(set(piv))


def _pivot_structure(
    basis: Set[Label], deps_md: Dict[Label, Dict[Label, int]],
    i: Label, d: Label, p: int
) -> Tuple[Set[Label], Dict[Label, Dict[Label, int]]]:
    # swap independent i with dependent d, assuming coeff of i in d is nonzero
    Sd = deps_md[d]
    m_di = Sd.get(i, 0) % p
    assert i in basis and d in deps_md and m_di != 0

    new_basis = (basis - {i}) | {d}
    new_deps = {}

    # helper: add scaled dict into target dict mod p
    def add_scaled(dst, src, scale):
        for j, m in src.items():
            mm = (dst.get(j, 0) + scale * m) % p
            if mm == 0:
                if j in dst: del dst[j]
            else:
                dst[j] = mm

    # For each other dependent x
    for x, Sx in deps_md.items():
        if x == d: continue
        Sx_new = dict(Sx)
        m_xi = Sx.get(i, 0) % p
        if m_xi != 0:
            # remove i term
            Sx_new_i = (Sx_new.get(i, 0) - m_xi) % p
            if Sx_new_i == 0: Sx_new.pop(i, None)
            else: Sx_new[i] = Sx_new_i
            # add m_xi * (Sd - m_di * {i})
            Sd_minus = dict(Sd)
            Sd_minus_i = (Sd_minus.get(i, 0) - m_di) % p
            if Sd_minus_i == 0: Sd_minus.pop(i, None)
            else: Sd_minus[i] = Sd_minus_i
            add_scaled(Sx_new, Sd_minus, m_xi)
            # add m_xi * {d}
            Sx_new[d] = (Sx_new.get(d, 0) + m_xi) % p
            if Sx_new[d] == 0: Sx_new.pop(d, None)
        new_deps[x] = Sx_new

    # New dependency for i: (Sd - m_di*i) + m_di*{d}
    Si_new = {}
    Sd_minus = dict(Sd)
    Sd_minus_i = (Sd_minus.get(i, 0) - m_di) % p
    if Sd_minus_i == 0: Sd_minus.pop(i, None)
    else: Sd_minus[i] = Sd_minus_i
    if Sd_minus: add_scaled(Si_new, Sd_minus, 1)
    Si_new[d] = (Si_new.get(d, 0) + m_di) % p
    if Si_new[d] == 0: Si_new.pop(d, None)
    new_deps[i] = Si_new

    return new_basis, new_deps


# ----------------------------
# NEW: finish a basis permutation to a full automorphism by matching dependents
# ----------------------------
def try_finish_to_full_perm(basis: Set[int],
                            deps_md: Dict[int, Dict[int, int]],
                            indep_map: Dict[int, int],
                            p: int) -> Optional[Dict[int, int]]:
    """
    Given a permutation on the *current* basis 'basis' (labels in this presentation),
    attempt to build a full label permutation by matching dependent-label signatures.
    Returns the full permutation dict if successful, else None.
    """
    dep_map = _implied_dep_map_for_indep_perm(basis, deps_md, indep_map, p)
    if dep_map is None:
        return None
    labels = _all_labels(basis, deps_md)
    full = _identity_perm(labels)
    for i in basis:
        full[i] = indep_map[i]
    for d in deps_md:
        full[d] = dep_map[d]
    return full


# ----------------------------
# Generator harvesting
# ----------------------------
def _cand_indep_transpositions(basis, deps_md, p):
    labels = _all_labels(basis, deps_md)
    for a, b in combinations(sorted(basis), 2):
        indep_map = {i: i for i in basis}; indep_map[a], indep_map[b] = b, a
        dep_map = _implied_dep_map_for_indep_perm(basis, deps_md, indep_map, p)
        if dep_map is None: continue
        full = _identity_perm(labels)
        for i in basis: full[i] = indep_map[i]
        for d in deps_md: full[d] = dep_map[d]
        yield full

def _cand_indep_3cycles(basis, deps_md, p):
    for a, b, c in combinations(sorted(basis), 3):
        for cyc in [(a, b, c), (a, c, b)]:
            indep_map = {i: i for i in basis}
            indep_map[cyc[0]] = cyc[1]; indep_map[cyc[1]] = cyc[2]; indep_map[cyc[2]] = cyc[0]
            dep_map = _implied_dep_map_for_indep_perm(basis, deps_md, indep_map, p)
            if dep_map is None: continue
            full = _identity_perm(_all_labels(basis, deps_md))
            for i in basis: full[i] = indep_map[i]
            for d in deps_md: full[d] = dep_map[d]
            yield full

# ---- FIXED: pivots now yield only completed automorphisms ----
def _cand_single_pivots_fixed(basis: Set[int], deps_md: Dict[int, Dict[int, int]], p: int):
    """
    For each legal pivot (i,d):
      1) Pivot to (B1, D1).
      2) Identity indep_map on B1.
      3) Complete to a full permutation on labels of (B1,D1) via dependent matching.
      4) Pull back to original labels by composing with the (i d) swap.
    """
    labels0 = _all_labels(basis, deps_md)
    for (i, d) in _allowed_pivots(basis, deps_md, p):
        B1, D1 = _pivot_structure(basis, deps_md, i, d, p)
        indep_map_B1 = {j: j for j in B1}
        full_on_B1 = try_finish_to_full_perm(B1, D1, indep_map_B1, p)
        if full_on_B1 is None:
            continue
        swap = _identity_perm(labels0)
        swap[i], swap[d] = d, i
        yield _compose_perm(swap, full_on_B1)

def _cand_pivot_then_indep_fixed(basis: Set[int], deps_md: Dict[int, Dict[int, int]], p: int):
    """
    One pivot, then an independent automorphism in the new basis, then completion.
    """
    labels0 = _all_labels(basis, deps_md)
    for (i, d) in _allowed_pivots(basis, deps_md, p):
        B1, D1 = _pivot_structure(basis, deps_md, i, d, p)

        # indep transpositions in B1
        for a, b in combinations(sorted(B1), 2):
            indep_map = {j: j for j in B1}
            indep_map[a], indep_map[b] = b, a
            full_on_B1 = try_finish_to_full_perm(B1, D1, indep_map, p)
            if full_on_B1 is None:
                continue
            swap = _identity_perm(labels0)
            swap[i], swap[d] = d, i
            yield _compose_perm(swap, full_on_B1)

        # indep 3-cycles in B1
        for a, b, c in combinations(sorted(B1), 3):
            for cyc in [(a, b, c), (a, c, b)]:
                indep_map = {j: j for j in B1}
                indep_map[cyc[0]] = cyc[1]
                indep_map[cyc[1]] = cyc[2]
                indep_map[cyc[2]] = cyc[0]
                full_on_B1 = try_finish_to_full_perm(B1, D1, indep_map, p)
                if full_on_B1 is None:
                    continue
                swap = _identity_perm(labels0)
                swap[i], swap[d] = d, i
                yield _compose_perm(swap, full_on_B1)

def harvest_generators(
    independent: List[Label],
    dependencies: DepPairs,
    *,
    p: int = 2,
    pivot_depth: int = 1,
    allow_swaps=True, allow_3cycles=True, allow_pivots=True, allow_pivot_then_indep=True,
) -> Tuple[List[Dict[Label, Label]], List[Label]]:
    """
    Return a small generating set discovered from the local presentation (and up to `pivot_depth` neighbors).
    Also returns the canonical label order used to key permutations.
    """
    basis, deps_md = _check_inputs(independent, dependencies, p)
    labels = _all_labels(basis, deps_md)

    gens = []
    # level-0 (current presentation)
    if allow_swaps:  gens += list(_cand_indep_transpositions(basis, deps_md, p))
    if allow_3cycles: gens += list(_cand_indep_3cycles(basis, deps_md, p))
    if allow_pivots:  gens += list(_cand_single_pivots_fixed(basis, deps_md, p))
    if allow_pivot_then_indep: gens += list(_cand_pivot_then_indep_fixed(basis, deps_md, p))

    # explore nearby presentations up to `pivot_depth`
    seen_pres = {_presentation_key(basis, deps_md, p)}
    frontier = deque([(basis, deps_md, 0)])
    while frontier:
        B, D, depth = frontier.popleft()
        if depth >= pivot_depth: continue
        for (i, d) in _allowed_pivots(B, D, p):
            B1, D1 = _pivot_structure(B, D, i, d, p)
            key = _presentation_key(B1, D1, p)
            if key in seen_pres: continue
            seen_pres.add(key)
            frontier.append((B1, D1, depth + 1))
            # harvest indep automorphisms in this neighbor (no raw pivot emission)
            if allow_swaps:  gens += list(_cand_indep_transpositions(B1, D1, p))
            if allow_3cycles: gens += list(_cand_indep_3cycles(B1, D1, p))

    # ---------- NEW: filter to true automorphisms ----------
    gens = [g for g in gens if _is_automorphism_simple(independent, dependencies, g, p)]

    # dedup generators
    seen = set(); uniq = []
    for g in gens:
        k = _perm_key(g, labels)
        if k not in seen:
            seen.add(k); uniq.append(g)
    return uniq, labels


def _presentation_key(basis: Set[Label], deps_md: Dict[Label, Dict[Label, int]], p: int) -> Tuple:
    b = tuple(sorted(basis))
    ds = tuple(sorted((d, _signature(md, p)) for d, md in deps_md.items()))
    return (b, ds)


# ----------------------------
# Short-lex enumerator (BFS on Cayley graph)
# ----------------------------
def iterate_shortlex_until_k(
    generators: List[Dict[Label, Label]],
    labels: List[Label],
    checker: Callable[[Dict[Label, Label]], bool],
    *,
    k: Optional[int] = 1,
    max_word_length: Optional[int] = None,
    max_group_size: Optional[int] = None,
) -> List[Dict[Label, Label]]:
    """
    Enumerate distinct permutations in increasing word length over `generators`,
    running `checker` on each; stop after `k` acceptances (or exhaust).
    """
    if not generators:
        return []

    # Dedup generators again (defensive)
    gens = []
    seen = set()
    for g in generators:
        key = _perm_key(g, labels)
        if key not in seen:
            seen.add(key); gens.append(g)

    # BFS
    accepted: List[Dict[Label, Label]] = []
    visited = set(_perm_key(g, labels) for g in gens)
    q = deque((g, 1) for g in gens)

    while q:
        perm, wlen = q.popleft()
        if max_word_length is not None and wlen > max_word_length:
            break

        try:
            ok = checker(perm)
        except Exception:
            ok = False

        if ok:
            accepted.append(perm)
            if k is not None and len(accepted) >= k:
                return accepted

        # expand to next layer
        for g in gens:
            comp = _compose_perm(perm, g)  # perm then g
            key = _perm_key(comp, labels)
            if key not in visited:
                visited.add(key)
                if max_group_size is not None and len(visited) >= max_group_size:
                    return accepted  # safety cutoff
                q.append((comp, wlen + 1))

    return accepted


# ----------------------------
# High-level API
# ----------------------------
def find_permutations_matroid(
    independent: List[Label],
    dependencies: DepPairs,
    checker: Callable[[Dict[Label, Label]], bool],
    *,
    p: int = 2,
    k: int = 1,
    pivot_depth: int = 1,
    max_word_length: Optional[int] = None,
    max_group_size: Optional[int] = None,
    allow_swaps=True, allow_3cycles=True, allow_pivots=True, allow_pivot_then_indep=True,
) -> List[Dict[Label, Label]]:
    """
    Matroid-based pipeline:
      1) Harvest a compact generator set from local (and nearby) presentations.
      2) Enumerate unique permutations in short-lex order until `k` pass `checker`.
    """
    gens, labels = harvest_generators(
        independent, dependencies, p=p, pivot_depth=pivot_depth,
        allow_swaps=allow_swaps, allow_3cycles=allow_3cycles,
        allow_pivots=allow_pivots, allow_pivot_then_indep=allow_pivot_then_indep,
    )
    return iterate_shortlex_until_k(
        gens, labels, checker, k=k, max_word_length=max_word_length, max_group_size=max_group_size
    )


def restricted_cycles(
    perm: dict[int, int],
    independent: list[int],
    dependencies: dict[int, list[tuple[int, int]]],
    *,
    include_full_cycles: bool = False,
):
    """
    Return cycle decompositions restricted to independents and dependents separately.
    Optionally include the full (mixed) cycle decomposition too.
    """
    indep_set = set(independent)
    dep_set = set(dependencies.keys())

    indep_cycles = perm_to_cycles(perm, domain=indep_set)
    dep_cycles = perm_to_cycles(perm, domain=dep_set)

    out = {
        "independent_cycles": indep_cycles,
        "dependent_cycles": dep_cycles,
    }
    if include_full_cycles:
        all_labels = indep_set | dep_set
        out["full_cycles"] = perm_to_cycles(perm, domain=all_labels)
    return out


# ------------------------- REPORTS -------------------------
def format_perm_report(
    perm: dict[int, int],
    independent: list[int],
    dependencies: dict[int, list[tuple[int, int]]],
    *,
    include_full_cycles: bool = False,
) -> dict:
    """
    Convenience wrapper: returns a dict with both the mapping and cycle summaries,
    with indep/dep cycles restricted to their respective domains.
    """
    cyc = restricted_cycles(perm, independent, dependencies, include_full_cycles=include_full_cycles)
    return {
        "perm_map": perm,                      # explicit mapping dict
        "independent_cycles": cyc["independent_cycles"],
        "dependent_cycles": cyc["dependent_cycles"],
        **({"full_cycles": cyc["full_cycles"]} if include_full_cycles else {}),
    }


# =========================
# TEST HELPERS (GF(p))
# =========================


def _build_vectors_original_basis(independent: List[int],
                                  dependencies: Dict[int, List[Tuple[int, int]]],
                                  p: int) -> Dict[int, Tuple[int, ...]]:
    """
    For each label ℓ in independent ∪ dependencies, return its vector coordinates
    in the ORIGINAL basis (sorted 'independent') over GF(p).
    """
    basis = sorted(independent)
    k = len(basis)
    idx = {b: i for i, b in enumerate(basis)}

    vecs: Dict[int, Tuple[int, ...]] = {}

    # Independents = unit vectors
    for b in basis:
        v = [0] * k
        v[idx[b]] = 1
        vecs[b] = tuple(v)

    # Dependents = linear combos of unit basis
    for d, pairs in dependencies.items():
        v = [0] * k
        for j, m in pairs:
            v[idx[j]] = (v[idx[j]] + (m % p)) % p
        vecs[d] = tuple(v)

    return vecs


def _apply_perm_to_presentation(independent: List[int],
                                dependencies: Dict[int, List[Tuple[int, int]]],
                                perm: Dict[int, int]) -> Tuple[List[int], Dict[int, List[Tuple[int, int]]]]:
    """
    Apply label permutation to the presentation:
      - independents are relabeled
      - dependent equations are relabeled on both LHS and RHS.
    """
    indep2 = [perm[i] for i in independent]
    deps2: Dict[int, List[Tuple[int, int]]] = {}
    for d, pairs in dependencies.items():
        d2 = perm[d]
        acc = defaultdict(int)
        for j, m in pairs:
            acc[perm[j]] = (acc[perm[j]] + m)
        # keep integer multiplicities; mod p happens when building vectors
        deps2[d2] = sorted(acc.items(), key=lambda x: x[0])
    return indep2, deps2
