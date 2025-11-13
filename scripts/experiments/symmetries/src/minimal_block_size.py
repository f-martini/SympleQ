import numpy as np
from typing import Dict, List, Tuple
from .modular_helpers import mod_p, rank_mod, _solve_linear, nullspace_mod


# --------------------------
# Polynomial arithmetic over GF(p): coeffs low->high, monic when expected
# --------------------------

def poly_trim(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    i = len(a) - 1
    while i > 0 and a[i] == 0:
        i -= 1
    return a[:i + 1]


def poly_monic(a: np.ndarray, p: int) -> np.ndarray:
    a = mod_p(poly_trim(a), p)
    if a.size == 0 or (a.size == 1 and a[0] == 0):
        return a
    inv = pow(int(a[-1]), p - 2, p)
    return mod_p(a * inv, p)


def poly_add(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    n = max(len(a), len(b))
    c = np.zeros(n, dtype=np.int64)
    c[:len(a)] += a
    c[:len(b)] += b
    return mod_p(poly_trim(c), p)


def poly_sub(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    n = max(len(a), len(b))
    c = np.zeros(n, dtype=np.int64)
    c[:len(a)] += a
    c[:len(b)] -= b
    return mod_p(poly_trim(c), p)


def poly_mul(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros(1, dtype=np.int64)
    c = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        c[i:i + len(b)] += ai * b
    return mod_p(poly_trim(c), p)


def poly_divmod(a: np.ndarray, b: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    # divide a by b (monic b is best)
    a = mod_p(poly_trim(a.copy()), p)
    b = mod_p(poly_trim(b.copy()), p)
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError("poly div by 0")
    if len(a) < len(b):
        return np.zeros(1, dtype=np.int64), a
    inv_lead = pow(int(b[-1]), p - 2, p)
    q = np.zeros(len(a) - len(b) + 1, dtype=np.int64)
    r = a.copy()
    while len(r) >= len(b) and not (len(r) == 1 and r[0] == 0):
        k = len(r) - len(b)
        c = mod_p(r[-1] * inv_lead, p)
        q[k] = c
        r[k: k + len(b)] -= c * b
        r = mod_p(poly_trim(r), p)
        if len(r) == 0:
            r = np.zeros(1, dtype=np.int64)
            break
        if len(r) < len(b):
            break
    return mod_p(poly_trim(q), p), mod_p(poly_trim(r), p)


def poly_mod(a: np.ndarray, m: np.ndarray, p: int) -> np.ndarray:
    _, r = poly_divmod(a, m, p)
    return r


def poly_gcd(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    a = mod_p(poly_trim(a), p)
    b = mod_p(poly_trim(b), p)
    while not (len(b) == 1 and b[0] == 0):
        _, r = poly_divmod(a, b, p)
        a, b = b, r
    return poly_monic(a, p)


def poly_pow_mod(a: np.ndarray, e: int, m: np.ndarray, p: int) -> np.ndarray:
    # exponentiation mod m (poly)
    res = np.array([1], dtype=np.int64)
    base = a.copy()
    while e > 0:
        if e & 1:
            res = poly_mod(poly_mul(res, base, p), m, p)
        base = poly_mod(poly_mul(base, base, p), m, p)
        e >>= 1
    return poly_monic(res, p)


def poly_eval_matrix(F: np.ndarray, poly: np.ndarray, p: int) -> np.ndarray:
    # Horner: poly(F) = a0 I + a1 F + ...; poly is low->high
    n2 = F.shape[0]
    M = np.zeros_like(F)
    P_pow = np.eye(n2, dtype=np.int64)
    for a in poly:
        if a != 0:
            M = mod_p(M + a * P_pow, p)
        P_pow = mod_p(P_pow @ F, p)
    return M


def poly_reciprocal(poly: np.ndarray, p: int) -> np.ndarray:
    poly = poly_trim(poly)
    rev = poly[::-1].copy()
    return poly_monic(rev, p)

# --------------------------
# Distinct-degree + equal-degree factorization (Cantor–Zassenhaus)
# --------------------------


def squarefree_decomposition(f: np.ndarray, p: int) -> List[Tuple[np.ndarray, int]]:
    # returns [(f1,e1), ...] with fi squarefree, pairwise coprime, f = prod fi^ei
    f = poly_monic(f, p)
    df = derivative_poly(f, p)
    if np.all(df == 0):
        # f = g(t^p), extract p-th roots
        g = poly_pth_root(f, p)
        sub = squarefree_decomposition(g, p)
        return [(h, e * p) for (h, e) in sub]
    g = poly_gcd(f, df, p)
    w, _ = poly_divmod(f, g, p)
    i, res = 1, []
    while not (len(w) == 1 and w[0] == 0):
        y = poly_gcd(w, g, p)
        fi, _ = poly_divmod(w, y, p)
        if not (len(fi) == 1 and fi[0] == 1):
            res.append((fi, i))
        w = y
        i += 1
        if len(g) == 1 and g[0] == 1:
            break
        g = poly_mod(g, np.array([0, 1], dtype=np.int64), p)  # noop to keep type
    if len(g) != 1 or g[0] != 1:
        # leftover perfect pth power
        g_root = poly_pth_root(g, p)
        sub = squarefree_decomposition(g_root, p)
        res += [(h, e * p) for (h, e) in sub]
    return res


def derivative_poly(f: np.ndarray, p: int) -> np.ndarray:
    if len(f) <= 1:
        return np.zeros(1, dtype=np.int64)
    df = np.array([(i * f[i]) % p for i in range(1, len(f))], dtype=np.int64)
    return poly_trim(df)


def poly_pth_root(f: np.ndarray, p: int) -> np.ndarray:
    # assumes f(x) = g(x^p); pull out pth root of exponents
    g = np.zeros((len(f) + p - 1) // p, dtype=np.int64)
    for i in range(0, len(f), p):
        g[i // p] = f[i]
    return poly_trim(g)


def distinct_degree_factorization(f: np.ndarray, p: int) -> List[Tuple[np.ndarray, int]]:
    # input squarefree monic f; returns [(h_d, d)] where h_d = product of irreducibles of degree d
    f = poly_monic(f, p)
    res = []
    x = np.array([0, 1], dtype=np.int64)   # x
    h = x.copy()
    n = len(f) - 1
    for d in range(1, n + 1):
        # h <- h^{p} mod f
        h = poly_pow_mod(h, p, f, p)
        g = poly_gcd(poly_sub(h, x, p), f, p)
        if len(g) > 1 and not np.array_equal(g, np.array([1], dtype=np.int64)) and not np.array_equal(g, f):
            res.append((poly_monic(g, p), d))
            q, r = poly_divmod(f, g, p)
            assert np.all(r == 0)
            f = poly_monic(q, p)
            h = poly_mod(h, f, p)
        if len(f) <= 1 or np.array_equal(f, np.array([1], dtype=np.int64)):
            break
    if len(f) > 1 and not np.array_equal(f, np.array([1], dtype=np.int64)):
        # whatever remains is a product of equal degree factors larger than loop range
        res.append((f, len(f) - 1))
    return res


def equal_degree_factorization(f: np.ndarray, d: int, p: int, rng=np.random.default_rng(2025)) -> List[np.ndarray]:
    # factor monic squarefree f that is a product of irreducibles all of degree d
    if len(f) - 1 == d:
        return [f]
    while True:
        # pick random a(x) of deg < deg f
        a = mod_p(rng.integers(0, p, size=len(f) - 1, dtype=np.int64), p)
        a = poly_trim(a)
        if len(a) == 0:
            a = np.array([1], dtype=np.int64)
        # compute g = gcd(a^{(p^d - 1)/2} - 1, f)
        exp = (p**d - 1) // 2
        a_pow = poly_pow_mod(a, exp, f, p)
        g = poly_gcd(poly_sub(a_pow, np.array([1], dtype=np.int64), p), f, p)
        if len(g) > 1 and not np.array_equal(g, f):
            return (equal_degree_factorization(g, d, p, rng) +
                    equal_degree_factorization(poly_monic(poly_divmod(f, g, p)[0], p), d, p, rng))


def factor_poly_over_fp(f: np.ndarray, p: int) -> List[np.ndarray]:
    # return list of monic irreducible factors (with multiplicity via repeats)
    f = poly_monic(f, p)
    if len(f) <= 1:
        return [f]
    sq = squarefree_decomposition(f, p)  # [(fi, ei)]
    out = []
    for fi, ei in sq:
        dd = distinct_degree_factorization(fi, p)  # [(h_d, d)]
        for hd, d in dd:
            parts = equal_degree_factorization(hd, d, p)
            for _ in range(ei):
                out.extend(parts)
    return [poly_monic(h, p) for h in out]

# --------------------------
# Krylov cyclic decomposition: companion polynomials of cyclic blocks
# --------------------------


def minimal_poly_for_vector(F: np.ndarray, v: np.ndarray, p: int) -> np.ndarray:
    """Return the monic minimal polynomial m_v(t) of F restricted to the cyclic subspace generated by v."""
    v = mod_p(v.reshape(-1, 1), p)
    # Build Krylov [v, Fv, ..., F^k v] until dependent; solve relation
    V = v.copy()
    powers = [v]
    while True:
        r = rank_mod(V, p)
        w = mod_p(F @ powers[-1], p)
        candidate = np.concatenate([V, w], axis=1)
        r2 = rank_mod(candidate, p)
        if r2 == r:
            # find coefficients c s.t. sum_{i=0}^{k} c_i F^i v = 0 with c_k = 1 (monic)
            # Solve V c = -w, where columns of V are [v, Fv, ..., F^{k-1}v]
            c = _solve_linear(mod_p(V, p), mod_p(-w, p), p)  # particular
            coeffs = np.concatenate([c.reshape(-1), np.array([1], dtype=np.int64)])
            return poly_monic(coeffs, p)
        V = candidate
        powers.append(w)


def cyclic_decomposition(F: np.ndarray, p: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (basis_block, m(t)) where basis_block columns span an F-invariant cyclic subspace
    and m(t) is its monic companion polynomial on that subspace.
    Blocks are disjoint and together span the whole space.
    Construction is projection-free: we only use rank tests to keep new independent columns.
    """
    n2 = F.shape[0]
    basis_blocks: List[np.ndarray] = []
    polys: List[np.ndarray] = []

    # Global accumulated basis
    B = np.zeros((n2, 0), dtype=np.int64)

    # Deterministic seeds: std basis + a few fixed randoms
    seeds = [np.eye(n2, dtype=np.int64)[:, i: i + 1] for i in range(n2)]
    rng = np.random.default_rng(2025)
    seeds += [rng.integers(0, p, size=(n2, 1), dtype=np.int64) for _ in range(min(8, n2))]

    s_idx = 0
    while B.shape[1] < n2:
        # Get or make a seed
        if s_idx < len(seeds):
            v = mod_p(seeds[s_idx], p)
            s_idx += 1
        else:
            v = rng.integers(0, p, size=(n2, 1), dtype=np.int64)

        # If v adds nothing, skip
        if rank_mod(np.concatenate([B, v], axis=1), p) == B.shape[1]:
            continue

        # Compute minimal polynomial on the cyclic subspace of v (ignores B, as it should)
        m_v = minimal_poly_for_vector(F, v, p)
        deg = len(m_v) - 1

        # Build a block basis by walking powers and only keeping independent columns
        block_cols = []
        C = B.copy()  # local accumulator = current global basis + this block
        w = v.copy()
        # We will attempt up to (deg + n2) steps to be robust if some powers land in span(C)
        steps_cap = deg + n2
        steps = 0
        while steps < steps_cap and len(block_cols) < deg:
            steps += 1
            if rank_mod(np.concatenate([C, w], axis=1), p) > C.shape[1]:
                block_cols.append(w.copy())
                C = np.concatenate([C, w], axis=1)
            w = mod_p(F @ w, p)

        # If we failed to collect anything (extremely unlikely), skip this seed
        if not block_cols:
            continue

        block = np.concatenate(block_cols, axis=1)
        B = np.concatenate([B, block], axis=1)
        basis_blocks.append(block)
        polys.append(m_v)

    # Sanity: we spanned all
    assert B.shape[1] == n2, "Cyclic decomposition failed to span the space"
    return list(zip(basis_blocks, polys))


# --------------------------
# Primary components via irreducible factors and exponents
# --------------------------

def primary_components_from_cyclic(F: np.ndarray, p: int) -> Dict:
    """
    Build primary components V^(q) for irreducible q(t) and sector grouping.
    """
    n2 = F.shape[0]
    cyclic_blocks = cyclic_decomposition(F, p)  # [(B_i, m_i(t))]
    # collect irreducibles + exponents
    mult: Dict[Tuple[int, ...], int] = {}   # key is tuple of coeffs
    irreducible_polys: Dict[Tuple[int, ...], np.ndarray] = {}
    for _, m_i in cyclic_blocks:
        factors = factor_poly_over_fp(m_i, p)  # list with multiplicities
        # m_i may include powers; multiplicities handled by repeats above
        # compute exponent per irreducible = max power over blocks
        # count multiplicity of each factor within m_i
        # (we can compute v_q(m_i) by dividing repeatedly)
        # unique irreducibles for this m_i
        uniq = {}
        for q in factors:
            key = tuple(q.tolist())
            irreducible_polys[key] = q
            uniq.setdefault(key, 0)
            uniq[key] += 1
        for key, e in uniq.items():
            mult[key] = max(mult.get(key, 0), e)

    # build V^(q) = ker (q(F)^{m_q})
    primaries: Dict[Tuple[int, ...], Dict] = {}
    for key, e_max in mult.items():
        q = irreducible_polys[key]
        Mq = poly_eval_matrix(F, q, p)
        # power by repeated squaring at matrix level
        P = np.eye(n2, dtype=np.int64)
        base = Mq.copy()
        e = e_max
        while e > 0:
            if e & 1:
                P = mod_p(P @ base, p)
            base = mod_p(base @ base, p)
            e >>= 1
        # nullspace over GF(p)
        Vq = nullspace_mod(P, p)  # n2 x dim
        primaries[key] = {
            "poly": q,
            "deg": len(q) - 1,
            "exponent": e_max,
            "V_basis": Vq
        }

    # reciprocal pairing
    keys = list(primaries.keys())
    for key in keys:
        q = primaries[key]["poly"]
        q_star = poly_reciprocal(q, p)
        k_star = tuple(q_star.tolist())
        primaries[key]["reciprocal_key"] = k_star
        primaries[key]["self_reciprocal"] = (k_star == key)

    # sectors
    used = set()
    sectors = []
    for key in primaries.keys():
        if key in used:
            continue
        data = primaries[key]
        if data["self_reciprocal"]:
            W = data["V_basis"]
            half_floor = data["deg"] * data["exponent"]
            sectors.append({
                "type": "self",
                "q_key": key,
                "W_basis": W,
                "half_dim_floor": half_floor
            })
            used.add(key)
        else:
            k_star = data["reciprocal_key"]
            if k_star not in primaries:
                # If the reciprocal irreducible doesn't occur, the pairing space is just V^(q)
                # (this can happen if invariant factors contain only q but not q*, but for symplectic F
                #  the reciprocal partner should appear; still, be robust.)
                W = data["V_basis"]
                half_floor = data["deg"] * data["exponent"]
                sectors.append({
                    "type": "self",  # fallback
                    "q_key": key,
                    "W_basis": W,
                    "half_dim_floor": half_floor
                })
                used.add(key)
            else:
                W = np.concatenate([data["V_basis"], primaries[k_star]["V_basis"]], axis=1)
                half_floor = data["deg"] * data["exponent"]
                sectors.append({
                    "type": "paired",
                    "q_key": key,
                    "qstar_key": k_star,
                    "W_basis": W,
                    "half_dim_floor": half_floor
                })
                used.add(key)
                used.add(k_star)

    L_min_star = max([sec["half_dim_floor"] for sec in sectors] + [0])

    return {
        "irreducibles": primaries,
        "sectors": sectors,
        "Lmin_star": int(L_min_star)
    }

# --------------------------
# Public entry
# --------------------------


def rcf_prepass(F: np.ndarray, p: int) -> Dict:
    """
    Deterministic structural pre-pass for symplectic F over GF(p).
    - Computes a cyclic (Frobenius) decomposition via Krylov subspaces.
    - Factors each block polynomial over GF(p).
    - Assembles primary components V^(q) = ker (q(F)^{m_q}).
    - Pairs reciprocals q and q* into sectors W_q (or self-recognizes q=q*).
    Returns a structure with per-irreducible metadata, sector bases, and the theoretical
    half-dimension floor L_min* for the largest block.
    """
    assert F.shape[0] == F.shape[1], "F must be square"
    return primary_components_from_cyclic(F, p)
