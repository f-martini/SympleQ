import numpy as np
import galois
from quaos.core.circuits.utils import is_symplectic

# The following code is taken from https://arxiv.org/pdf/1406.2170


def direct_sum(m1, m2):
    n1 = len(m1[0])
    n2 = len(m2[0])
    output = np.zeros((n1 + n2, n1 + n2), dtype=np.int8)
    for i in range(0, n1):
        for j in range(0, n1):
            output[i, j] = m1[i, j]
    for i in range(0, n2):
        for j in range(0, n2):
            output[i + n1, j + n1] = m2[i, j]
    return output


def inner(v, w):  # symplectic inner product
    t = 0
    for i in range(0, np.size(v) >> 1):
        t += v[2 * i] * w[2 * i + 1]
        t += w[2 * i] * v[2 * i + 1]
    return t % 2


def transvection(k, v):  # applies transvection Z k to v
    return (v + inner(k, v) * k) % 2


def int2bits(i, n):  # converts integer i to an length n array of bits
    output = np.zeros(n, dtype=np.int8)
    for j in range(0, n):
        output[j] = i & 1
        i >>= 1
    return output


def find_transvection(x, y):  # finds h1, h2 such that y = Z_h1 Z_h2 x
    # Lemma 2 it the text
    # # Note that if only one transvection i s required output [1] will be # zero and applying the all −zero
    # transvection does nothing .
    output = np.zeros((2, np.size(x)), dtype=np.int8)
    if np.all(x == y):
        return output
    if inner(x, y) == 1:
        output[0] = (x + y) % 2
        return output
    # find a pair where they are both not 00

    z = np.zeros(np.size(x))
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            z[ii] = (x[ii] + y[ii]) % 2
            z[ii + 1] = (x[ii + 1] + y[ii + 1]) % 2
            if (z[ii] + z[ii + 1]) == 0:  # they were the same so they added to 00
                z[ii + 1] = 1
                if x[ii] != x[ii + 1]:
                    z[ii] = 1
            output[0] = (x + z) % 2
            output[1] = (y + z) % 2
            return output
    # didn’t find a pair # so look for two places where x has 00 and y doesn’t , and vice versa #
    # first y==00 and x doesn’t
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) == 0):  # found the pair
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break
    # finally x==00 and y doesn’t
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) == 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break
    output[0] = (x + z) % 2
    output[1] = (y + z) % 2
    return output


def symplectic(i, n):
    """
    Returns a symplectic canonical matrix of size 2n x 2n, following a recursive construction.
    Note: the transpose of the symplectic matrix is returned, differing from the convention in some texts.
    """
    nn = 2 * n

    # Step 1
    s = (1 << nn) - 1
    k = (i % s) + 1
    i //= s  # integer division

    # Step 2
    f1 = int2bits(k, nn)

    # Step 3
    e1 = np.zeros(nn, dtype=np.int8)
    e1[0] = 1
    T = find_transvection(e1, f1)

    # Step 4
    bits = int2bits(i % (1 << (nn - 1)), nn - 1)

    # Step 5
    e_prime = np.copy(e1)
    for j in range(2, nn):
        e_prime[j] = bits[j - 1]
    h0 = transvection(T[0], e_prime)
    h0 = transvection(T[1], h0)

    # Step 6
    if bits[0] == 1:
        f1 *= 0  # zero it out

    # Step 7
    id2 = np.zeros((2, 2), dtype=np.int8)
    id2[0, 0] = 1
    id2[1, 1] = 1

    if n != 1:
        g = direct_sum(id2, symplectic(i >> (nn - 1), n - 1))
    else:
        g = id2

    for j in range(nn):
        g[j] = transvection(T[0], g[j])
        g[j] = transvection(T[1], g[j])
        g[j] = transvection(h0, g[j])
        g[j] = transvection(f1, g[j])

    return g


###########


# ------------------------ Field-Agnostic Helpers ------------------------
def int2digits_gf(i, n, p):
    digits = []
    for _ in range(n):
        digits.append(i % p)
        i //= p
    return np.array(digits, dtype=np.uint8) if p == 2 else galois.GF(p)(digits)

# ------------------------ GF(2) Optimized Path ------------------------


def inner_mod2(v, w):
    return np.sum(v[0::2] * w[1::2] + v[1::2] * w[0::2]) % 2


def transvection_mod2(k, v):
    return (v + inner_mod2(k, v) * k) % 2


def find_transvection_mod2(x, y):
    n = len(x)
    out = np.zeros((2, n), dtype=np.uint8)
    if np.all(x == y):
        return out
    if inner_mod2(x, y) == 1:
        out[0] = (x + y) % 2
        return out

    z = np.zeros(n, dtype=np.uint8)
    for i in range(n // 2):
        ii = 2 * i
        if (x[ii] | x[ii + 1]) and (y[ii] | y[ii + 1]):
            z[ii] = (x[ii] + y[ii]) % 2
            z[ii + 1] = (x[ii + 1] + y[ii + 1]) % 2
            if not (z[ii] | z[ii + 1]):
                z[ii + 1] = 1
                if x[ii] != x[ii + 1]:
                    z[ii] = 1
            out[0] = (x + z) % 2
            out[1] = (y + z) % 2
            return out

    for i in range(n // 2):
        ii = 2 * i
        if (x[ii] | x[ii + 1]) and not (y[ii] | y[ii + 1]):
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break

    for i in range(n // 2):
        ii = 2 * i
        if not (x[ii] | x[ii + 1]) and (y[ii] | y[ii + 1]):
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break

    out[0] = (x + z) % 2
    out[1] = (y + z) % 2
    return out


def direct_sum_mod2(A, B):
    m, n = A.shape[0], B.shape[0]
    out = np.zeros((m + n, m + n), dtype=np.uint8)
    out[:m, :m] = A
    out[m:, m:] = B
    return out


def symplectic_mod2(i, n):
    nn = 2 * n
    s = (1 << nn) - 1
    k = (i % s) + 1
    i //= s

    f1 = int2digits_gf(k, nn, 2)
    e1 = np.zeros(nn, dtype=np.uint8)
    e1[0] = 1
    T = find_transvection_mod2(e1, f1)

    bits = int2digits_gf(i % (1 << (nn - 1)), nn - 1, 2)
    e_prime = e1.copy()
    e_prime[2:] = bits[1:]
    h0 = transvection_mod2(T[0], e_prime)
    h0 = transvection_mod2(T[1], h0)

    if bits[0] == 1:
        f1 *= 0

    id2 = np.eye(2, dtype=np.uint8)
    g = direct_sum_mod2(id2, symplectic_mod2(i >> (nn - 1), n - 1)) if n > 1 else id2

    for j in range(nn):
        g[j] = transvection_mod2(T[0], g[j])
        g[j] = transvection_mod2(T[1], g[j])
        g[j] = transvection_mod2(h0, g[j])
        g[j] = transvection_mod2(f1, g[j])
    return g

# ------------------------ GF(p) General Path ------------------------


def inner_gf(v, w):
    return (v[0::2] * w[1::2] + v[1::2] * w[0::2]).sum()


def transvection_gf(k, v):
    inner = inner_gf(k, v)  # should already be in GF
    return v + inner * k


def find_transvection_gf(x, y):
    GF = type(x)
    n = len(x)
    out = GF.Zeros((2, n))
    if np.all(x == y):
        return out
    if inner_gf(x, y) != 0:
        out[0] = x + y
        return out

    z = GF.Zeros(n)
    for i in range(n // 2):
        ii = 2 * i
        if (x[ii] != 0 or x[ii + 1] != 0) and (y[ii] != 0 or y[ii + 1] != 0):
            z[ii] = x[ii] + y[ii]
            z[ii + 1] = x[ii + 1] + y[ii + 1]
            if z[ii] == 0 and z[ii + 1] == 0:
                z[ii + 1] = GF(1)
                if x[ii] != x[ii + 1]:
                    z[ii] = GF(1)
            out[0] = x + z
            out[1] = y + z
            return out

    for i in range(n // 2):
        ii = 2 * i
        if (x[ii] != 0 or x[ii + 1] != 0) and (y[ii] == 0 and y[ii + 1] == 0):
            if x[ii] == x[ii + 1]:
                z[ii + 1] = GF(1)
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break

    for i in range(n // 2):
        ii = 2 * i
        if (x[ii] == 0 and x[ii + 1] == 0) and (y[ii] != 0 or y[ii + 1] != 0):
            if y[ii] == y[ii + 1]:
                z[ii + 1] = GF(1)
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break

    out[0] = x + z
    out[1] = y + z
    return out


def direct_sum_gf(A, B, GF):
    m, n = A.shape[0], B.shape[0]
    out = GF.Zeros((m + n, m + n))
    out[:m, :m] = A
    out[m:, m:] = B
    return out


def symplectic_from_index(index: int, n_qudits: int, field: int) -> np.ndarray:

    GF = galois.GF(field)

    if GF.characteristic == 2:
        return symplectic_mod2(index, n_qudits)

    nn = 2 * n_qudits
    p = GF.characteristic
    s = p ** nn - 1
    k = (index % s) + 1
    index //= s

    f1 = int2digits_gf(k, nn, p)
    f1 = GF(f1)
    e1 = GF.Zeros(nn)
    e1[0] = GF(1)
    T = find_transvection_gf(e1, f1)

    bits = int2digits_gf(index % (p ** (nn - 1)), nn - 1, p)
    bits = GF(bits)
    e_prime = e1.copy()
    e_prime[2:] = bits[1:]
    h0 = transvection_gf(T[0], e_prime)
    h0 = transvection_gf(T[1], h0)

    if bits[0] != 0:
        f1 = GF.Zeros(nn)

    id2 = GF.Identity(2)
    g = direct_sum_gf(id2, symplectic_from_index(index >> (nn - 1), n_qudits - 1, p), GF) if n_qudits > 1 else id2

    for j in range(nn):
        g[j] = transvection_gf(T[0], g[j])
        g[j] = transvection_gf(T[1], g[j])
        g[j] = transvection_gf(h0, g[j])
        g[j] = transvection_gf(f1, g[j])

    return g


def bits2int_gf(b_gf):
    p = type(b_gf).characteristic
    # Force to plain NumPy array first (as integers), then do dot product
    b_np = np.array(b_gf, dtype=np.int64)
    powers = np.array([p ** i for i in range(len(b_gf))], dtype=np.int64)
    return int(np.dot(b_np, powers))


def number_of_cosets(n):
    return (2 ** (2 * n - 1)) * (2 ** (2 * n) - 1)


def number_of_symplectics(n_qudits):
    # needs GF(p) implementation of number of cosets, fine for GF(2) for now
    x = 1
    for j in range(1, n_qudits + 1):
        x *= number_of_cosets(j)
    return x


def index_from_symplectic(n_qudits: int, symplectic: np.ndarray | galois.FieldArray, dimension: int) -> int:
    """
    Returns the index of a symplectic matrix gn_gf ∈ Sp(2n, GF(p)) under canonical enumeration.
    Assumes gn_gf is a FieldArray from galois.GF(p).
    """

    if isinstance(symplectic, np.ndarray):
        symplectic = galois.GF(dimension)(symplectic)

    GF = type(symplectic)
    nn = 2 * n_qudits

    # Step 1
    v_g = symplectic[0]
    w_g = symplectic[1]

    # Step 2
    e1_g = GF.Zeros(nn)
    e1_g[0] = 1
    T_g = find_transvection_gf(v_g, e1_g)

    # Step 3
    tw_g = transvection_gf(T_g[0], w_g)
    tw_g = transvection_gf(T_g[1], tw_g)

    b_g = tw_g[0]
    h0_g = GF.Zeros(nn)
    h0_g[0] = 1
    h0_g[1] = 0
    h0_g[2:] = tw_g[2:]

    # Step 4
    bb_g = GF.Zeros(nn - 1)
    bb_g[0] = b_g
    bb_g[1:] = tw_g[2:]

    zv = bits2int_gf(v_g) - 1
    zw = bits2int_gf(bb_g)

    p = GF.characteristic
    cvw = zw * (p ** (2 * n_qudits) - 1) + zv

    if n_qudits == 1:
        return cvw

    g_prime_gf = symplectic.copy()
    # if b_g == 0:
    #     for j in range(nn):
    #         g_prime_gf[j] = transvection_gf(T_g[0], g_prime_gf[j])
    #         g_prime_gf[j] = transvection_gf(T_g[1], g_prime_gf[j])
    #         g_prime_gf[j] = transvection_gf(h0_g, g_prime_gf[j])
    #         g_prime_gf[j] = transvection_gf(e1_g, g_prime_gf[j])
    # else:
    #     for j in range(nn):
    #         g_prime_gf[j] = transvection_gf(T_g[0], g_prime_gf[j])
    #         g_prime_gf[j] = transvection_gf(T_g[1], g_prime_gf[j])
    #         g_prime_gf[j] = transvection_gf(h0_g, g_prime_gf[j])

    if b_g == 0:
        T_total = (
            transvection_matrix(e1_g)
            @ transvection_matrix(h0_g)
            @ transvection_matrix(T_g[1])
            @ transvection_matrix(T_g[0])
        )
    else:
        T_total = (
            transvection_matrix(h0_g)
            @ transvection_matrix(T_g[1])
            @ transvection_matrix(T_g[0])
        )

    g_prime_gf = T_total @ g_prime_gf

    # Step 7: recurse
    g_new_gf = g_prime_gf[2:, 2:]
    return index_from_symplectic(n_qudits - 1, g_new_gf, dimension) * number_of_cosets(n_qudits) + cvw


def test_symplectic_index_inverse(p: int, n: int, max_tests: int = 100):

    GF = galois.GF(p)
    N = number_of_symplectics(n)
    if max_tests is not None:
        N = min(N, max_tests)

    print(f"Testing symplectic inversion over GF({p}) with n={n}, total cases = {N}")

    for i in range(N):
        g = symplectic_from_index(i, n, p)
        idx = index_from_symplectic(n, GF(g), p)
        assert idx == i, f"Test failed: index {i} was inverted to {idx}"

    print(f"✅ All {N} tests passed for GF({p}) and n={n}")


def random_symplectic(n: int, p: int, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    if p != 2:
        raise NotImplementedError("Only implemented for GF(2) - just the number of symplectics needs to be updated")
    return symplectic_from_index(np.random.randint(0, number_of_symplectics(n) - 1), n, p)


def construct_omega(n, p):
    GF = galois.GF(p)
    Id = GF.Identity(n)
    Omega = GF.Zeros((2 * n, 2 * n))
    Omega[:n, n:] = Id
    Omega[n:, :n] = Id if GF.characteristic == 2 else -Id
    return Omega


def test_inner_gf():
    GF = galois.GF(2)
    v = GF([1, 0, 1, 1])
    w = GF([0, 1, 1, 1])
    half = len(v) // 2
    inner = (v[0::2][:half] * w[1::2][:half] + v[1::2][:half] * w[0::2][:half]).sum()
    assert isinstance(inner, GF), "Inner product result not in field"


def transvection_matrix(v):
    GF = type(v)
    n = len(v) // 2
    Omega = construct_omega(n, GF.characteristic)

    v = v.reshape((-1, 1))  # column
    T = GF.Identity(2 * n) + Omega @ (v @ v.T)
    return T


def test_transvection_preserves_symplecticity():
    GF = galois.GF(2)
    n = 2
    nn = 2 * n
    v = GF.Random(nn)

    print(v)
    F = GF.Identity(nn)
    for j in range(nn):
        F[j] = transvection_gf(v, F[j])
    # F = transvection_matrix(v)
    # print(F)

    assert is_symplectic(F, 2), "Transvection did not preserve symplecticity"


if __name__ == '__main__':
    # loc_sym = [1, 2, 3, 4]
    # print(embed_symplectic_single_pauli_string(loc_sym, [1, 3], 5))
    import time
    ts = 0
    ts_g = 0
    wrong = 0
    not_identical = 0

    n = 5

    dim = 2
    for i in range(100):
        t_0 = time.time()
        index = np.random.randint(0, number_of_symplectics(n) - 1)
        s = symplectic(index, n)
        t_1 = time.time()
        ts += t_1 - t_0
        t_0 = time.time()
        s_g = symplectic_from_index(index, n, dim)
        if not is_symplectic(galois.GF(dim)(s_g), dim):
            wrong += 1
            # print(s_g)
        t_1 = time.time()
        ts_g += t_1 - t_0
        if not np.array_equal(s, s_g):
            not_identical += 1

    print(wrong, not_identical)

    # print(ts, ts_g)
    # test_symplectic_index_inverse(p=2, n=2)
    # for _ in range(100):
    #     print(_)
    #     test_inner_gf()
    #     test_transvection_preserves_symplecticity()
    # print('Done')

    # test_symplectic_index_inverse(p=3, n=2, max_tests=100)
