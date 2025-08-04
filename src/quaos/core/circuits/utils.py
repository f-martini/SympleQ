# TODO: Refine quaos.circuit_utils and move here
from math import gcd
import numpy as np
import galois


def solve_modular_linear(x: int, z: int, d: int) -> int:
    """
    Find smallest non-negative integer n such that (x + z*n) % d == 0.
    Returns None if no solution exists
    """
    x, z, d = int(x), int(z), int(d)
    g = gcd(z, d)
    if x % g != 0:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    # Reduce the equation modulo d // g
    z_ = z // g
    d_ = d // g
    x_ = (-x // g) % d_

    # compute modular inverse
    try:
        z_inv = pow(z_, -1, d_)
    except ValueError:
        raise ValueError(f"No solution of (x + z*n) % d == 0 exists for x ={x}, z ={z}, d ={d}")

    n = (x_ * z_inv) % d_
    return n


def is_symplectic(symplectic: np.ndarray, dimension: int = 2) -> bool:
    """
    Check if the gate is symplectic, i.e., it preserves the symplectic structure of the Pauli group.
    """

    n_q = symplectic.shape[0] // 2

    id = np.eye(n_q, dtype=int)
    J = np.zeros([2 * n_q, 2 * n_q], dtype=int)
    J[n_q:, :n_q] = id
    J[:n_q, n_q:] = id

    return bool(np.all(J == symplectic.T @ J @ symplectic % dimension))


def embed_symplectic_single_pauli_string(symplectic_local, qudit_indices, n_qudits):
    """Embed a local Clifford (F, h) into full 2n space"""
    full_F = np.zeros(2 * n_qudits, dtype=int)

    n_qudits_local = len(qudit_indices)

    if 2 * n_qudits_local != len(symplectic_local):
        raise ValueError(
            f"symplectic_local must have 2 * n_qudits_local = {n_qudits_local}, but has length {len(symplectic_local)}")

    x_in = symplectic_local[:n_qudits_local]
    z_in = symplectic_local[n_qudits_local:]

    x_image = np.zeros(n_qudits, dtype=int)
    z_image = np.zeros(n_qudits, dtype=int)

    for i in range(n_qudits_local):
        x_image[qudit_indices[i]] = x_in[i]
        z_image[qudit_indices[i]] = z_in[i]

    full_F[:n_qudits] = x_image
    full_F[n_qudits:] = z_image

    return full_F


def embed_symplectic(symplectic_local, phase_vector_local, qudit_indices, n_qudits, dimension):
    """Embed a local Clifford (F, h) into full 2n space"""
    full_F = np.eye(2 * n_qudits, dtype=int)
    full_v = np.zeros(2 * n_qudits, dtype=int)

    n_loc_qudits = len(qudit_indices)
    x_in = symplectic_local[:n_loc_qudits, :]
    z_in = symplectic_local[n_loc_qudits:, :]

    for image_row_index in range(n_loc_qudits):

        full_F[image_row_index, :] = embed_symplectic_single_pauli_string(x_in[image_row_index, :],
                                                                          qudit_indices, n_qudits)
        full_F[n_qudits + image_row_index, :] = embed_symplectic_single_pauli_string(z_in[image_row_index, :],
                                                                                     qudit_indices, n_qudits)

    for i, ind in enumerate(qudit_indices):  # check
        full_v[ind] = phase_vector_local[i]

    return np.mod(full_F, dimension), np.mod(full_v, dimension)


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
    return v + inner_gf(k, v) * k


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
        f1 *= 0

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

    gprime_gf = symplectic.copy()
    if b_g == 0:
        for j in range(nn):
            gprime_gf[j] = transvection_gf(T_g[0], gprime_gf[j])
            gprime_gf[j] = transvection_gf(T_g[1], gprime_gf[j])
            gprime_gf[j] = transvection_gf(h0_g, gprime_gf[j])
            gprime_gf[j] = transvection_gf(e1_g, gprime_gf[j])
    else:
        for j in range(nn):
            gprime_gf[j] = transvection_gf(T_g[0], gprime_gf[j])
            gprime_gf[j] = transvection_gf(T_g[1], gprime_gf[j])
            gprime_gf[j] = transvection_gf(h0_g, gprime_gf[j])

    # Step 7: recurse
    gnew_gf = gprime_gf[2:, 2:]
    return index_from_symplectic(n_qudits - 1, gnew_gf, dimension) * number_of_cosets(n_qudits) + cvw


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




if __name__ == '__main__':
    # loc_sym = [1, 2, 3, 4]
    # print(embed_symplectic_single_pauli_string(loc_sym, [1, 3], 5))
    import time
    ts = 0
    ts_g = 0
    wrong = 0

    dim = 2
    for i in range(10):
        t_0 = time.time()
        # s = symplectic(i, 25)
        t_1 = time.time()
        ts += t_1 - t_0
        t_0 = time.time()
        s_g = symplectic_from_index(i, 3, dim)
        if not is_symplectic(s_g):
            print(s_g)
        t_1 = time.time()
        ts_g += t_1 - t_0
        # if not np.array_equal(s, s_g):
        #     wrong += 1
    print('Done')
    # print(wrong)
    # print(ts, ts_g)
    # test_symplectic_index_inverse(p=2, n=2)

    # test_symplectic_index_inverse(p=3, n=2, max_tests=100)
