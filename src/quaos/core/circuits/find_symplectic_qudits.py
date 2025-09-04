"""
Currently this module implements Lemma 5.1 in https://kups.ub.uni-koeln.de/50465/1/dissertation_heinrich.pdf 
to build the necessary transvection(s) to map one PauliSum in GF(p) to another. 
It also assumes p is prime dimension. 

Future extensions: input symplectic matrix -> output symplectic matrix 
"""

import numpy as np
import galois
from itertools import islice
from quaos.core.circuits.utils import transvection_matrix, symplectic_product


def modinv(a, p):
    """Modular inverse in GF(p)"""
    return pow(int(a), p-2, p)

def pair(vec, i, n):
    """Return 2x1 pair at index i"""
    return np.array([vec[i], vec[i+n]])

def solve_non_proportional(u_i, v_i, p):
    """Solve B x = y for non-proportional 2x2 vectors"""

    GF = galois.GF(p)
    B = np.array([[u_i[0], -u_i[1]], [-v_i[0], v_i[1]]]) % p
    y = GF([1, 1])
    B = GF(B)
    x = np.linalg.solve(B, y).astype(np.int32).view(np.ndarray) % p
    return x

def proportionality_constant_mod(u_i, v_i, p):
    """
    Compute the proportionality constant k mod p such that v_i = k * u_i (mod p),
    assuming u_i and v_i are proportional.
    """
    a, b = u_i
    c, d = v_i

    if not galois.is_prime(p):
        raise ValueError(f"{p} is not prime!")

    # Convert to Python ints to avoid NumPy overflow
    a, b, c, d = int(a), int(b), int(c), int(d)

    # Case 1: all entries nonzero
    if a != 0 and b != 0 and c != 0 and d != 0:
        k1 = (a * pow(c, p-2, p)) % p  # k = a / c mod p
        k2 = (b * pow(d, p-2, p)) % p  # k = b / d mod p
        if k1 != k2:
            raise ValueError("Vectors are not proportional")
        return k1

    # Case 2: only second component nonzero
    elif b != 0 and d != 0:
        k = (b * pow(d, p-2, p)) % p
        return k

    # Case 3: only first component nonzero
    elif a != 0 and c != 0:
        k = (a * pow(c, p-2, p)) % p
        return k

    else:
        raise ValueError("Cannot determine proportionality constant: one of the components is zero")


def solve_proportional(u_i, v_i, p, proportionality_const):
    """Solve B x = y for proportional pair with free parameter"""

    GF = galois.GF(p)
    B = np.array([[u_i[0], -u_i[1]], [-v_i[0], v_i[1]]]) % p
    y = GF([1, proportionality_const])
    B = GF(B)
    row1 = B[0]
    t_free = proportionality_const
    solutions = []

    if row1[0] != 0 and row1[1] != 0:
        x1 = (y[0] - row1[1]*t_free) * GF(int(row1[0]))**(-1)
        solutions.append(GF([x1, t_free]))
    elif row1[0] != 0 and row1[1] == 0:
        x2 = (y[0] - row1[0]*t_free) * GF(int(row1[0]))**(-1)
        solutions.append(GF([t_free, x2]))
    elif row1[0] == 0 and row1[1] != 0:
        x2 = (y[0] - row1[0]*t_free) * GF(int(row1[1]))**(-1)
        solutions.append(GF([x2, t_free]))

    return solutions[0].astype(np.int32).view(np.ndarray)


def build_symplectic_for_transvection(u, v, p):
    """
    Build the intermediate vector w for two-step transvections if [u, v]=0 mod p 
    Assumes qudit dimension p is prime and symplectic_product(u, v, p) == 0
    """
    # --- Input validation ---
    if np.all(u == 0) or np.all(v == 0):
        raise ValueError("Input vectors cannot be zero")
    if not galois.is_prime(p):
        raise NotImplementedError(f"Prime dimension expected, got p={p}")
    if symplectic_product(u, v, p) != 0:
        raise ValueError("Symplectic product must be zero")

    n = len(u) // 2
    w = np.zeros(2*n, dtype=int)
    GF = galois.GF(p)

    # --- First pass: find non-proportional pair ---
    for i, (ui, vi) in enumerate(islice(zip(u, v), n)):
        u_i = pair(u, i, n)
        v_i = pair(v, i, n)
        nz_u, nz_v = np.count_nonzero(u_i), np.count_nonzero(v_i)
        sp_uv = symplectic_product(u_i, v_i, p)

        if nz_u and nz_v and sp_uv != 0:
            x = solve_non_proportional(u_i, v_i, p)
            w[i+n], w[i] = x[0], x[1]
            break

    # --- Second pass: proportional pair ---
    if np.all(w == 0):
        for i, (ui, vi) in enumerate(islice(zip(u, v), n)):
            u_i = pair(u, i, n)
            v_i = pair(v, i, n)
            nz_u, nz_v = np.count_nonzero(u_i), np.count_nonzero(v_i)
            sp_uv = symplectic_product(u_i, v_i, p)

            if nz_u and nz_v and sp_uv == 0:
                k = proportionality_constant_mod(u_i, v_i, p)
                x = solve_proportional(u_i, v_i, p,  k)
                w[i+n], w[i] = x[0], x[1]
                break

    # --- Third pass: handle single nonzero entries manually ---
    if np.all(w == 0):
        for i in range(n):
            u_i, v_i = pair(u, i, n), pair(v, i, n)

            # Case u_i nonzero, v_i zero
            if np.count_nonzero(u_i) != 0 and np.count_nonzero(v_i) == 0:
                w_u = np.zeros(2, dtype=int)
                if u_i[0] != 0:
                    w_u[1] = modinv(u_i[0],p)
                else:
                    w_u[0] = modinv(-u_i[1],p)
                assert symplectic_product(u_i, w_u, p) == 1
                w[i], w[i+n] = w_u[0], w_u[1]
                break

        for i in range(n):
            u_i, v_i = pair(u, i, n), pair(v, i, n)

            # Case u_i zero, v_i nonzero
            if np.count_nonzero(u_i) == 0 and np.count_nonzero(v_i) != 0:
                w_v = np.zeros(2, dtype=int)
                if v_i[0] != 0:
                    w_v[1] = (-modinv(v_i[0],p)) % p
                else:
                    w_v[0] = modinv(v_i[1],p)
                assert symplectic_product(w_v, v_i, p) == 1
                w[i], w[i+n] = w_v[0], w_v[1]
                break

    # --- Final validation ---
    if symplectic_product(u, w, p) == 0 or symplectic_product(w, v, p) == 0:
        raise ValueError("Failed to construct valid symplectic vector w")

    return w


def Find_transvection_map(input_ps, output_ps, p):
    """
    Provides the map to transfer one paulisum in GF(p) to another paulisum.
    Currently works only for prime dimension. 
    Returns combined transvection (final map)
    """

    # --- Input validation ---
    if np.all(input_ps == 0) or np.all(output_ps == 0):
        raise ValueError("Vectors cannot be zero")
    if not galois.is_prime(p):
        raise NotImplementedError(f"Prime dimension expected, got p={p}")

    n= int(len(input_ps)//2)
    a=symplectic_product(input_ps, output_ps,p)

    if a != 0:
        ainv = modinv(a, p)
        h=(-input_ps + output_ps) % p
        F_h= transvection_matrix(h, p, multiplier=ainv)
        if (input_ps @ F_h % p != output_ps).all():
            raise ValueError("Failed to construct valid transvection for nonzero symplectic product")

    elif a == 0:
        w=build_symplectic_for_transvection(input_ps, output_ps, p)
        a_w= symplectic_product(input_ps, w,p)
        a_w_inv = modinv(a_w, p)
        h=(-input_ps + w) % p
        F_h_1= transvection_matrix(h, p, multiplier=a_w_inv)
        if (input_ps @ F_h_1 % p != w).all():
            raise ValueError("Failed to construct valid transvection for u->w")
        b_w= symplectic_product(w, output_ps, p)
        b_w_inv = modinv(b_w, p)
        h=(-w + output_ps) % p
        F_h_2= transvection_matrix(h, p, multiplier=b_w_inv)
        if (w @ F_h_2 % p != output_ps).all():
            raise ValueError("Failed to construct valid transvection for w->v")
        F_h= F_h_1 @ F_h_2 %p
        if (input_ps @ F_h %p != output_ps).all():
            raise ValueError("Failed to construct valid transvection map: for u->v")

    return F_h



