import numpy as np

# --- core functions from Koenig & Smolin appendix ---
def direct_sum(m1, m2):
    n1, n2 = m1.shape[0], m2.shape[0]
    out = np.zeros((n1+n2, n1+n2), dtype=np.int8)
    out[:n1, :n1] = m1
    out[n1:, n1:] = m2
    return out

def int2bits(i, n):
    out = np.zeros(n, dtype=np.int8)
    for j in range(n):
        out[j] = i & 1
        i >>= 1
    return out

def inner(v, w):
    n = v.size // 2
    t = 0
    for i in range(n):
        t += v[2*i] * w[2*i+1]
        t += w[2*i] * v[2*i+1]
    return t % 2

def transvection(k, v):
    return (v + inner(k, v) * k) % 2

def find_transvection(x, y):
    n = x.size
    out = np.zeros((2, n), dtype=np.int8)
    if np.array_equal(x, y):
        return out
    if inner(x, y) == 1:
        out[0] = (x + y) % 2
        return out
    out[0] = (x + y) % 2
    out[1] = x.copy()
    return out

def symplectic_interleaved(i, n):
    """Recursive algorithm from the appendix (interleaved ordering)."""
    nn = 2*n
    s = (1 << nn) - 1
    k = (i % s) + 1
    i //= s

    f1 = int2bits(k, nn)
    e1 = np.zeros(nn, dtype=np.int8)
    e1[0] = 1
    T = find_transvection(e1, f1)

    bits = int2bits(i % (1 << (nn-1)), nn-1)
    i //= (1 << (nn-1))

    e_prime = e1.copy()
    for j in range(2, nn):
        e_prime[j] = bits[j-1]

    h0 = transvection(T[0], e_prime)
    h0 = transvection(T[1], h0)

    if bits[0] == 1:
        h0 = (h0 + f1) % 2

    id2 = np.eye(2, dtype=np.int8)
    if n > 1:
        g = direct_sum(id2, symplectic_interleaved(i, n-1))
    else:
        g = id2

    for j in range(nn):
        col = g[:, j]
        col = transvection(T[0], col)
        col = transvection(T[1], col)
        col = transvection(h0, col)
        col = transvection(f1, col)
        g[:, j] = col

    return g

# --- conversion and checker ---
def interleaved_to_grouped(F):
    """Convert interleaved [x0,z0,x1,z1,...] to grouped [x0,...,xn-1, z0,...,zn-1]."""
    n2 = F.shape[0]
    n = n2 // 2
    perm = [i//2 + (i%2)*n for i in range(n2)]
    return F[np.ix_(perm, perm)]

def is_symplectic_grouped(F):
    n = F.shape[0] // 2
    Omega = np.zeros((2*n, 2*n), dtype=np.int8)
    Omega[:n, n:] = np.eye(n, dtype=np.int8)
    Omega[n:, :n] = np.eye(n, dtype=np.int8)
    lhs = (F.T @ Omega @ F) % 2
    return np.array_equal(lhs, Omega)

# --- wrapper ---
def symplectic_grouped(i, n):
    """Return a symplectic matrix in grouped ordering."""
    F_interleaved = symplectic_interleaved(i, n)
    F_grouped = interleaved_to_grouped(F_interleaved)
    assert is_symplectic_grouped(F_grouped), "Resulting matrix is not symplectic!"
    return F_grouped

# --- example usage ---
if __name__ == "__main__":
    F = symplectic_grouped(12345, 3)
    print("F (grouped ordering):\n", F)
    print("Is symplectic (grouped check)?", is_symplectic_grouped(F))
