from __future__ import annotations
from typing import Optional, List
import numpy as np

def in_image_mod2(M: np.ndarray, b: np.ndarray) -> bool:
    x = gauss_solve_mod2(M % 2, b % 2)
    return x is not None

def inv_mod_prime(a: int, p: int) -> int:
    p = int(p); a = int(a) % p
    if a == 0:
        raise ZeroDivisionError("no inverse")
    return pow(a, p - 2, p)

def gauss_solve_mod_prime(M_int: np.ndarray, b_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    m, n = M_int.shape
    M = (M_int % p).astype(int).copy()
    b = (b_int % p).astype(int).copy()
    row = 0
    pivots: List[tuple[int,int]] = []
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r; break
        if pivot == -1:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
            b[row], b[pivot] = b[pivot], b[row]
        inv = inv_mod_prime(int(M[row, col]), p)
        M[row, col] = 1
        if col + 1 < n:
            M[row, col+1:] = (M[row, col+1:] * inv) % p
        b[row] = (b[row] * inv) % p
        for r in range(m):
            if r == row: continue
            f = M[r, col] % p
            if f != 0:
                M[r, col] = 0
                if col + 1 < n:
                    M[r, col+1:] = (M[r, col+1:] - f * M[row, col+1:]) % p
                b[r] = (b[r] - f * b[row]) % p
        pivots.append((row, col))
        row += 1
        if row == m:
            break
    # consistency
    for r in range(row, m):
        if np.all(M[r, :] % p == 0) and (b[r] % p != 0):
            return None
    x = np.zeros(n, dtype=int)
    for r, c in reversed(pivots):
        s = 0 if c + 1 >= n else int(np.dot(M[r, c+1:] % p, x[c+1:] % p) % p)
        x[c] = (b[r] - s) % p
    return x % p

def gauss_inverse_mod_prime(M_int: np.ndarray, p: int) -> Optional[np.ndarray]:
    p = int(p)
    n = int(M_int.shape[0])
    assert M_int.shape == (n, n)
    A = (M_int % p).astype(int).copy()
    I = np.eye(n, dtype=int)
    A_aug = np.concatenate([A, I], axis=1)
    row = 0
    for col in range(n):
        pivot = -1
        for r in range(row, n):
            if A_aug[r, col] % p != 0:
                pivot = r; break
        if pivot == -1:
            return None
        if pivot != row:
            A_aug[[row, pivot]] = A_aug[[pivot, row]]
        inv = inv_mod_prime(int(A_aug[row, col]), p)
        A_aug[row, col:] = (A_aug[row, col:] * inv) % p
        for r in range(n):
            if r == row: continue
            f = A_aug[r, col] % p
            if f != 0:
                A_aug[r, col:] = (A_aug[r, col:] - f * A_aug[row, col:]) % p
        row += 1
        if row == n:
            break
    if not np.array_equal(A_aug[:, :n] % p, np.eye(n, dtype=int)):
        return None
    return A_aug[:, n:] % p

def gauss_solve_mod2(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    A = (M.astype(np.uint8) & 1).copy()
    y = (b.astype(np.uint8) & 1).copy()
    m, n = A.shape
    piv_cols, piv_rows = [], []
    r = 0
    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if A[i, c]:
                pivot = i; break
        if pivot == -1:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            y[r], y[pivot] = y[pivot], y[r]
        piv_cols.append(c); piv_rows.append(r)
        # eliminate column c elsewhere
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
                y[i] ^= y[r]
        r += 1
        if r == m:
            break
    # inconsistency: 0 = 1
    for i in range(m):
        if not A[i].any() and y[i]:
            return None
    # back-substitute (free vars = 0)
    x = np.zeros(n, dtype=np.uint8)
    for k in reversed(range(len(piv_cols))):
        i = piv_rows[k]; c = piv_cols[k]
        s = 0
        row = A[i]
        for j in range(n):
            if j != c and row[j]:
                s ^= x[j]
        x[c] = y[i] ^ s
    return x.astype(int)

def nullspace_mod2(M: np.ndarray) -> list[np.ndarray]:
    A = (M % 2).astype(np.uint8)
    m, n = A.shape
    piv_col = [-1]*m
    r = 0
    used = [False]*n
    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if A[i, c]:
                pivot = i; break
        if pivot == -1: continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        piv_col[r] = c
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        used[c] = True
        r += 1
        if r == m: break
    free_cols = [j for j in range(n) if not used[j]]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i in reversed(range(r)):
            c = piv_col[i]
            s = 0
            row = A[i]
            for j in range(n):
                if j != c and row[j]:
                    s ^= v[j]
            v[c] = s
        basis.append(v.astype(int))
    return basis

if __name__ == "__main__":
    # smoke
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(5,7))
    x = rng.integers(0, 2, size=7)
    b = (A @ x) % 2
    x2 = gauss_solve_mod2(A, b); assert x2 is not None and ((A @ x2) % 2 == b).all()
    print("[algebra] ok")
