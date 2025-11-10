import numpy as np
from sklearn.decomposition import TruncatedSVD
import math
from src.common.common import time_wrapper


def error_func_frobenius(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def error_func_absolute(A, B):
    return np.count_nonzero(B != A) / np.count_nonzero(A)


def error_at_k(A, k, error_func, t=0.5):
    svd = TruncatedSVD(n_components=k, random_state=42)

    L = svd.fit_transform(A)
    R = svd.components_
    B = (np.matmul(L, R) > t).astype(int)
    return error_func(A, B)

def decomposition_at_k(A, k):
    svd = TruncatedSVD(n_components=k, n_iter=50, random_state=42)

    L = svd.fit_transform(A)
    R = svd.components_
    return L, R


def pad_A(A, k):
    n = A.shape[0]
    pad_width = ((0, k - n), (0, k - n))

    return np.pad(A, pad_width, mode='constant', constant_values=0)


def init_svd(k, n_iter=100, seed=0):
    return TruncatedSVD(n_components=k, n_iter=n_iter, random_state=seed)


@time_wrapper
def decomposition_at_k_with_error(svd, A, k):
    n = A.shape[0]
    A_padded = pad_A(A, k) if n < k else A

    L = svd.fit_transform(A_padded)[:n, :]
    R = svd.components_[:, :n]
    return error_func_frobenius(A, round(L, R)), L, R


def round(L, R, t=0.5):
    return (L @ R > t).astype(int)


@time_wrapper
def calculate_rr(A, error_func=error_func_frobenius):
    errors = []
    
    n_nodes = A.shape[0]
    k = math.floor(n_nodes * 0.6)
    L, R = decomposition_at_k(A, k)
    error = error_func_frobenius(A, round(L, R))

    # make sure we picked a big enough k
    while error > 0 and k <= n_nodes:
        k += math.ceil((n_nodes - k) / 2)
        L, R = decomposition_at_k(A, k)
        error = error_func_frobenius(A, round(L, R))

    # find the rr
    while error == 0 and k > 1:
        k -= 1
        error = error_func_frobenius(A, round(L[:,:k], R[:k,]))
    
    rr = k + 1
    errors.append(0)
    
    # collect all remaining errors:
    while k >= 1:
        errors.append(error_func_frobenius(A, round(L[:,:k], R[:k,])))
        k -= 1

    return rr, errors[::-1]
