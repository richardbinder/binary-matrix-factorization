import numpy as np
import scipy as sp
from scipy.special import expit
from src.common.common import time_wrapper, measure_encoding_similarity


def sim_grad(W, L):
    diff = L[:, None, :] - L[None, :, :]  # (n, n, k)

    inv_W = 1.0 / ((W + 1) ** 2)

    grad = 2 * np.sum(inv_W[:, :, None] * diff, axis=1)  # (n, k)

    return grad


def sim_loss(W, L_diff, R_diff):
    """
    all matrices shape n,n
    W is the neighbourhood similarity matrix

    divide by half since it is symmetric
    """
    L_dist = np.sum(L_diff**2, axis=2)
    R_dist = np.sum(R_diff**2, axis=2)
    return 0.5 * np.sum((L_dist + R_dist) / ((W + 1) ** 2))


def normalize_enc(L, eps=1e-8):
    norms = np.linalg.norm(L, axis=1, keepdims=True)  # shape: (n, 1)
    return L / (norms + eps)


def lpca_sim_loss(factors, adj_s, A, k, gamma=0.2):
    # adj_s = shifted adj with -1's and +1's
    n, _ = adj_s.shape

    L = factors[: n * k].reshape(n, k)
    R = factors[n * k :].reshape(k, n)

    # lpca loss and grads
    logits = L @ R
    prob_wrong = expit(-logits * adj_s)  # (n, n)
    l_loss = (np.logaddexp(0, -logits * adj_s)).sum()
    L_lpca_grad = -((prob_wrong) * adj_s) @ R.T  # (n, k)
    R_lpca_grad = -L.T @ (prob_wrong * adj_s)  # (k, n)

    if gamma == 0:
        return l_loss, np.concatenate(
            (
                L_lpca_grad.flatten(),
                R_lpca_grad.flatten(),
            )
        )

    # sim loss and grads
    L_diff = L[:, None, :] - L[None, :, :]  # (n, k)
    R_diff = R.T[:, None, :] - R.T[None, :, :]  # (n, k)
    W = np.sum(np.bitwise_xor(A[:, None, :], A[None, :, :]), axis=2)  # (n, n)
    s_loss = sim_loss(W, L_diff, R_diff)
    L_sim_grad = sim_grad(W, L)
    R_sim_grad = sim_grad(W, R.T).T

    return l_loss + gamma * s_loss, np.concatenate(
        (
            (L_lpca_grad + gamma * L_sim_grad).flatten(),
            (R_lpca_grad + gamma * R_sim_grad).flatten(),
        )
    )


@time_wrapper
def lpca_encoding(A, k, bound=None, gamma=0.5):
    n = A.shape[0]
    size = 2 * n * k
    factors = -1 + 2 * np.random.random(size=size)

    bounds = None
    if bound is not None:
        lb, ub = -bound, bound
        bounds = sp.optimize.Bounds(lb, ub)

    adj = np.array(A.todense())
    res = sp.optimize.minimize(
        lambda x, adj_s, A, k: lpca_sim_loss(x, adj_s, A, k, gamma),
        x0=factors,
        args=(-1 + 2 * adj, adj, k),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000},
        bounds=bounds,
    )
    L = res.x[: n * k].reshape(n, k)
    R = res.x[n * k :].reshape(k, n)

    enc = normalize_enc(np.hstack((L, R.T)))
    L_n, R_n = enc[:, :k], enc[:, k:]
    A_reconstructed = 1.0 * (L_n @ R_n.T > 0)
    error = np.linalg.norm(1.0 * (A_reconstructed > 0) - A) / sp.sparse.linalg.norm(A)
    d_mean = []
    d_std = []

    sim = measure_encoding_similarity(A.todense(), enc)
    for _, x in sorted(sim.items()):
        d_mean.append(np.mean(x))
        d_std.append(np.std(x))
    return error, d_mean, d_std, res.nit, enc
