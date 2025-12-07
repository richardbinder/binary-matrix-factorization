import torch

from src.compute.svd import psd_factor_svd
from src.compute.compute_properties import best_low_rank_approx_error

global count
count = 0

def pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    # X: (n, k)
    # returns D where D[i,j] = ||X[i]-X[j]||^2, shape: (n, n)
    G = X @ X.T                         # Gram (n, n)
    diag = torch.diag(G)                # (n,)
    return diag[:, None] + diag[None, :] - 2.0 * G

def pairwise_dot_products(X: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise dot products between all rows of X.

    X: (n, k)
    returns: D where D[i, j] = X[i] · X[j], shape: (n, n)
    """
    return X @ X.T

def lpca_loss(L, R, adj_s):
    """
    L: (n, k) torch tensor
    R: (k, n) torch tensor
    adj_s: (n, n) torch tensor with entries in {-1, +1}
    W: (n, n) torch tensor with neighborhood dissimilarities (>= 0)
    gamma: float

    returns: scalar loss tensor
    """

    # LPCA part
    logits = L @ R  # (n, n)
    neg_logits_y = -logits * adj_s  # (n, n)

    # log(1 + exp(-y f(x))) = logaddexp(0, -y f(x))
    lpca_loss = torch.logaddexp(
        torch.zeros_like(neg_logits_y),
        neg_logits_y
    ).mean()

    return lpca_loss, lpca_loss, 0


def lpca_dist_loss(L, R, adj_s, W, weights, params, gamma=0.2):
    """
    L: (n, k) torch tensor
    R: (k, n) torch tensor
    adj_s: (n, n) torch tensor with entries in {-1, +1}
    W: (n, n) torch tensor with neighborhood dissimilarities (>= 0)
    gamma: float

    returns: scalar loss tensor
    """

    # LPCA part
    logits = L @ R  # (n, n)
    neg_logits_y = -logits * adj_s  # (n, n)

    # log(1 + exp(-y f(x))) = logaddexp(0, -y f(x))
    lpca_loss = torch.logaddexp(
        torch.zeros_like(neg_logits_y),
        neg_logits_y
    ).mean()

    if gamma == 0:
        return lpca_loss, lpca_loss, 0

    L_dist = pairwise_sq_dists(L)              # (n, n)
    R_dist = pairwise_sq_dists(R.t())          # (n, n)
    dist = L_dist + R_dist
    dist = dist / dist.max()

    W = W.pow(2)
    W = W / W.max()

    sim_loss = ((dist - W).abs() * weights).mean()

    # inv_W = 1.0 / ((W + 1.0) ** 2)     # (n, n)
    # sim_loss = 0.5 * ((L_dist + R_dist) * inv_W).sum()

    global count
    if count >= 299:
        count = 0
    else:
        count += 1

    return lpca_loss + gamma * sim_loss, lpca_loss, sim_loss

def lpca_sim_loss(L, R, adj_s, W, gamma=0.2):
    """
    L: (n, k) torch tensor
    R: (k, n) torch tensor
    adj_s: (n, n) torch tensor with entries in {-1, +1}
    W: (n, n) torch tensor with neighborhood dissimilarities (>= 0)
    gamma: float

    returns: scalar loss tensor
    """

    # LPCA part
    logits = L @ R  # (n, n)
    neg_logits_y = -logits * adj_s  # (n, n)

    # log(1 + exp(-y f(x))) = logaddexp(0, -y f(x))
    lpca_loss = torch.logaddexp(
        torch.zeros_like(neg_logits_y),
        neg_logits_y
    ).mean()

    if gamma == 0:
        return lpca_loss, lpca_loss, 0

    # (n, n) pairwise squared distances; no (n,n,k) tensors created
    norms = (L.pow(2) + R.T.pow(2)).sum(dim=1).pow(1/2).unsqueeze(1)
    norms = norms + 1e-12
    L = L/norms
    R = R/norms.T

    L_sim = pairwise_dot_products(L)              # (n, n)
    R_sim = pairwise_dot_products(R.T)          # (n, n)
    sim = L_sim + R_sim
    # sim = sim.abs()

    sim_loss = ((sim - W).pow(2)).mean() / W.max()

    global count
    if count >= 999:
        count = 0
    else:
        count += 1

    return lpca_loss + gamma * sim_loss, lpca_loss, sim_loss

def lpca_sim_loss_params(L, R, adj_s, W, params, gamma=0.2):
    """
    L: (n, k) torch tensor
    R: (k, n) torch tensor
    adj_s: (n, n) torch tensor with entries in {-1, +1}
    W: (n, n) torch tensor with neighborhood dissimilarities (>= 0)
    gamma: float

    returns: scalar loss tensor
    """

    # LPCA part
    logits = L @ R  # (n, n)
    neg_logits_y = -logits * adj_s  # (n, n)

    # log(1 + exp(-y f(x))) = logaddexp(0, -y f(x))
    lpca_loss = torch.logaddexp(
        torch.zeros_like(neg_logits_y),
        neg_logits_y
    ).mean()

    if gamma == 0:
        return lpca_loss, lpca_loss, 0

    # (n, n) pairwise squared distances; no (n,n,k) tensors created
    norms = (L.pow(2) + R.T.pow(2)).sum(dim=1).pow(1/2).unsqueeze(1)
    norms = norms + 1e-12
    L = L/norms
    R = R/norms.T

    L_sim = pairwise_dot_products(L)              # (n, n)
    R_sim = pairwise_dot_products(R.T)          # (n, n)
    sim = L_sim + R_sim
    # sim = sim.abs()

    # sim = sim @ params
    # sim = torch.relu(sim)

    sim_loss = ((sim - W).pow(2)).mean().pow(1/2) / W.max()

    global count
    if count >= 999:
        count = 0
    else:
        count += 1

    return lpca_loss + gamma * sim_loss, lpca_loss, sim_loss

def lpca_p_loss(L, R, adj_s, W, gamma=0.2):
    """
    L: (n, k) torch tensor
    R: (k, n) torch tensor
    adj_s: (n, n) torch tensor with entries in {-1, +1}
    W: (n, n) torch tensor with neighborhood dissimilarities (>= 0)
    gamma: float

    returns: scalar loss tensor
    """

    # LPCA part
    logits = L @ R  # (n, n)
    neg_logits_y = -logits * adj_s  # (n, n)

    # log(1 + exp(-y f(x))) = logaddexp(0, -y f(x))
    lpca_loss = torch.logaddexp(
        torch.zeros_like(neg_logits_y),
        neg_logits_y
    ).mean()

    if gamma == 0:
        return lpca_loss, lpca_loss, 0

    # (n, n) pairwise squared distances; no (n,n,k) tensors created
    # norms = (L.pow(2) + R.T.pow(2)).sum(dim=1).pow(1/2).unsqueeze(1)
    # norms = norms + 1e-12
    # L = L/norms
    # R = R/norms.T

    L_sim = pairwise_dot_products(L)              # (n, n)
    R_sim = pairwise_dot_products(R.T)          # (n, n)
    logits = L_sim + R_sim

    S = torch.nn.functional.log_softmax(logits, dim=1)

    # Cross entropy:  -sum_i W_i * log(S_i)
    p_loss = -torch.sum(W * S, dim=1)
    p_loss = p_loss.mean()

    global count
    if count >= 99:
        count = 0
    else:
        count += 1

    return gamma * p_loss, lpca_loss, p_loss