import torch

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

def lpca_sim_loss(L, R, adj_s, W, weights, params, gamma=0.2):
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
    L = L/norms
    R = R/norms.T

    L_sim = pairwise_dot_products(L)              # (n, n)
    R_sim = pairwise_dot_products(R.t())          # (n, n)
    sim = L_sim + R_sim
    sim = sim.abs()

    W = W / W.max()

    # e = 0.01
    # try:
    #     assert torch.max(dist) <= 1+e
    #     assert torch.min(dist) >= 0-e
    #     assert torch.max(W) <= 1+e
    #     assert torch.min(W) >= 0-e
    # except AssertionError:
    #     print(W)
    #
    # try:
    #     assert torch.max(W) <= 1+e
    #     assert torch.min(W_sim) >= -1-e
    # except AssertionError:
    #     print(W_sim)

    sim_loss = ((sim - W).abs() * weights).mean()

    global count
    if count >= 299:
        count = 0
    else:
        count += 1

    return lpca_loss + gamma * sim_loss, lpca_loss, sim_loss