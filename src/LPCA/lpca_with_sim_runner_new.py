import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from common_new import (
    construct_adjacency_matrix,
    load_dataset,
    time_wrapper,
    measure_encoding_similarity,
)


def normalize_enc_torch(L, eps=1e-8):
    """
    Row-wise L2-normalize a 2D tensor.

    L: (n, d) torch tensor
    returns: (n, d) tensor
    """
    norms = torch.norm(L, dim=1, keepdim=True)
    return L / (norms + eps)

def pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    # X: (n, k)
    # returns D where D[i,j] = ||X[i]-X[j]||^2, shape: (n, n)
    G = X @ X.T                         # Gram (n, n)
    diag = torch.diag(G)                # (n,)
    return diag[:, None] + diag[None, :] - 2.0 * G


def lpca_sim_loss_torch(L, R, adj_s, W, gamma=0.2):
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
    ).sum()

    if gamma == 0:
        return lpca_loss

    # (n, n) pairwise squared distances; no (n,n,k) tensors created
    L_dist = pairwise_sq_dists(L)              # (n, n)
    R_dist = pairwise_sq_dists(R.t())          # (n, n)
    # sim_loss = (L_dist + R_dist - W.pow(2)).abs().mean()



    return lpca_loss + gamma * sim_loss


@time_wrapper
def lpca_encoding(A, k, bound=None, gamma=0.5, device=None):
    """
    A: torch tensor (n, n) with 0/1 entries (dense adjacency)
    k: embedding dimension
    bound: if not None, clamp parameters to [-bound, bound]
    gamma: similarity loss weight
    max_iter: max L-BFGS iterations
    device: 'cpu' or 'cuda'; if None, auto-select
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Move adjacency to device, ensure float
    if isinstance(A, torch.Tensor):
        adj = A.to(device=device, dtype=torch.float32)
    else:
        adj = torch.from_numpy(np.asarray(A, dtype=np.float32)).to(device)

    n = adj.shape[0]

    # shifted adjacency: -1 for 0, +1 for 1
    adj_s = -1.0 + 2.0 * adj  # (n, n), in {-1, +1}

    # W: neighborhood dissimilarity matrix via XOR over adjacency rows
    A_bool = adj.bool()
    W = (A_bool.unsqueeze(1) ^ A_bool.unsqueeze(0)).sum(dim=2).to(torch.float32)  # (n, n)

    # Initialize factors L, R in [-1, 1]
    L = torch.empty((n, k), device=device).uniform_(-1.0, 1.0)
    R = torch.empty((k, n), device=device).uniform_(-1.0, 1.0)

    L.requires_grad_(True)
    R.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [L, R],
        max_iter=3000,
        line_search_fn="strong_wolfe",
    )

    # optimizer = torch.optim.Adam(
    #     [L, R],
    #     lr=1e-2
    # )

    def closure():
        optimizer.zero_grad()

        loss = lpca_sim_loss_torch(L, R, adj_s, W, gamma=gamma)
        loss.backward()

        # approximate bound handling by projection
        if bound is not None:
            with torch.no_grad():
                L.clamp_(-bound, bound)
                R.clamp_(-bound, bound)

        return loss

    # for _ in range(3000):
    #     optimizer.step(closure)

    # Try to read iterations from optimizer state (may not always be present)
    state = optimizer.state.get(L, {})
    nit = int(state.get("n_iter", 0))

    # Build final normalized encoding on CPU for downstream numpy-based stuff
    with torch.no_grad():
        enc = torch.cat([L, R.t()], dim=1)  # (n, 2k)
        enc = normalize_enc_torch(enc)      # (n, 2k)
        enc_np = enc.cpu().numpy()

    # Reconstruct adjacency from normalized encodings
    L_n = enc_np[:, :k]
    R_n = enc_np[:, k:]
    A_reconstructed = (L_n @ R_n.T > 0).astype(np.float32)

    A_dense_np = adj.cpu().numpy()  # original adjacency
    num = np.linalg.norm(A_reconstructed - A_dense_np)
    denom = np.linalg.norm(A_dense_np)
    error = num / denom if denom != 0 else 0.0

    # similarity stats (kept as in the original)
    sim = measure_encoding_similarity(A_dense_np, enc_np)
    d_mean = []
    d_std = []
    for _, x in sorted(sim.items()):
        d_mean.append(np.mean(x))
        d_std.append(np.std(x))

    return error, d_mean, d_std, nit, enc_np


def compute_encodings(data, k, out_path, bound=None, gamma=0.5, n_samples=None, device=None):
    matrices = {}
    results = []

    idx_max = len(data) if n_samples is None else n_samples

    for i in tqdm(range(idx_max)):
        # Now returns a dense torch tensor adjacency
        A = construct_adjacency_matrix(data[i])

        t, error, d_mean, d_std, nit, enc = lpca_encoding(A, k, bound, gamma, device)
        matrices[f"idx_{i}"] = enc

        print(error)

        results.append(
            {
                "graph_id": i,
                "n_nodes": data[i].x.shape[0],
                "nit": nit,
                "error": error,
                "time": t,
                "d_mean": d_mean,
                "d_std": d_std,
            }
        )

    np.savez_compressed(out_path + ".npz", **matrices)
    pd.DataFrame(results).to_parquet(out_path + ".parquet")


if __name__ == "__main__":
    # python lpca_with_sim_runner_torch.py ZINC 4 8 10 1000
    name = sys.argv[1]
    data = load_dataset(name)

    bound = None
    if sys.argv[2].lower() != "none":
        bound = int(sys.argv[2])

    k = int(sys.argv[3])
    gamma = float(sys.argv[4])
    n_samples = None

    if len(sys.argv) > 5:
        n_samples = int(sys.argv[5])

    out_path = f"lpca_out/lpca_with_sim_{name}_k{k}_b{bound}_gamma{gamma}_s{n_samples}"

    compute_encodings(data, k, out_path, bound, gamma, n_samples, "cpu")

    print("computed encodings:", out_path)
