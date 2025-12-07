import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, LRGBDataset
from src.compute.compute_properties import get_sim_targets


def construct_adjacency_matrix(data: torch.Tensor):
    """
    Build a dense adjacency matrix for a torch_geometric data object.

    Returns:
        A (n_nodes, n_nodes) torch.FloatTensor with entries in {0, 1}.
    """
    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]

    # edge_index is (2, n_edges), values are all ones
    values = torch.ones(n_edges, dtype=torch.float32, device=data.edge_index.device)
    s = torch.sparse_coo_tensor(
        data.edge_index,
        values,
        (n_nodes, n_nodes),
        dtype=torch.float32,
    )
    # return dense tensor on CPU (safer for numpy-based downstream)
    return s.to_dense().cpu()


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return duration, *result
    return wrapper


def plot_nodes_error_k(data, k_list):
    plt.figure(figsize=(18, 9))
    for k in k_list:
        col_name = "k_" + str(k)
        plt.errorbar(
            data.index,
            data[col_name]["mean"],
            yerr=data[col_name]["std"],
            fmt='-o',
            capsize=0.2,
            capthick=1,
            label=col_name,
        )
        if len(data.index) > 200:
            plt.xticks(
                range(data.index.min(), data.index.max(), len(data.index) // 50),
                rotation=60,
            )
        else:
            plt.xticks(data.index, rotation=60)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Relative Reconstruction Error")
    plt.legend()
    plt.show()


def neighbourhood_symmetric_difference(u_neigh, v_neigh):
    """
    u_neigh, v_neigh: 1D numpy arrays (0/1)
    """
    u_bool = np.asarray(u_neigh).astype(bool)
    v_bool = np.asarray(v_neigh).astype(bool)
    return np.count_nonzero(np.logical_xor(u_bool, v_bool))


def neighbourhood_symmetric_difference(u_neigh, v_neigh):
    """
    u_neigh, v_neigh: 1D numpy arrays (0/1)
    """
    u_bool = np.asarray(u_neigh).astype(bool)
    v_bool = np.asarray(v_neigh).astype(bool)
    return np.count_nonzero(np.logical_xor(u_bool, v_bool))


def pairwise_euclidean(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    X: (n, d) tensor, each row is a d-dim vector.
    Returns: (n, n) tensor D where D[i, j] = ||X[i] - X[j]||_2
    """
    # Gram matrix (dot products)
    G = X @ X.T                 # (n, n)

    # Squared norms of each row: ||x_i||^2
    sq_norms = torch.diag(G)            # (n,)

    # Use (x_i - x_j)^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    dist_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * G

    # Numerical stability: clamp small negatives to 0 before sqrt
    dist_sq = torch.clamp(dist_sq, min=0.0)

    return torch.sqrt(dist_sq + eps)


def measure_encoding_similarity(A, encodings, enc_method="Dist"):
    if enc_method == "None":
        D_sim = torch.zeros(A.shape)
    elif enc_method == "Dist":
        D_sim = pairwise_euclidean(encodings, encodings)
    elif enc_method == "Sim":
        D_sim = encodings @ encodings.T
    elif enc_method == "SimDegree":
        D_sim = encodings @ encodings.T
    elif enc_method == "SimPaths":
        D_sim = encodings @ encodings.T
    else:
        raise ValueError("Unknown encoding method")

    return D_sim


def bin_and_stats(XY):
    """
    XY: list of (x, y) pairs, with x roughly in [0, 1].
    Returns:
      L: list of 20 lists, each containing the (x, y) pairs in that bin
      stats: list of 20 triples (x_k, y_mean, y_std)
             where x_k is the bin center.
    """
    n_bins = 20
    bin_width = 1.0 / n_bins

    # 1) Collect pairs into bins
    L = [[] for _ in range(n_bins)]

    for x, y in XY:
        if x < 0 or x > 1 or math.isnan(x):
            continue  # skip out-of-range; adjust if you want different behavior

        # Map x to bin index k
        k = int(x / bin_width)
        if k == n_bins:  # catch edge case x == 1.0
            k = n_bins - 1
        L[k].append((x, y))

    # 2) Compute stats per bin
    x_list = []
    y_mean_list = []
    y_std_list = []
    for k in range(n_bins):
        bin_pairs = L[k]
        x_k = (k + 0.5) * bin_width  # bin center; use k*bin_width for left edge if preferred

        if bin_pairs:
            ys = np.array([y for _, y in bin_pairs])
            y_mean = float(ys.mean())
            y_std = float(ys.std(ddof=0))  # population std; use ddof=1 for sample std
        else:
            y_mean = math.nan
            y_std = math.nan

        x_list.append(x_k)
        y_mean_list.append(y_mean)
        y_std_list.append(y_std)

    return x_list, y_mean_list, y_std_list


def bin_and_stats_torch(X: torch.Tensor,
                        Y: torch.Tensor,
                        n_bins: int = 20):
    """
    X, Y: PyTorch tensors of the same shape, representing x and y values.
          x is assumed roughly in [0, 1].
          They can be any shape; they will be flattened.

    Returns (all as 1D tensors on the same device as X/Y):
      bin_centers: shape [n_bins]
      y_mean:      shape [n_bins]
      y_std:       shape [n_bins]
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} vs {Y.shape}")

    device = X.device
    dtype = Y.dtype

    # Flatten
    x = X.reshape(-1)
    y = Y.reshape(-1)

    # Mask out invalid x (out of range or NaN)
    valid = (x >= 0) & (x <= 1) & (~torch.isnan(x))
    x = x[valid]
    y = y[valid]

    if x.numel() == 0:
        # No valid data: return NaNs everywhere
        bin_centers = (torch.arange(n_bins, device=device, dtype=X.dtype) + 0.5) / n_bins
        nan = torch.full((n_bins,), float('nan'), device=device, dtype=dtype)
        return bin_centers, nan, nan

    # Map x to bin index k in [0, n_bins-1]
    # Equivalent to int(x / bin_width) with bin_width = 1.0 / n_bins
    # but cheaper and more numerically stable:
    k = (x * n_bins).long()
    k.clamp_(0, n_bins - 1)

    # Prepare accumulators
    y_sum = torch.zeros(n_bins, device=device, dtype=dtype)
    y_sq_sum = torch.zeros(n_bins, device=device, dtype=dtype)
    counts = torch.zeros(n_bins, device=device, dtype=torch.long)

    # Sum y, y^2 and counts per bin using scatter_add_
    y_sum.scatter_add_(0, k, y)
    y_sq_sum.scatter_add_(0, k, y * y)
    counts.scatter_add_(0, k, torch.ones_like(k, dtype=torch.long))

    # Compute mean and std (population std, ddof=0)
    counts_float = counts.clamp(min=1).to(dtype)
    y_mean = y_sum / counts_float
    var = y_sq_sum / counts_float - y_mean ** 2
    var.clamp_(min=0)  # numerical safety
    y_std = torch.sqrt(var)

    # Set NaN where there were no elements in the bin
    empty = counts == 0
    y_mean = y_mean.masked_fill(empty, float('nan'))
    y_std = y_std.masked_fill(empty, float('nan'))

    # Bin centers
    bin_centers = (torch.arange(n_bins, device=device, dtype=X.dtype) + 0.5) / n_bins

    return bin_centers, y_mean, y_std


def mean_of_y_std(stats):
    """
    stats: output of bin_and_stats (list of dicts with key 'y_std')
    Returns: std of all finite y_std values.
    """
    return torch.nanmean(stats)


def load_dataset(name):
    train, val, test = None, None, None
    if name == "ZINC":
        train = ZINC(subset=True, root='data', split='train')
        val   = ZINC(subset=True, root='data', split='val')
        test  = ZINC(subset=True, root='data', split='test')
    elif name == "CIFAR":
        train = GNNBenchmarkDataset(name='CIFAR10', root='data', split='train')
        val   = GNNBenchmarkDataset(name='CIFAR10', root='data', split='val')
        test  = GNNBenchmarkDataset(name='CIFAR10', root='data', split='test')
    elif name == "Peptides":
        train = LRGBDataset(name='Peptides-func', root='data', split='train')
        val   = LRGBDataset(name='Peptides-func', root='data', split='val')
        test  = LRGBDataset(name='Peptides-func', root='data', split='test')

    if train is not None and val is not None and test is not None:
        return train + val + test
    return None


if __name__ == "__main__":
    u = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    v = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    w = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    assert neighbourhood_symmetric_difference(u, v) == 3
    assert neighbourhood_symmetric_difference(u, w) == 3
    assert neighbourhood_symmetric_difference(w, v) == 2
    assert neighbourhood_symmetric_difference(v, v) == 0
