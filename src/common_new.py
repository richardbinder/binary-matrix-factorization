import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, LRGBDataset


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


def measure_encoding_similarity(A, encodings):
    """
    A: adjacency matrix of shape (n, n)
       - can be numpy array or torch tensor
    encodings: numpy array of shape (n, d)

    Returns:
        dict: distance (neighborhood symmetric difference) -> list of encoding distances
    """
    # Make sure A is a numpy array
    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy()
    else:
        A_np = np.asarray(A)

    enc_np = np.asarray(encodings)

    similarity = {}
    n_nodes = A_np.shape[0]

    for v in range(n_nodes):
        for w in range(v + 1, n_nodes):
            d = neighbourhood_symmetric_difference(A_np[v], A_np[w])

            if d not in similarity:
                similarity[d] = []
            similarity[d].append(np.linalg.norm(enc_np[v] - enc_np[w]))

    return similarity


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
