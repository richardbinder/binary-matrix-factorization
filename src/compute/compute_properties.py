from src.common.common import construct_adjacency_matrix, load_dataset, neighbourhood_symmetric_difference
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
import torch


def get_sim_targets(A, enc_method="Dist", eps=1e-1, device=None):
    A = torch.tensor(A).to(device)
    neighbourhood_diff, neighbourhood_sim, neighbourhood_path3 = compute_neighbourhood_properties(A)

    if enc_method == "Dist":
        W = neighbourhood_diff
    elif enc_method == "Sim":
        W = 10*neighbourhood_sim + neighbourhood_path3
        w_max = W.max()

        # max over rows: shape (4, 1)
        row_max = W.max(dim=1, keepdim=True).values
        # max over columns: shape (1, 5)
        col_max = W.max(dim=0, keepdim=True).values
        # for each (i, j): denom[i, j] = max( row_max[i], col_max[j] )
        denom = torch.maximum(row_max, col_max)
        W = W / denom
        W = W * w_max
    else:
        raise ValueError(f"Unknown method {enc_method}")

    r_D = torch.linalg.matrix_rank(neighbourhood_diff.float())
    r_A = torch.linalg.matrix_rank(A.float())
    r_W = torch.linalg.matrix_rank(W.float())

    return W, neighbourhood_diff, neighbourhood_sim, neighbourhood_path3


def compute_neighbourhood_properties(A):
    A_float = A.float()
    product = torch.matmul(A_float, A_float)
    product = torch.matmul(product, A_float)

    A_bool = A.bool()
    xor_all = torch.logical_xor(A_bool[:, None, :], A_bool[None, :, :])  # (n,n,d)
    and_all = torch.logical_and(A_bool[:, None, :], A_bool[None, :, :])  # (n,n,d)

    neighbourhood_diff = xor_all.sum(dim=2)
    neighbourhood_sim = and_all.sum(dim=2)
    neighbourhood_path3 = product

    return neighbourhood_diff, neighbourhood_sim, neighbourhood_path3


def compute_properties(A, device):
    A = torch.tensor(A).to(device)
    W, neighbourhood_diff, neighbourhood_sim, neighbourhood_path3 = get_sim_targets(A, enc_method="Sim", device=device)

    # upper-triangular indices
    idx = torch.triu_indices(A.shape[0], A.shape[0], offset=1)

    w = W[idx[0], idx[1]].cpu().numpy()
    diff = neighbourhood_diff[idx[0], idx[1]].cpu().numpy()
    sim = neighbourhood_sim[idx[0], idx[1]].cpu().numpy()
    path3 = neighbourhood_path3[idx[0], idx[1]].cpu().numpy()
    return w, diff, sim, path3

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    data = load_dataset(dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for i in tqdm(range(len(data))):
        A = construct_adjacency_matrix(data[i])
        w, neighbourhood_diff, neighbourhood_sim, neighbourhood_path3 = compute_properties(A, device)
        results.append(
            {
                "graph_id": i,
                "d": neighbourhood_diff,
                "sim": neighbourhood_sim,
                "path3": neighbourhood_path3,
                "w": w
            }
        )

    pd.DataFrame(results).to_parquet("output/properties/properties_" + dataset_name + '.parquet')
