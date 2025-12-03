from src.common.common import construct_adjacency_matrix, load_dataset, neighbourhood_symmetric_difference
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
import torch


def singular_values(A: torch.Tensor) -> torch.Tensor:
    """Returns singular values in descending order (float64 for stability)."""
    A = A.double()
    return torch.linalg.svdvals(A)  # sorted desc


def stable_rank(A: torch.Tensor) -> float:
    """
    srank(A) = ||A||_F^2 / ||A||_2^2 = sum s_i^2 / s_1^2
    """
    s = singular_values(A)
    return float((s.pow(2).sum() / (s[0]**2)).item())


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    norms_A = A.pow(2).sum(dim=1).pow(1/2).unsqueeze(1)
    norms_B = B.pow(2).sum(dim=1).pow(1/2).unsqueeze(1)
    A_normed = A / norms_A
    B_normed = B / norms_B
    sim = A_normed @ B_normed.T
    return sim


def get_neighbourhood_diff(A):
    A_bool = A.bool()
    xor_all = torch.logical_xor(A_bool[:, None, :], A_bool[None, :, :])  # (n,n,d)
    neighbourhood_diff = xor_all.sum(dim=2)
    return neighbourhood_diff


def get_neighbourhood_sim(A):
    A_bool = A.bool()
    and_all = torch.logical_and(A_bool[:, None, :], A_bool[None, :, :])  # (n,n,d)
    neighbourhood_sim = and_all.sum(dim=2)
    return neighbourhood_sim


def get_paths(A, length):
    length -= 1
    A = A.float()
    A_paths = A
    for i in range(length):
        A_paths = A_paths @ A
    return A_paths


def get_jaccard_index(A):
    diff = get_neighbourhood_diff(A)
    sim = get_neighbourhood_sim(A)
    W = sim / (diff + sim)
    return W


def get_degree_similarity(A):
    # Degree similarity
    A = A.float()
    v = A.sum(dim=0)
    W_asymmetric = v[:, None] @ (1 / v[None, :])
    W = torch.min(W_asymmetric, W_asymmetric.T)
    return W


def get_sim_targets(A, enc_method="Dist", device=None):
    A = torch.tensor(A).to(device)

    if enc_method == "Dist":
        W = get_neighbourhood_diff
    elif enc_method == "Sim":
        W = get_jaccard_index(A)
    elif enc_method == "SimDegree":
        W = get_degree_similarity(A)
    else:
        raise ValueError(f"Unknown method {enc_method}")

    # r_D = torch.linalg.matrix_rank(neighbourhood_diff.float())
    # r_S = torch.linalg.matrix_rank(neighbourhood_sim.float())
    # r_A = torch.linalg.matrix_rank(A.float())
    # r_W = torch.linalg.matrix_rank(W.float())
    #
    # rs_D = stable_rank(neighbourhood_diff.float())
    # rs_S = stable_rank(neighbourhood_sim.float())
    # rs_A = stable_rank(A)
    # rs_W = stable_rank(W)

    return W


def compute_properties(A, device):
    A = torch.tensor(A).to(device)

    jaccard_index = get_jaccard_index(A)
    degree_similarity = get_degree_similarity(A)
    neighbourhood_sim = get_neighbourhood_sim(A)
    neighbourhood_diff = get_neighbourhood_diff(A)
    paths_3 = get_paths(A, 3)

    # upper-triangular indices
    idx = torch.triu_indices(A.shape[0], A.shape[0], offset=1)

    jaccard_index = jaccard_index[idx[0], idx[1]].cpu().numpy()
    degree_similarity = degree_similarity[idx[0], idx[1]].cpu().numpy()
    neighbourhood_diff = neighbourhood_diff[idx[0], idx[1]].cpu().numpy()
    neighbourhood_sim = neighbourhood_sim[idx[0], idx[1]].cpu().numpy()
    paths_3 = paths_3[idx[0], idx[1]].cpu().numpy()

    return jaccard_index, degree_similarity, neighbourhood_sim, neighbourhood_diff, paths_3

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    data = load_dataset(dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for i in tqdm(range(len(data))):
        A = construct_adjacency_matrix(data[i])
        jaccard_index, degree_similarity, neighbourhood_sim, neighbourhood_diff, paths_3 = compute_properties(A, device)

        results.append(
            {
                "graph_id": i,
                "neighbourhood_diff": neighbourhood_diff,
                "neighbourhood_sim": neighbourhood_sim,
                "paths_3": paths_3,
                "jaccard_index": jaccard_index,
                "degree_similarity": degree_similarity
            }
        )

    pd.DataFrame(results).to_parquet("output/properties/properties_" + dataset_name + '.parquet')
