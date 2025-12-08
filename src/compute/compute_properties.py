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


def best_low_rank_approx_error(A: torch.Tensor, rank) -> float:
    """
    Returns the best possible relative low rank approximation error of the given rank for matrix A,
    i.e. sqrt( sum_{i>rank} s_i^2 ) / ||A||_F
    The return value is between 0 and 1, where 0 is the best and 1 the worst
    Best possible refers to min_{rank(B) <= rank} ||A-B||_F
    """
    A = A.double()
    s = singular_values(A)
    min_Frobenius = s[rank:].pow(2).sum().pow(1/2)
    min_Frobenius_normed = min_Frobenius / ( A.max() * A.shape[0] + 1e-12)
    return min_Frobenius_normed.item()


def stable_rank(A: torch.Tensor) -> float:
    """
    srank(A) = ||A||_F^2 / ||A||_2^2 = sum s_i^2 / s_1^2
    """
    s = singular_values(A)
    return float((s.pow(2).sum() / (s[0]**2)).item())


def stable_rank_cut(A: torch.Tensor, rank) -> float:
    """
    srank(A) = ||A||_F^2 / ||A||_2^2 = sum s_i^2 / s_1^2
    """
    s = singular_values(A)[1:rank-1]
    return float((s.pow(2).sum() / (s[1]**2)).item())


def stable_rank_cut_rel(A: torch.Tensor, rank) -> float:
    """
    srank(A) = ||A||_F^2 / ||A||_2^2 = sum s_i^2 / s_1^2
    """
    return stable_rank_cut(A, rank)/rank


def stable_rank_relative(A: torch.Tensor, rank) -> float:
    return stable_rank(A)/rank


def similarity_metric_quality(A: torch.Tensor, rank) -> float:
    return stable_rank_relative(A, rank) - best_low_rank_approx_error(A, rank)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    norms_A = A.pow(2).sum(dim=1).pow(1/2).unsqueeze(1)
    norms_B = B.pow(2).sum(dim=1).pow(1/2).unsqueeze(1)
    A_normed = A / (norms_A + 1e-12)
    B_normed = B / (norms_B + 1e-12)
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


def get_path_counts(A, length):
    A = A.float()
    A_paths = torch.matrix_power(A, length)
    return A_paths

def get_path_probabilities(A, length):
    A = A.float()
    A = A / (A.sum(dim=1) + 1e-12)
    A = A.T
    A_paths = torch.matrix_power(A, length)
    return A_paths


def get_path_probabilities_reduce_error(A):
    A = A.float()
    A = A / (A.sum(dim=1) + 1e-12)
    A = A.T
    path_length = 20
    A_paths_20 = get_path_counts(A, path_length)
    A_paths = A_paths_20
    i = 0
    while best_low_rank_approx_error(A_paths, 24) > 0.01:
        A_paths = A_paths @ A_paths_20
        i += 1
    final_path_length = path_length * i

    print("Path length:", final_path_length)
    return A_paths


def symmetrize_probabilities(M):
    return (M * M.T).pow(1/2)


def get_path_similarity(A, length_min, length_max):
    p_sum = torch.zeros(A.shape).to(A.device)
    for i in range(length_min, length_max+1):
        p = get_path_probabilities(A, i)
        p_sum = p_sum + p
    mean_p = p_sum / (length_max+1 - length_min)
    norms = mean_p.pow(2).sum(dim=1).pow(1/2).unsqueeze(1)
    norms = norms + 1e-12
    mean_p = mean_p / norms
    sim = mean_p @ mean_p.T
    return sim


def get_path_distance(A, length_min, length_max):
    p_concat = torch.zeros((A.shape[0], length_max - length_min + 1)).to(A.device)
    for i in range(length_min, length_max+1):
        p = get_path_probabilities(A, i)
        p = p.sum(dim=0)
        p_concat[:, i-length_min] = p
    return p_concat


def get_path_probabilities_mean(A, length_min, length_max):
    p_sum = torch.zeros(A.shape).to(A.device)
    for i in range(length_min, length_max+1):
        p = get_path_probabilities(A, i)
        p_sum = p_sum + p
    index = p_sum / (length_max+1 - length_min)
    return index


def get_jaccard_index(A):
    diff = A @ (1-A.T)
    sim = A @ A.T
    W = sim / (diff + sim + 1e-12)
    return W


def get_degree_similarity(A):
    # Degree similarity
    A = A.float()
    v = A.sum(dim=0)
    W_asymmetric = v[:, None] @ (1 / (v[None, :] + 1e-12))
    W = torch.min(W_asymmetric, W_asymmetric.T)
    return W


def get_sim_targets(A, enc_method="Dist", device=None):
    A = torch.tensor(A).to(device)

    if enc_method == "None":
        W = torch.zeros(A.shape)
    elif enc_method == "Dist":
        W = get_neighbourhood_diff
    elif enc_method == "Sim":
        W = get_jaccard_index(A)
    elif enc_method == "SimDegree":
        W = get_degree_similarity(A)
    elif enc_method == "SimPaths":
        W = get_path_similarity(A, 10, 15)
    else:
        raise ValueError(f"Unknown method {enc_method}")

    if torch.isnan(W).sum() > 0:
        raise RuntimeError("W contains NaN values")

    return W

class properties:
    def __init__(self, A, device):
        self.device = device
        self.A = torch.tensor(A).to(device)

        self.jaccard_index = None
        self.degree_similarity = None
        self.neighbourhood_sim = None
        self.neighbourhood_diff = None
        self.paths_count = None
        self.paths_probabilities = None
        self.paths_similarity = None
        self.paths_distance = None

        self.jaccard_index_flat = None
        self.degree_similarity_flat = None
        self.neighbourhood_diff_flat = None
        self.neighbourhood_sim_flat = None
        self.paths_count_flat = None
        self.paths_probabilities_flat = None
        self.paths_similarity_flat = None
        self.paths_distance_flat = None

        self.jaccard_index_rank = None
        self.degree_similarity_rank = None
        self.neighbourhood_diff_rank = None
        self.neighbourhood_sim_rank = None
        self.paths_count_rank = None
        self.paths_probabilities_rank = None
        self.paths_similarity_rank = None
        self.paths_distance_rank = None

        self.jaccard_index_stable_rank = None
        self.degree_similarity_stable_rank = None
        self.neighbourhood_diff_stable_rank = None
        self.neighbourhood_sim_stable_rank = None
        self.paths_count_stable_rank = None
        self.paths_probabilities_stable_rank = None
        self.paths_similarity_stable_rank = None
        self.paths_distance_stable_rank = None

        self.jaccard_index_min_error = None
        self.degree_similarity_min_error = None
        self.neighbourhood_diff_min_error = None
        self.neighbourhood_sim_min_error = None
        self.paths_count_min_error = None
        self.paths_probabilities_min_error = None
        self.paths_similarity_min_error = None
        self.paths_distance_min_error = None


    def compute(self):
        self.jaccard_index = get_jaccard_index(self.A).double()
        self.degree_similarity = get_degree_similarity(self.A).double()
        self.neighbourhood_sim = get_neighbourhood_sim(self.A).double()
        self.neighbourhood_diff = get_neighbourhood_diff(self.A).double()

        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)

        self.jaccard_index_flat = self.jaccard_index[idx[0], idx[1]].cpu().numpy()
        self.degree_similarity_flat = self.degree_similarity[idx[0], idx[1]].cpu().numpy()
        self.neighbourhood_diff_flat = self.neighbourhood_diff[idx[0], idx[1]].cpu().numpy()
        self.neighbourhood_sim_flat = self.neighbourhood_sim[idx[0], idx[1]].cpu().numpy()

        self.jaccard_index_rank = torch.linalg.matrix_rank(self.jaccard_index)
        self.degree_similarity_rank = torch.linalg.matrix_rank(self.degree_similarity)
        self.neighbourhood_diff_rank = torch.linalg.matrix_rank(self.neighbourhood_diff)
        self.neighbourhood_sim_rank = torch.linalg.matrix_rank(self.neighbourhood_sim)

        self.jaccard_index_stable_rank = stable_rank(self.jaccard_index)
        self.degree_similarity_stable_rank = stable_rank(self.degree_similarity)
        self.neighbourhood_diff_stable_rank = stable_rank(self.neighbourhood_diff)
        self.neighbourhood_sim_stable_rank = stable_rank(self.neighbourhood_sim)

        rank = 8

        self.jaccard_index_min_error = best_low_rank_approx_error(self.jaccard_index, rank=rank)
        self.degree_similarity_min_error = best_low_rank_approx_error(self.degree_similarity, rank=rank)
        self.neighbourhood_diff_min_error = best_low_rank_approx_error(self.neighbourhood_diff, rank=rank)
        self.neighbourhood_sim_min_error = best_low_rank_approx_error(self.neighbourhood_sim, rank=rank)

    def compute_paths_count(self, length, rank):
        self.paths_count = None
        self.paths_count = get_path_counts(self.A, length).double()
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_count_flat =self.paths_count[idx[0], idx[1]].cpu().numpy()
        self.paths_count_rank = torch.linalg.matrix_rank(self.paths_count)
        self.paths_count_stable_rank = stable_rank(self.paths_count)
        self.paths_count_min_error = best_low_rank_approx_error(self.paths_count, rank=rank)
        print(f"Paths count, Rank: {self.paths_count_rank}, Stable rank: {self.paths_count_stable_rank}, Min error: {self.paths_count_min_error}")

    def compute_paths_probabilities(self, length, rank):
        self.paths_probabilities = None
        self.paths_probabilities = get_path_probabilities(self.A, length).double()
        self.paths_probabilities = (self.paths_probabilities * self.paths_probabilities.T).pow(1/2)
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_probabilities_flat =self.paths_probabilities[idx[0], idx[1]].cpu().numpy()
        self.paths_probabilities_rank = torch.linalg.matrix_rank(self.paths_probabilities)
        self.paths_probabilities_stable_rank = stable_rank(self.paths_probabilities)
        self.paths_probabilities_min_error = best_low_rank_approx_error(self.paths_probabilities, rank=rank)
        print(f"Paths probabilities, Rank: {self.paths_probabilities_rank}, Stable rank: {self.paths_probabilities_stable_rank}, Min error: {self.paths_probabilities_min_error}")

    def compute_paths_probabilities_mean(self, length_min, length_max, rank):
        self.paths_probabilities = None
        self.paths_probabilities = get_path_probabilities_mean(self.A, length_min, length_max).double()
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_probabilities_flat =self.paths_probabilities[idx[0], idx[1]].cpu().numpy()
        self.paths_probabilities_rank = torch.linalg.matrix_rank(self.paths_probabilities)
        self.paths_probabilities_stable_rank = stable_rank(self.paths_probabilities)
        self.paths_probabilities_min_error = best_low_rank_approx_error(self.paths_probabilities, rank=rank)
        print(f"Paths probabilities, Rank: {self.paths_probabilities_rank}, Stable rank: {self.paths_probabilities_stable_rank}, Min error: {self.paths_probabilities_min_error}")

    def compute_paths_similarity(self, length_min, length_max, rank):
        self.paths_similarity = None
        self.paths_similarity = get_path_similarity(self.A, length_min, length_max).double()
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_similarity_flat =self.paths_similarity[idx[0], idx[1]].cpu().numpy()
        self.paths_similarity_rank = torch.linalg.matrix_rank(self.paths_similarity)
        self.paths_similarity_stable_rank = stable_rank(self.paths_similarity)
        self.paths_similarity_min_error = best_low_rank_approx_error(self.paths_similarity, rank=rank)
        print(f"Paths similarity, Rank: {self.paths_similarity_rank}, Stable rank: {self.paths_similarity_stable_rank}, Min error: {self.paths_similarity_min_error}")

    def compute_paths_distance(self, length_min, length_max, rank):
        self.paths_distance = None
        self.paths_distance = get_path_distance(self.A, length_min, length_max).double()
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_distance_flat =self.paths_distance[idx[0], idx[1]].cpu().numpy()
        self.paths_distance_rank = torch.linalg.matrix_rank(self.paths_distance)
        self.paths_distance_stable_rank = stable_rank(self.paths_distance)
        self.paths_distance_min_error = best_low_rank_approx_error(self.paths_similarity, rank=rank)
        print(f"Paths similarity, Rank: {self.paths_distance_rank}, Stable rank: {self.paths_distance_stable_rank}, Min error: {self.paths_distance_min_error}")

    def compute_paths_probabilities_optimize(self, rank):
        self.paths_probabilities = None
        self.paths_probabilities = get_path_probabilities_reduce_error(self.A).double()
        self.paths_probabilities = symmetrize_probabilities(self.paths_probabilities * self.paths_probabilities.T)
        # upper-triangular indices
        idx = torch.triu_indices(self.A.shape[0], self.A.shape[0], offset=1)
        self.paths_probabilities_flat =self.paths_probabilities[idx[0], idx[1]].cpu().numpy()
        self.paths_probabilities_rank = torch.linalg.matrix_rank(self.paths_probabilities)
        self.paths_probabilities_stable_rank = stable_rank(self.paths_probabilities)
        self.paths_probabilities_min_error = best_low_rank_approx_error(self.paths_probabilities, rank=rank)
        print(f"Optimized probabilities, Rank: {self.paths_probabilities_rank}, Stable rank: {self.paths_probabilities_stable_rank}, Min error: {self.paths_probabilities_min_error}")


if __name__ == "__main__":
    dataset_name = sys.argv[1]

    data = load_dataset(dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    results = []
    for i in tqdm(range(len(data))):
        A = construct_adjacency_matrix(data[i])
        p = properties(A, device)
        p.compute()

        p.compute_paths_count(15, 12)
        p.compute_paths_probabilities_mean(10, 15, 8)
        p.compute_paths_similarity(10, 15, 8)
        p.compute_paths_distance(1, 15, 8)
        print("\n")

        results.append(
            {
                "graph_id": i,
                "neighbourhood_diff_flat": p.neighbourhood_diff_flat,
                "neighbourhood_sim_flat": p.neighbourhood_sim_flat,
                "paths_count_flat": p.paths_count_flat,
                "jaccard_index_flat": p.jaccard_index_flat,
                "degree_similarity_flat": p.degree_similarity_flat
            }
        )

    pd.DataFrame(results).to_parquet("output/properties/properties_" + dataset_name + '.parquet')
