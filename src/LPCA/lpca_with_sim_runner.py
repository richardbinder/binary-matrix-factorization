import sys
from time import process_time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.compute.compute_properties import get_sim_targets, best_low_rank_approx_error
from src.LPCA.lpca_losses import lpca_dist_loss, lpca_sim_loss_params, lpca_loss, lpca_p_loss, lpca_sim_loss

from src.common.common_lpca import (
    construct_adjacency_matrix,
    load_dataset,
    time_wrapper,
    measure_encoding_similarity,
    bin_and_stats,
    mean_of_y_std, bin_and_stats_torch
)
from src.compute.svd import psd_factor_svd

enc_method = "Dist"


def normalize_enc_torch(L, eps=1e-8):
    """
    Row-wise L2-normalize a 2D tensor.

    L: (n, d) torch tensor
    returns: (n, d) tensor
    """
    norms = torch.norm(L, dim=1, keepdim=True)
    return L / (norms + eps)


def handle_bound(L, R, bound=None):
    # approximate bound handling by projection
    pass
    # if bound is not None:
    #     with torch.no_grad():
    #         L.clamp_(-bound, bound)
    #         R.clamp_(-bound, bound)


def closure(optimizer, handle_bound_fnc, loss_fnc):
    optimizer.zero_grad()

    loss, lpca_loss, sim_loss = loss_fnc()
    loss.backward()

    handle_bound_fnc()

    return loss, lpca_loss, sim_loss


@time_wrapper
def lpca_encoding(A, k, W, bound=None, gamma=0.5, device=None):
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
    A = A.to(device=device, dtype=torch.float32)

    n = A.shape[0]

    # shifted adjacency: -1 for 0, +1 for 1
    adj_s = -1.0 + 2.0 * A  # (n, n), in {-1, +1}

    # Initialize factors L, R in [-1, 1]
    # L = torch.empty((n, k), device=device).uniform_(-1.0, 1.0)
    # R = torch.empty((k, n), device=device).uniform_(-1.0, 1.0)

    X = psd_factor_svd(W, k=k * 2)
    L = X[:, :k]
    R = X[:, k:].T

    L.requires_grad_(True)
    R.requires_grad_(True)

    params = torch.empty_like(A).uniform_(-1.0, 1.0)
    params.requires_grad_(True)

    # optimizer = torch.optim.LBFGS(
    #     [L, R],
    #     max_iter=300,
    #     line_search_fn="strong_wolfe",
    # )

    optimizer = torch.optim.AdamW(
        [L, R, params],
        lr=5e-1,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,       # first restart after 5 epochs
        T_mult=1,    # 5, 10, 20, ...
        eta_min=1e-4 # minimum lr
    )

    # not used
    weights = torch.zeros(A.shape)

    final_loss = 0
    final_lpca_loss = 0
    final_sim_loss = 0

    if enc_method == "None":
        loss_fnc = lambda: lpca_loss(L, R, adj_s)
    elif enc_method == "Dist":
        loss_fnc = lambda: lpca_dist_loss(L, R, adj_s, W, weights, params, gamma=gamma)
    elif enc_method == "Sim":
        loss_fnc = lambda: lpca_sim_loss(L, R, adj_s, W, gamma=gamma)
    elif enc_method == "SimDegree":
        loss_fnc = lambda: lpca_sim_loss(L, R, adj_s, W, gamma=gamma)
    elif enc_method == "SimPaths":
        loss_fnc = lambda: lpca_sim_loss_params(L, R, adj_s, W, params, gamma=gamma)
    else:
        raise ValueError(f"Unknown method {enc_method}")

    for epoch in range(100):
        handle_bound_fnc = lambda: handle_bound(L, R, bound)
        final_loss, final_lpca_loss, final_sim_loss = closure(optimizer, handle_bound_fnc, loss_fnc)
        optimizer.step()
        scheduler.step(epoch)

    # optimizer.step(closure)

    # Try to read iterations from optimizer state (may not always be present)
    state = optimizer.state.get(L, {})
    nit = int(state.get("n_iter", 0))

    # Build final normalized encoding on CPU for downstream numpy-based stuff
    with torch.no_grad():
        enc = torch.cat([L, R.t()], dim=1)  # (n, 2k)
        enc_norm = normalize_enc_torch(enc)
        enc = enc

    # Reconstruct adjacency from normalized encodings
    L_n = enc_norm[:, :k]
    R_n = enc_norm[:, k:]
    A_reconstructed = (L_n @ R_n.T > 0).float()

    # compute reconstruction error
    num = torch.linalg.norm(A_reconstructed - A)
    denom = torch.linalg.norm(A)
    error = num / denom if denom != 0 else 0.0

    # similarity stats
    sim_pairs = measure_encoding_similarity(A, enc_norm, enc_method)
    d_list, sim_mean_list, sim_std_list = bin_and_stats_torch(sim_pairs, W)

    best_possible_sim_loss = best_low_rank_approx_error(W, rank=k*2)

    return final_loss, final_lpca_loss, final_sim_loss, best_possible_sim_loss, error, d_list, sim_mean_list, sim_std_list, nit, enc_norm, enc


def compute_encodings(data, k, out_path, bound=None, gamma=0.5, n_samples=None, device=None):
    matrices_norm = {}
    matrices = {}
    results = []

    idx_max = len(data) if n_samples is None else n_samples

    Ws = []
    for i in tqdm(range(idx_max)):
        A = construct_adjacency_matrix(data[i])
        W = get_sim_targets(A, enc_method, device=device)
        Ws.append(W)

    tqdm.write("\n")

    for i in tqdm(range(idx_max)):
        # Now returns a dense torch tensor adjacency
        A = construct_adjacency_matrix(data[i])

        t, final_loss, final_lpca_loss, final_sim_loss, best_possible_sim_loss, error, d_list, sim_mean_list, sim_std_list, nit, enc_norm, enc = lpca_encoding(A, k, Ws[i], bound, gamma, device)

        tqdm.write(f"Rec. Error: {error:.2e}, "
                   f"Sim mean std: {mean_of_y_std(sim_std_list):.2e}, "
                   f"Final Loss: {final_loss:.2e}, "
                   f"Final LPCA Loss: {final_lpca_loss:.2e}, "
                   f"Final Sim Loss: {final_sim_loss:.2e}, "
                   f"Best possible Sim Loss: {best_possible_sim_loss:.2e}")

        results.append(
            {
                "graph_id": i,
                "n_nodes": data[i].x.shape[0],
                "nit": nit,
                "error": error.item(),
                "time": t,
                "d": d_list.cpu().numpy(),
                "sim_mean": sim_mean_list.cpu().numpy(),
                "sim_std": sim_std_list.cpu().numpy(),
            }
        )

        matrices_norm[f"idx_{i}"] = enc_norm.cpu().numpy()
        matrices[f"idx_{i}"] = enc.cpu().numpy()

    np.savez_compressed(out_path + "_norm.npz", **matrices_norm)
    np.savez_compressed(out_path + ".npz", **matrices)
    pd.DataFrame(results).to_parquet(out_path + ".parquet")


if __name__ == "__main__":
    # python lpca_with_sim_runner_torch.py ZINC 4 8 10 1000
    name = sys.argv[1]

    print("\n")
    print("#################################")
    print(f"Computing Encodings for {name}")
    print("#################################\n")

    enc_method = sys.argv[2]

    data = load_dataset(name)

    bound = None
    if sys.argv[3].lower() != "none":
        bound = int(sys.argv[3])

    k = int(sys.argv[4])
    gamma = float(sys.argv[5])
    n_samples = None

    if len(sys.argv) > 6:
        n_samples = int(sys.argv[6])

    out_path = f"lpca_out/lpca_with_sim_new_{name}_method{enc_method}_k{k}_b{bound}_gamma{gamma}_s{n_samples}"

    compute_encodings(data, k, out_path, bound, gamma, n_samples, "cpu")

    print("computed encodings:", out_path)
