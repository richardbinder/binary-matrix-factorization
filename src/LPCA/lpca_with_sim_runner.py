import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import sys
from common import construct_adjacency_matrix, load_dataset
from lpca_with_sim import lpca_encoding


def compute_encodings(data, k, out_path, bound=None, gamma=0.5, n_samples=None):
    matrices = {}
    results = []

    idx_max = len(data) if n_samples is None else n_samples

    for i in tqdm(range(idx_max)):
        A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))
        t, error, d_mean, d_std, nit, enc = lpca_encoding(A, k, bound, gamma)
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
    # python lpca_with_sim_runner.py ZINC 4 8 10 1000
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

    compute_encodings(data, k, out_path, bound, gamma, n_samples)

    print("computed encodings:", out_path)
