import numpy as np
import scipy as sp
import pandas as pd
from src.common.common import construct_adjacency_matrix, load_dataset
from LPCA.lpca import decomposition_at_k
from tqdm import tqdm
import sys
from tSVD.tSVD import decomposition_at_k_with_error, init_svd


def stack_matrices_encoding(L, R):
    return np.hstack((L, np.transpose(R)))

def format_LPCA_encoding(data):
    return stack_matrices_encoding(data['U'], data['V'])

def format_LPCA_encoding_fixed(data):
    return data['U']


def save_encodings(matrices, out_path):
    names = [f"idx_{i}" for i in range(len(matrices))]
    np.savez_compressed(out_path, **dict(zip(names, matrices)))


def compute_encodings(data, k, out_path, format_fun, config, method="LPCA",n_samples=None):
    matrices = {}
    results = []

    n_hops = config["n_hops"]
    
    idx_max = len(data) if n_samples is None else n_samples 
    print(idx_max)
    for i in tqdm(range(idx_max)):
        A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))

        while n_hops > 1:
            A = A @ A
            n_hops -= 1
        

        if method == "LPCA":
            A = sp.sparse.csr_matrix(construct_adjacency_matrix(data[i]))
            t, error, nit, res = decomposition_at_k(A, k, config)
            matrices[f"idx_{i}"] = format_fun(res)
        elif method == "tSVD":
            A = construct_adjacency_matrix(data[i])
            t, error, L, R = decomposition_at_k_with_error(config["svd"], A, k)
            nit = None
            matrices[f"idx_{i}"] = stack_matrices_encoding(L, R)
        else:
            raise ValueError("unknown method")
        
        results.append(
            {
                "graph_id": i,
                "n_nodes": data[i].x.shape[0],
                "nit": nit,
                "error": error,
                "time": t
            }
        )
    
    np.savez_compressed(out_path + '.npz', **matrices)
    pd.DataFrame(results).to_parquet(out_path + '.parquet')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the dataset name.")
        exit()
    
    if len(sys.argv) < 3:
        print("Please provide the bounds range.")
        exit()

    if len(sys.argv) < 4:
        print("Please provide the dimension k.")
        exit()

    name = sys.argv[1]
    data = load_dataset(name)

    config = {
        "n_hops": 1,
        "is_single": False,
        "fixed_V": None,
        "bounds": None,
        "max_iter": 5000
    }

    method="LPCA"

    if sys.argv[2].lower() != "none":
        b = int(sys.argv[2])
        config["bounds"] = (-b, b)
    
    k = int(sys.argv[3])
    
    if len(sys.argv) > 4:
        config["n_hops"] = int(sys.argv[4])

    if len(sys.argv) > 5:
        config["is_single"] = sys.argv[5] == "True"

    if len(sys.argv) > 6 and sys.argv[6] == "True":
        config["fixed_V"] = -1+2*np.random.random(size=k*k).reshape((k, k))
    
    if len(sys.argv) > 7:
        config["p_lambda"] = float(sys.argv[7])
    
    n_samples = None
    if len(sys.argv) > 8:
        n_samples = int(sys.argv[8])

    if len(sys.argv) > 9:
        method = sys.argv[9]

    # lb, ub = bounds
    # N = 40
    # k = 16
    # V = lb+ub+1*np.random.random(size=N*k).reshape((k, N))
    #compute_encodings(data, 2, 'lpca_out/lpca2b_4_4', format_LPCA_encoding, bounds)

    # python compute_encodings.py ZINC 4 16 1 False True
    
    if method == "LPCA":
        out_path = f"lpca_out/lpca_{name}_{k}_b{config['bounds'][1] if config['bounds'] is not None else 'N'}_{config['n_hops']}hop_single{config['is_single']}_fixedV{config['fixed_V'] is not None}_lambda{config.get('p_lambda')}"
    else:
        out_path = f"tsvd_out/tsvd_dim{k}"
    print("Computing endocing:", out_path)

    if method == "tSVD":
        config["svd"] = init_svd(k)
    compute_encodings(data, k, out_path, 
                      format_LPCA_encoding if config['fixed_V'] is None else format_LPCA_encoding_fixed, 
                      config,
                      method, 
                      n_samples)
    