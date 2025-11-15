from src.common.common_new import measure_encoding_similarity, construct_adjacency_matrix, load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    dir_path = sys.argv[2]
    file_name = sys.argv[3]

    data = load_dataset(dataset_name)
    encoding = np.load(dir_path + file_name + ".npz")

    results = []
    for i in tqdm(range(len(encoding))):
        A = construct_adjacency_matrix(data[i])
        sim_measures = measure_encoding_similarity(A, encoding[f"idx_{i}"], "Dist")
        for d, similarities in sim_measures.items():
            for s in similarities:
                results.append(
                    {
                        "graph_id": i,
                        "d": d,
                        "sim": s
                    }
                )

    encoding_norm = np.load(dir_path + file_name + "_norm.npz")
    results_norm = []
    for i in tqdm(range(len(encoding_norm))):
        A = construct_adjacency_matrix(data[i])
        sim_measures_norm = measure_encoding_similarity(A, encoding_norm[f"idx_{i}"], "Sim")
        for d, similarities in sim_measures_norm.items():
            for s in similarities:
                results_norm.append(
                    {
                        "graph_id": i,
                        "d": d,
                        "sim": s
                    }
                )
    
    pd.DataFrame(results).to_parquet("output/similarity_res/similarity_" + file_name + '.parquet')
    pd.DataFrame(results_norm).to_parquet("output/similarity_res/similarity_" + file_name + '_norm.parquet')
