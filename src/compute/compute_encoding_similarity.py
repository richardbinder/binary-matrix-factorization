from src.common.common import measure_encoding_similarity, construct_adjacency_matrix, load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def plot_neighbourhood_similarity(df_path, title):
    df = pd.read_parquet(df_path)
    data = df.groupby("d")["sim"].agg(["mean", "std"])

    plt.errorbar(data.index, data["mean"], yerr=data["std"], fmt='-o', capsize=0.2, capthick=1)
    plt.xlabel("neighbourhood symmetric difference")
    plt.ylabel("average distance between \n distinct node encoding pairs")
    plt.title(title)
    plt.savefig("neighbourhood_similarity_plot.png")
    plt.show()

def plot_neighbourhood_similarity_multiple(df_paths, title):
    plt.rcParams.update({'font.size': 16})
    for label, df_path in df_paths.items():
        df = pd.read_parquet(df_path)
        data = df.groupby("d")["sim"].agg(["mean", "std"])

        plt.errorbar(data.index, data["mean"], yerr=data["std"], fmt='-o', capsize=0.2, capthick=1, label=label)
    plt.xlabel("neighbourhood symmetric difference")
    plt.ylabel("average distance between \n distinct node encoding pairs")
    plt.title(title)
    plt.legend()
    plt.savefig("neighbourhood_similarity_plot.png")
    plt.show()


def plot_diff_histogram(input_path, title):
    plt.bar(*np.unique(pd.read_parquet(input_path)["d"], return_counts=True))
    plt.title(title)
    plt.xlabel("neighbourhood symmetric difference")
    plt.ylabel("number of node pairs")
    plt.show()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    dir_path = sys.argv[2]
    file_name = sys.argv[3]

    data = load_dataset(dataset_name)
    encoding = np.load(dir_path + file_name + ".npz")

    results = []

    for i in tqdm(range(len(encoding))):
        A = construct_adjacency_matrix(data[i])
        sim_measures = measure_encoding_similarity(A, encoding[f"idx_{i}"])
        for d, similarities in sim_measures.items():
            for s in similarities:
                results.append(
                    {
                        "graph_id": i,
                        "d": d,
                        "sim": s
                    }
                )
    
    pd.DataFrame(results).to_parquet("output/similarity_res/similarity_" + file_name + '.parquet')
