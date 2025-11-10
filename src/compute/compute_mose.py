from src.common.common import load_dataset
import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# SOURCE: https://github.com/linusbao/MoSE/blob/main/hombasis-gt/hombasis-bench/data/get_data.py
def add_zinc_subhom(
    name, hom_files, idx_list, sub_file, root, dataset):
    if name == "ZINC":
        original_data = [dataset[i] for i in range(len(dataset))]

    matrices = {}

    all_hom_data = []
    for hom_file in hom_files:
        hom_path = os.path.join(root, hom_file)
        hom_data = json.load(open(hom_path))
        all_hom_data.append(hom_data)

    sub_data = json.load(open(os.path.join(root, sub_file)))

    for graph_idx in tqdm(range(len(original_data))):

        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):

            vertex_counts = []
            for hom_list in all_hom_data:
                homcounts = hom_list[str(graph_idx)]["homcounts"][str(v_idx)]
                vertex_counts += homcounts

            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()

            if "anchor" in sub_file:
                sub_counts = sub_data[str(graph_idx)]["subcounts"][str(v_idx)][
                    :-2
                ]  # for anchored spasm
            else:
                sub_counts = sub_data[str(graph_idx)][str(v_idx)]  # for spasm
            vertex_counts += sub_counts
            graph_counts.append(vertex_counts)

        matrices[f"idx_{graph_idx}"] = torch.Tensor(graph_counts)

    np.savez_compressed("mose_" + name + '.npz', **matrices)


if __name__ == "__main__":
    dataset = load_dataset("ZINC")
    
    count_files = ['zinc_with_homs_c7.json', 'zinc_with_homs_c8.json']
    idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 15, 20, 21, 22, 24, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    sub_file = 'zinc_3to8C_multhom.json'

    add_zinc_subhom(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root="data", dataset=dataset)
