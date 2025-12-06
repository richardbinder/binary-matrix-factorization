#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from compute_properties import stable_rank, stable_rank_relative, best_low_rank_approx_error, singular_values, stable_rank_cut, stable_rank_cut_rel

def cosine_sim_matrix(X: torch.Tensor, block: int = 0) -> torch.Tensor:
    # Normalize rows to unit length
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    n = X.shape[0]
    if block <= 0 or block >= n:
        return X @ X.T
    S = torch.empty((n, n), device=X.device, dtype=X.dtype)
    for i in range(0, n, block):
        j = min(i + block, n)
        S[i:j] = X[i:j] @ X.T
    return S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glove-wiki-gigaword-100",
                    help='gensim-data model name (e.g. "glove-wiki-gigaword-100")')
    ap.add_argument("--max-words", type=int, default=300,
                    help="Only use first N words (full similarity is O(N^2) memory).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "float64"])
    ap.add_argument("--block", type=int, default=0, help="Block size for similarity matmul (0 = no blocking).")
    ap.add_argument("--out", default="similarity.npy", help="Output .npy file.")
    ap.add_argument("--list-models", action="store_true", help="List available gensim-data models and exit.")
    args = ap.parse_args()

    import gensim.downloader as api

    if args.list_models:
        info = api.info()  # dict of datasets/models available via gensim-data
        print("Available models:")
        for k in sorted(info.get("models", {}).keys()):
            print("  ", k)
        return

    # Downloads once, then loads from local cache on later runs
    kv = api.load(args.model)  # KeyedVectors :contentReference[oaicite:1]{index=1}

    words = kv.index_to_key[:args.max_words]
    X = kv.vectors[:args.max_words].astype(np.float32)

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "float64": torch.float64}[args.dtype]
    Xt = torch.from_numpy(X).to(device=args.device, dtype=torch_dtype)

    with torch.inference_mode():
        S = cosine_sim_matrix(Xt, block=args.block)

    svalues = singular_values(S)
    srank = stable_rank(S)
    srank_cut = stable_rank_cut(S, rank=Xt.shape[1])
    srank_cut_rel = stable_rank_cut_rel(S, rank=Xt.shape[1])
    rank = torch.linalg.matrix_rank(S)
    srank_rel = stable_rank_relative(S, rank=Xt.shape[1])
    min_error = best_low_rank_approx_error(S, rank=Xt.shape[1])
    quality = srank_rel - min_error

    print(f"rank: {rank}, "
          f"Stable rank: {srank}, "
          f"Rel. stable rank: {srank_rel}, "
          f"Stable rank cut: {srank_cut}, "
          f"Stable rank cut rel: {srank_cut_rel}, "
          f"min_error: {min_error}, "
          f"quality: {quality}")

    # S_cpu = S.float().cpu().numpy()
    # np.save(args.out, S_cpu)
    # print(f"Saved {args.out} shape={S_cpu.shape} for model={args.model}, vocab={len(words)}")

if __name__ == "__main__":
    main()
