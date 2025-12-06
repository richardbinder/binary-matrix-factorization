import torch


if __name__ == "__main__":
    n = 80
    W = torch.randn(n, n)  # (m,n) float tensor
    r = 19

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Wr = (U[:, :r] * S[:r]) @ Vh[:r, :]

    E = W - Wr

    print("sigma_{r}   =", S[r-1].item())
    print("sigma_{r+1} =", S[r].item())
    print("||E||_2     =", torch.linalg.matrix_norm(E, ord=2).item())
    print("||E||_F     =", torch.linalg.matrix_norm(E, ord='fro').item())
    print("RMSE entry  =", (torch.linalg.matrix_norm(E, ord='fro') / (W.numel()**0.5)).item())