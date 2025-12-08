import torch


def psd_factor_svd(W: torch.Tensor, k: int | None = None, eps: float = 1e-8) -> torch.Tensor:
    # Symmetrize
    W_sym = 0.5 * (W + W.T)

    try:
        U, S, Vh = torch.linalg.svd(W_sym)
    except RuntimeError as e:
        raise RuntimeError() from e
    # For symmetric PSD: U ≈ V, S ≈ eigenvalues

    S_clamped = S.clamp(min=0.0)
    pos = S_clamped > eps
    S_pos = S_clamped[pos]
    U_pos = U[:, pos]
    r = S_pos.numel()

    if k is None:
        k_use = r
    else:
        k_use = min(k, r)

    S_k = S_pos[:k_use]
    U_k = U_pos[:, :k_use]

    X = U_k * torch.sqrt(S_k).unsqueeze(0)

    if k is not None and k > k_use:
        n = W.shape[0]
        pad = torch.zeros(n, k - k_use, dtype=W.dtype, device=W.device)
        X = torch.cat([X, pad], dim=1)

    return X