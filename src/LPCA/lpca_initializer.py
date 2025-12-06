import torch

from src.compute.compute_properties import stable_rank, singular_values, best_low_rank_approx_error


def symmetrize(W: torch.Tensor) -> torch.Tensor:
    return 0.5 * (W + W.T)


def project_to_psd(A: torch.Tensor, eps: float = 0.0):
    """
    Projects a symmetric matrix A onto the PSD cone by zeroing negative eigenvalues.
    """
    # A should be symmetric; enforce for numerical stability
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)  # evals ascending
    evals_psd = torch.clamp(evals, min=eps)
    A_psd = (evecs * evals_psd) @ evecs.T
    return A_psd, evals, evals_psd


def low_rank_psd_approx(A_psd: torch.Tensor, rank: int):
    """
    Best rank-'rank' approximation in Frobenius norm for a PSD matrix
    by truncating eigen-decomposition.
    """
    A_psd = symmetrize(A_psd)
    evals, evecs = torch.linalg.eigh(A_psd)  # ascending
    # take top-r
    idx = torch.argsort(evals, descending=True)[:rank]
    lam_r = evals[idx].clamp(min=0.0)
    U_r = evecs[:, idx]
    G = (U_r * lam_r) @ U_r.T
    G = symmetrize(G)
    return G, U_r, lam_r


def factor_gram(G: torch.Tensor, rank: int | None = None):
    """
    Given symmetric PSD G, produce X so that X X^T = G (or its rank-truncated version).
    If rank is None, uses all nonzero eigenvalues (within numerical tolerance).
    """
    G = symmetrize(G)
    evals, evecs = torch.linalg.eigh(G)  # ascending
    # Filter positive eigenvalues
    tol = 1e-10 * torch.max(torch.abs(evals)).clamp(min=1.0)
    pos = evals > tol
    evals_pos = evals[pos]
    U_pos = evecs[:, pos]

    if rank is not None:
        # take top 'rank' among positive ones
        order = torch.argsort(evals_pos, descending=True)[:rank]
        evals_pos = evals_pos[order]
        U_pos = U_pos[:, order]

    X = U_pos * torch.sqrt(evals_pos)
    # X has shape (n, r). Then X @ X.T == G (numerically).
    return X, evals, pos


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(0)

    # Example "known matrix W" (can be any real matrix)
    n = 80
    W = torch.randn(n, n)
    # Turn W into a symmetric matrix
    W = symmetrize(W)

    # --- Step 1: from W get low-rank approximation G ---
    # We'll interpret "low rank approximation G" as: take a Gram-like PSD matrix derived from W,
    # then truncate it to rank r.
    r = 10

    # project to PSD (Gram-like).
    W_psd, evals_raw, evals_psd = project_to_psd(W, eps=0.0)

    # Best rank-r approximation of this PSD matrix
    G, U_r, lam_r = low_rank_psd_approx(W_psd, rank=r)

    # --- Step 2: from G get X such that X X^T = G ---
    X, evals_G, pos_mask = factor_gram(G, rank=r)

    # Checks
    recon = X @ X.T
    err_fro_G = torch.linalg.norm(G - W, ord='fro') / (torch.linalg.norm(G) + 1e-12)
    err_fro_recon = torch.linalg.norm(recon - W, ord='fro') / (torch.linalg.norm(W) + 1e-12)
    rank_G = torch.linalg.matrix_rank(G)
    rank_W = torch.linalg.matrix_rank(W)
    rank_recon = torch.linalg.matrix_rank(recon)
    ranks_G = stable_rank(G)
    ranks_W = stable_rank(W)
    svd_r_G = best_low_rank_approx_error(G, rank=r)
    svd_r_W = best_low_rank_approx_error(W, rank=r)

    print(f"W shape: {W.shape}")
    print(f"Target rank r: {r}")
    print(f"rank(recon) (numeric): {int(rank_recon.item())}")
    print(f"rank(G) (numeric): {int(rank_G.item())}")
    print(f"rank(W) (numeric): {int(rank_W.item())}")
    print(f"srank(G) (numeric): {ranks_G}")
    print(f"srank(W) (numeric): {ranks_W}")
    print(f"svd(G)[r+1] (numeric): {svd_r_G}")
    print(f"svd(W)[r+1] (numeric): {svd_r_W}")
    print(f"Relative Frobenius reconstruction error ||G - W||_F / ||G||_F: {err_fro_G.item()}")
    print(f"Relative Frobenius reconstruction error ||XX^T - W||_F / ||W||_F: {err_fro_recon.item()}")

    # If you want to see that G is PSD (within numerical tolerance):
    min_eig = torch.min(torch.linalg.eigvalsh(G)).item()
    print(f"min eigenvalue of G: {min_eig:.3e}")


if __name__ == "__main__":
    main()