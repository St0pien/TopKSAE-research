import torch


def single_dim_cross_dcor(X: torch.Tensor):
    # X: (batch_size, n_features) == (n, d) (n, v)
    n, d = X.shape

    # ---- 1. Compute pairwise distances for all variables ----

    # Compute |x_i - x_j| for each variable
    # Result shape: (d, n, n)
    D = torch.abs(X.T.unsqueeze(2) - X.T.unsqueeze(1))

    # ---- 2. Double center each distance matrix ----
    row_mean = D.mean(dim=2, keepdim=True)  # (d, n, 1)
    col_mean = D.mean(dim=1, keepdim=True)  # (d, 1, n)
    grand_mean = D.mean(dim=(1, 2), keepdim=True)  # (d, 1, 1)

    A = D - row_mean - col_mean + grand_mean  # double centered (d, n, n)

    # ---- 3. Pairwise distance covariances ----
    # dCov(i,j) = mean(A_i * A_j)
    dCov = torch.einsum("vab,wab->vw", A, A) / (n * n)  # (d, d)

    # ---- 4. Distance variances ----
    dVar = dCov.diag()  # (d,)

    # ---- 5. Distance correlation matrix ----
    denom = torch.sqrt(dVar[:, None] * dVar[None, :])  # (d, d)

    # Avoid divide-by-zero
    dCor = dCov / (denom + 1e-8)

    # Remove distance variances
    dCor -= torch.eye(d).to(dCor.device)

    return dCor.mean()


def pdist(x, eps=1e-8):
    """
    Pairwise Euclidean distances for a matrix of shape (n, d).
    Fully differentiable.
    """
    x_norm = (x**2).sum(dim=1).unsqueeze(1)
    dist = torch.sqrt(torch.clamp(x_norm + x_norm.t() - 2 * x @ x.t(), min=eps))
    return dist


def double_center(distance_matrix):
    """
    Double-centers a distance matrix A into:
      A_ij - row_mean_i - col_mean_j + grand_mean
    """
    row_mean = distance_matrix.mean(dim=1, keepdim=True)
    col_mean = distance_matrix.mean(dim=0, keepdim=True)
    grand_mean = distance_matrix.mean()
    return distance_matrix - row_mean - col_mean + grand_mean


def distance_correlation(x, y, eps=1e-8):
    """
    Computes the differentiable distance correlation between
    two tensors x and y of shape (n, d_x) and (n, d_y).
    """
    # Pairwise distances
    a = pdist(x)
    b = pdist(y)

    # Double-centering
    A = double_center(a)
    B = double_center(b)

    # Distance covariance
    dcov_xy = (A * B).mean()
    dcov_xx = (A * A).mean()
    dcov_yy = (B * B).mean()

    # Distance correlation
    dcor = dcov_xy / torch.sqrt(torch.clamp(dcov_xx * dcov_yy, min=eps))
    return dcor
