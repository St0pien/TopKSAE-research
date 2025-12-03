import torch
from functools import partial


class SAELoss(torch.nn.Module):
    """
    Combined loss function for Sparse Autoencoders (SAE).

    Combines a reconstruction loss term with a sparsity-inducing regularization term.
    The reconstruction loss measures how well the autoencoder reconstructs its inputs,
    while the sparsity loss encourages sparse activations in the latent space.

    Attributes:
        reconstruction_loss (str): Name of the reconstruction loss function.
        reconstruction_loss_fn (callable): Function for computing reconstruction loss.
        sparse_loss (str): Name of the sparsity loss function.
        sparse_loss_fn (callable): Function for computing sparsity loss.
        sparse_weight (float): Coefficient for the sparsity loss term.
        mean_input (torch.Tensor, optional): Mean input vector used for normalization.
    """

    def __init__(
        self,
        reconstruction_loss: str = "mse",
        sparse_loss: str = "l1",
        sparse_weight: float = 0.0,
        independence_loss: str = None,
        independence_weight: float = 0.0,
        mean_input: torch.Tensor = None,
    ):
        """
        Initialize the SAE loss function.

        Args:
            reconstruction_loss (str): Type of reconstruction loss ("mse" or "cosine").
            sparse_loss (str): Type of sparsity regularization ("l1", "l0", or "tanh").
            sparse_weight (float): Weight coefficient for the sparsity loss term.
            mean_input (torch.Tensor, optional): Mean input vector used for normalization
                                                in certain loss functions.
        """
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_fn = get_recon_loss_fn(reconstruction_loss)
        self.sparse_loss = sparse_loss
        self.sparse_loss_fn = get_sparse_loss_fn(sparse_loss)
        self.sparse_weight = sparse_weight
        self.independence_loss = independence_loss
        self.independence_loss_fn = get_independence_loss_fn(independence_loss)
        self.independence_weight = independence_weight
        self.mean_input = mean_input

    def forward(
        self,
        reconstruction: torch.Tensor,
        original_input: torch.Tensor,
        latent_activations: torch.Tensor,
        decoder = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss for the sparse autoencoder.

        Args:
            reconstruction: Output of Autoencoder.decode (shape: [batch, n_inputs])
            original_input: Input of Autoencoder.encode (shape: [batch, n_inputs])
            latent_activations: Output of Autoencoder.encode (shape: [batch, n_latents])

        Returns:
            tuple: (total_loss, reconstruction_loss, sparsity_loss)
                - total_loss: Combined weighted loss
                - reconstruction_loss: Loss measuring reconstruction quality
                - sparsity_loss: Loss measuring latent space sparsity
        """
        recon_loss = self.reconstruction_loss_fn(
            reconstruction, original_input, self.mean_input
        )
        sparse_loss = self.sparse_loss_fn(latent_activations, original_input)

        independence_loss = (
            self.independence_loss_fn(latent_activations, decoder)
            if self.independence_loss_fn is not None
            else torch.tensor([0]).to(latent_activations.device)
        )

        return (
            recon_loss
            + self.sparse_weight * sparse_loss
            + independence_loss * self.independence_weight,
            recon_loss,
            sparse_loss,
            independence_loss,
        )

    def __repr__(self):
        """String representation for debugging."""
        return f"SAELoss(reconstruction_loss={self.reconstruction_loss}, sparse_loss={self.sparse_loss}, sparse_weight={self.sparse_weight})"

    def __str__(self):
        """Human-readable string representation."""
        return f"{self.reconstruction_loss}_{self.sparse_loss}_{self.sparse_weight}"


def mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    mean_vector: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute mean squared error between reconstruction and original input.

    This loss measures the average squared difference between the reconstructed
    and original input, providing a direct measure of reconstruction quality.
    """
    return torch.nn.functional.mse_loss(
        reconstruction, original_input, reduction="mean"
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    mean_vector: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute normalized mean squared error between reconstruction and original input.

    This loss normalizes the MSE by the variance of the input, which helps
    account for the inherent difficulty of reconstructing high-variance data.
    Also known as the Fraction of Variance Unexplained (FVU).

    Args:
        reconstruction: Output of Autoencoder.decode (shape: [batch, n_inputs])
        original_input: Input of Autoencoder.encode (shape: [batch, n_inputs])
        mean_vector: Mean vector for normalization (defaults to zero vector)

    Returns:
        torch.Tensor: Normalized mean squared error (scalar)
    """
    if mean_vector is None:
        mean_vector = torch.zeros_like(original_input)
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1)
        / ((mean_vector - original_input) ** 2).mean(dim=1)
    ).mean()


def cosine_similarity_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    mean_vector: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute squared cosine distance between reconstruction and original input.

    This loss focuses on directional similarity rather than exact value matching,
    which can be useful when the scale of the vectors is less important than
    their orientation in the embedding space.

    Args:
        reconstruction: Output of Autoencoder.decode (shape: [batch, n_inputs])
        original_input: Input of Autoencoder.encode (shape: [batch, n_inputs])
        mean_vector: Not used in this loss, included for API consistency

    Returns:
        torch.Tensor: Cosine similarity loss (scalar)
    """
    # Normalize vectors to unit length
    norm_reconstruction = reconstruction / reconstruction.norm(dim=-1, keepdim=True)
    norm_original_input = original_input / original_input.norm(dim=-1, keepdim=True)

    # Compute cosine similarity matrix
    cos_sim = norm_original_input @ norm_reconstruction.T

    # Extract diagonal elements (similarity between corresponding pairs)
    diag_cos_sim = torch.diagonal(cos_sim)

    # Convert similarity to squared distance and average
    return ((1 - diag_cos_sim) ** 2).mean()


def L0_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L0 loss (number of non-zero elements) for latent activations.

    This loss directly measures the average number of active neurons per sample,
    which is a direct measure of sparsity. Lower values indicate higher sparsity.

    Args:
        latent_activations: Output of Autoencoder.encode (shape: [batch, n_latents])
        original_input: Input of Autoencoder.encode (shape: [batch, n_inputs])
                       (not used, included for API consistency)

    Returns:
        torch.Tensor: Average number of active neurons per sample (scalar)
    """
    return latent_activations.sum(dim=-1).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L1 norm of latent activations normalized by input magnitude.

    This loss encourages sparsity by penalizing the sum of absolute values
    of latent activations, while normalizing by input magnitude to make
    the loss scale-invariant.

    Args:
        latent_activations: Output of Autoencoder.encode (shape: [batch, n_latents])
        original_input: Input of Autoencoder.encode (shape: [batch, n_inputs])

    Returns:
        torch.Tensor: Normalized L1 loss (scalar)
    """
    return (latent_activations.abs().sum(dim=-1) / original_input.norm(dim=-1)).mean()


def dcor_latent_loss(latent_activations: torch.Tensor, decoder=None, k=16):
    """
    Compute distance correlation between latent dimmensions and average them

    This loss aims to enforce independence of latent neurons

    Args:
        latent_activations: Output of Autoencoder.encode (shape: [batch, n_latents])

    Returns:
        torch.Tensor: Mean distance correlation between features
    """

    # Pick k most active neurons to reduce matrix size
    mean_abs = latent_activations.abs().mean(dim=0)
    topk_indices = torch.topk(mean_abs, k=k).indices
    X = latent_activations[:, topk_indices]

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


def dcor_reconstruction_loss(latent_activations: torch.Tensor, decoder, k=8):
    B, d = latent_activations.shape

    most_active = latent_activations.mean(dim=0).topk(k).indices

    expanded = latent_activations.unsqueeze(0).expand(k, B, d)
    mask = torch.zeros(k, d, device=latent_activations.device)
    mask[torch.arange(k), most_active] = True
    mask = mask.unsqueeze(1)

    isolated_concepts = (expanded * mask).reshape(k*B, d)

    decoder_output = decoder(isolated_concepts)
    reconstructions = torch.split(decoder_output, B, dim=0)
    
    dcors = []
    for i in range(k):
        for j in range(i+1, k):
            dcors.append(distance_correlation(reconstructions[i], reconstructions[j]))

    dcor_loss = torch.stack(dcors).mean()
    return dcor_loss


# Mapping of reconstruction loss function names to their implementations
RECON_LOSSES_MAP = {
    "mse": mean_squared_error,
    "nmse": normalized_mean_squared_error,
    "cosine": cosine_similarity_loss,
}


# Mapping of sparsity loss function names to their implementations
SPARSITY_LOSSES_MAP = {
    "l1": normalized_L1_loss,
    "l0": L0_loss,
}

INDEPENDENCE_LOSSES_MAP = {
    "DcorLatent": dcor_latent_loss,
    "DcorRecon": dcor_reconstruction_loss,
}


def get_recon_loss_fn(recon_loss: str) -> callable:
    """
    Get the reconstruction loss function by name.

    Args:
        recon_loss: Name of the reconstruction loss function

    Returns:
        callable: The corresponding reconstruction loss function

    Raises:
        KeyError: If the specified reconstruction loss is not supported
    """
    return RECON_LOSSES_MAP[recon_loss]


def get_sparse_loss_fn(sparse_loss: str) -> callable:
    """
    Get the sparsity loss function by name.

    Supports parameterized tanh loss in the format "tanh_scale_saturation"
    (e.g., "tanh_2.0_0.5" for tanh loss with scale=2.0 and saturation=0.5).

    Args:
        sparse_loss: Name of the sparsity loss function

    Returns:
        callable: The corresponding sparsity loss function

    Raises:
        KeyError: If the specified sparsity loss is not supported
    """
    if "_" in sparse_loss and sparse_loss.startswith("tanh"):
        # Parse parameters for parameterized tanh loss
        sparse_loss, scale, saturation = sparse_loss.split("_")
        return partial(
            SPARSITY_LOSSES_MAP[sparse_loss],
            scale=float(scale),
            saturation=float(saturation),
        )

    return SPARSITY_LOSSES_MAP[sparse_loss]


def get_independence_loss_fn(ind_loss: str) -> callable:
    """
    Get the independence loss function by name.

    Args:
        independence_loss: Name of the independence loss function

    Returns:
        callable: The corresponding independence loss function

    Raises:
        KeyError: If the specified independence loss is not supported
    """
    if ind_loss is None:
        return None
    return INDEPENDENCE_LOSSES_MAP[ind_loss]
