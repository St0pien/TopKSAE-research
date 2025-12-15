import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from sae import load_model, Autoencoder
from utils import SAEDataset, set_seed, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the representation extraction and evaluation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract and evaluate representations from Sparse Autoencoder models"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file (.pt)",
    )
    parser.add_argument(
        "-d", "--data", type=str, required=True, help="Path to the dataset file (.npy)"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for processing data",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=".",
        help="Directory path to save orthogonalities",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def get_orthogonalities(model: Autoencoder, dataset, output_file, batch_size):
    device = get_device()
    model.eval().to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    mean_orthogonalities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating orthogonality"):
            batch = batch.to(device, non_blocking=True)

            # Encode
            sparse_repr = model(batch)[1]  # (B, D)
            B, D = sparse_repr.shape

            # Get active indices
            active_mask = sparse_repr != 0  # (B, D)
            counts = active_mask.sum(dim=1)  # (B,)

            if counts.max() < 2:
                continue

            # Build isolated latents without loops
            # Shape: (total_active, D)
            idx_b, idx_d = active_mask.nonzero(as_tuple=True)
            isolated = torch.zeros(idx_b.shape[0], D, device=device)
            isolated[torch.arange(idx_b.shape[0]), idx_d] = sparse_repr[idx_b, idx_d]

            # Decode all at once
            decoded = model.decode(isolated)
            decoded = torch.nn.functional.normalize(decoded, dim=1)

            # Split per-sample and compute mean similarity
            start = 0
            for k in counts.tolist():
                if k < 2:
                    start += k
                    continue

                X = decoded[start : start + k]  # (k, F)
                sim_sum = (X @ X.T).sum() - k
                mean_sim = sim_sum / (k * (k - 1))
                mean_orthogonalities.append(mean_sim.item())

                start += k

    np_orthogonalities = np.array(mean_orthogonalities)
    min_idx = np_orthogonalities.argmin()
    max_idx = np_orthogonalities.argmax()
    ort_mean = np_orthogonalities.mean()
    ort_std = np_orthogonalities.std()

    logger.info(f"Minimal orthogonality at index {min_idx} = {np_orthogonalities[min_idx]:.6f}")
    logger.info(f"Maximal orthogonality at index {max_idx} = {np_orthogonalities[max_idx]:.6f}")
    logger.info(f"Mean orthogonality = {ort_mean}")
    logger.info(f"Standard deviation of orthogonality = {ort_std}")

    np.save(output_file, np_orthogonalities)
    logger.info(f"Orthogonalities saved into: {output_file}")


def main(args):
    set_seed(args.seed)

    # Load the trained model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    logger.info("Model loaded")

    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or (
        "image" in args.model and "image" in args.data
    ):
        logger.info("Using model mean and scalling factor")
        dataset = SAEDataset(args.data)
        dataset.mean = mean_center.cpu()
        dataset.scaling_factor = scaling_factor
    else:
        logger.info("Computing mean and scalling factor")
        dataset = SAEDataset(
            args.data,
            mean_center=True if mean_center.sum() != 0.0 else False,
            target_norm=target_norm,
        )

    logger.info(f"Dataset loaded with length: {len(dataset)}")
    logger.info(
        f"Dataset mean center: {dataset.mean.mean()}, Scaling factor: {dataset.scaling_factor} with target norm {dataset.target_norm}"
    )
    # Construct output filename from model and data names
    model_path_name = args.model.split("/")[-1].replace(".pt", "")
    data_path_name = args.data.split("/")[-1].replace(".npy", "")
    ort_output = os.path.join(
        args.output_path, f"ort_{data_path_name}_{model_path_name}.npy"
    )

    # Extract representations and compute metrics
    get_orthogonalities(model, dataset, ort_output, args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
