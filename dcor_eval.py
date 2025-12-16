import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from sae import load_model, Autoencoder
from utils import SAEDataset, set_seed, get_device
from dcor import single_dim_cross_dcor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the distance correlation evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate distance correlation of activated neurons in Sparse Autoencoders"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file (.pt)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Path to the dataset file (.npy)",
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
        help="Directory path to save distance correlations",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def get_distance_correlations(
    model: Autoencoder,
    dataset,
    output_file: str,
    batch_size: int,
):
    device = get_device()
    model.eval().to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )

    sample_distcorrs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating distance correlation"):
            batch = batch.to(device, non_blocking=True)

            # Encode
            sparse_repr = model(batch)[1]  # (B, D)
            B, D = sparse_repr.shape

            dcor = single_dim_cross_dcor(sparse_repr)

    distcorrs = np.array(sample_distcorrs)

    min_idx = distcorrs.argmin()
    max_idx = distcorrs.argmax()

    logger.info(
        f"Minimal distance correlation at index {min_idx} = {distcorrs[min_idx]:.6f}"
    )
    logger.info(
        f"Maximal distance correlation at index {max_idx} = {distcorrs[max_idx]:.6f}"
    )
    logger.info(f"Mean distance correlation = {distcorrs.mean()}")
    logger.info(f"Standard deviation = {distcorrs.std()}")

    np.save(output_file, distcorrs)
    logger.info(f"Distance correlations saved into: {output_file}")


def main(args):
    set_seed(args.seed)

    # Load trained model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    logger.info("Model loaded")

    # Load dataset with correct preprocessing
    if ("text" in args.model and "text" in args.data) or (
        "image" in args.model and "image" in args.data
    ):
        logger.info("Using model mean and scaling factor")
        dataset = SAEDataset(args.data)
        dataset.mean = mean_center.cpu()
        dataset.scaling_factor = scaling_factor
    else:
        logger.info("Computing mean and scaling factor")
        dataset = SAEDataset(
            args.data,
            mean_center=True if mean_center.sum() != 0.0 else False,
            target_norm=target_norm,
        )

    logger.info(f"Dataset loaded with length: {len(dataset)}")
    logger.info(
        f"Dataset mean center: {dataset.mean.mean()}, "
        f"Scaling factor: {dataset.scaling_factor}, "
        f"Target norm: {dataset.target_norm}"
    )

    # Construct output filename
    model_name = args.model.split("/")[-1].replace(".pt", "")
    data_name = args.data.split("/")[-1].replace(".npy", "")
    output_file = os.path.join(
        args.output_path, f"dcor_{data_name}_{model_name}.npy"
    )

    get_distance_correlations(model, dataset, output_file, args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
