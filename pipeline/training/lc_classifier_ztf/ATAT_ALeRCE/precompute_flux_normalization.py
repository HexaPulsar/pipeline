"""Precompute and cache flux normalization statistics for efficient training.

This script computes mean/std for each sample and band once, storing them
in an HDF5 file for zero-overhead normalization during training.

Usage:
    python precompute_flux_normalization.py \
        --data-root path/to/dataset.h5 \
        --output-root path/to/cache/ \
        --observation-key flux \
        --mask-key mask
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_flux_stats(data_path, obs_key='flux', mask_key='mask', sets=['training', 'validation', 'test']):
    """Compute and cache normalization statistics for each sample.

    Args:
        data_path: Path to H5 file
        obs_key: Key for observation data in H5
        mask_key: Key for mask in H5
        sets: Which splits to process

    Returns:
        dict mapping set_name -> {sample_idx -> {band -> (mean, std)}}
    """
    stats = {}

    with h5py.File(data_path, 'r') as f:
        for set_name in sets:
            if set_name not in f:
                logger.warning(f"Set {set_name} not found in {data_path}, skipping")
                continue

            group = f[set_name]
            data = group[obs_key][:]  # (N_samples, T, num_bands)
            mask = group[mask_key][:]  # (N_samples, T, num_bands)

            N_samples, T, num_bands = data.shape
            logger.info(f"Processing {set_name}: {N_samples} samples, {T} timepoints, {num_bands} bands")

            set_stats = {}
            for i in range(N_samples):
                sample_stats = {}
                for band in range(num_bands):
                    valid = mask[i, :, band]
                    if valid.sum() > 1:
                        band_data = data[i, valid, band]
                        mean = float(band_data.mean())
                        std = float(band_data.std())
                        std = max(std, 1e-8)  # Avoid zero std
                    else:
                        mean, std = 0.0, 1.0
                    sample_stats[band] = (mean, std)

                set_stats[i] = sample_stats
                if (i + 1) % max(1, N_samples // 10) == 0:
                    logger.info(f"  Processed {i+1}/{N_samples} samples")

            stats[set_name] = set_stats

    return stats


def save_stats_to_h5(stats, output_path, data_path):
    """Save computed statistics to HDF5 cache file.

    Args:
        stats: dict mapping set_name -> {sample_idx -> {band -> (mean, std)}}
        output_path: Path to cache H5 file
        data_path: Original data file path (stored as metadata)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.attrs['source_data'] = str(data_path)
        f.attrs['description'] = 'Precomputed flux normalization statistics (mean, std per sample per band)'

        for set_name, set_stats in stats.items():
            grp = f.create_group(set_name)

            # Get shape info
            max_idx = max(set_stats.keys()) + 1
            max_band = max(max(s.keys()) for s in set_stats.values()) + 1

            # Store as arrays: (N_samples, num_bands, 2) where last dim is [mean, std]
            means = np.zeros((max_idx, max_band), dtype=np.float32)
            stds = np.zeros((max_idx, max_band), dtype=np.float32)

            for sample_idx, band_stats in set_stats.items():
                for band_idx, (mean, std) in band_stats.items():
                    means[sample_idx, band_idx] = mean
                    stds[sample_idx, band_idx] = std

            grp.create_dataset('means', data=means, compression='gzip')
            grp.create_dataset('stds', data=stds, compression='gzip')

    logger.info(f"Saved statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Precompute flux normalization statistics')
    parser.add_argument('--data-root', required=True, help='Path to dataset H5 file')
    parser.add_argument('--output-root', required=True, help='Path to save cached statistics')
    parser.add_argument('--observation-key', default='flux', help='Key for flux data')
    parser.add_argument('--mask-key', default='mask', help='Key for mask data')
    args = parser.parse_args()

    logger.info(f"Computing normalization stats for {args.data_root}")
    stats = compute_flux_stats(
        args.data_root,
        obs_key=args.observation_key,
        mask_key=args.mask_key
    )

    output_path = Path(args.output_root) / f"{Path(args.data_root).stem}_norm_stats.h5"
    save_stats_to_h5(stats, output_path, args.data_root)
    logger.info("Done!")


if __name__ == '__main__':
    main()
