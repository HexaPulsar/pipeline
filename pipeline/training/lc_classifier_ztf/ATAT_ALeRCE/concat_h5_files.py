#!/usr/bin/env python3
"""Concatenate multiple H5 files from different years into one combined file."""

import h5py
import numpy as np
from pathlib import Path

input_files = [
    '/home/magdalena/Desktop/sambashare/H5_files/2019/200_2019.h5',
    '/home/magdalena/Desktop/sambashare/H5_files/2020/200_2020.h5',
    '/home/magdalena/Desktop/sambashare/H5_files/2021/200_2021.h5',
    '/home/magdalena/Desktop/sambashare/H5_files/2022/200_2022.h5',
]

output_file = '/home/magdalena/Desktop/sambashare/H5_files/200_COMBINED_2019-2022.h5'

# Data arrays to concatenate (with same dimensions)
concat_keys = ['flux', 'flux_err', 'time', 'mask', 'mask_detection', 'mask_photometry', 'extracted_features', 'metadata_feat']

# Index arrays to update (store sample indices)
index_keys = ['training_0', 'training_1', 'training_2', 'training_3', 'training_4',
              'validation_0', 'validation_1', 'validation_2', 'validation_3', 'validation_4']

# Scalars that should be preserved
scalar_keys = ['oid']

print("Loading data from input files...")
all_data = {}
offset = 0
offsets = [0]  # Track offset for each file

for f in input_files:
    print(f"  Reading {Path(f).name}...")
    with h5py.File(f, 'r') as h5:
        n_samples = h5['flux'].shape[0]

        for key in concat_keys:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(h5[key][:])

        for key in scalar_keys:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(h5[key][:])

        for key in index_keys:
            if key not in all_data:
                all_data[key] = []
            # Shift indices by current offset
            indices = h5[key][:]
            all_data[key].append(indices + offset)

        offset += n_samples
        offsets.append(offset)

print(f"\nConcatenating data (total samples: {offset})...")

# Create output file
with h5py.File(output_file, 'w') as out:
    # Concatenate array data
    for key in concat_keys:
        print(f"  Writing {key}...")
        concatenated = np.concatenate(all_data[key], axis=0)
        out.create_dataset(key, data=concatenated, compression='gzip', compression_opts=4)

    # Concatenate scalar data
    for key in scalar_keys:
        print(f"  Writing {key}...")
        concatenated = np.concatenate(all_data[key], axis=0)
        out.create_dataset(key, data=concatenated)

    # Concatenate and shift index arrays
    for key in index_keys:
        print(f"  Writing {key}...")
        concatenated = np.concatenate(all_data[key], axis=0)
        out.create_dataset(key, data=concatenated)

print(f"\nDone! Combined file: {output_file}")
print(f"  Total samples: {offset}")
print(f"  File size: {Path(output_file).stat().st_size / 1e9:.1f} GB")
