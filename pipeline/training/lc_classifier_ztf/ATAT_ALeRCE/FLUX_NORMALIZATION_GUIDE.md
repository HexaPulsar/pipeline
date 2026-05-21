# Per-Sample Flux Normalization with Precomputation

## Overview

Flux values are normalized independently per sample and per band using **precomputed statistics**:
- Compute once before training (offline)
- Apply at dataset load time (zero overhead)
- Invertible at generation time using stored mean/std

## Setup: Precompute Statistics

### Step 1: Compute and cache statistics
```bash
python precompute_flux_normalization.py \
    --data-root /path/to/dataset.h5 \
    --output-root ./norm_cache/ \
    --observation-key flux \
    --mask-key mask
```

This creates `norm_cache/dataset_norm_stats.h5` with precomputed mean/std for each sample and band.

### Step 2: Enable in launcher
```bash
# In launch_ar_pretrain.sh, add:
++ATATConfig.datamodule.dataset.norm_stats_path=/path/to/norm_stats.h5
```

Or programmatically:
```python
config.datamodule.dataset.norm_stats_path = "path/to/norm_stats.h5"
```

## How Normalization Works

**During training (in CustomDataset.__getitem__):**
```
1. Load flux for sample idx: shape (T, num_bands)
2. Retrieve precomputed mean/std for that sample
3. Apply: normalized = (flux - mean) / std
4. Feed normalized flux to model
5. Store (mean, std) in batch dict for denormalization reference
```

**During generation (inference):**
```python
from src.models.PretrainARModule import PretrainARModule

# Load model and initial context
model = PretrainARModule.load_from_checkpoint(...)
context_data = ...  # shape: (num_context, num_bands)

# Retrieve stored mean/std from batch (set during training)
mean, std = flux_norm_stats[band_idx]

# Generate predictions (outputs normalized values)
normalized_pred = model.reconstruction_head(embeddings)

# Denormalize using the stored stats
physical_pred = PretrainARModule.denormalize_flux(
    normalized_pred.item(), mean, std
)
```

## Properties

✓ **Zero overhead at train time** - precomputed statistics, simple lookup/apply
✓ **Invertible** - store mean/std per sample, denormalize at generation
✓ **Consistent** - same normalization across train/val/test if stats precomputed
✓ **Flexible** - each sample gets its own scale factor (robust to magnitude variations)

## Denormalization at Generation

```python
# Denormalize: x_original = normalized * std + mean
denormalized = PretrainARModule.denormalize_flux(
    normalized_flux=prediction,
    mean=stored_mean,
    std=stored_std
)
```

The `flux_norm_stats` dict is automatically included in batch dicts by CustomDataset for easy access during generation.
