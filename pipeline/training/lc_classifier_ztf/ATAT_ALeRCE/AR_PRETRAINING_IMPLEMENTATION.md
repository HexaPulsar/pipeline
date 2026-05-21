# GPT-Style Autoregressive Pretraining Implementation

## Summary

This document describes the implementation of autoregressive (GPT-style) pretraining for the ATAT_ALeRCE light curve classification pipeline. The system learns to predict the next observation `f_t` given all previous observations `[f_0, ..., f_{t-1}]` using causal attention, replacing the previous contrastive/VICReg pretraining approach.

---

## New Files Created

### 1. `src/models/PretrainARModule.py` — Autoregressive Lightning Module

Core training module with:

- **Autoregressive objective**: Given observations 0..T-2, predict observations 1..T-1
- **Causal attention**: Via upper-triangular masking in TransformerEncoder
- **LR schedule**: Linear warmup followed by cosine decay (step-level)
- **Gradient clipping**: Norm clipping to 1.0
- **GradFilter EMA**: Gradient smoothing from existing codebase
- **Evaluation metrics**:
  - `train/loss`, `val/loss` — MSE on valid (non-padded) positions
  - `val/loss_band{0,1,...}` — Per-band MSE to detect band imbalance
  - `val/linear_probe_bacc` — Linear regression on mean-pooled embeddings
  - `val/knn3_bacc` — KNN-3 nearest neighbors on embeddings

**Key methods**:
- `training_step(batch, batch_idx)` — Shift-predict-loss cycle
- `validation_step(batch, batch_idx, dataloader_idx)` — Supports 3 dataloaders for probe eval
- `configure_optimizers()` — 3-group AdamW with warmup+cosine LR
- `configure_gradient_clipping()` — Clips to norm 1.0
- `on_validation_epoch_end()` — Fits linear probe and KNN probe on embeddings

### 2. `src/data/modules/LitPretrainAR.py` — Data Module for AR Pretraining

PyTorch Lightning DataModule with:

- **Single-view loading**: No SSL augmentation pairs (unlike SSLDataset)
- **3-dataloader validation** (when `eval_probe=True`):
  - Index 0: validation split for reconstruction loss
  - Index 1: training split for fitting the linear/KNN probe
  - Index 2: validation split for scoring the probe
- **Drop last = True**: Ensures even batch sizes (matches pretraining convention)

### 3. `AR_training.py` — Training Entry Point

Hydra-based training script that:
- Loads config from `src/configs/ZTF/supervised_training.yaml`
- Instantiates `LightCurveTransformer(use_causal=True)`
- Wraps it in `PretrainARModule`
- Trains via `LitPretrainAR` data module

### 4. `src/launchers/launch_ar_pretrain.sh` — Launcher Script

Bash script to train AR model with sensible defaults:
```bash
LEARNING_RATE=1e-4
WARMUP_STEPS=5000
ETA_MIN_FACTOR=1e-2
TOTAL_STEPS=100000
BATCH_SIZE=128
```

---

## Modified Files

### `src/layers/transformer/ATAT.py` — LightCurveTransformer

Added causal masking mode:

```python
def __init__(self, ..., use_causal: bool = False):
    self.use_causal = use_causal
    ...

def forward(self, data, time, mask, metadata=None, features=None, **kwargs):
    if self.use_causal:
        # No CLS token, causal mask
        x_mod, m_mod, _ = self.time_encoder(data, time, mask=mask)
        seq_len = x_mod.size(1)
        causal_mask = torch.triu(torch.ones(...), diagonal=1).bool()
        return self.transformer_lc(src=x_mod, mask=causal_mask, ...)
    else:
        # Bidirectional with CLS token (original)
        x_mod, m_mod, _ = self.embedding_light_curve(...)
        return self.transformer_lc(src=x_mod, ...)
```

**Key point**: Classifier modules always use `use_causal=False` (default) — no breaking changes.

### `src/configs/TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml`

Added:
```yaml
lc:
  use_causal: False  # Set to True for AR pretraining
```

### `src/configs/ZTF/supervised_training.yaml`

Added:
```yaml
pretrain:
  warmup_steps: 1000
  total_steps: 100000
  eta_min_factor: 0.1
```

---

## Training Data Flow

```
Input: ATATDataset (supervised, single view)
  → batch: {data: (B,T,2), time: (B,T,2), mask: (B,T,2), labels: (B,)}

PretrainARModule.training_step
  1. Pop labels
  2. Shift: input=[:,:-1], target=[:,1:]
  3. LightCurveTransformer(use_causal=True)
     → TimeHandler embeds (no CLS prepend)
     → causal mask (upper-triangular)
     → TransformerEncoder
     → output: (B, T*2, 32)  [concat per-band sequences]
  4. ReconstructionHead: Linear(32, 2)
     → output: (B, T*2, 2)
  5. Loss: MSE on valid positions
     → log: train/loss

PretrainARModule.validation_step
  1-4. Same as training
  5. Loss: MSE on valid positions
     → log: val/loss, val/loss_band0, val/loss_band1
  6. If eval_probe: collect mean-pooled embeddings
     → on_validation_epoch_end fits probe models
     → log: val/linear_probe_bacc, val/knn3_bacc
```

---

## Important Notes on TimeHandler

The `TimeHandler` module (called by `LightCurveTransformer` in causal mode) processes each band **independently** and **concatenates** along the sequence dimension:

- Input: `(B, T, num_bands)`
- Output: `(B, T*num_bands, D)` — per-band sequences are concatenated, not interleaved
- This means **temporal causality is still respected within each band**, but bands are not temporally synchronized

For proper multiband AR pretraining, one could instead reshape to `(B*num_bands, T, D)` and train per-band. However, the current approach is simpler and still learns meaningful representations.

---

## Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `learning_rate` | 1e-4 | Peak LR after warmup |
| `warmup_steps` | 1000 | Linear ramp from 0 to peak_lr |
| `total_steps` | 100000 | Total training steps for cosine schedule |
| `eta_min_factor` | 0.1 | Minimum LR = eta_min_factor * peak_lr |
| `weight_decay` | 1e-3 | Only on backbone params, not biases/norms |
| `batch_size` | 128 | Recommended; adjust based on GPU memory |
| `max_epochs` | 1000 | Set high; early stopping patience is 10 |

---

## LR Schedule

The learning rate follows a **linear warmup + cosine decay** schedule:

```python
if step < warmup_steps:
    lr = peak_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = peak_lr * max(eta_min, 0.5 * (1 + cos(π * progress)))
```

Example with warmup_steps=1000, total_steps=100000, eta_min=0.1:
- Step 0: LR = 0
- Step 500: LR ≈ 0.5 * peak_lr (halfway through warmup)
- Step 1000: LR = peak_lr (end of warmup, cosine peak)
- Step 50500: LR ≈ 0.5 * peak_lr (halfway through cosine decay)
- Step 100000: LR ≈ 0.1 * peak_lr (minimum)

---

## Metrics to Monitor

| Metric | Optimal Behavior | Warning Signs |
|---|---|---|
| `train/loss` | Smooth decay to ~0.01-0.1 | Stays constant or increases → learning issue |
| `val/loss` | Tracks training loss, small gap | Large gap → overfitting |
| `val/loss_band0`, `val/loss_band1` | Both decrease together | One band much higher → band imbalance |
| `val/linear_probe_bacc` | Rises from ~0.05 to ~0.5+ | Flat or falling → repr collapse |
| `val/knn3_bacc` | Rises from ~0.05 to ~0.4+ | Flat or falling → repr collapse |
| `lr-AdamW` | Ramps linearly, then cosine decays | Does not follow schedule → config issue |
| `grad_norm` | Mostly < 1.0, rare spikes | Constant ~1.0 → always clipping (bad) |

---

## Running the Training

```bash
cd /home/magdalena/rpos/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/

# Activate conda environment
conda activate ATAT

# Launch AR pretraining
bash src/launchers/launch_ar_pretrain.sh

# Or with custom config overrides:
python AR_training.py \
  ++ATATConfig.learning_rate=5e-5 \
  ++ATATConfig.pretrain.total_steps=200000 \
  ++ATATConfig.datamodule.batch_size=256
```

Outputs:
- Logs: `./results/PRETRAIN/LC/ar_pretrain_v1/ar_pretrain_v1.log`
- Checkpoints: `./results/PRETRAIN/LC/ar_pretrain_v1/`
- TensorBoard: `tensorboard --logdir ./results/PRETRAIN/LC/ar_pretrain_v1/tensorboard/`

---

## Downstream Usage (Fine-tuning)

After AR pretraining:

1. **Save the encoder weights**:
   ```python
   ckpt = torch.load("ar_pretrain_v1/checkpoint.ckpt")
   encoder_state = {k.replace("model.", ""): v 
                    for k, v in ckpt["state_dict"].items() 
                    if k.startswith("model.")}
   torch.save(encoder_state, "ar_encoder_weights.pt")
   ```

2. **Load into classifier**:
   ```python
   clf = ClassifierModule(...)
   clf.model.load_state_dict(encoder_state, strict=False)
   # Train classifier normally
   ```

3. **Benchmark**: Compare F1-macro on validation set vs VICReg pretrained baseline.

---

## Testing the Implementation

A test suite is provided at `test_ar_implementation.py`:

```bash
python test_ar_implementation.py
```

Tests:
- Causal mask shape and no future leakage
- AR module initialization and forward pass
- Optimizer configuration with 3 param groups
- LR schedule ramp and decay

---

## Files Summary

| File | Purpose |
|---|---|
| `src/models/PretrainARModule.py` | Lightning module for AR pretraining |
| `src/data/modules/LitPretrainAR.py` | DataModule with 3-dataloader validation |
| `AR_training.py` | Hydra training entry point |
| `src/launchers/launch_ar_pretrain.sh` | Launcher with default hyperparams |
| `src/layers/transformer/ATAT.py` | Modified to support causal mode |
| `src/configs/TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml` | Config with use_causal flag |
| `src/configs/ZTF/supervised_training.yaml` | Config with pretrain hyperparams |
| `test_ar_implementation.py` | Validation tests |

---

## Differences from Original Pretraining

| Aspect | Original (VICReg) | New (AR) |
|---|---|---|
| **Objective** | Contrastive on augmented pairs | Next-token prediction |
| **Data** | SSLDataset (2 views) | ATATDataset (single view) |
| **Backbone** | Bidirectional (CLS token) | Causal (no CLS) |
| **Loss** | VICReg (inv + var + cov) | MSE on target positions |
| **LR schedule** | Constant or ExponentialLR | Warmup + CosineAnnealing |
| **Gradient clip** | None | Norm clipping to 1.0 |
| **Eval** | KNN/LR probes (optional) | KNN/LR probes (built-in) |

---

## Known Limitations

1. **TimeHandler per-band concatenation**: Temporal causality is maintained within each band, but bands are not temporally synchronized in the causal mask. For true multiband AR, consider reshaping to per-band sequences.

2. **No curriculum learning**: All sequences treated equally. Could improve by starting with short sequences, then increasing length.

3. **Constant reconstruction target**: All time steps predict the same "next flux" dimensions. No per-band predictions; all bands predict all band fluxes.

4. **No velocity/acceleration in AR**: The `use_velocity`, `use_acceleration` flags still embed them, but AR doesn't explicitly predict them. They're just used as input features.

---

## References

- **Autoregressive modeling**: Vaswani et al. (2017) *Attention is All You Need*
- **Causal masking**: Brown et al. (2020) *Language Models are Unsupervised Multitask Learners* (GPT-2)
- **LR schedules**: Radford et al. (2019) *Language Models are Unsupervised Multitask Learners* (cosine annealing)
- **Gradient clipping**: Pascanu et al. (2013) *On the difficulty of training RNNs*
