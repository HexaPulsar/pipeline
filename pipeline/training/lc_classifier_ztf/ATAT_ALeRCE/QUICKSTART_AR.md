# Quick Start: Autoregressive Pretraining

## One-Liner

```bash
cd /home/magdalena/rpos/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/
conda activate ATAT && bash src/launchers/launch_ar_pretrain.sh
```

## What It Does

Trains a transformer to predict the next observation in a light curve using causal attention (GPT-style). Outputs:
- **Checkpoints** in `./results/PRETRAIN/LC/ar_pretrain_v1/`
- **Logs** in `./results/PRETRAIN/LC/ar_pretrain_v1/ar_pretrain_v1.log`
- **TensorBoard** at `./results/PRETRAIN/LC/ar_pretrain_v1/tensorboard/`

## Monitor Training

```bash
# In a separate terminal
tensorboard --logdir ./results/PRETRAIN/LC/ar_pretrain_v1/tensorboard/
# Open browser to http://localhost:6006
```

Watch these metrics:
- `train/loss` — should decay smoothly
- `val/loss` — should track training loss
- `val/linear_probe_bacc` — should improve (0.05 → 0.5+)
- `lr-AdamW` — should ramp up, then cosine down

## After Training

1. **Extract encoder weights**:
   ```python
   import torch
   ckpt = torch.load("./results/PRETRAIN/LC/ar_pretrain_v1/classifier_ckpt_*.ckpt")
   state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() 
            if k.startswith("model.")}
   torch.save(state, "./ar_encoder.pt")
   ```

2. **Fine-tune for classification**:
   ```python
   from src.models.ClassifierModule import ClassifierModule
   clf = ClassifierModule(...)
   clf.model.load_state_dict(torch.load("./ar_encoder.pt"), strict=False)
   # Train with supervised_training.yaml
   ```

3. **Compare to baseline**:
   ```bash
   # VICReg baseline
   python supervised_training.py ++pretrain_checkpoint=<vicrig_ckpt>
   
   # AR baseline
   python supervised_training.py ++pretrain_checkpoint=./ar_encoder.pt
   
   # Compare F1-macro scores
   ```

## Customize (Optional)

Edit the launcher before running:

```bash
# src/launchers/launch_ar_pretrain.sh
LEARNING_RATE=5e-5        # Lower for slower, more stable training
WARMUP_STEPS=10000        # Longer warmup for larger models
TOTAL_STEPS=200000        # More steps = longer decay
BATCH_SIZE=256            # Larger for more GPU memory
```

Or pass overrides directly:

```bash
python AR_training.py \
  ++ATATConfig.learning_rate=5e-5 \
  ++ATATConfig.pretrain.total_steps=200000 \
  ++ATATConfig.datamodule.batch_size=256
```

## Troubleshooting

| Issue | Fix |
|---|---|
| OOM | Reduce `batch_size` in launcher |
| Loss NaN | Reduce `learning_rate` or increase `warmup_steps` |
| Slow probe metrics | Normal; linear regression fits every epoch |
| Checkpoints not saving | Check `./results/PRETRAIN/LC/ar_pretrain_v1/` exists |

## Next Steps

See `AR_PRETRAINING_IMPLEMENTATION.md` for full details.
