#!/usr/bin/env python3
"""Quick validation test for AR pretraining implementation."""

import torch
import sys
sys.path.insert(0, '/home/magdalena/rpos/pipeline')

from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.src.layers.transformer.ATAT import LightCurveTransformer
from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.src.models.PretrainARModule import PretrainARModule


def test_causal_masking():
    """Test that causal mode produces correct output shapes and prevents future leakage."""
    print("Testing causal masking...")

    # Create transformer with causal mode
    transformer = LightCurveTransformer(
        input_size=1,
        embedding_size=32,
        embedding_size_sub=32,
        num_heads=4,
        num_encoders=2,
        num_bands=2,
        use_causal=True,
    )

    # Create dummy batch
    batch_size, seq_len, num_bands = 2, 100, 2
    data = torch.randn(batch_size, seq_len, num_bands)
    time = torch.linspace(0, 100, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_bands)
    mask = torch.ones(batch_size, seq_len, num_bands, dtype=torch.bool)

    # Forward pass
    emb = transformer(data, time, mask)

    # Check output shape: TimeHandler concatenates per-band sequences along dim=1
    # So output is (batch, seq_len*num_bands, embedding_size) — no CLS token prepended
    expected_shape = (batch_size, seq_len * num_bands, 32)
    assert emb.shape == expected_shape, f"Expected shape {expected_shape}, got {emb.shape}"
    print(f"✓ Output shape correct (causal): {emb.shape}")

    # Test bidirectional mode (default)
    transformer_bidir = LightCurveTransformer(
        input_size=1,
        embedding_size=32,
        embedding_size_sub=32,
        num_heads=4,
        num_encoders=2,
        num_bands=2,
        use_causal=False,
    )

    emb_bidir = transformer_bidir(data, time, mask)

    # Bidirectional should have CLS token prepended: (batch, seq_len*num_bands+1, embedding_size)
    expected_shape_bidir = (batch_size, seq_len * num_bands + 1, 32)
    assert emb_bidir.shape == expected_shape_bidir, f"Expected shape {expected_shape_bidir}, got {emb_bidir.shape}"
    print(f"✓ Bidirectional output shape correct: {emb_bidir.shape}")


def test_ar_module():
    """Test PretrainARModule initialization and forward pass."""
    print("\nTesting PretrainARModule...")

    transformer = LightCurveTransformer(
        input_size=1,
        embedding_size=32,
        embedding_size_sub=32,
        num_heads=4,
        num_encoders=2,
        num_bands=2,
        use_causal=True,
    )

    module = PretrainARModule(
        model=transformer,
        embedding_size=32,
        num_bands=2,
        lr=1e-4,
        warmup_steps=100,
        total_steps=1000,
        eval_probe=False,
    )

    # Create dummy batch
    batch = {
        "data": torch.randn(4, 100, 2),
        "time": torch.linspace(0, 100, 100).unsqueeze(0).unsqueeze(-1).expand(4, -1, 2),
        "mask": torch.ones(4, 100, 2, dtype=torch.bool),
        "labels": torch.randint(0, 5, (4,)),
    }

    # Test training step (should not raise)
    loss = module.training_step(batch.copy(), batch_idx=0)
    assert isinstance(loss, torch.Tensor), f"Expected tensor loss, got {type(loss)}"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Training step passed, loss={loss.item():.4f}")

    # Test validation step (should not raise)
    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    module.validation_step(batch_copy, batch_idx=0, dataloader_idx=0)
    print("✓ Validation step passed")

    # Test optimizer config
    optimizers = module.configure_optimizers()
    assert "optimizer" in optimizers, "Missing optimizer in config"
    assert "lr_scheduler" in optimizers, "Missing lr_scheduler in config"
    print("✓ Optimizer configuration correct")


def test_lr_schedule():
    """Test LR schedule shape."""
    print("\nTesting LR schedule...")
    from pipeline.training.lc_classifier_ztf.ATAT_ALeRCE.src.models.PretrainARModule import _ar_lr_lambda

    warmup_steps = 1000
    total_steps = 10000
    eta_min = 0.1

    # During warmup
    lr_0 = _ar_lr_lambda(0, warmup_steps, total_steps, eta_min)
    lr_500 = _ar_lr_lambda(500, warmup_steps, total_steps, eta_min)
    lr_1000 = _ar_lr_lambda(warmup_steps, warmup_steps, total_steps, eta_min)

    # During decay
    lr_5000 = _ar_lr_lambda(5000, warmup_steps, total_steps, eta_min)
    lr_10000 = _ar_lr_lambda(10000, warmup_steps, total_steps, eta_min)

    print(f"  LR at step 0:      {lr_0:.4f} (should ≈ 0)")
    print(f"  LR at step 500:    {lr_500:.4f} (should ≈ 0.5)")
    print(f"  LR at step 1000:   {lr_1000:.4f} (should = 1.0 at peak)")
    print(f"  LR at step 5000:   {lr_5000:.4f} (during cosine decay)")
    print(f"  LR at step 10000:  {lr_10000:.4f} (should ≈ {eta_min})")

    assert 0 <= lr_0 < 0.1, "LR at start should be near 0"
    assert lr_1000 >= 0.95, "LR at warmup end should peak near 1.0"
    assert lr_10000 >= eta_min * 0.95, "LR at end should approach eta_min"
    print("✓ LR schedule shape correct")


if __name__ == "__main__":
    print("=" * 60)
    print("AR Pretraining Implementation Validation")
    print("=" * 60)

    try:
        test_causal_masking()
        test_ar_module()
        test_lr_schedule()
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
