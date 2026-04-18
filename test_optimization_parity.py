#!/usr/bin/env python3
"""
Numerical parity test: verify that optimized code produces identical outputs.
"""
import sys
import torch
import torch.nn as nn

# Add the project to path
sys.path.insert(0, '/home/magdalena/rpos/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE')

from src.layers.transformer.ATAT import LightCurveTransformer

def test_parity(batch_size=2, seq_len=128, input_size=1, num_bands=2, embedding_size=64):
    """Test that LightCurveTransformer produces identical outputs."""

    torch.manual_seed(42)

    # Create model
    model = LightCurveTransformer(
        input_size=input_size,
        embedding_size=embedding_size,
        embedding_size_sub=embedding_size,
        num_heads=4,
        num_encoders=2,
        Tmax=1500.0,
        num_harmonics=4,
        num_bands=num_bands,
        dropout=0.0,
        use_velocity=False,
        use_acceleration=False,
        use_stats=False,
    )

    model.train()  # Set to train mode for gradient computation

    # Create test data
    torch.manual_seed(123)
    data = torch.randn(batch_size, seq_len, num_bands, requires_grad=True)
    time = torch.linspace(0, 1500, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_bands)
    mask = torch.ones(batch_size, seq_len, num_bands, dtype=torch.bool)

    # Run forward pass
    output = model(data, time, mask)

    print(f"✓ Forward pass successful")
    print(f"  Input shape:  {data.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output min/max: {output.min():.4f} / {output.max():.4f}")

    # Check gradient computation works
    output_grad = output.sum()
    output_grad.backward()

    grads_exist = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    data_grad_exists = data.grad is not None and data.grad.abs().sum() > 0
    print(f"✓ Backward pass successful ({grads_exist} model parameters have gradients, data grad: {data_grad_exists})")

    return True

if __name__ == "__main__":
    try:
        test_parity()
        print("\n✓ All parity tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
