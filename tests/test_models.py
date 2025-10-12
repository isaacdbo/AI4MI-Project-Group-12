#!/usr/bin/env python3

"""
Shape smoke tests for all model architectures.
Tests that models produce correct output shapes for various input sizes.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ENet import ENet
from segformer_b0 import SegFormerB0


def _check(model, C=1, H=256, W=256, K=3, name="Model"):
    """Helper function to check model output shape"""
    x = torch.randn(2, C, H, W)
    y = model(x)
    expected_shape = (2, K, H, W)

    if y.shape == expected_shape:
        print(f"✓ {name}: Input {x.shape} -> Output {y.shape} (Expected {expected_shape}) - PASS")
        return True
    else:
        print(f"✗ {name}: Input {x.shape} -> Output {y.shape} (Expected {expected_shape}) - FAIL")
        return False


def test_enet_baseline_shape():
    """Test baseline ENet without SE blocks"""
    print("\n=== Testing ENet Baseline ===")
    m = ENet(in_dim=1, out_dim=5, kernels=8, factor=2, use_se=False)
    m.eval()
    return _check(m, C=1, H=256, W=256, K=5, name="ENet (baseline)")


def test_enet_se_shape():
    """Test ENet with Squeeze-and-Excitation blocks"""
    print("\n=== Testing ENet + SE ===")
    m = ENet(in_dim=1, out_dim=5, kernels=8, factor=2, use_se=True)
    m.eval()
    return _check(m, C=1, H=256, W=256, K=5, name="ENet + SE")


def test_segformer_b0_shape():
    """Test SegFormer-B0 architecture"""
    print("\n=== Testing SegFormer-B0 ===")
    m = SegFormerB0(num_classes=5, in_ch=1)
    m.eval()
    return _check(m, C=1, H=256, W=256, K=5, name="SegFormer-B0")


def test_various_sizes():
    """Test models with various input sizes"""
    print("\n=== Testing Various Input Sizes ===")
    sizes = [(128, 128), (256, 256), (512, 512)]
    results = []

    for H, W in sizes:
        print(f"\nTesting size {H}x{W}:")
        m = ENet(in_dim=1, out_dim=5, kernels=8, factor=2, use_se=False)
        m.eval()
        results.append(_check(m, C=1, H=H, W=W, K=5, name=f"  ENet {H}x{W}"))

    return all(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Running Model Shape Smoke Tests")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(test_enet_baseline_shape())
    results.append(test_enet_se_shape())
    results.append(test_segformer_b0_shape())
    results.append(test_various_sizes())

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all(results):
        print("✓ All tests PASSED")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
