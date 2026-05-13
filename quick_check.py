"""
quick_check.py — Verify the v2 density estimator produces distinguishable signals.

Usage:
    python quick_check.py
"""

import torch
import numpy as np

import sys, os, importlib.util

# Load density_estimator.py directly, bypassing model/__init__.py entirely
_spec = importlib.util.spec_from_file_location(
    "density_estimator",
    os.path.join(os.path.dirname(__file__), "model", "density_estimator.py"),
)
_density_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_density_mod)
DensityEstimator = _density_mod.DensityEstimator


def make_static_latent():
    """Static scene: T frames nearly identical."""
    base = torch.randn(1, 16, 1, 60, 104)       # single frame
    z = base.expand(-1, -1, 3, -1, -1).clone()   # repeat T=3 times
    z += torch.randn_like(z) * 0.01               # tiny perturbation
    return z


def make_action_latent():
    """High-dynamic scene: T frames are independent."""
    return torch.randn(1, 16, 3, 60, 104)


def make_complex_latent():
    """Spatially complex scene: high-freq spatial noise."""
    return torch.randn(1, 16, 3, 60, 104) * 2.0


def make_smooth_latent():
    """Spatially smooth scene: low-freq only (large Gaussian blobs)."""
    z = torch.randn(1, 16, 3, 60, 104)
    # Low-pass filter via heavy avg_pool then upsample
    z = torch.nn.functional.avg_pool3d(z, kernel_size=(1, 4, 4), stride=(1, 4, 4))
    z = torch.nn.functional.interpolate(z, size=(3, 60, 104), mode='trilinear', align_corners=False)
    return z


print("=" * 60)
print("  Density Estimator v2 — Quick Health Check")
print("=" * 60)

estimator = DensityEstimator()

scenarios = [
    ("Static (temporal)", make_static_latent),
    ("Action (temporal)", make_action_latent),
    ("Smooth (spatial)",  make_smooth_latent),
    ("Complex (spatial)", make_complex_latent),
]

for name, make_fn in scenarios:
    estimator.reset()
    raw_motions = []
    raw_complexities = []
    scores = []
    tiers_list = []

    for _ in range(40):
        z = make_fn()
        tier, score, info = estimator(z)
        raw_motions.append(info["raw_motion"])
        raw_complexities.append(info["raw_complexity"])
        scores.append(score)
        tiers_list.append(tier)

    # Skip warmup
    raw_motions = np.array(raw_motions[5:])
    raw_complexities = np.array(raw_complexities[5:])
    scores = np.array(scores[5:])
    from collections import Counter
    cnt = Counter(tiers_list[5:])

    print(f"\n[{name}]")
    print(f"  raw_motion:      mean={raw_motions.mean():.4f}  std={raw_motions.std():.4f}")
    print(f"  raw_complexity:  mean={raw_complexities.mean():.4f}  std={raw_complexities.std():.4f}")
    print(f"  density_score:   mean={scores.mean():.3f}  std={scores.std():.3f}")
    print(f"  tiers:  H={cnt.get('high',0)}  M={cnt.get('mid',0)}  L={cnt.get('low',0)}")

# Verdict
print("\n" + "=" * 60)
print("  Verdict:")
print("  PASS: Static vs Action motion scores differ by >10x")
print("  PASS: Smooth vs Complex complexity scores differ visibly")
print("  FAIL:  All scenarios produce similar scores")
print("=" * 60)
