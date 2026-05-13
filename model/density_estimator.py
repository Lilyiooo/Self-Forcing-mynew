# model/density_estimator.py
#
# Information density estimator v3.
#
# Single signal: temporal variance within a block (motion).
# Fixed min-max normalization based on observed real-latent distribution.
# No adaptive normalization — preserves absolute value differences across content.

import torch
from typing import Literal

DensityLevel = Literal["high", "mid", "low"]

# Observed range of raw_motion on real generation latents (7 prompts × 40 blocks)
# Action mean ≈ 0.87, Static mean ≈ 0.76; full range ≈ [0.26, 1.11]
# Pad slightly to avoid clipping at boundaries.
_MOTION_MIN = 0.25
_MOTION_MAX = 1.10


class DensityEstimator:
    """
    Information density estimator for heterogeneous KV cache compression.

    Usage:
        estimator = DensityEstimator()
        estimator.reset()           # call at the start of each video
        tier, score, info = estimator(z_block)
    """

    def __init__(
        self,
        high_threshold: float = 0.67,
        low_threshold: float = 0.33,
        motion_weight: float = 1.0,       # kept for config compat, ignored
        complexity_weight: float = 0.0,   # kept for config compat, ignored
        window_size: int = 20,            # kept for config compat, ignored
        motion_min: float = _MOTION_MIN,
        motion_max: float = _MOTION_MAX,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.motion_min = motion_min
        self.motion_max = motion_max

    def reset(self):
        """No-op, kept for API compatibility."""
        pass

    # Keep backward-compatible alias
    def reset_stats(self):
        self.reset()

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _motion_score(self, z: torch.Tensor) -> float:
        """
        Temporal variance within a block.
        z: (B, C, T, H, W) → var along dim=2 (time), then mean.
        High motion → large variance between T frames.
        Static scene → T frames nearly identical → variance ≈ 0.
        """
        return z.float().var(dim=2).mean().item()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def __call__(self, z: torch.Tensor):
        """
        Args:
            z: latent block, shape (B, C, T, H, W)

        Returns:
            tier:          "high" / "mid" / "low"
            density_score: float in [0, 1]
            info:          dict with raw signal values (for logging)
        """
        raw_motion = self._motion_score(z)

        # Fixed-range normalization
        density_score = (raw_motion - self.motion_min) / (self.motion_max - self.motion_min)
        density_score = float(max(0.0, min(1.0, density_score)))

        if density_score >= self.high_threshold:
            tier: DensityLevel = "high"
        elif density_score <= self.low_threshold:
            tier: DensityLevel = "low"
        else:
            tier: DensityLevel = "mid"

        info = {
            "raw_motion": raw_motion,
            "raw_complexity": 0.0,
            "norm_motion": density_score,
            "norm_complexity": 0.0,
        }

        return tier, density_score, info

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def estimate(self, z_current, z_prev=None, complexity_score=0.0):
        """
        Backward-compatible interface.
        Returns (tier, density_score) — same as old API.
        """
        tier, density_score, _info = self(z_current)
        return tier, density_score
