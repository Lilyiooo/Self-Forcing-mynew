"""
calibrate_thresholds.py — Analyze density score distribution and compute thresholds.

Usage:
    python calibrate_thresholds.py --results_dir results/density_exp
"""

import json
import glob
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/density_exp")
    parser.add_argument("--low_percentile", type=int, default=33)
    parser.add_argument("--high_percentile", type=int, default=67)
    args = parser.parse_args()

    all_scores = []
    all_motion = []
    all_complexity = []

    files = sorted(glob.glob(f"{args.results_dir}/*_density.json"))
    if not files:
        print(f"No *_density.json files found in {args.results_dir}")
        return

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        all_scores.extend([d["density_score"] for d in data])
        all_motion.extend([d["motion_score"] for d in data])
        all_complexity.extend([d["complexity_score"] for d in data])

    all_scores = np.array(all_scores)
    all_motion = np.array(all_motion)
    all_complexity = np.array(all_complexity)

    print(f"Total blocks: {len(all_scores)}  from {len(files)} videos")
    print()
    print("=== Density Score ===")
    print(f"  mean={all_scores.mean():.4f}  std={all_scores.std():.4f}")
    print(f"  min={all_scores.min():.4f}  max={all_scores.max():.4f}")
    print()
    print("=== Motion Score (raw) ===")
    print(f"  mean={all_motion.mean():.4f}  std={all_motion.std():.4f}")
    print(f"  min={all_motion.min():.4f}  max={all_motion.max():.4f}")
    print()
    print("=== Complexity Score (raw) ===")
    print(f"  mean={all_complexity.mean():.4f}  std={all_complexity.std():.4f}")
    print(f"  min={all_complexity.min():.4f}  max={all_complexity.max():.4f}")
    print()
    print("=== Density Score Percentiles ===")
    for p in [10, 20, 25, 33, 40, 50, 60, 67, 75, 80, 90]:
        print(f"  {p:3d}th: {np.percentile(all_scores, p):.4f}")

    low_thresh = np.percentile(all_scores, args.low_percentile)
    high_thresh = np.percentile(all_scores, args.high_percentile)

    print()
    print(f"=== Recommended Thresholds ({args.low_percentile}th / {args.high_percentile}th) ===")
    print(f"  LOW_THRESHOLD  = {low_thresh:.4f}")
    print(f"  HIGH_THRESHOLD = {high_thresh:.4f}")

    # Save
    out = {
        "total_blocks": int(len(all_scores)),
        "density_mean": float(all_scores.mean()),
        "density_std": float(all_scores.std()),
        "low_threshold": float(low_thresh),
        "high_threshold": float(high_thresh),
    }
    out_path = f"{args.results_dir}/calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print config snippet
    print("\n=== Config snippet (copy to yaml) ===")
    print(f"delta_high: {high_thresh:.4f}")
    print(f"delta_low:  {low_thresh:.4f}")


if __name__ == "__main__":
    main()
