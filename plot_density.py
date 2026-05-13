"""
plot_density.py — Visualize density distribution from Experiment A results.

Usage:
    python plot_density.py --results_dir results/density_exp --output_dir figures/
"""

import json
import os
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Color / style constants ───────────────────────────────────────────
TIER_COLORS = {
    "high": "#E74C3C",   # red
    "mid":  "#F39C12",   # orange
    "low":  "#2ECC71",   # green
}
TIER_LABELS = {
    "high": "High-density (8×, ~780 tokens)",
    "mid":  "Mid-density  (32×, ~182 tokens)",
    "low":  "Low-density  (128×, ~48 tokens)",
}


def print_stats(prompt_id, data):
    """Quick sanity check printed to stdout."""
    tiers = [d["tier"] for d in data]
    total = len(tiers)
    cnt = Counter(tiers)
    density_arr = [d["density_score"] for d in data]
    print(f"\n[{prompt_id}]  Total blocks: {total}")
    print(f"  High (8×):  {cnt.get('high',0):3d} blocks  ({100*cnt.get('high',0)/total:.1f}%)")
    print(f"  Mid  (32×): {cnt.get('mid',0):3d} blocks  ({100*cnt.get('mid',0)/total:.1f}%)")
    print(f"  Low  (128×):{cnt.get('low',0):3d} blocks  ({100*cnt.get('low',0)/total:.1f}%)")
    print(f"  Density score — mean: {np.mean(density_arr):.3f}  std: {np.std(density_arr):.3f}  "
          f"min: {np.min(density_arr):.3f}  max: {np.max(density_arr):.3f}")


def plot_single(data, title, save_path):
    """Three-subplot figure for a single prompt."""
    times      = [d["time_sec"]      for d in data]
    density    = [d["density_score"]  for d in data]
    motion     = [d["motion_score"]   for d in data]
    complexity = [d["complexity_score"] for d in data]
    tiers      = [d["tier"]          for d in data]
    tokens     = [d["tokens_allocated"] for d in data]
    colors     = [TIER_COLORS.get(t, "#999") for t in tiers]

    # Block duration for shading (derived from time gaps or fixed)
    block_dur = (times[1] - times[0]) if len(times) > 1 else 0.75

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", wrap=True)

    # ── (a) Density score + tier shading ──────────────────────────────
    ax = axes[0]
    ax.plot(times, density, color="#2C3E50", linewidth=1.8, zorder=3, label="Density score")
    ax.axhline(0.7, color=TIER_COLORS["high"], ls="--", lw=1, alpha=0.7, label="High threshold (0.7)")
    ax.axhline(0.3, color=TIER_COLORS["low"],  ls="--", lw=1, alpha=0.7, label="Low threshold (0.3)")
    for i, (t, tier) in enumerate(zip(times, tiers)):
        ax.axvspan(t, t + block_dur, alpha=0.25, color=TIER_COLORS.get(tier, "#ccc"), lw=0)
    ax.set_ylabel("Density Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(a) Information Density Score over Time", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # ── (b) Raw signals (normalized) ──────────────────────────────────
    ax = axes[1]
    def norm(arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)
    ax.plot(times, norm(motion),     color="#3498DB", lw=1.5, label="Motion score (normalized)")
    ax.plot(times, norm(complexity), color="#9B59B6", lw=1.5, ls="--", label="Complexity score (normalized)")
    ax.set_ylabel("Score (normalized)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(b) Raw Density Signals", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # ── (c) Token allocation bar chart ────────────────────────────────
    ax = axes[2]
    ax.bar(times, tokens, width=block_dur * 0.9, color=colors, align="edge")
    ax.axhline(182, color="black", ls=":", lw=1.5, label="PackForcing fixed (182 tokens)")
    patches = [mpatches.Patch(color=TIER_COLORS[k], label=TIER_LABELS[k])
               for k in ["high", "mid", "low"]]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.set_ylabel("Tokens Allocated", fontsize=11)
    ax.set_xlabel("Video Time (s)", fontsize=11)
    ax.set_title("(c) Token Budget Allocation per Block", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_comparison(all_data_dict, save_path):
    """Overlay density curves from all prompts, styled by category."""
    fig, ax = plt.subplots(figsize=(14, 5))

    style_map = {
        "mixed":  {"linestyle": "-",  "linewidth": 2.2},
        "action": {"linestyle": "--", "linewidth": 1.8},
        "static": {"linestyle": ":",  "linewidth": 1.8},
    }
    palette = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22"]

    max_time = 0
    for idx, (label, data) in enumerate(all_data_dict.items()):
        times   = [d["time_sec"]     for d in data]
        density = [d["density_score"] for d in data]
        max_time = max(max_time, max(times))

        cat = "mixed" if "mixed" in label else ("action" if "action" in label else "static")
        s = style_map.get(cat, {"linestyle": "-", "linewidth": 1.5})
        ax.plot(times, density, label=label, color=palette[idx % len(palette)], **s)

    ax.axhline(0.7, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(0.3, color="gray", ls="--", lw=1, alpha=0.6)
    ax.fill_between([0, max_time], 0.7, 1.05, alpha=0.06, color=TIER_COLORS["high"])
    ax.fill_between([0, max_time], 0.3, 0.7,  alpha=0.06, color=TIER_COLORS["mid"])
    ax.fill_between([0, max_time], -0.05, 0.3, alpha=0.06, color=TIER_COLORS["low"])
    ax.set_xlabel("Video Time (s)", fontsize=12)
    ax.set_ylabel("Density Score", fontsize=12)
    ax.set_title("Information Density Score across Different Video Types", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/density_exp")
    parser.add_argument("--output_dir",  default="figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_data = {}

    for json_file in sorted(Path(args.results_dir).glob("*_density.json")):
        prompt_id = json_file.stem.replace("_density", "")
        with open(json_file) as f:
            data = json.load(f)
        all_data[prompt_id] = data

        print_stats(prompt_id, data)
        plot_single(
            data,
            title=f"Density Analysis: {prompt_id}",
            save_path=os.path.join(args.output_dir, f"{prompt_id}_density.png"),
        )

    if len(all_data) > 1:
        plot_summary_comparison(
            all_data,
            save_path=os.path.join(args.output_dir, "summary_density_comparison.png"),
        )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
