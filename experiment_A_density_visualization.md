# 实验A：信息密度分布可视化

## 实验目标

验证核心假设：**视频不同 block 的信息密度存在显著且有规律的差异**，从而证明"固定压缩率对所有 block 一视同仁"是次优的，为内容感知动态压缩提供 motivation。

产出物是一张可直接放进 paper Introduction/Method 的 **Figure**，展示密度分数随视频时间的分布结构。

---

## 背景与假设

当前方法（HeterogeneousCompressor）在推理时对每个 mid block 计算一个信息密度分数 `d`，由两个轻量信号合成：

- **运动强度**：相邻 block latent 的余弦距离（`motion_score`）
- **结构复杂度**：LR 分支的重建误差（`complexity_score`）

合成后按阈值划分三档：
- `d > 0.7` → 高密度，8× 压缩（~780 tokens）
- `0.3 ≤ d ≤ 0.7` → 中密度，32× 压缩（~182 tokens，PackForcing 原版）
- `d < 0.3` → 低密度，128× 压缩（~48 tokens）

**假设**：密度分数应该呈现"波峰/波谷"结构，与视频内容的运动/静止节奏吻合；而不是一条平线（那意味着固定压缩率已经是最优的）。

---

## 实验设置

### Step 1：准备测试视频 Prompt（共 3 类，各 2~3 条）

**目的**：覆盖"动静分明"、"持续高动态"、"持续低动态"三种典型场景，以验证密度分数确实能区分内容类型。

**A 类 —— 动静混合（最重要）**：密度分数应出现明显波峰/波谷，是最有说服力的图

```
"A person runs at full speed across a field, then stops and stands still watching the sunset."
"A car accelerates through a winding mountain road, then parks and the driver steps out to look at the view."
"A dancer performs an energetic routine, then takes a bow and holds a still pose."
```

**B 类 —— 持续高动态**：密度分数应全程高位，验证高密度块确实被分配更多 token

```
"A skateboarder performs continuous tricks and jumps in a skate park."
"Fast-paced street traffic with cars and motorcycles constantly moving."
```

**C 类 —— 持续低动态**：密度分数应全程低位，验证静止内容被激进压缩

```
"A still life of flowers on a table in soft morning light, gentle breeze makes petals barely move."
"A mountain lake at dawn, completely calm water reflecting the sky."
```

### Step 2：生成视频并提取密度分数

每条 prompt 生成 **30s** 视频（Wan2.1-T2V-1.3B，832×480，16 FPS，与 PackForcing 实验设置一致）。

**关键：在推理代码中 hook 以下数据，每个 block 记录一次**：

```python
# 在 HeterogeneousCompressor 的 forward / compress block 方法中插入以下记录逻辑
# 假设你的压缩函数签名类似 compress(latent_block, prev_latent_block)

record = {
    "block_idx": int,          # 当前 block 的索引（0, 1, 2, ...）
    "time_sec": float,         # block_idx * Bf / 16.0，换算成秒
    "motion_score": float,     # 相邻 block latent 余弦距离
    "complexity_score": float, # LR 分支重建误差（MSE 或 L1）
    "density_score": float,    # 合成后的最终 d 值
    "tier": str,               # "high" / "mid" / "low"
    "tokens_allocated": int,   # 实际分配的 token 数（780 / 182 / 48）
}
```

每条 prompt 将所有 block 的 record 保存为 `results/{prompt_id}_density.json`。

**数据提取的代码插入位置**（根据你的实际代码调整路径）：

```python
# 在 model/compress.py 或调用 DensityEstimator 的地方加入：
import json, os

density_log = []  # 全局 list，推理开始前清空

def log_density(block_idx, motion_score, complexity_score, density_score, tier, tokens):
    density_log.append({
        "block_idx": block_idx,
        "time_sec": block_idx * 4 / 16.0,   # Bf=4 frames, 16 FPS
        "motion_score": float(motion_score),
        "complexity_score": float(complexity_score),
        "density_score": float(density_score),
        "tier": tier,
        "tokens_allocated": tokens,
    })

# 推理结束后保存
def save_density_log(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(density_log, f, indent=2)
```

### Step 3：生成可视化图表

用以下 Python 脚本对每条 prompt 的 JSON 数据绘图，最终产出 Figure。

```python
"""
plot_density.py
用法：python plot_density.py --results_dir results/ --output_dir figures/
"""
import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 颜色定义（与三档对应）
TIER_COLORS = {
    "high": "#E74C3C",   # 红色 - 高密度
    "mid":  "#F39C12",   # 橙色 - 中密度
    "low":  "#2ECC71",   # 绿色 - 低密度
}
TIER_LABELS = {
    "high": "High-density (8× compression, ~780 tokens)",
    "mid":  "Mid-density  (32× compression, ~182 tokens)",
    "low":  "Low-density  (128× compression, ~48 tokens)",
}

def plot_single(data, title, save_path):
    times      = [d["time_sec"] for d in data]
    density    = [d["density_score"] for d in data]
    motion     = [d["motion_score"] for d in data]
    complexity = [d["complexity_score"] for d in data]
    tiers      = [d["tier"] for d in data]
    tokens     = [d["tokens_allocated"] for d in data]
    colors     = [TIER_COLORS[t] for t in tiers]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", wrap=True)

    # --- 子图1：density_score 折线 + 背景色块标注档位 ---
    ax = axes[0]
    ax.plot(times, density, color="#2C3E50", linewidth=1.8, zorder=3, label="Density score")
    ax.axhline(0.7, color=TIER_COLORS["high"], linestyle="--", linewidth=1, alpha=0.7, label="High threshold (0.7)")
    ax.axhline(0.3, color=TIER_COLORS["low"],  linestyle="--", linewidth=1, alpha=0.7, label="Low threshold (0.3)")
    # 背景色块
    for i, (t, tier) in enumerate(zip(times, tiers)):
        ax.axvspan(t, t + 4/16.0, alpha=0.25, color=TIER_COLORS[tier], linewidth=0)
    ax.set_ylabel("Density Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(a) Information Density Score over Time", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # --- 子图2：两个原始信号 ---
    ax = axes[1]
    # 归一化到 [0,1] 以便对比
    def norm(arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)
    ax.plot(times, norm(motion),     color="#3498DB", linewidth=1.5, label="Motion score (normalized)")
    ax.plot(times, norm(complexity), color="#9B59B6", linewidth=1.5, linestyle="--", label="Complexity score (normalized)")
    ax.set_ylabel("Score (normalized)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(b) Raw Density Signals", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # --- 子图3：token 分配柱状图（颜色对应档位）---
    ax = axes[2]
    ax.bar(times, tokens, width=4/16.0 * 0.9, color=colors, align="edge")
    # PackForcing 固定 token 数基准线
    ax.axhline(182, color="black", linestyle=":", linewidth=1.5, label="PackForcing fixed (182 tokens)")
    patches = [mpatches.Patch(color=TIER_COLORS[k], label=TIER_LABELS[k]) for k in ["high","mid","low"]]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.set_ylabel("Tokens Allocated", fontsize=11)
    ax.set_xlabel("Video Time (s)", fontsize=11)
    ax.set_title("(c) Token Budget Allocation per Block", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_comparison(all_data_dict, save_path):
    """
    多条 prompt 的密度分数叠加对比图（用于展示 A/B/C 三类差异）
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    style_map = {
        "mixed":  {"linestyle": "-",  "linewidth": 2.2},
        "action": {"linestyle": "--", "linewidth": 1.8},
        "static": {"linestyle": ":",  "linewidth": 1.8},
    }
    color_list = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C", "#E67E22"]

    for idx, (label, data) in enumerate(all_data_dict.items()):
        times   = [d["time_sec"]     for d in data]
        density = [d["density_score"] for d in data]
        # 猜测类别（可在 label 前缀约定 mixed_ / action_ / static_）
        cat = "mixed" if "mixed" in label else ("action" if "action" in label else "static")
        s = style_map.get(cat, {"linestyle": "-", "linewidth": 1.5})
        ax.plot(times, density, label=label, color=color_list[idx % len(color_list)], **s)

    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.fill_between([0, max(times)], 0.7, 1.05, alpha=0.06, color=TIER_COLORS["high"])
    ax.fill_between([0, max(times)], 0.3, 0.7,  alpha=0.06, color=TIER_COLORS["mid"])
    ax.fill_between([0, max(times)], -0.05, 0.3, alpha=0.06, color=TIER_COLORS["low"])
    ax.set_xlabel("Video Time (s)", fontsize=12)
    ax.set_ylabel("Density Score", fontsize=12)
    ax.set_title("Information Density Score across Different Video Types", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir",  default="figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_data = {}

    for json_file in sorted(Path(args.results_dir).glob("*_density.json")):
        prompt_id = json_file.stem.replace("_density", "")
        with open(json_file) as f:
            data = json.load(f)
        all_data[prompt_id] = data

        # 单条 prompt 的详细三子图
        plot_single(
            data,
            title=f"Density Analysis: {prompt_id}",
            save_path=os.path.join(args.output_dir, f"{prompt_id}_density.png")
        )

    # 所有 prompt 的汇总对比图
    if len(all_data) > 1:
        plot_summary_comparison(
            all_data,
            save_path=os.path.join(args.output_dir, "summary_density_comparison.png")
        )

if __name__ == "__main__":
    main()
```

---

## 预期输出与判断标准

### 每条 Prompt 产出一张三子图（单图）

| 子图 | 内容 | 预期结果 |
|---|---|---|
| (a) Density Score | 折线 + 背景色块 | A类出现明显波峰/波谷；B类全程高位；C类全程低位 |
| (b) Raw Signals | motion + complexity 归一化曲线 | 两条曲线走势大致吻合，说明两个信号互补 |
| (c) Token Allocation | 柱状图 + PackForcing 基准线 | A类出现大量高于/低于182的柱，证明异构分配有效 |

### 汇总对比图（summary图）

所有 prompt 的 density score 叠加，三类场景的曲线应呈现明显分层：
- 动静混合类：波动最大，曲线穿越两条阈值线
- 高动态类：曲线集中在 0.7 以上
- 低动态类：曲线集中在 0.3 以下

**如果看到以上结果 → 假设成立，可以写进 paper。**

**如果密度分数是一条基本平线 → 密度估计信号失效，需要先 debug DensityEstimator 是否正常工作（检查 motion_score 和 complexity_score 的数值范围是否合理，以及两个阈值 0.7/0.3 是否需要重新标定）。**

---

## 补充：快速 debug 用的统计量

在绘图脚本运行时，额外打印以下统计信息，方便快速判断数据是否合理：

```python
def print_stats(prompt_id, data):
    tiers = [d["tier"] for d in data]
    total = len(tiers)
    from collections import Counter
    cnt = Counter(tiers)
    print(f"\n[{prompt_id}]  Total blocks: {total}")
    print(f"  High (8×):  {cnt['high']:3d} blocks  ({100*cnt['high']/total:.1f}%)")
    print(f"  Mid  (32×): {cnt['mid']:3d} blocks  ({100*cnt['mid']/total:.1f}%)")
    print(f"  Low  (128×):{cnt['low']:3d} blocks  ({100*cnt['low']/total:.1f}%)")
    density_arr = [d["density_score"] for d in data]
    print(f"  Density score — mean: {np.mean(density_arr):.3f}  std: {np.std(density_arr):.3f}  "
          f"min: {np.min(density_arr):.3f}  max: {np.max(density_arr):.3f}")
```

**健康状态参考**（30s 视频，约 120 blocks）：

| 视频类型 | 预期 High 占比 | 预期 Low 占比 | density std |
|---|---|---|---|
| 动静混合 | 20~40% | 20~40% | > 0.15 |
| 持续高动态 | > 60% | < 10% | < 0.12 |
| 持续低动态 | < 10% | > 60% | < 0.12 |

---

## 文件结构约定

```
project_root/
├── results/
│   ├── mixed_run_1_density.json
│   ├── mixed_run_2_density.json
│   ├── action_run_1_density.json
│   ├── action_run_2_density.json
│   ├── static_run_1_density.json
│   └── static_run_2_density.json
├── figures/
│   ├── mixed_run_1_density.png        ← 三子图
│   ├── mixed_run_2_density.png
│   ├── ...
│   └── summary_density_comparison.png ← 汇总对比图（放进 paper）
└── plot_density.py
```

---

## 注意事项

1. **不需要重新训练模型**。这个实验是纯推理侧的观测，在现有 checkpoint 上直接跑即可。

2. **LR 分支的 complexity_score 提取**：如果当前代码中 LR 分支没有显式返回 `complexity_score`，需要在 `LRBranch.forward()` 里额外计算并返回重建误差（`F.mse_loss(reconstructed, original).item()`），不需要任何额外训练。

3. **motion_score 计算**：相邻 block 的 latent 做全局平均池化后计算 cosine distance：
   ```python
   # prev_latent, curr_latent: (B, C, T, H, W)
   prev_feat = prev_latent.mean(dim=[2,3,4])  # (B, C)
   curr_feat = curr_latent.mean(dim=[2,3,4])  # (B, C)
   motion_score = 1 - F.cosine_similarity(prev_feat, curr_feat, dim=1).mean().item()
   ```

4. **阈值标定**：如果实验发现绝大多数 block 落在同一档位（比如 95% 以上都是 mid），说明阈值 0.3/0.7 需要根据实际 density score 的分布重新设置。可以先对 10 条不同类型视频的 density score 做统计，用 25th/75th 百分位数作为低/高阈值。

5. **Prompt 命名约定**：JSON 文件名前缀请按 `mixed_` / `action_` / `static_` 开头，`plot_summary_comparison` 函数会据此自动区分类别。
