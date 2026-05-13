# 内容感知异构 KV Cache 压缩 —— Self-Forcing 代码改造方案

---

## 背景与动机

### 问题：PackForcing 的均等压缩假设是错的

PackForcing 解决了长视频自回归生成的内存瓶颈，核心手段是将所有中间历史块（mid partition）统一用双分支压缩模块压缩 **32×**，把每块从 6240 个 token 压到 182 个 token，从而在有限显存下支持更长的生成序列。

但这个设计隐含了一个假设：**所有历史块的信息密度是均等的，应该被同等压缩**。

这个假设在真实视频中并不成立。视频在时间轴上信息分布极度不均：

- 一段 0.25 秒的快速动作携带的独特信息，可能远超一段 3 秒的静止背景。
- 对静止块用 182 个 token 表示，大量 token 在描述几乎没有变化的内容，是压缩预算的浪费。
- 对高动作块同样只用 182 个 token，关键帧的运动细节被过度压缩，attention 从这些 token 里能获取的信息精度下降。

这个 trade-off 直接反映在 PackForcing 的评测数据上：**Dynamic Degree 最高（56.25），但 Subject Consistency 比 LongLive 低约 1.5 分**——说明高动作块的主体细节因过度压缩而丢失，而静止块又占用了不必要的 token 预算。

### 核心思路：在固定总预算下，按信息密度自适应分配压缩比

本方案提出**内容感知的异构 KV Cache 压缩**：在给定 mid buffer 总 token 预算不变的前提下，根据每个历史块的信息密度动态选择压缩比——密度高的块少压（保留更多细节），密度低的块多压（节省预算），让 token 预算集中在最重要的地方。

具体分三档：

| 档位 | 触发条件 | 压缩比 | 每块 Token 数（原 6240） |
|------|----------|--------|--------------------------|
| 高密度（High） | density score > 0.7 | 8×   | ~780 tokens |
| 中密度（Mid）  | 0.3 ≤ score ≤ 0.7  | 32×  | ~182 tokens（PackForcing 原版） |
| 低密度（Low）  | score < 0.3        | 128× | ~48 tokens  |

**总 token 预算守恒**：mid buffer 从"块数 × 182"的固定大小，改为"所有块 token 数之和 ≤ Nmid_tokens"的动态分配。超出预算时，优先驱逐信息密度最低的块（而非简单 FIFO）。

### 信息密度如何估计

不引入任何额外大模型，使用两个轻量代理信号：

**运动强度**（主要信号，权重 0.7）：计算相邻块 latent 的余弦距离。只需一次向量运算，计算量可忽略。

```python
motion_score = 1 - cosine_similarity(z_i.mean(dim=[1,2,3]), z_{i-1}.mean(dim=[1,2,3]))
```

**结构复杂度**（辅助信号，权重 0.3）：LR 分支在做 decode → pool → re-encode 时顺带计算重建误差，作为内容复杂度的代理指标，**完全零额外开销**。

两个信号加权融合并归一化到 \[0, 1\]，按阈值分入三档。

### 与 PackForcing 的关系

本方案**不是修改 PackForcing**，而是提出了一个更通用的内容感知压缩框架，PackForcing 的均等 32× 压缩是其中的一个特例（mid 档）。Self-Forcing 作为代码基础，PackForcing 作为主要对比基线。

核心对比实验：在**相同总 token 预算**下，异构压缩 vs 固定 32× 压缩，预期结果是 Subject Consistency 提升（高密度块保留了更精确的主体细节），Dynamic Degree 维持或提升（低密度块释放的预算流向了高密度块），显存开销不变。

---

## 0. 改动总览

```
self-forcing/
├── models/
│   ├── kv_cache.py              ← 【核心改动】三分区 cache + token 预算管理
│   ├── compress.py              ← 【新建】双分支压缩模块（三套压缩头）
│   ├── density_estimator.py     ← 【新建】信息密度估计器
│   └── transformer.py           ← 【小改】attention 前注入压缩 KV
├── training/
│   └── trainer.py               ← 【小改】两阶段训练逻辑
└── configs/
    └── heterogeneous_cache.yaml ← 【新建】超参配置
```

---

## 1. 新建：`models/density_estimator.py`

### 职责
在每个历史块的 latent 生成后、进入压缩模块之前，计算该块的信息密度分，输出 `"high"` / `"mid"` / `"low"` 三档路由标签。

### 完整实现

```python
# models/density_estimator.py

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal

DensityLevel = Literal["high", "mid", "low"]


@dataclass
class DensityConfig:
    delta_high: float = 0.7      # 高密度阈值
    delta_low: float = 0.3       # 低密度阈值
    motion_weight: float = 0.7   # 运动分数权重
    complexity_weight: float = 0.3  # 复杂度分数权重


class DensityEstimator:
    """
    轻量信息密度估计器。无可学习参数，纯计算。
    
    两个代理信号：
      1. motion_score    —— 当前块与上一块 latent 的余弦距离（主要信号）
      2. complexity_score —— LR 分支重建误差（辅助信号，由压缩模块外部传入）
    """

    def __init__(self, config: DensityConfig = None):
        self.cfg = config or DensityConfig()
        # 滑动统计，用于归一化（在线 min-max，初始化为 None）
        self._motion_min = None
        self._motion_max = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def estimate(
        self,
        z_current: torch.Tensor,    # (C, T, H, W)  当前块 latent
        z_prev: torch.Tensor | None,  # (C, T, H, W)  上一块 latent；第一块传 None
        complexity_score: float = 0.0,  # 由 LR 分支重建误差外部传入
    ) -> tuple[DensityLevel, float]:
        """
        返回 (density_level, raw_density_score)
        raw_density_score ∈ [0, 1]，越高越密集
        """
        motion = self._motion_score(z_current, z_prev)
        # 对 motion_score 做在线 min-max 归一化
        motion_norm = self._normalize_motion(motion)

        # complexity_score 外部已归一化到 [0,1]
        density = (
            self.cfg.motion_weight * motion_norm
            + self.cfg.complexity_weight * complexity_score
        )
        density = float(torch.clamp(torch.tensor(density), 0.0, 1.0))

        if density > self.cfg.delta_high:
            level: DensityLevel = "high"
        elif density < self.cfg.delta_low:
            level: DensityLevel = "low"
        else:
            level: DensityLevel = "mid"

        return level, density

    def reset_stats(self):
        """每个视频生成开始时调用，重置在线统计"""
        self._motion_min = None
        self._motion_max = None

    # ------------------------------------------------------------------
    # 内部计算
    # ------------------------------------------------------------------

    def _motion_score(
        self,
        z_current: torch.Tensor,
        z_prev: torch.Tensor | None,
    ) -> float:
        """余弦距离 ∈ [0, 2]，第一块默认返回中等值 0.5"""
        if z_prev is None:
            return 0.5
        # 全局平均池化到向量
        v_curr = z_current.float().mean(dim=[1, 2, 3])   # (C,)
        v_prev = z_prev.float().mean(dim=[1, 2, 3])       # (C,)
        cos_sim = F.cosine_similarity(v_curr.unsqueeze(0), v_prev.unsqueeze(0))
        return float(1.0 - cos_sim)  # ∈ [0, 2]，实际视频通常 ∈ [0, 1]

    def _normalize_motion(self, raw: float) -> float:
        """在线 min-max 归一化，保证 motion_score ∈ [0, 1]"""
        eps = 1e-6
        if self._motion_min is None:
            self._motion_min = raw
            self._motion_max = raw + eps
        else:
            self._motion_min = min(self._motion_min, raw)
            self._motion_max = max(self._motion_max, raw + eps)
        return (raw - self._motion_min) / (self._motion_max - self._motion_min + eps)
```

---

## 2. 新建：`models/compress.py`

### 职责
实现双分支（HR + LR）压缩模块，三套 HR 压缩头分别对应 8×、32×、128× 压缩比。LR 分支的 VAE encoder 三档共用且冻结。

### 接口约定

```
输入：  latent block  (B, C, T, H, W)   其中 C=16, T=2, H=W=24（以 latent 空间计）
输出：  compressed KV tokens  (B, N_compressed, D)
        reconstruction_error  float（供密度估计器用作 complexity_score）
```

> **注意**：`T`、`H`、`W` 的具体值取决于 Self-Forcing 的 latent 分辨率，请以仓库实际值为准，下面用变量表示。

### 完整实现

```python
# models/compress.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

DensityLevel = Literal["high", "mid", "low"]

# ---------------------------------------------------------------
# HR 压缩头（三套，只有这部分有可学习参数）
# ---------------------------------------------------------------

class HRHead8x(nn.Module):
    """8× 压缩头：空间 2×，时间 2× → net 8×"""
    def __init__(self, in_ch: int = 16, d_model: int = 2048):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 32,   kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(32,   128,  kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(128,  d_model, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        y = self.conv(x)                          # (B, D, T', H', W')
        B, D, t, h, w = y.shape
        return y.permute(0, 2, 3, 4, 1).reshape(B, t * h * w, D)  # (B, N, D)


class HRHead32x(nn.Module):
    """32× 压缩头（PackForcing 原版）：空间 4×，时间 2×"""
    def __init__(self, in_ch: int = 16, d_model: int = 2048):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 32,   kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(32,   128,  kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(128,  512,  kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(512,  2048, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(2048, d_model, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        B, D, t, h, w = y.shape
        return y.permute(0, 2, 3, 4, 1).reshape(B, t * h * w, D)


class HRHead128x(nn.Module):
    """128× 压缩头：空间 8×，时间 2×"""
    def __init__(self, in_ch: int = 16, d_model: int = 2048):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 32,   kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(32,   128,  kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(128,  512,  kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(512,  2048, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(2048, 4096, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(4096, d_model, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        B, D, t, h, w = y.shape
        return y.permute(0, 2, 3, 4, 1).reshape(B, t * h * w, D)


# ---------------------------------------------------------------
# LR 分支（三档共用 VAE，冻结；只改 avgpool stride）
# ---------------------------------------------------------------

class LRBranch(nn.Module):
    """
    LR 分支：decode → avgpool → VAE encode → patch embed
    VAE 部分从主模型复用（传入引用），不新增参数，冻结不更新。
    """
    def __init__(
        self,
        vae,                    # 复用主模型的 VAE，外部传入
        patch_embedder: nn.Module,  # 复用主模型的 patch embed
        pool_stride_hw: int = 4,    # 空间池化 stride，三档分别传 2/4/8
        d_model: int = 2048,
    ):
        super().__init__()
        self.vae = vae           # 冻结，不在 compress.py 内 requires_grad_(False)
        self.patch_embedder = patch_embedder
        self.pool_stride_hw = pool_stride_hw
        self.d_model = d_model

    def forward(
        self, x_latent: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        返回:
          lr_tokens: (B, N_lr, D)
          complexity_score: float ∈ [0, 1]，重建误差归一化后的值
        """
        with torch.no_grad():
            # decode latent → pixel space
            x_pixel = self.vae.decode(x_latent)          # (B, 3, T_full, H_full, W_full)
        
        # 空间降采样
        B, C, T, H, W = x_pixel.shape
        x_pooled = F.avg_pool3d(
            x_pixel,
            kernel_size=(1, self.pool_stride_hw, self.pool_stride_hw),
            stride=(1, self.pool_stride_hw, self.pool_stride_hw),
        )                                                 # (B, 3, T, H', W')

        with torch.no_grad():
            # re-encode
            x_re_latent = self.vae.encode(x_pooled)      # (B, C', T', H'', W'')
        
        # complexity_score：重建误差（原 latent pool 后与重编码的 L2 距离）
        x_latent_pooled = F.avg_pool3d(
            x_latent,
            kernel_size=(1, self.pool_stride_hw, self.pool_stride_hw),
            stride=(1, self.pool_stride_hw, self.pool_stride_hw),
        )
        # 对齐空间尺寸以便计算误差
        if x_re_latent.shape != x_latent_pooled.shape:
            x_re_latent = F.interpolate(
                x_re_latent, size=x_latent_pooled.shape[2:], mode="trilinear", align_corners=False
            )
        recon_err = F.mse_loss(x_re_latent.float(), x_latent_pooled.float())
        complexity_score = float(torch.sigmoid(recon_err - 0.1))  # 粗归一化到 [0,1]

        # patch embed → tokens
        lr_tokens = self.patch_embedder(x_re_latent)     # (B, N_lr, D)
        return lr_tokens, complexity_score


# ---------------------------------------------------------------
# 主入口：HeterogeneousCompressor
# ---------------------------------------------------------------

class HeterogeneousCompressor(nn.Module):
    """
    统一压缩入口。根据 density_level 路由到对应 HR 头 + LR 分支，
    拼接 HR tokens 和 LR tokens 后返回。
    """

    def __init__(
        self,
        vae,
        patch_embedder: nn.Module,
        d_model: int = 2048,
        in_ch: int = 16,
    ):
        super().__init__()
        # 三套 HR 头（有梯度）
        self.hr_high = HRHead8x(in_ch, d_model)
        self.hr_mid  = HRHead32x(in_ch, d_model)
        self.hr_low  = HRHead128x(in_ch, d_model)

        # 三套 LR 分支（VAE 共用且冻结，只有 patch_embedder 可更新）
        self.lr_high = LRBranch(vae, patch_embedder, pool_stride_hw=2, d_model=d_model)
        self.lr_mid  = LRBranch(vae, patch_embedder, pool_stride_hw=4, d_model=d_model)
        self.lr_low  = LRBranch(vae, patch_embedder, pool_stride_hw=8, d_model=d_model)

        # HR + LR 拼接后投影到 d_model（可选，如果维度一致可省去）
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        x: torch.Tensor,              # (B, C, T, H, W)  latent block
        density_level: DensityLevel,
    ) -> tuple[torch.Tensor, float]:
        """
        返回:
          compressed_tokens: (B, N_compressed, D)
          complexity_score:  float（供密度估计器用）
        """
        if density_level == "high":
            hr_tokens = self.hr_high(x)
            lr_tokens, complexity = self.lr_high(x)
        elif density_level == "mid":
            hr_tokens = self.hr_mid(x)
            lr_tokens, complexity = self.lr_mid(x)
        else:  # "low"
            hr_tokens = self.hr_low(x)
            lr_tokens, complexity = self.lr_low(x)

        # 拼接并投影
        # 对齐序列长度（HR 和 LR token 数可能不同，取 HR 长度为准，LR 插值或裁剪）
        N_hr = hr_tokens.shape[1]
        if lr_tokens.shape[1] != N_hr:
            lr_tokens = lr_tokens[:, :N_hr, :] if lr_tokens.shape[1] > N_hr else \
                F.pad(lr_tokens, (0, 0, 0, N_hr - lr_tokens.shape[1]))

        combined = torch.cat([hr_tokens, lr_tokens], dim=-1)  # (B, N_hr, 2D)
        out = self.proj(combined)                               # (B, N_hr, D)
        return out, complexity
```

---

## 3. 核心改动：`models/kv_cache.py`

### 改动说明
在 Self-Forcing 原有 rolling KV cache 基础上，扩展为三分区结构，mid buffer 改为以 **token 数**为单位管理。

> **关键差异**：原版按块数驱逐（固定 182 tokens/块），新版按总 token 预算驱逐（每块 token 数可变），驱逐顺序按信息密度从低到高（先丢 low 档块）。

### 新增/修改的数据结构

```python
# 在原有 KV cache 类中，新增以下属性（在 __init__ 中初始化）

# ── mid buffer 的元数据（每个历史块一条记录）──────────────────────────
self.mid_meta: list[dict] = []
# 每条 dict 的结构：
# {
#   "density_level": "high" | "mid" | "low",
#   "density_score": float,   # raw density score
#   "n_tokens": int,          # 该块压缩后的 token 数
#   "kv_slice": slice,        # 在 mid_kv_buffer 张量中的切片位置
# }

# ── mid buffer 张量（预分配，避免频繁 cat）────────────────────────────
# 尺寸：(2, B, Nmid_tokens, n_heads, head_dim)
#   dim 0: K / V
self.mid_kv_buffer: torch.Tensor | None = None   # 懒初始化
self.mid_token_count: int = 0                    # 当前已用 token 数
self.Nmid_tokens: int = 5000                     # 总预算（可配置）
```

### `push_mid_block` 方法（新增）

```python
def push_mid_block(
    self,
    kv_compressed: torch.Tensor,   # (2, B, N_new, n_heads, head_dim)
    density_level: str,
    density_score: float,
) -> None:
    """
    将一个新的压缩块推入 mid buffer。
    若超出 token 预算，先按密度从低到高驱逐旧块，再插入。
    """
    N_new = kv_compressed.shape[2]

    # 1. 驱逐：优先驱逐 density_score 最低的块，直到有足够空间
    while self.mid_token_count + N_new > self.Nmid_tokens and self.mid_meta:
        self._evict_lowest_density_block()

    # 2. 如果驱逐后仍不够（N_new 本身超过预算），截断
    if N_new > self.Nmid_tokens:
        kv_compressed = kv_compressed[:, :, :self.Nmid_tokens, :, :]
        N_new = self.Nmid_tokens

    # 3. 懒初始化 buffer
    if self.mid_kv_buffer is None:
        B = kv_compressed.shape[1]
        n_heads = kv_compressed.shape[3]
        head_dim = kv_compressed.shape[4]
        self.mid_kv_buffer = torch.zeros(
            2, B, self.Nmid_tokens, n_heads, head_dim,
            dtype=kv_compressed.dtype,
            device=kv_compressed.device,
        )

    # 4. 写入 buffer
    start = self.mid_token_count
    end = start + N_new
    self.mid_kv_buffer[:, :, start:end, :, :] = kv_compressed
    self.mid_token_count = end

    # 5. 记录元数据
    self.mid_meta.append({
        "density_level": density_level,
        "density_score": density_score,
        "n_tokens": N_new,
        "kv_slice": slice(start, end),
    })


def _evict_lowest_density_block(self) -> None:
    """
    驱逐 density_score 最低的块（原地压缩 buffer）。
    """
    if not self.mid_meta:
        return

    # 找最低密度块的索引
    min_idx = min(range(len(self.mid_meta)), key=lambda i: self.mid_meta[i]["density_score"])
    evicted = self.mid_meta.pop(min_idx)
    ev_slice = evicted["kv_slice"]
    ev_n = evicted["n_tokens"]

    # 原地移动：把 ev_slice 之后的 token 往前移
    end = self.mid_token_count
    ev_start = ev_slice.start
    ev_end = ev_slice.stop

    if ev_end < end:
        self.mid_kv_buffer[:, :, ev_start:end - ev_n, :, :] = \
            self.mid_kv_buffer[:, :, ev_end:end, :, :].clone()

    self.mid_token_count -= ev_n

    # 更新后续块的 kv_slice 偏移
    for meta in self.mid_meta:
        s = meta["kv_slice"]
        if s.start >= ev_end:
            meta["kv_slice"] = slice(s.start - ev_n, s.stop - ev_n)


def get_mid_kv(self) -> torch.Tensor:
    """
    返回当前 mid buffer 中有效的 KV tokens。
    shape: (2, B, mid_token_count, n_heads, head_dim)
    """
    if self.mid_kv_buffer is None or self.mid_token_count == 0:
        return None
    return self.mid_kv_buffer[:, :, :self.mid_token_count, :, :]


def reset_mid_buffer(self) -> None:
    """每个视频生成开始时调用"""
    self.mid_meta = []
    self.mid_token_count = 0
    # 不清零 mid_kv_buffer 张量，下次写入会覆盖
```

---

## 4. 小改：`models/transformer.py`（attention 注入）

### 改动位置
在 causal attention 的 KV 构建阶段，将三分区 KV 拼接后送入 attention。原版只拼接 recent window，新版拼接顺序为：`[sink | mid_compressed | recent]`。

### 改动示意（找到对应的 attention forward 逻辑）

```python
# ── 原版（示意，以实际仓库代码为准）────────────────────────────────────
# k = torch.cat([self.kv_cache.sink_k, self.kv_cache.recent_k], dim=2)
# v = torch.cat([self.kv_cache.sink_v, self.kv_cache.recent_v], dim=2)

# ── 新版 ─────────────────────────────────────────────────────────────
mid_kv = self.kv_cache.get_mid_kv()   # (2, B, N_mid, n_heads, head_dim) or None

if mid_kv is not None:
    mid_k, mid_v = mid_kv[0], mid_kv[1]
    k = torch.cat([self.kv_cache.sink_k, mid_k, self.kv_cache.recent_k], dim=2)
    v = torch.cat([self.kv_cache.sink_v, mid_v, self.kv_cache.recent_v], dim=2)
else:
    k = torch.cat([self.kv_cache.sink_k, self.kv_cache.recent_k], dim=2)
    v = torch.cat([self.kv_cache.sink_v, self.kv_cache.recent_v], dim=2)
```

### Dynamic Top-K（继承 PackForcing 逻辑，token 粒度）

```python
# 在 mid_k、mid_v 拼入之前，对 mid tokens 做 Top-K 筛选
# （只在第一个去噪步计算 affinity，后续步复用缓存的 indices）

if mid_kv is not None and self.use_top_k:
    if self.is_first_denoise_step:
        # 用当前 query 计算对所有 mid token 的 affinity
        q_probe = self.kv_cache.recent_q[:, :, -1:, :, :]   # 最新一帧 query 作为 probe
        # affinity: (B, n_heads, N_mid)
        affinity = torch.einsum("b h 1 d, b h n d -> b h n", q_probe, mid_k)
        # 取 top-k（以 token 数为单位，而非块数）
        topk_indices = affinity.mean(dim=1).topk(
            min(self.top_k_tokens, mid_k.shape[2]), dim=-1
        ).indices                                             # (B, k)
        self.kv_cache.cached_topk_indices = topk_indices
    else:
        topk_indices = self.kv_cache.cached_topk_indices

    # 按 top-k 索引筛选 mid tokens
    B, n_heads, N_mid, head_dim = mid_k.shape
    idx = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B, n_heads, -1, head_dim)
    mid_k = torch.gather(mid_k, 2, idx)
    mid_v = torch.gather(mid_v, 2, idx)
```

---

## 5. 推理循环改造（`inference.py` 或 `generate.py`）

### 改动位置
在生成每个新 chunk 后，原本把该 chunk 滚入 recent window；改为：先计算密度，再路由到对应压缩头，把压缩后的 KV 推入 mid buffer。

```python
# 伪代码，基于 Self-Forcing 推理循环的结构

from models.density_estimator import DensityEstimator, DensityConfig
from models.compress import HeterogeneousCompressor

# ── 初始化（在推理主函数中）────────────────────────────────────────────
density_estimator = DensityEstimator(DensityConfig(
    delta_high=cfg.delta_high,
    delta_low=cfg.delta_low,
))
density_estimator.reset_stats()
kv_cache.reset_mid_buffer()

z_prev_chunk = None   # 用于计算 motion score

# ── 逐帧/逐块生成循环 ──────────────────────────────────────────────────
for chunk_idx in range(num_chunks):

    # 1. 去噪得到当前块 latent（Self-Forcing 原有逻辑）
    z_current = denoise_chunk(...)    # (B, C, T, H, W)

    # 2. 判断该块是否应进入 mid buffer（recent window 满了才开始压缩）
    if chunk_idx < recent_window_size:
        # 直接放入 recent window，不压缩
        kv_cache.push_recent(z_current)
        z_prev_chunk = z_current
        continue

    # 3. recent window 中最老的块即将被驱逐，先把它压缩推入 mid
    chunk_to_compress = kv_cache.pop_oldest_recent()   # (B, C, T, H, W) latent

    # 4. 估计信息密度（用即将压缩的块，而非当前新块）
    complexity_score_placeholder = 0.0   # 先用 0，压缩后更新
    density_level, density_score = density_estimator.estimate(
        z_current=chunk_to_compress,
        z_prev=z_prev_chunk,
        complexity_score=complexity_score_placeholder,
    )

    # 5. 压缩
    compressed_tokens, complexity_score = compressor(chunk_to_compress, density_level)
    # 用真实的 complexity_score 重新估计（可选，二次修正）
    density_level, density_score = density_estimator.estimate(
        z_current=chunk_to_compress,
        z_prev=z_prev_chunk,
        complexity_score=complexity_score,
    )

    # 6. 将压缩后的 KV 推入 mid buffer
    # 注意：这里需要用 transformer 对 compressed_tokens 做一次 KV projection
    kv_compressed = project_to_kv(compressed_tokens)   # (2, B, N, n_heads, head_dim)
    kv_cache.push_mid_block(kv_compressed, density_level, density_score)

    # 7. 更新 recent window（加入当前新块）
    kv_cache.push_recent(z_current)
    z_prev_chunk = chunk_to_compress
```

---

## 6. 新建：`configs/heterogeneous_cache.yaml`

```yaml
# configs/heterogeneous_cache.yaml

kv_cache:
  Nsink: 8           # sink 分区大小（latent frames）
  Nrecent: 4         # recent window 大小（latent frames）
  Nmid_tokens: 5000  # mid buffer 总 token 预算

density_estimator:
  delta_high: 0.7
  delta_low: 0.3
  motion_weight: 0.7
  complexity_weight: 0.3

compressor:
  d_model: 2048      # 与主 Transformer 的隐层维度一致
  in_ch: 16          # latent channel 数

top_k:
  enabled: true
  top_k_tokens: 1000  # mid buffer 中保留的 top-k token 数（送入 attention）
```

---

## 7. 训练策略改动（`training/trainer.py`）

### 阶段一：ODE 初始化（不改动，复用 Self-Forcing 原版）

### 阶段二：压缩头预训练（新增）

```python
# 在 trainer.py 中新增 pretrain_compressor 方法

def pretrain_compressor(self, dataloader, num_epochs=5):
    """
    固定 Transformer 主干，只训练三套 HR 压缩头。
    Loss：压缩前后 attention output 的 L2 重建误差。
    """
    # 冻结主干
    for name, param in self.model.named_parameters():
        if "hr_high" not in name and "hr_mid" not in name and "hr_low" not in name:
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in self.model.parameters() if p.requires_grad],
        lr=1e-4
    )

    for epoch in range(num_epochs):
        for batch in dataloader:
            z_chunks = batch["latents"]   # (B, N_chunks, C, T, H, W)

            total_loss = 0.0
            for i in range(z_chunks.shape[1]):
                z = z_chunks[:, i]   # (B, C, T, H, W)

                # 用三档分别过一遍，计算各自的重建 loss
                for level in ["high", "mid", "low"]:
                    compressed, _ = self.compressor(z, level)
                    # 重建 loss：压缩后再解压（用线性逆投影）与原 KV 的 L2
                    # 需要在 HeterogeneousCompressor 中额外实现一个 decode 头
                    loss = self._reconstruction_loss(compressed, z, level)
                    total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # 解冻
    for param in self.model.parameters():
        param.requires_grad_(True)
```

### 阶段三：端到端联合训练（小改）

```python
# 在原有 DMD loss 计算之后，加上压缩重建的辅助 loss

total_loss = dmd_loss + self.lambda_recon * reconstruction_loss
# lambda_recon 建议初始值 0.1，可以 warmup 后降到 0.01
```

---

## 8. 实现注意事项

### 8.1 LR 分支的 VAE 调用
LR 分支需要调用 VAE 的 decode 和 encode。Self-Forcing 的 VAE 通常是外部传入的冻结模块。在实例化 `HeterogeneousCompressor` 时，将主模型的 `self.vae` 直接传入，**不要** 在 `compress.py` 内部 `requires_grad_(False)`——交由主训练循环统一管理梯度。

### 8.2 incremental RoPE 调整
PackForcing 的 incremental RoPE 在每次驱逐时只对 sink 的 temporal 维度做一次旋转补偿。新版的驱逐发生在 `_evict_lowest_density_block` 中（按密度驱逐，非 FIFO），因此 RoPE 调整逻辑需要感知被驱逐块的**时间戳**，而非位置顺序。具体改法：

```python
# 在每条 mid_meta 中额外记录 temporal_position
meta["temporal_position"] = chunk_idx   # 生成时的绝对帧索引

# 驱逐时，根据 evicted["temporal_position"] 计算 RoPE 补偿量
# 而非按 buffer 中的相对顺序计算
```

### 8.3 token 预算的校验
建议在每次 `push_mid_block` 后加一个 assert：

```python
assert self.mid_token_count <= self.Nmid_tokens, \
    f"mid buffer overflow: {self.mid_token_count} > {self.Nmid_tokens}"
```

### 8.4 三套 HR 头的 stride 校验
Conv3d 的 stride 会改变输出尺寸，建议在 `HeterogeneousCompressor.__init__` 中用一个 dummy input 做前向验证：

```python
dummy = torch.zeros(1, in_ch, 2, 24, 24)
for name, head in [("8x", self.hr_high), ("32x", self.hr_mid), ("128x", self.hr_low)]:
    out = head(dummy)
    print(f"HR {name} output tokens: {out.shape[1]}")
    # 预期：8x→~288，32x→~18（即~182），128x→~4
```

### 8.5 单元测试建议
在接入主训练循环之前，建议对以下三个模块分别写单元测试：
1. `DensityEstimator.estimate` —— 验证三档路由阈值、在线归一化稳定性
2. `HeterogeneousCompressor.forward` —— 验证三档 token 数符合预期
3. `KVCache.push_mid_block` + `_evict_lowest_density_block` —— 验证 token 预算守恒、kv_slice 偏移更新正确

---

## 9. 文件改动清单（快速索引）

| 文件 | 类型 | 改动摘要 |
|------|------|----------|
| `models/density_estimator.py` | **新建** | 信息密度估计器，运动分 + 复杂度分，输出三档标签 |
| `models/compress.py` | **新建** | 三套 HR 压缩头 + LR 分支 + 统一入口 `HeterogeneousCompressor` |
| `models/kv_cache.py` | **改动** | 新增 mid buffer token 预算管理、按密度驱逐逻辑 |
| `models/transformer.py` | **改动** | attention 中注入三分区 KV，可选 Dynamic Top-K |
| `inference.py` / `generate.py` | **改动** | 推理循环接入密度估计 + 压缩路由 |
| `training/trainer.py` | **改动** | 新增压缩头预训练阶段，端到端训练加辅助 loss |
| `configs/heterogeneous_cache.yaml` | **新建** | 超参配置 |
