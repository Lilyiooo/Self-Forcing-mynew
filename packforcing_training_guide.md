# PackForcing 训练层改动完整指导书

> **写给实现 AI 的说明**：你目前只改了推理（inference）部分——三分区 KV cache、双分支压缩、Dynamic Top-K、RoPE Adjustment 这些在推理时都能工作。但 PackForcing 的核心优势（短训练→长推理的泛化能力）来自训练时让模型**见过压缩 context 的样子**。如果训练时 context 仍然是 Self-Forcing 的 rolling KV cache（全精度），推理时突然换成压缩的 mid token，模型没见过这种 KV 分布，attention 的输出会出现严重 OOD（out-of-distribution）问题，效果会很差。本文档告诉你训练循环需要改哪些地方，以及为什么。

---

## 0. 核心原则：训练和推理的 context 必须一致

PackForcing 成功的根本原因，论文 Section 4.4 明确说了两点：

> "First, the framework enforces **context size invariance**. By systematically compressing and managing the KV cache, the attention context remains strictly bounded (~27,872 tokens) during **both training and inference**."
>
> "Second, the architecture ensures **representational compatibility**. Jointly training the dual-branch compression layer aligns the compressed tokens with the full-resolution tokens within the same latent subspace."

第一点：训练时的 context 大小必须和推理时完全一致（都是定长的三分区 context，不是 Self-Forcing 的 rolling 全精度 cache）。

第二点：压缩模块必须和 Transformer 主干**联合训练**，才能让压缩后的 token 和原始 token 处于同一 latent 子空间，attention 才能有效利用压缩 KV。

只改推理不改训练，违反了这两点。

---

## 1. 训练目标：DMD Loss

PackForcing 沿用 Self-Forcing 的 DMD（Distribution Matching Distillation）训练范式，损失函数不变：

```
L_DMD = E_{t, x_hat_t, x_hat} [
    (1/2) * || x_hat - sg[x_hat - (f_psi(x_hat_t, t) - f_phi(x_hat_t, t))] ||^2
]
```

- `f_phi`：teacher 模型（Wan2.1-T2V-1.3B，双向，冻结）
- `f_psi`：fake score estimator（critic，可学习）
- `sg[·]`：stop_gradient
- 梯度**只流经当前生成 block** 的参数，历史帧的 KV 全部 detach

**这一部分不需要改。改的是训练循环中 context 的构建方式。**

---

## 2. 训练超参数（来自论文 Section 4.1）

| 参数 | 值 |
|------|-----|
| 基础模型 | Wan2.1-T2V-1.3B |
| 训练窗口长度 | 20 latent frames（5 秒） |
| Batch size | 8 |
| Generator lr | 2.0 × 10⁻⁶ |
| Critic lr | 1.0 × 10⁻⁶ |
| Generator:Critic 更新比 | 1:5 |
| Optimizer | AdamW，β₁=0，β₂=0.999 |
| 训练 iterations | 3,000 |
| 去噪步数 S | 4 |
| Block size Bf | 4 latent frames |
| CFG scale | 3.0 |
| Timestep shift | 5.0 |

**cache 分区参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| Nsink | 8 latent frames（2 blocks） | sink 大小 |
| Nrecent | 4 latent frames（1 block） | recent window 大小 |
| Ntop | 16 blocks | top-k 选择数量 |
| Nc | 182 tokens/block | 每个压缩 block 的 token 数 |

---

## 3. 训练循环的完整逻辑

### 3.1 整体结构（伪代码）

```python
# 外层：训练主循环
for iteration in range(3000):
    prompt = sample_prompt(vidprom_dataset)

    # ============================================================
    # Step 1: 初始化三分区 KV cache（训练时和推理时结构完全一致）
    # ============================================================
    kv_cache = ThreePartitionKVCache(
        Nsink=8, Nrecent=4, Ntop=16, Nc=182
    )

    # ============================================================
    # Step 2: 自回归 rollout，生成 5s（20 latent frames = 5 blocks）
    #         使用三分区 cache，与推理完全一致
    # ============================================================
    for block_idx in range(5):  # 5 blocks × 4 frames = 20 latent frames
        # 2a. 初始化噪声
        x_t = torch.randn(Bf, C, H, W)  # [4, 16, 60, 104]

        # 2b. 多步去噪（S=4 步）
        for step in [t4, t3, t2, t1]:   # [1000, 750, 500, 250]
            ctx = kv_cache.get_context()  # 三分区拼接后的 context（detach）
            x0_hat = model(x_t, step, ctx, prompt)
            if step > t1:
                x_t = add_noise(x0_hat, step - 1)

        # 2c. 获取当前 block 的干净 latent（detach，不保留梯度）
        clean_latent = x0_hat.detach()

        # 2d. 更新三分区 KV cache（核心改动！详见第 4 节）
        kv_cache.update(clean_latent, block_idx)

    # ============================================================
    # Step 3: 在最后一个 block 上计算 DMD loss 并反传
    # ============================================================
    # 取最后一个 block 做损失计算（梯度只流经这个 block）
    t = sample_timestep()
    x_t_last = add_noise(clean_latent, t)

    ctx_for_loss = kv_cache.get_context()  # 全部 detach
    student_pred = model(x_t_last, t, ctx_for_loss, prompt)  # 有梯度
    teacher_pred = teacher_model(x_t_last, t, ctx_for_loss, prompt).detach()

    loss = dmd_loss(student_pred, teacher_pred)
    loss.backward()

    optimizer_G.step()
    optimizer_critic.step()
```

---

## 4. 三分区 KV Cache 在训练时的更新逻辑

这是和原版 Self-Forcing **最关键的差异**。原版只有 rolling cache，PackForcing 有三分区。

### 4.1 `ThreePartitionKVCache.update()` 的完整逻辑

```python
class ThreePartitionKVCache:
    def __init__(self, Nsink, Nrecent, Ntop, Nc, dual_branch_compressor):
        self.Nsink = Nsink          # 8 latent frames
        self.Nrecent = Nrecent      # 4 latent frames
        self.Ntop = Ntop            # 16 blocks
        self.Nc = Nc                # 182 tokens/block

        self.sink_kv = None         # 全精度，永不驱逐
        self.mid_archive = []       # 压缩 KV 列表，soft eviction
        self.recent_kv = []         # 全精度，滚动

        self.compressor = dual_branch_compressor  # 双分支压缩模块
        self.block_count = 0

    def update(self, clean_latent, block_idx):
        """
        每生成一个 block 后调用。
        clean_latent: [Bf, C, H, W]，已 detach，无梯度
        """
        # 1. 用 clean_latent 计算当前 block 的全精度 KV（detach）
        with torch.no_grad():
            full_kv = compute_kv_from_latent(clean_latent)
            # shape: [num_layers, 2, Bf*h*w, num_heads, head_dim]

        # 2. 并发计算压缩 KV（Dual-Resolution Shifting）
        #    与 full_kv 的计算同步进行，不引入额外延迟
        with torch.no_grad():
            compressed_kv = self.compressor(clean_latent)
            # shape: [num_layers, 2, Nc, num_heads, head_dim]
            # Nc = 182

        # 3. 决定当前 block 进哪个分区
        if self.block_count < (self.Nsink // self.Bf):
            # 前 2 个 block → sink（全精度，永不驱逐）
            if self.sink_kv is None:
                self.sink_kv = full_kv
            else:
                self.sink_kv = concat_along_token_dim(self.sink_kv, full_kv)

        else:
            # 后续 block → 先进 recent（全精度）
            self.recent_kv.append(full_kv)

            # recent 满了（超过 Nrecent/Bf 个 block）→ 最旧的 recent 滑入 mid
            if len(self.recent_kv) > (self.Nrecent // self.Bf):
                aging_block_idx = 0  # 最旧的 recent block
                # 用预先计算好的 compressed_kv 放入 mid archive
                # （注意：这里用的是"上一个 block"的压缩 KV，
                #  因为 Dual-Resolution Shifting 是并发计算的）
                self.mid_archive.append(self.prev_compressed_kv)
                self.recent_kv.pop(0)

        # 4. 保存当前 block 的压缩 KV 供下一步使用
        self.prev_compressed_kv = compressed_kv
        self.block_count += 1

    def get_context(self):
        """
        返回用于 attention 的完整三分区 context（全部 detach）。
        顺序：sink || mid_selected || recent
        """
        parts = []

        # Sink（全精度）
        if self.sink_kv is not None:
            parts.append(self.sink_kv)

        # Mid（压缩，top-k 筛选）
        if self.mid_archive:
            selected_mid = self._dynamic_top_k_select()
            parts.append(selected_mid)

        # Recent（全精度）
        if self.recent_kv:
            recent_cat = torch.cat(self.recent_kv, dim=2)
            parts.append(recent_cat)

        if not parts:
            return None
        return torch.cat(parts, dim=2)  # 沿 token 维拼接

    def _dynamic_top_k_select(self):
        """
        从 mid_archive 中选出最重要的 Ntop 个 block。
        训练时：用上一个 block 的 query 做 affinity scoring。
        只在每个 block 的第一个去噪步计算一次，后续步骤复用索引。
        """
        if len(self.mid_archive) <= self.Ntop:
            # archive 未超出预算，全部保留
            return torch.cat(self.mid_archive, dim=2)

        # 计算 affinity score（subsampled query × all mid keys）
        # 这里用 cached_query（上一步的 query，已 detach）
        scores = compute_affinity_scores(
            self.cached_query,         # [num_heads//2, q_sub, head_dim]
            self.mid_archive,          # list of [num_layers, 2, Nc, heads, dim]
        )
        top_k_indices = scores.topk(self.Ntop).indices
        selected = [self.mid_archive[i] for i in sorted(top_k_indices)]

        # RoPE 调整（见第 5 节）
        selected = self._rope_adjustment(selected)

        return torch.cat(selected, dim=2)
```

---

## 5. 双分支压缩模块在训练时的梯度处理

### 5.1 两阶段训练方案

**第一阶段（预热，约 500 steps）：单独训练压缩模块**

```python
# 第一阶段：只训练 compressor，主干冻结
for param in transformer_model.parameters():
    param.requires_grad = False
for param in compressor.parameters():
    param.requires_grad = True

# 损失：重建 loss
compressed_kv = compressor(clean_latent)
reconstructed = decompressor(compressed_kv)  # 可以是简单的线性解压
recon_loss = F.mse_loss(reconstructed, original_kv)
recon_loss.backward()
optimizer_compressor.step()
```

这一阶段确保压缩模块的输出和原始 KV 在同一 latent 子空间，后续 Transformer 才能有效 attend。

**第二阶段（主训练，3000 steps）：联合训练**

```python
# 第二阶段：compressor 和 transformer 联合训练
# compressor 的输出 detach 后送入 attention（切断梯度流）
# DMD loss 的梯度只流经当前 block 的 transformer 参数
# compressor 的梯度来自 DMD loss 对压缩质量的隐式反馈

for param in transformer_model.parameters():
    param.requires_grad = True
for param in compressor.parameters():
    param.requires_grad = True  # compressor 继续微调

# 关键：压缩 KV 在 get_context() 里已经是 detach 的
# 所以 DMD loss 的梯度不会穿越 compressor 的计算图
# compressor 用独立的重建 loss（可用更小的 lr，比如 1e-7）继续更新
```

### 5.2 梯度流的完整示意

```
clean_latent (detach)
    │
    ├──→ compute_kv_from_latent() → full_kv.detach() → sink/recent
    │
    └──→ compressor(clean_latent) → compressed_kv.detach() → mid archive
                │
                └──→ recon_loss（独立，compressor 的梯度来源）

get_context() 返回全部 detach 的 KV
    │
    ↓
model(x_t, t, ctx_detach, prompt)  ← 有梯度（当前 block）
    │
    ↓
dmd_loss → loss.backward()
    │
    └──→ 梯度只流经 transformer 的参数（当前 block 对应的层）
         不流向 compressor，不流向历史帧
```

---

## 6. Dual-Resolution Shifting 在训练时的实现

Dual-Resolution Shifting 的关键是**并发计算**——在生成当前 block 的同时，后台计算上一个 block 的压缩 KV，从而零延迟地把 recent 中老化的 block 滑入 mid。

```python
# 训练时的并发实现（简化版）

prev_compress_future = None  # 上一个 block 的压缩任务（延迟执行）

for block_idx in range(num_blocks):

    # 生成当前 block（全精度）
    clean_latent = generate_block(model, kv_cache, prompt)

    # 同步等待上一个 block 的压缩结果（如果有）
    if prev_compress_future is not None:
        prev_compressed_kv = prev_compress_future.result()
        # 把压缩好的 KV 滑入 mid archive
        kv_cache.slide_into_mid(prev_compressed_kv)

    # 启动当前 block 的压缩（异步，供下一个 block 使用）
    with torch.no_grad():
        # 在实际实现中可以用线程池或 CUDA stream 做异步
        prev_compress_future = compressor_pool.submit(
            compressor, clean_latent.detach()
        )

    # 更新 recent（全精度）
    kv_cache.update_recent(clean_latent.detach())
```

**简化实现（如果不做真正的异步）**：

训练阶段可以不做真正的并发，直接同步计算压缩 KV。延迟问题在推理时才关键，训练时吞吐量允许一定的串行开销。

```python
# 简化版：串行计算（训练时可用）
for block_idx in range(num_blocks):
    clean_latent = generate_block(model, kv_cache, prompt)

    # 计算全精度 KV（detach）
    full_kv = compute_full_kv(clean_latent)

    # 计算压缩 KV（detach）
    compressed_kv = compressor(clean_latent.detach()).detach()

    # 更新三分区（顺序：recent 满了就把最旧的 recent 以压缩态送入 mid）
    kv_cache.update(full_kv, compressed_kv, block_idx)
```

---

## 7. Incremental RoPE Adjustment 在训练时

训练时 mid archive 通常不会超出 Ntop=16 个 block（5s 训练窗口只有 5 个 block，不存在 mid 溢出的情况）。所以 RoPE Adjustment 在训练时**通常不会被触发**。

但为了训练-推理一致性，代码里必须有这个逻辑，并且在推理时超长视频生成时会被触发：

```python
def incremental_rope_adjustment(sink_kv, n_evicted_blocks):
    """
    当 mid archive 超出容量驱逐了 n_evicted_blocks 个 block 时，
    对 sink_kv 做时间维的 RoPE 旋转补偿。
    
    利用 RoPE 的乘法性质：e^{iθp} · e^{iθδ} = e^{iθ(p+δ)}
    只调整时间维度（θt），空间维度（θh, θw）不动。
    
    Args:
        sink_kv: [num_layers, 2, Nsink_tokens, num_heads, head_dim]
        n_evicted_blocks: 被驱逐的 block 数量
    
    Returns:
        adjusted_sink_kv: 位置调整后的 sink_kv
    """
    delta = n_evicted_blocks * Bf  # 被驱逐的 latent frames 数

    # 计算时间维度的旋转因子 e^{i * theta_t * delta}
    # theta_t 来自模型的 RoPE 配置（Wan2.1 用 44 维 temporal）
    rotation = compute_temporal_rope_rotation(delta, dim=44)

    # 只旋转时间维度（空间维度乘以单位旋转 = 不变）
    adjusted_keys = sink_kv[:, 0] * rotation  # [layers, Nsink_tokens, heads, dim]
    adjusted_sink_kv = torch.stack([adjusted_keys, sink_kv[:, 1]], dim=1)

    return adjusted_sink_kv
```

---

## 8. 需要修改的文件汇总

| 文件 | 修改内容 | 优先级 |
|------|---------|-------|
| `train.py` | 训练主循环：把 rolling cache 换成 `ThreePartitionKVCache`，在 rollout 的每个 block 后调用 `cache.update()` | **必须** |
| `kv_cache.py`（新建） | `ThreePartitionKVCache` 类的完整实现 | **必须** |
| `compressor.py`（新建） | 双分支压缩模块（HR: 4层Conv3D；LR: decode→pool→VAE encode→patch embed） | **必须** |
| `train.py` | 两阶段训练：第一阶段单独训练 compressor（重建 loss），第二阶段联合训练 | **必须** |
| `attention.py` | forward 里把固定 sink_kv 换成 `cache.get_context()[layer_idx]` | **必须**（已在推理里改了，训练时要确保同路径） |
| `inference.py` | 已改，确认 `get_context()` 和训练时调用的是同一个方法 | 检查对齐 |

---

## 9. 最常见的错误：训练时 context 和推理时不一致

下面是几种会导致训练-推理 mismatch 的错误写法，**避免这些**：

**错误 1：训练时用全精度 rolling cache，推理时用三分区压缩 cache**

```python
# ❌ 错误
# train.py
ctx = rolling_full_kv_cache.get()  # 全精度

# inference.py
ctx = three_partition_cache.get_context()  # 包含压缩 mid token
```

模型训练时从没见过压缩 KV，推理时 attention 输入的分布完全不同，效果会很差。

**错误 2：训练时 compressor 的输出没有 detach**

```python
# ❌ 错误
compressed_kv = compressor(clean_latent)  # 有梯度
ctx = concat(sink, compressed_kv, recent)  # 梯度会流入历史帧的 compressor
dmd_loss.backward()  # 反传会错误地更新 compressor 的参数（历史帧方向）
```

历史帧的压缩 KV 应该是 detach 的，DMD loss 只应该流经当前 block。

**错误 3：训练时不做 Dynamic Top-K，推理时做**

```python
# ❌ 错误
# 训练时
ctx = concat(sink, all_mid, recent)  # 用所有 mid token

# 推理时
ctx = concat(sink, top_k_selected_mid, recent)  # 只用 top-k
```

训练时 attention 见到的 mid context 大小和推理时不同，context size invariance 被打破。

**正确做法**：训练时和推理时都用 `ThreePartitionKVCache.get_context()`，同一套逻辑，同一个 context 大小。

---

## 10. 快速验证训练是否正确的检查项

实现完成后按顺序核对：

- [ ] `ThreePartitionKVCache.get_context()` 的输出 token 数量在训练和推理时完全一致（目标：约 27,872 tokens）
- [ ] 所有历史 KV（sink、mid、recent）在送入 attention 前均已 `detach()`
- [ ] `compressor` 的第一阶段（重建 loss）单独训练能收敛（重建误差下降）
- [ ] 第二阶段联合训练时，设置 `compressor` 的学习率 ≤ `1e-7`，避免过度更新
- [ ] 训练 loss 曲线：DMD loss 在 3000 steps 内稳定下降
- [ ] 用相同 prompt 跑 5s 短视频，和原版 Self-Forcing 效果相当（验证 ODE 初始化 + 训练正确）
- [ ] 逐步放开推理长度（10s → 30s → 60s → 120s），每阶段都确认 context 大小符合预期

---

*文档结束*
