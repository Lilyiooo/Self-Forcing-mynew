# Compressor Training Plan

Current PackForcing-style experiments use per-layer compressed KV. Main DMD
training still keeps:

```yaml
compressor_training:
  lambda_recon: 0.0
  end_to_end: false
```

So the compressor is trained only by explicit pretrain stages before the main
training loop. This is intentional until the end-to-end path matches real cache
semantics.

## Implemented

`Trainer.pretrain_compressor()` provides a weak reconstruction warmup for the
HR heads/decoders and a non-degenerate regularizer for all per-layer
`kv_k_proj` / `kv_v_proj`.

`Trainer.pretrain_compressor_kv_distill()` now provides representation
pretraining for:

- compressor encoder/fusion path,
- HR/LR projection path,
- every per-layer `kv_k_proj` / `kv_v_proj`.

It uses two aligned objectives:

1. **Unroped KV distillation**: capture raw normalized teacher K before RoPE and
   V from each self-attention layer, pool them to the compressed grid, and match
   unroped student compressed K/V with MSE plus cosine loss.
2. **Optional attention-output distillation**: capture roped teacher Q/K/V,
   apply the same temporal-only RoPE used by inference to student compressed K,
   and match sampled `Attn(Q, K, V)` outputs. Enable with:

```yaml
compressor_training:
  kv_distill_attn_output_weight: 0.1
  kv_distill_attn_query_tokens: 128
  kv_distill_attn_max_layers: 8
```

The split is important: KV MSE does not force temporal-only student K to fit a
full spatial+temporal RoPE teacher target, while attention-output distillation
still trains the representation in the roped attention space used at inference.

The distillation latent source is configurable:

```yaml
compressor_training:
  kv_distill_latent_source: random    # default
```

The first generated-latent source is single-block denoising:

```yaml
compressor_training:
  kv_distill_latent_source: denoised
  kv_distill_denoise_timestep: 750
```

This runs the frozen generator on a noisy latent block, uses its predicted
`x0` as the distillation block, and then applies the same raw-KV and
attention-output objectives. It is closer to generated clean latent
distribution than pure Gaussian random blocks, while avoiding full long-video
cache rollout inside the pretrain loop.

This is **not** AR-cache rollout. `kv_distill_latent_source: rollout` is not a
valid source name; use `denoised` for the single-block diagnostic above.

The first AR-cache-aware source is:

```yaml
compressor_training:
  kv_distill_latent_source: ar_rollout
  kv_distill_rollout_blocks: 4
  kv_distill_target_block_index: 2
```

It runs a short frozen-generator rollout with a real self-attention KV cache.
By default it skips the `Nsink=8` sink frames and uses block index 2
(`start_frame=8`) as the compressed-past target. The attention-output loss then
uses the following block's query to read the target block:

```text
Attn(Q_future, K/V_compressed_past) ~= Attn(Q_future, K/V_full_past)
```

This is still a minimal implementation: it uses a short single-step denoise per
block rather than the full inference denoising schedule, but it trains the key
PackForcing behavior that compressed mid blocks are read by future queries under
real cache state.

Prompt conditioning can be enabled for this cheap diagnostic:

```yaml
compressor_training:
  kv_distill_use_real_prompts: true
  kv_distill_prompt_path: prompts/vidprom_filtered_extended.txt
```

This replaces the empty prompt used by early distillation runs with prompts
sampled from the training prompt file. It tests whether the logic drop from
single-block denoised distillation is partly caused by missing semantic prompt
conditioning. It still does not provide AR cache history.

## Remaining Work

- Move AR rollout distillation from short single-step rollout toward the full
  inference denoising schedule and longer cache histories.
- Re-evaluate `cachepath`, `mideviction`, and DCS with the stronger compressor.
- Implement query-affinity DCS after compressed KV quality is stable.
- Revisit end-to-end compressor fine-tuning only with a real-cache path:
  `[sink | selected_mid | recent]`, bounded archive, RoPE adjustment, and the
  same active-mid selection used at inference.
