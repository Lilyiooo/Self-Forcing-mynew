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

The first rollout-like source is:

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

## Remaining Work

- Move distillation data from single-block denoised latents toward generated
  long-rollout latents with realistic cache history.
- Re-evaluate `cachepath`, `mideviction`, and DCS with the stronger compressor.
- Implement query-affinity DCS after compressed KV quality is stable.
- Revisit end-to-end compressor fine-tuning only with a real-cache path:
  `[sink | selected_mid | recent]`, bounded archive, RoPE adjustment, and the
  same active-mid selection used at inference.
