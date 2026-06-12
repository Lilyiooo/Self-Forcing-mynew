# Compressor Training Plan

Current cache experiments use per-layer compressed KV, but the main baseline
configs keep:

```yaml
compressor_training:
  pretrain_epochs: 0
  lambda_recon: 0.0
  end_to_end: false
```

That means the compressor participates in forward/inference but is not
attention-trained in the main runs.

## Current Pretrain Coverage

`Trainer.pretrain_compressor()` trains:

- HR heads and HR decoders with latent reconstruction.
- All per-layer `kv_k_proj` / `kv_v_proj` with a non-degenerate variance
  regularizer.

It still does not train:

- LR branch / VAE fusion through the main compressor forward.
- `proj` fusion from HR+LR tokens.
- Per-layer compressed KV to match the transformer's real per-layer K/V space.

So this is only a weak lower-bound pretrain. It can make parts of the
compressor less random, but it is not PackForcing-style attention alignment.

## Next Representation Step

Add a KV distillation pretrain:

1. Run a clean latent block through the base generator attention path and
   capture each layer's full-resolution K/V teacher.
2. Run the same latent block through `HeterogeneousCompressor.forward()`.
3. Run `project_to_kv()` to get per-layer compressed K/V.
4. Pool or sample each teacher K/V to the compressed token grid.
5. Optimize MSE/cosine loss over every layer's compressed K/V.

This directly trains:

- compressor encoder,
- HR/LR fusion projection,
- every per-layer `kv_k_proj` / `kv_v_proj`.

Only after this representation line is stable should old `end_to_end=true` be
revisited, and only if the differentiable path matches real cache semantics.
