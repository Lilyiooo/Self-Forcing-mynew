import torch


def temporal_rope_dim(head_dim: int) -> int:
    """Return the complex RoPE dimensions assigned to time."""
    half = head_dim // 2
    return half - 2 * (half // 3)


def _as_complex_heads(x: torch.Tensor) -> torch.Tensor:
    B, L, H, D = x.shape
    return torch.view_as_complex(x.float().reshape(B, L, H, D // 2, 2))


def _from_complex_heads(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.view_as_real(x).flatten(-2).to(dtype=dtype)


def apply_temporal_rope_to_unrotated(
    k: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int,
    grid_shape: tuple[int, int, int],
    temporal_stride: int = 2,
) -> torch.Tensor:
    """
    Apply temporal-only RoPE to raw compressed keys.

    Args:
        k: raw key tensor, shape (B, N, num_heads, head_dim).
        freqs: complex RoPE frequencies, shape (max_pos, head_dim / 2).
        start_frame: absolute latent-frame index of the original block.
        grid_shape: compressed token grid (T, H, W), matching flattened order.
        temporal_stride: temporal stride represented by each compressed step.
    """
    t_comp, h_comp, w_comp = grid_shape
    expected_tokens = t_comp * h_comp * w_comp
    if k.shape[1] != expected_tokens:
        raise ValueError(
            f"compressed key token count {k.shape[1]} does not match grid {grid_shape}"
        )

    if expected_tokens == 0:
        return k

    dtype = k.dtype
    x = _as_complex_heads(k)
    t_dim = temporal_rope_dim(k.shape[-1])

    positions = start_frame + torch.arange(
        t_comp, device=k.device, dtype=torch.long
    ) * temporal_stride
    positions = positions.clamp(max=freqs.shape[0] - 1)
    phase = freqs.to(device=k.device)[positions, :t_dim]
    phase = phase.view(t_comp, 1, 1, t_dim).expand(t_comp, h_comp, w_comp, t_dim)
    phase = phase.reshape(expected_tokens, t_dim).view(1, expected_tokens, 1, t_dim)

    x_t = x[..., :t_dim] * phase
    x = torch.cat([x_t, x[..., t_dim:]], dim=-1)
    return _from_complex_heads(x, dtype)


def apply_temporal_rope_shift(
    k: torch.Tensor,
    freqs: torch.Tensor,
    delta: int,
) -> torch.Tensor:
    """
    Rotate already-RoPE'd keys by a temporal delta.

    This is intended for full-resolution sink keys that were cached with
    absolute RoPE already applied. It must not be used as initial RoPE for raw K.
    """
    if delta == 0:
        return k

    dtype = k.dtype
    x = _as_complex_heads(k)
    t_dim = temporal_rope_dim(k.shape[-1])

    delta = min(int(delta), freqs.shape[0] - 1)
    phase = freqs.to(device=k.device)[delta, :t_dim].view(1, 1, 1, t_dim)
    x_t = x[..., :t_dim] * phase
    x = torch.cat([x_t, x[..., t_dim:]], dim=-1)
    return _from_complex_heads(x, dtype)
