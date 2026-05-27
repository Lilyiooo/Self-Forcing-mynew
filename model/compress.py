# model/compress.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

DensityLevel = Literal["high", "mid", "low"]

# ---------------------------------------------------------------
# HR compression heads (three variants, only these have learnable params)
# ---------------------------------------------------------------

class HRHead8x(nn.Module):
    """8x compression head: spatial 2x, temporal 2x -> net 8x"""
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
    """32x compression head (PackForcing original): spatial 4x, temporal 2x"""
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
    """128x compression head: spatial 8x, temporal 2x"""
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
# HR decoders (symmetric mirrors of HR heads, only used in pretrain)
# ---------------------------------------------------------------

class HRDecoder8x(nn.Module):
    """Decoder mirror of HRHead8x. Reconstructs latent from compressed tokens.
    Only used during pretrain, discarded for inference."""
    def __init__(self, d_model: int = 2048, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(d_model, 128, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(32, out_ch, kernel_size=3, stride=(2, 1, 1), padding=1),
        )

    def forward(self, tokens: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        T, H, W = target_shape
        t = (T - 1) // 2 + 1
        h = (H - 1) // 2 + 1
        w = (W - 1) // 2 + 1
        B = tokens.shape[0]
        x = tokens.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        if x.shape[2:] != (T, H, W):
            x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)
        return x


class HRDecoder32x(nn.Module):
    """Decoder mirror of HRHead32x. Reconstructs latent from compressed tokens.
    Only used during pretrain, discarded for inference."""
    def __init__(self, d_model: int = 2048, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(d_model, 2048, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.ConvTranspose3d(2048, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(512, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(32, out_ch, kernel_size=3, stride=(2, 1, 1), padding=1),
        )

    def forward(self, tokens: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        T, H, W = target_shape
        t = (T - 1) // 2 + 1
        h, w = H, W
        for _ in range(3):
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        B = tokens.shape[0]
        x = tokens.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        if x.shape[2:] != (T, H, W):
            x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)
        return x


class HRDecoder128x(nn.Module):
    """Decoder mirror of HRHead128x. Reconstructs latent from compressed tokens.
    Only used during pretrain, discarded for inference."""
    def __init__(self, d_model: int = 2048, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(d_model, 4096, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.ConvTranspose3d(4096, 2048, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(2048, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(512, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(32, out_ch, kernel_size=3, stride=(2, 1, 1), padding=1),
        )

    def forward(self, tokens: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        T, H, W = target_shape
        t = (T - 1) // 2 + 1
        h, w = H, W
        for _ in range(4):
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        B = tokens.shape[0]
        x = tokens.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        if x.shape[2:] != (T, H, W):
            x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)
        return x


# ---------------------------------------------------------------
# LR branch (shared VAE across tiers, frozen; only avgpool stride differs)
# ---------------------------------------------------------------

class LRBranch(nn.Module):
    """
    LR branch: decode -> avgpool -> re-encode -> project to tokens
    VAE is reused from the main model (passed by reference), frozen.
    Uses its own linear projection instead of reusing the FSDP-wrapped patch_embedding.
    """
    def __init__(
        self,
        vae,                    # reuse main model VAE, passed from outside
        pool_stride_hw: int = 4,    # spatial pooling stride, 2/4/8 for different tiers
        in_ch: int = 16,            # latent channel count
        d_model: int = 2048,
    ):
        super().__init__()
        self.vae = vae
        self.pool_stride_hw = pool_stride_hw
        self.d_model = d_model
        # Own projection layer (avoids FSDP sharding issues with shared patch_embedding)
        self.proj = nn.Linear(in_ch, d_model)

    def forward(
        self, x_latent: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Args:
          x_latent: (B, C, T, H, W) latent block in C-first format (from compressor)
        Returns:
          lr_tokens: (B, N_lr, D)
          complexity_score: float in [0, 1], normalized reconstruction error
        """
        # x_latent is (B, C, T, H, W) from HeterogeneousCompressor
        # WanVAEWrapper.decode_to_pixel expects (B, T, C, H, W)
        # Determine VAE's dtype from its parameters
        vae_dtype = next(self.vae.parameters()).dtype
        x_latent_btc = x_latent.to(dtype=vae_dtype).permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        with torch.no_grad():
            # decode latent -> pixel space
            x_pixel = self.vae.decode_to_pixel(x_latent_btc)  # (B, T, 3, H', W')

        # x_pixel is (B, T, C, H, W), need (B, C, T, H, W) for avg_pool3d
        x_pixel_5d = x_pixel.to(dtype=vae_dtype).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Spatial downsampling
        B, C, T, H, W = x_pixel_5d.shape
        x_pooled = F.avg_pool3d(
            x_pixel_5d,
            kernel_size=(1, self.pool_stride_hw, self.pool_stride_hw),
            stride=(1, self.pool_stride_hw, self.pool_stride_hw),
        )                                                 # (B, C=3, T, H', W')

        # Re-encode using VAE: encode_to_latent expects (B, C=3, T, H, W)
        # x_pooled is already in (B, C, T, H, W) format, no permute needed
        with torch.no_grad():
            x_re_latent = self.vae.encode_to_latent(x_pooled)  # (B, T, C=16, H'', W'')

        # complexity_score: reconstruction error (pooled original latent vs re-encoded)
        # Pool the original latent (C-first) for comparison
        x_latent_pooled = F.avg_pool3d(
            x_latent.float(),
            kernel_size=(1, self.pool_stride_hw, self.pool_stride_hw),
            stride=(1, self.pool_stride_hw, self.pool_stride_hw),
        )
        x_re_chw = x_re_latent.permute(0, 2, 1, 3, 4)  # (B, C, T, H'', W'')

        # Align spatial dimensions for error computation
        if x_re_chw.shape != x_latent_pooled.shape:
            x_re_chw = F.interpolate(
                x_re_chw, size=x_latent_pooled.shape[2:], mode="trilinear", align_corners=False
            )
        recon_err = F.mse_loss(x_re_chw.float(), x_latent_pooled.float())
        complexity_score = float(torch.sigmoid(recon_err - 0.1))  # rough normalization to [0,1]

        # project re-encoded latent to tokens using own linear projection
        # x_re_latent: (B, T, C, H, W) -> flatten spatial dims and project
        B, T, C, H_, W_ = x_re_latent.shape
        # (B, T, C, H_, W_) -> (B, T*H_*W_, C) -> project -> (B, T*H_*W_, D)
        proj_dtype = self.proj.weight.dtype
        lr_tokens = x_re_latent.permute(0, 1, 3, 4, 2).reshape(B, T * H_ * W_, C).to(dtype=proj_dtype)
        lr_tokens = self.proj(lr_tokens)  # (B, N, D)

        return lr_tokens, complexity_score


# ---------------------------------------------------------------
# Main entry: HeterogeneousCompressor
# ---------------------------------------------------------------

class HeterogeneousCompressor(nn.Module):
    """
    Unified compression entry point. Routes to corresponding HR head + LR branch
    based on density_level, concatenates HR tokens and LR tokens, and projects.
    """

    def __init__(
        self,
        vae,
        d_model: int = 2048,
        in_ch: int = 16,
    ):
        super().__init__()
        # Three HR heads (trainable)
        self.hr_high = HRHead8x(in_ch, d_model)
        self.hr_mid  = HRHead32x(in_ch, d_model)
        self.hr_low  = HRHead128x(in_ch, d_model)

        # Three LR branches (VAE shared and frozen, own projection layers)
        self.lr_high = LRBranch(vae, pool_stride_hw=2, in_ch=in_ch, d_model=d_model)
        self.lr_mid  = LRBranch(vae, pool_stride_hw=4, in_ch=in_ch, d_model=d_model)
        self.lr_low  = LRBranch(vae, pool_stride_hw=8, in_ch=in_ch, d_model=d_model)

        # Decoders for pretrain phase only (discarded after pretrain, not used in inference)
        self.decoder_high = HRDecoder8x(d_model=d_model, out_ch=in_ch)
        self.decoder_mid  = HRDecoder32x(d_model=d_model, out_ch=in_ch)
        self.decoder_low  = HRDecoder128x(d_model=d_model, out_ch=in_ch)

        # HR + LR concat projection to d_model
        self.proj = nn.Linear(d_model * 2, d_model)

        # KV projection heads: project compressed tokens to K and V for attention
        # Shared across all transformer layers (each layer's attention produces
        # different queries, so per-layer K/V specialization is not strictly needed)
        self.kv_k_proj = nn.Linear(d_model, d_model)
        self.kv_v_proj = nn.Linear(d_model, d_model)

        # Validate output token counts with a dummy input
        self._validated = False

    def compressed_grid_shape(self, density_level: DensityLevel, latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the HR-branch compressed token grid for (T, H, W)."""
        T, H, W = latent_shape
        t = (T - 1) // 2 + 1
        spatial_layers = {"high": 1, "mid": 3, "low": 4}[density_level]
        h, w = H, W
        for _ in range(spatial_layers):
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return t, h, w

    def _validate_heads(self, in_ch: int, device: torch.device):
        """Validate HR head output shapes with a dummy input."""
        # Wan2.1-T2V-1.3B latent: C=16, T varies (1 or num_frame_per_block), H=60, W=104
        # After patch_embedding with stride (1,2,2): spatial becomes 30x52
        # For compression, input is the raw latent before patch embedding
        dummy = torch.zeros(1, in_ch, 2, 60, 104, device=device)
        for name, head in [("8x", self.hr_high), ("32x", self.hr_mid), ("128x", self.hr_low)]:
            try:
                out = head(dummy)
                print(f"  HR {name} output tokens: {out.shape[1]} (shape: {out.shape})")
            except Exception as e:
                print(f"  HR {name} validation failed: {e}")
        self._validated = True

    def forward(
        self,
        x: torch.Tensor,              # (B, C, T, H, W) latent block (raw, before patch embedding)
        density_level: DensityLevel,
    ) -> tuple[torch.Tensor, float]:
        """
        Returns:
          compressed_tokens: (B, N_compressed, D)
          complexity_score:  float (for density estimator)
        """
        # Validate on first call
        if not self._validated:
            self._validate_heads(x.shape[1], x.device)

        if density_level == "high":
            hr_tokens = self.hr_high(x)
            lr_tokens, complexity = self.lr_high(x)
        elif density_level == "mid":
            hr_tokens = self.hr_mid(x)
            lr_tokens, complexity = self.lr_mid(x)
        else:  # "low"
            hr_tokens = self.hr_low(x)
            lr_tokens, complexity = self.lr_low(x)

        # Align sequence lengths (use HR length as reference, pad/truncate LR)
        N_hr = hr_tokens.shape[1]
        N_lr = lr_tokens.shape[1]
        if N_lr != N_hr:
            if N_lr > N_hr:
                lr_tokens = lr_tokens[:, :N_hr, :]
            else:
                lr_tokens = F.pad(lr_tokens, (0, 0, 0, N_hr - N_lr))

        combined = torch.cat([hr_tokens, lr_tokens], dim=-1)  # (B, N_hr, 2D)
        out = self.proj(combined)                               # (B, N_hr, D)
        return out, complexity

    def project_to_kv(self, compressed_tokens, num_layers, num_heads):
        """
        Project compressed tokens to per-layer KV pairs for attention.
        Uses learned k_proj and v_proj (shared across layers).
        compressed_tokens: (B, N, D)
        Returns: list of (2, B, N, num_heads, head_dim) per layer
        """
        B, N, D = compressed_tokens.shape
        head_dim = D // num_heads
        k = self.kv_k_proj(compressed_tokens).view(B, N, num_heads, head_dim)
        v = self.kv_v_proj(compressed_tokens).view(B, N, num_heads, head_dim)
        kv = torch.stack([k, v], dim=0)  # (2, B, N, num_heads, head_dim)
        return [kv] * num_layers
