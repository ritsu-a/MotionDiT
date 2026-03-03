"""
DiT (Diffusion Transformer)：以 audio 为 condition，预测去噪后的 motion (x0)。
输入为 patch 化的 (noisy_motion, audio)，输出为 motion 的 x0 预测（同形状）。
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


def patchify(x: torch.Tensor, patch_size: int):
    """(B, T, D) -> (B, n_patches, patch_size * D)"""
    B, T, D = x.shape
    assert T % patch_size == 0
    n = T // patch_size
    x = rearrange(x, "b (n p) d -> b n (p d)", n=n, p=patch_size)
    return x


def unpatchify(x: torch.Tensor, patch_size: int, d: int):
    """(B, n_patches, patch_size * D) -> (B, T, D)"""
    B, n, pd = x.shape
    x = rearrange(x, "b n (p d) -> b (n p) d", n=n, p=patch_size, d=d)
    return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor):
        x = x + self._attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attn(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]


class MotionDiT(nn.Module):
    """
    以 audio 为 condition 的 Motion DiT。
    - motion: (B, T, motion_dim), 训练时输入为加噪后的 motion
    - audio:  (B, T, audio_dim), condition，与 motion 时间对齐
    输出: (B, T, motion_dim)，预测的 x0 (干净 motion)。
    """

    def __init__(
        self,
        motion_dim: int = 60,
        audio_dim: int = 512,
        motion_frames: int = 300,
        patch_size: int = 5,
        dim: int = 256,
        depth: int = 12,
        heads: int = 8,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.patch_size = patch_size
        self.cond_drop_prob = cond_drop_prob
        assert motion_frames % patch_size == 0
        self.num_patches = motion_frames // patch_size
        patch_motion_dim = patch_size * motion_dim
        patch_audio_dim = patch_size * audio_dim
        self.patch_motion_dim = patch_motion_dim
        self.patch_audio_dim = patch_audio_dim
        input_dim = patch_motion_dim + patch_audio_dim  # concat condition

        self.proj_in = nn.Linear(input_dim, dim)
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, patch_motion_dim)

    def forward(self, motion: torch.Tensor, audio: torch.Tensor, t: torch.LongTensor):
        """
        motion: (B, T_m, motion_dim), 加噪后的 motion
        audio:  (B, T_a, audio_dim)，若 T_a != T_m 会在时间维上插值到 T_m
        t:      (B,) 时间步
        """
        B = motion.shape[0]
        T_m = motion.shape[1]
        if audio.shape[1] != T_m:
            # audio 时间维与 motion 对齐（如 25→40），便于 patch 后 concat
            audio = torch.nn.functional.interpolate(
                audio.permute(0, 2, 1), size=T_m, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        # Patchify
        x_m = patchify(motion, self.patch_size)   # (B, n, patch_motion_dim)
        x_a = patchify(audio, self.patch_size)   # (B, n, patch_audio_dim)
        # Classifier-free: 随机 drop condition
        if self.training and self.cond_drop_prob > 0:
            drop = torch.rand(B, device=motion.device) < self.cond_drop_prob
            x_a = torch.where(drop.view(B, 1, 1), torch.zeros_like(x_a), x_a)
        x = torch.cat([x_m, x_a], dim=-1)  # (B, n, input_dim)
        x = self.proj_in(x)
        t_emb = self.t_embed(t)  # (B, dim)
        x = x + t_emb[:, None, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.proj_out(x)  # (B, n, patch_motion_dim)
        motion_pred = unpatchify(x, self.patch_size, self.motion_dim)
        return motion_pred
