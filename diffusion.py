"""
高斯扩散过程：对 motion 加噪与去噪。
"""
import numpy as np
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    t = np.linspace(0, 1, steps)
    alphas_cumprod = np.cos(((t + s) / (1 + s)) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.9999).astype(np.float32)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.timesteps = timesteps
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        sqrt_alphas = np.sqrt(alphas)
        self.register_buffer("betas", torch.from_numpy(betas))
        self.register_buffer("alphas", torch.from_numpy(alphas))
        self.register_buffer("alphas_cumprod", torch.from_numpy(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", torch.from_numpy(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", torch.from_numpy(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.from_numpy(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas", torch.from_numpy(np.sqrt(1.0 / alphas)))
        self.register_buffer("posterior_variance", torch.from_numpy(betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)))
        # 后验均值系数: mean = coef_x0 * x0 + coef_xt * x_t
        self.register_buffer("posterior_mean_coef_x0", torch.from_numpy(np.sqrt(alphas_cumprod_prev) * betas / (1 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef_xt", torch.from_numpy(sqrt_alphas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """前向扩散：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """后验 q(x_{t-1} | x_t, x_0) 的均值和方差."""
        view = lambda x: x[t].view(-1, *([1] * (x_start.dim() - 1)))
        posterior_mean = view(self.posterior_mean_coef_x0) * x_start + view(self.posterior_mean_coef_xt) * x_t
        posterior_variance = view(self.posterior_variance)
        return posterior_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """单步去噪采样."""
        pred = model(x_t, cond, t)
        x_start = pred.clamp(-1, 1)
        posterior_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        noise = torch.randn_like(x_t, device=x_t.device)
        # t=0 时直接返回 x_start，不加噪声
        out = posterior_mean + torch.sqrt(posterior_variance + 1e-8) * noise
        mask = (t == 0).view(-1, *([1] * (x_t.dim() - 1)))
        return torch.where(mask, x_start, out)

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple, cond: torch.Tensor):
        """从 x_T ~ N(0,I) 逐步去噪到 x_0."""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t_batch)
        return x
