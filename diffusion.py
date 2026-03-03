"""
高斯扩散过程：对 motion 加噪与去噪。
采样使用 DDIM（确定性，可减少步数）；训练仍用同一前向过程。
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

    def _get_sqrt_alpha_bar(self, t, device):
        """取 t 对应的 sqrt(alpha_bar_t)，形状 (1,1,1) 便于与 (B,T,D) 广播."""
        if isinstance(t, int):
            return (self.alphas_cumprod[t] ** 0.5).to(device).view(1, 1, 1)
        return (self.alphas_cumprod[t] ** 0.5).to(device).view(-1, 1, 1)

    def _get_sqrt_one_minus_alpha_bar(self, t, device):
        if isinstance(t, int):
            return (1 - self.alphas_cumprod[t]).clamp(min=1e-8).sqrt().to(device).view(1, 1, 1)
        return (1 - self.alphas_cumprod[t]).clamp(min=1e-8).sqrt().to(device).view(-1, 1, 1)

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        pred_x0: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM 单步：由 x_t 与预测的 x0 得到 x_{t_prev}（确定性当 eta=0）。
        pred_x0: 模型预测的 x0，已 clamp 到 [-1,1]。
        """
        device = x_t.device
        pred_x0 = pred_x0.clamp(-1, 1)
        sqrt_alpha_bar = self._get_sqrt_alpha_bar(t, device)
        sqrt_one_minus_alpha_bar = self._get_sqrt_one_minus_alpha_bar(t, device)
        eps = (x_t - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar

        if t_prev < 0:
            return pred_x0
        sqrt_alpha_bar_prev = self._get_sqrt_alpha_bar(t_prev, device)
        sqrt_one_minus_alpha_bar_prev = self._get_sqrt_one_minus_alpha_bar(t_prev, device)
        # 确定性: x_{t_prev} = sqrt(alpha_bar_{t_prev})*x0 + sqrt(1-alpha_bar_{t_prev})*eps
        x_prev = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * eps
        return x_prev

    def ddim_timestep_sequence(self, ddim_steps: int):
        """DDIM 使用的降序时间步序列 [T-1, ..., 0]（等间隔）。"""
        if ddim_steps >= self.timesteps:
            return list(range(self.timesteps - 1, -1, -1))
        steps = np.linspace(self.timesteps - 1, 0, ddim_steps).astype(int)
        return steps.tolist()

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        model: nn.Module,
        shape: tuple,
        cond: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM 采样：从 x_T ~ N(0,I) 沿子序列去噪到 x_0（步数可小于 T）。"""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        seq = self.ddim_timestep_sequence(ddim_steps)
        for i in range(len(seq) - 1):
            t = seq[i]
            t_prev = seq[i + 1]
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_x0 = model(x, cond, t_batch).clamp(-1, 1)
            x = self.ddim_step(x, t, t_prev, pred_x0, eta=eta)
        return x
