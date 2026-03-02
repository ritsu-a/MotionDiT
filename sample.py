"""
从训练好的 DiT 采样：给定 audio condition，生成 motion。
支持 classifier-free guidance (cfg_scale)。
"""
import os
import argparse
import yaml
import torch
import numpy as np

from dataset import BEATSegmentDataset
from dit import MotionDiT
from diffusion import GaussianDiffusion


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def sample(
    model: MotionDiT,
    diffusion: GaussianDiffusion,
    audio: torch.Tensor,
    device: torch.device,
    cfg_scale: float = 1.0,
):
    """
    audio: (B, T, audio_dim)
    返回 motion (B, T, motion_dim)
    """
    B, T, _ = audio.shape
    shape = (B, T, model.motion_dim)
    model.eval()
    x = torch.randn(shape, device=device)
    for t in reversed(range(diffusion.timesteps)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        if cfg_scale <= 1.0 or cfg_scale is None:
            pred = model(x, audio, t_batch)
        else:
            # Classifier-free guidance: pred = uncond + scale * (cond - uncond)
            uncond = model(x, torch.zeros_like(audio), t_batch)
            cond = model(x, audio, t_batch)
            pred = uncond + cfg_scale * (cond - uncond)
        pred = pred.clamp(-1, 1)
        posterior_mean, posterior_var = diffusion.q_posterior_mean_variance(pred, x, t_batch)
        noise = torch.randn_like(x, device=device)
        out = posterior_mean + torch.sqrt(posterior_var + 1e-8) * noise
        mask = (t == 0).view(B, 1, 1)
        x = torch.where(mask, pred, out)
    return x.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionDiT(
        motion_dim=cfg["data"]["motion_dim"],
        audio_dim=cfg["data"]["audio_dim"],
        motion_frames=cfg["data"]["motion_frames"],
        patch_size=cfg["model"]["patch_size"],
        dim=cfg["model"]["dim"],
        depth=cfg["model"]["depth"],
        heads=cfg["model"]["heads"],
        cond_drop_prob=0.0,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )

    dataset = BEATSegmentDataset(root=cfg["data"]["root"], train=False)
    os.makedirs(args.output_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    for i in range(num):
        batch = dataset[i]
        audio = batch["audio"].unsqueeze(0).to(device)  # (1, T, 512)
        motion_gen = sample(model, diffusion, audio, device, cfg_scale=args.cfg_scale)
        motion_gen = motion_gen[0]  # (T, 60)
        out_path = os.path.join(args.output_dir, f"motion_{i:04d}.npy")
        np.save(out_path, motion_gen.astype(np.float32))
        print(f"Saved {out_path} shape {motion_gen.shape}")

    print(f"Done. Generated {num} motions in {args.output_dir}")


if __name__ == "__main__":
    main()
