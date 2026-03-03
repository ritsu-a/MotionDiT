"""
从训练好的 DiT 采样：给定 audio condition，生成 motion。
支持 classifier-free guidance (cfg_scale)。
Diffusion 在低帧率上生成，输出会上采样回 motion_frames。
可选：用 GMR 将生成的 npz 渲染为视频。
"""
import os
import argparse
import subprocess
import yaml
import torch
import numpy as np

from dataset import BEATSegmentDataset
from dit import MotionDiT
from diffusion import GaussianDiffusion


def upsample_motion(motion: np.ndarray, target_frames: int) -> np.ndarray:
    """(T_low, D) 上采样到 (target_frames, D)，线性插值。"""
    if motion.shape[0] == target_frames:
        return motion
    # (T, D) -> (1, D, T) for interpolate
    x = torch.from_numpy(motion).float().unsqueeze(0).permute(0, 2, 1)
    x = torch.nn.functional.interpolate(x, size=target_frames, mode="linear", align_corners=False)
    return x.permute(0, 2, 1).squeeze(0).numpy()


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
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
):
    """
    使用 DDIM 采样。audio: (B, T, audio_dim)，返回 motion (B, T, motion_dim)。
    """
    B, T, _ = audio.shape
    shape = (B, T, model.motion_dim)
    model.eval()
    x = torch.randn(shape, device=device)
    seq = diffusion.ddim_timestep_sequence(ddim_steps)
    for i in range(len(seq) - 1):
        t = seq[i]
        t_prev = seq[i + 1]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        if cfg_scale <= 1.0 or cfg_scale is None:
            pred = model(x, audio, t_batch)
        else:
            uncond = model(x, torch.zeros_like(audio), t_batch)
            cond = model(x, audio, t_batch)
            pred = uncond + cfg_scale * (cond - uncond)
        pred = pred.clamp(-1, 1)
        x = diffusion.ddim_step(x, t, t_prev, pred, eta=ddim_eta)
    return x.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--ddim_steps", type=int, default=None, help="DDIM 采样步数，默认用 config 或 50")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM 随机性，0 为确定性")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_video", action="store_true", help="对每个生成的 npz 用 GMR 渲染视频")
    parser.add_argument("--motion_fps", type=int, default=60, help="渲染视频的 motion FPS（与 --render_video 配合）")
    parser.add_argument("--robot", type=str, default="g1_brainco", help="GMR 渲染的机器人类型（与 --render_video 配合）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionDiT(
        motion_dim=cfg["data"]["motion_dim"],
        audio_dim=cfg["data"]["audio_dim"],
        motion_frames=cfg["data"]["motion_frames_low"],
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

    data_cfg = cfg["data"]
    dataset = BEATSegmentDataset(
        root=data_cfg["root"],
        train=False,
        motion_frames=data_cfg["motion_frames"],
        motion_frames_low=data_cfg["motion_frames_low"],
        audio_frames_low=data_cfg["audio_frames_low"],
        motion_dim=data_cfg["motion_dim"],
        audio_dim=data_cfg["audio_dim"],
    )
    os.makedirs(args.output_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    for i in range(num):
        batch = dataset[i]
        audio = batch["audio"].unsqueeze(0).to(device)  # (1, T, 512)
        ddim_steps = args.ddim_steps or cfg.get("generate", {}).get("ddim_steps", 50)
        motion_gen = sample(
            model, diffusion, audio, device,
            cfg_scale=args.cfg_scale,
            ddim_steps=ddim_steps,
            ddim_eta=args.ddim_eta,
        )
        motion_gen = motion_gen[0]  # (motion_frames_low, 60)
        motion_gen = upsample_motion(motion_gen, data_cfg["motion_frames"])  # 还原到 300 帧
        out_path = os.path.join(args.output_dir, f"motion_{i:04d}.npz")
        np.savez(out_path, qpos=motion_gen.astype(np.float32))
        print(f"Saved {out_path} qpos shape {motion_gen.shape}")

        if args.render_video:
            video_path = os.path.join(args.output_dir, f"motion_{i:04d}.mp4")
            gmr_script = os.path.join(os.path.dirname(__file__), "external", "GMR", "scripts", "vis_npz_motion.py")
            if not os.path.isfile(gmr_script):
                print(f"Warning: GMR script not found {gmr_script}, skip rendering.")
            else:
                cmd = [
                    "python", gmr_script,
                    "--npz_path", os.path.abspath(out_path),
                    "--video_path", os.path.abspath(video_path),
                    "--motion_fps", str(args.motion_fps),
                    "--robot", args.robot,
                ]
                ret = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), "external", "GMR"))
                if ret.returncode == 0:
                    print(f"Rendered {video_path}")
                else:
                    print(f"Render failed for {out_path} (exit code {ret.returncode})")

    print(f"Done. Generated {num} motions in {args.output_dir}")


if __name__ == "__main__":
    main()
