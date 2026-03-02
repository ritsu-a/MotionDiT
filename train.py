"""
训练 DiT：以 audio (whisper) 为 condition，生成 motion。
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from dit import MotionDiT
from diffusion import GaussianDiffusion


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, diffusion, loader, opt, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        motion = batch["motion"].to(device)
        audio = batch["audio"].to(device)
        B = motion.shape[0]
        t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
        noise = torch.randn_like(motion, device=device)
        x_t = diffusion.q_sample(motion, t, noise)
        pred = model(x_t, audio, t)
        loss = nn.functional.mse_loss(pred, motion)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(
        root=cfg["data"]["root"],
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
    )

    model = MotionDiT(
        motion_dim=cfg["data"]["motion_dim"],
        audio_dim=cfg["data"]["audio_dim"],
        motion_frames=cfg["data"]["motion_frames"],
        patch_size=cfg["model"]["patch_size"],
        dim=cfg["model"]["dim"],
        depth=cfg["model"]["depth"],
        heads=cfg["model"]["heads"],
        cond_drop_prob=cfg["model"]["cond_drop_prob"],
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])

    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        avg_loss = train_one_epoch(model, diffusion, train_loader, opt, device, epoch)
        print(f"Epoch {epoch} train loss: {avg_loss:.6f}")
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            path = os.path.join(out_dir, f"dit_epoch{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "optimizer": opt.state_dict(),
            }, path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
