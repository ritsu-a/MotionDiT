"""
BEAT v2 Segment 数据集：以 audio (whisper) 为 condition，加载 motion。
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class BEATSegmentDataset(Dataset):
    """
    从 BEAT_v2/segment 的 npz 文件加载：
    - motion: (T_motion, 60), 作为生成目标
    - whisper: (T_audio, 512), 作为 condition (audio)
    将 audio 时间维对齐到 motion 长度（线性插值）。
    """

    def __init__(
        self,
        root: str,
        motion_frames: int = 300,
        motion_dim: int = 60,
        audio_dim: int = 512,
        train: bool = True,
        train_ratio: float = 0.95,
        seed: int = 42,
    ):
        self.root = root
        self.motion_frames = motion_frames
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.train = train

        files = sorted([f for f in os.listdir(root) if f.endswith(".npz")])
        n = len(files)
        np.random.seed(seed)
        perm = np.random.permutation(n)
        split = int(n * train_ratio)
        if train:
            self.files = [files[i] for i in perm[:split]]
        else:
            self.files = [files[i] for i in perm[split:]]

    def __len__(self):
        return len(self.files)

    def _align_audio_to_motion(self, audio: np.ndarray, target_frames: int) -> np.ndarray:
        """将 audio (T_audio, D) 插值到 target_frames 帧，与 motion 对齐。"""
        t_audio, d = audio.shape
        if t_audio == target_frames:
            return audio.astype(np.float32)
        scale = target_frames / t_audio
        # zoom: (T, D) -> 只对时间维插值
        aligned = zoom(audio, (scale, 1.0), order=1)
        return aligned.astype(np.float32)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        data = np.load(path, allow_pickle=True)
        motion = data["motion"]  # (T_motion, 60)
        whisper = data["whisper"]  # (T_audio, 512)

        # 对齐 audio 到 motion 长度
        audio = self._align_audio_to_motion(whisper, motion.shape[0])

        # 若 motion 长度与设定不一致，可裁剪或插值（这里假定固定 300）
        if motion.shape[0] != self.motion_frames:
            motion = self._align_audio_to_motion(motion, self.motion_frames)
        else:
            motion = motion.astype(np.float32)

        if audio.shape[0] != self.motion_frames:
            audio = self._align_audio_to_motion(audio, self.motion_frames)

        return {
            "motion": torch.from_numpy(motion),   # (T, 60)
            "audio": torch.from_numpy(audio),    # (T, 512)
        }


def get_dataloaders(root: str, batch_size: int = 32, train_ratio: float = 0.95, num_workers: int = 0):
    from torch.utils.data import DataLoader
    train_ds = BEATSegmentDataset(root=root, train=True, train_ratio=train_ratio)
    val_ds = BEATSegmentDataset(root=root, train=False, train_ratio=train_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
