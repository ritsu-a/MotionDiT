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
        motion_frames_low: int = 40,
        audio_frames_low: int = 25,
        motion_dim: int = 60,
        audio_dim: int = 512,
        train: bool = True,
        train_ratio: float = 0.95,
        seed: int = 42,
    ):
        self.root = root
        self.motion_frames = motion_frames
        self.motion_frames_low = motion_frames_low
        self.audio_frames_low = audio_frames_low
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.train = train

        files = sorted([f for f in os.listdir(root) if f.endswith(".npz")])
        n = len(files)
        np.random.seed(seed)
        perm = np.random.permutation(n)
        split = int(n * train_ratio)
        if train:
            candidate_files = [files[i] for i in perm[:split]]
        else:
            candidate_files = [files[i] for i in perm[split:]]
        self.files = self._filter_valid_files(root, candidate_files, motion_dim, audio_dim)
        if len(self.files) < len(candidate_files):
            skipped = len(candidate_files) - len(self.files)
            print(f"[BEATSegmentDataset] 过滤 {skipped} 个异常片段，保留 {len(self.files)} 个 (train={train})")

    def _filter_valid_files(
        self, root: str, file_list: list, motion_dim: int, audio_dim: int
    ) -> list:
        """检测并只保留 motion/whisper 形状正确的 npz。"""
        valid = []
        for f in file_list:
            path = os.path.join(root, f)
            try:
                data = np.load(path, allow_pickle=True)
                motion = data["motion"]
                whisper = data["whisper"]
                if motion.ndim != 2 or motion.shape[1] != motion_dim or motion.shape[0] == 0:
                    continue
                if whisper.ndim != 2 or whisper.shape[1] != audio_dim or whisper.shape[0] == 0:
                    continue
                valid.append(f)
            except Exception:
                continue
        return valid

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
        motion = np.asarray(data["motion"])
        whisper = np.asarray(data["whisper"])
        if motion.ndim != 2 or whisper.ndim != 2:
            raise RuntimeError(
                f"{path}: motion.ndim={getattr(motion, 'ndim', '?')}, whisper.ndim={getattr(whisper, 'ndim', '?')}"
            )

        # 对齐 audio 到 motion 长度（原始 300 帧）
        audio = self._align_audio_to_motion(whisper, motion.shape[0])

        if motion.shape[0] != self.motion_frames:
            motion = self._align_audio_to_motion(motion, self.motion_frames)
        else:
            motion = motion.astype(np.float32)

        if audio.shape[0] != self.motion_frames:
            audio = self._align_audio_to_motion(audio, self.motion_frames)

        # 进 diffusion 前下采样：motion 8 倍、audio 10 倍（与 config motion_frames_low / audio_frames_low 一致）
        motion_low = zoom(motion, (self.motion_frames_low / motion.shape[0], 1.0), order=1).astype(np.float32)
        audio_low = zoom(audio, (self.audio_frames_low / audio.shape[0], 1.0), order=1).astype(np.float32)

        return {
            "motion": torch.from_numpy(motion_low),   # (motion_frames_low, 60)
            "audio": torch.from_numpy(audio_low),    # (audio_frames_low, 512)
        }


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    train_ratio: float = 0.95,
    num_workers: int = 0,
    motion_frames: int = 300,
    motion_frames_low: int = 40,
    audio_frames_low: int = 25,
    motion_dim: int = 60,
    audio_dim: int = 512,
):
    from torch.utils.data import DataLoader
    train_ds = BEATSegmentDataset(
        root=root, train=True, train_ratio=train_ratio,
        motion_frames=motion_frames, motion_frames_low=motion_frames_low, audio_frames_low=audio_frames_low,
        motion_dim=motion_dim, audio_dim=audio_dim,
    )
    val_ds = BEATSegmentDataset(
        root=root, train=False, train_ratio=train_ratio,
        motion_frames=motion_frames, motion_frames_low=motion_frames_low, audio_frames_low=audio_frames_low,
        motion_dim=motion_dim, audio_dim=audio_dim,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
