# DiT: Audio-Conditioned Motion Generation

基于 **Diffusion Transformer (DiT)** 的以 **音频 (audio)** 为条件的人体动作生成，在 BEAT v2 segment 数据集上训练。

## 数据

- **路径**: `data/BEAT_v2/segment`
- 每个 `.npz` 包含:
  - `motion`: `(T, 60)` 动作序列（生成目标）
  - `whisper`: `(T_audio, 512)` 音频特征（Whisper 编码，作为 condition）
- 加载时会将 audio 时间维对齐到 motion 长度（线性插值）。

## 环境

```bash
pip install -r requirements.txt
```

## 配置

主配置: `configs/default.yaml`

- `data`: 数据路径、motion/audio 维度、帧数、划分比例
- `model`: DiT 维度、层数、patch 大小、cond_drop_prob
- `diffusion`: 时间步、beta 调度
- `train` / `generate`: 训练与采样参数

## 训练

```bash
python train.py --config configs/default.yaml
```

从检查点恢复:

```bash
python train.py --config configs/default.yaml --resume checkpoints/dit_epoch100.pt
```

## 采样（生成 motion）

在验证集上取 audio，生成对应 motion；结果保存为 **npz**（键 `qpos`，可用 GMR 可视化）:

```bash
python sample.py --checkpoint checkpoints/dit_epoch200.pt --output_dir samples --num_samples 16 --cfg_scale 3.0
```

- `--cfg_scale`: classifier-free guidance 强度，越大越贴合 condition，一般 2.0–5.0。
- `--render_video`: 生成 npz 后自动用 GMR 渲染为 mp4（需已安装 GMR 环境）。
- `--motion_fps`、`--robot`: 渲染时的 FPS 与机器人类型，默认 30、`g1_brainco`。

示例（生成并直接渲染视频）:

```bash
python sample.py --checkpoint checkpoints/dit_epoch200.pt --output_dir samples --num_samples 4 --cfg_scale 3.0 --render_video
```

仅可视化已有 npz（不跑 sample）:

```bash
python external/GMR/scripts/vis_npz_motion.py --npz_path samples/motion_0000.npz --video_path samples/motion_0000.mp4 --motion_fps 30
```

## 结构概览

| 文件 | 说明 |
|------|------|
| `dataset.py` | BEAT segment 数据集，返回 `motion` + `audio` |
| `dit.py` | MotionDiT：patch 化 motion+audio，Transformer 预测 x0 |
| `diffusion.py` | 高斯扩散：q_sample、后验、p_sample |
| `train.py` | 训练脚本 |
| `sample.py` | 推理脚本，支持 CFG |

## 引用

- DiT: Scalable Diffusion Models with Transformers  
- BEAT: BEAT dataset for audio-driven gesture generation  
