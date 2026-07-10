#!/usr/bin/env python3
"""
Knowledge distillation: DPT DenseHead (teacher) → MLPDenseHead (student).

Architecture
------------
  Aggregator (frozen)  ──→  patch tokens
       ↓                         ↓
  DPT head (frozen)       MLP head (trainable)
  depth_teacher            depth_student
       └──── SILog(depth_student, depth_teacher) ────┘

Training data: random 10-second clips sampled on-the-fly from Ego4D full_scale videos.
No depth ground-truth is required; the pretrained DPT head acts as a strong teacher.

Frame extraction uses subprocess ffmpeg (same as inference scripts) to avoid
library conflicts with decord inside the Singularity container.
"""
import argparse
import glob
import gc
import json
import math
import os
import random
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TF

from vggt_omega.models import VGGTOmega
from vggt_omega.models.heads.dense_head import MLPDenseHead
from vggt_omega.utils.load_fn import load_and_preprocess_images


# ── Constants ─────────────────────────────────────────────────────────────────

VIDEO_ROOT   = '/workspace/data/Ego4D/v2/full_scale'
EGO4D_JSON   = '/workspace/data/Ego4D/ego4d.json'
CHECKPOINT   = '/workspace/outputs/checkpoints/vggt-omega/pretrain/vggt_omega_1b_512.pt'
OUTPUT_DIR   = '/workspace/outputs/checkpoints/vggt-omega/mlp_distill'

IMAGE_RES    = 512
PATCH_SIZE   = 16
SAMPLE_FPS   = 6.0
NUM_FRAMES   = 8
MIN_CLIP_DUR = 20.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class Ego4DClipDataset(Dataset):
    """
    Infinite virtual dataset.
    Each __getitem__ samples a random 10-second window from a random Ego4D video,
    extracts frames with ffmpeg, and returns a preprocessed image tensor.
    num_workers=0 is required (no multiprocessing) to avoid CUDA/fork conflicts.
    """

    def __init__(
        self,
        video_root: str,
        ego4d_json: str,
        num_frames: int = NUM_FRAMES,
        fps: float = SAMPLE_FPS,
        image_resolution: int = IMAGE_RES,
        min_dur: float = MIN_CLIP_DUR,
        virtual_size: int = 200_000,
    ):
        self.video_root      = video_root
        self.num_frames      = num_frames
        self.fps             = fps
        self.image_resolution = image_resolution
        self.virtual_size    = virtual_size
        self.clip_dur        = (num_frames - 1) / fps

        available = {
            os.path.splitext(f)[0]
            for f in os.listdir(video_root)
            if f.endswith('.mp4')
        }
        with open(ego4d_json) as fh:
            meta = json.load(fh)
        self.videos = [
            (v['video_uid'], v['duration_sec'])
            for v in meta['videos']
            if v['video_uid'] in available
            and v.get('duration_sec', 0) >= min_dur + self.clip_dur
        ]
        if not self.videos:
            raise RuntimeError('No usable videos found.')
        print(f'[Dataset] {len(self.videos)} usable videos')

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, _idx: int) -> torch.Tensor:
        for attempt in range(10):
            try:
                return self._load_clip()
            except Exception as exc:
                if attempt == 9:
                    raise RuntimeError(f'Failed after 10 attempts: {exc}')
        raise RuntimeError('unreachable')

    def _load_clip(self) -> torch.Tensor:
        uid, dur = random.choice(self.videos)
        video_path = os.path.join(self.video_root, uid + '.mp4')

        max_start = dur - self.clip_dur - 2.0
        if max_start <= 0:
            raise ValueError('video too short')
        start = random.uniform(0.0, max_start)

        with tempfile.TemporaryDirectory() as tmp:
            ret = subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-ss', f'{start:.3f}', '-i', video_path,
                '-t', f'{self.clip_dur + 1.0:.3f}',
                '-vf', f'fps={self.fps}', '-q:v', '2',
                os.path.join(tmp, 'frame_%06d.jpg'),
            ], capture_output=True)
            if ret.returncode != 0:
                raise RuntimeError(f'ffmpeg failed: {ret.stderr.decode()[:200]}')

            paths = sorted(glob.glob(os.path.join(tmp, 'frame_*.jpg')))
            if len(paths) < self.num_frames:
                raise ValueError(f'only {len(paths)} frames extracted')

            # Trim to exactly num_frames
            step = max(1, (len(paths) - 1) // (self.num_frames - 1))
            selected = [paths[min(i * step, len(paths) - 1)] for i in range(self.num_frames)]

            images = load_and_preprocess_images(selected, image_resolution=self.image_resolution)

        return images   # [S, 3, H, W]


# ── Loss ──────────────────────────────────────────────────────────────────────

def silog_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """
    Scale-Invariant Log loss (Eigen et al., 2014).
    pred, target: [..., 1] positive depth values.
    """
    pred   = pred.squeeze(-1).float()
    target = target.squeeze(-1).float()

    valid = (target > 1e-6) & (pred > 1e-6) & torch.isfinite(target) & torch.isfinite(pred)
    if valid.sum() == 0:
        return pred.sum() * 0.0

    log_pred   = torch.log(pred[valid])
    log_target = torch.log(target[valid])
    d = log_pred - log_target
    return d.pow(2).mean() - lam * d.mean().pow(2)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_teacher(checkpoint_path: str, device: str) -> VGGTOmega:
    model = VGGTOmega(dense_head_type='dpt').eval()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_student(dim_in: int = 2048, patch_size: int = PATCH_SIZE,
                 mlp_dim: int = 1024) -> MLPDenseHead:
    return MLPDenseHead(dim_in=dim_in, patch_size=patch_size, mlp_dim=mlp_dim)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int,
              lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(out_dir: str, step: int, student: nn.Module,
              optimizer: torch.optim.Optimizer, loss: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    kept = sorted(f for f in os.listdir(out_dir) if f.startswith('mlp_head_step'))
    for old in kept[:-2]:
        os.remove(os.path.join(out_dir, old))
    path = os.path.join(out_dir, f'mlp_head_step{step:07d}.pt')
    torch.save({
        'step': step,
        'model_state_dict': student.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f'  [ckpt] → {path}')


def load_ckpt(path: str, student: nn.Module,
              optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    step = ckpt.get('step', 0)
    print(f'  [ckpt] resumed from step {step}')
    return step


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f'Device: {device}  AMP: {amp_dtype}')
    print(f'Loading teacher from {args.checkpoint} …')
    teacher = load_teacher(args.checkpoint, device)
    print('Teacher loaded and frozen.\n')

    student = make_student(mlp_dim=args.mlp_dim).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'Student MLPDenseHead: {n_params:,} params\n')

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_ckpt(args.resume, student, optimizer)

    dataset = Ego4DClipDataset(
        video_root=args.video_root,
        ego4d_json=args.ego4d_json,
        num_frames=args.num_frames,
        fps=args.fps,
        image_resolution=args.image_resolution,
    )
    # num_workers=0: ffmpeg subprocess is called in the main process.
    # Avoids all fork/spawn CUDA conflicts; GPU compute dominates anyway (~3-5s/step).
    loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'train.log')

    print(f'Training  steps {start_step} → {args.total_steps}')
    print(f'  grad_accum={args.grad_accum}  lr={args.lr}  warmup={args.warmup_steps}')
    print(f'  save_every={args.save_every}  log_every={args.log_every}\n')

    step       = start_step
    accum_loss = 0.0
    t0         = time.time()
    student.train()
    optimizer.zero_grad()

    for images in loader:
        if step >= args.total_steps:
            break

        # images: [1, S, 3, H, W]
        images = images.to(device, non_blocking=True)

        # ── Teacher forward (no grad) ────────────────────────────────────────
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                agg_tokens, patch_token_start = teacher.aggregator(images)

            with torch.autocast(device_type='cuda', enabled=False):
                depth_teacher, _ = teacher.dense_head(
                    agg_tokens, images=images,
                    patch_token_start=patch_token_start,
                    frames_chunk_size=args.num_frames,
                )

        depth_teacher = depth_teacher.detach()

        # Detach cached tokens so no grad flows back into the aggregator
        agg_tokens_detached = [
            t.detach() if t is not None else None for t in agg_tokens
        ]

        # ── Student forward (gradients on) ───────────────────────────────────
        depth_student, _ = student(
            agg_tokens_detached, images=images,
            patch_token_start=patch_token_start,
            frames_chunk_size=args.num_frames,
        )

        loss = silog_loss(depth_student, depth_teacher) / args.grad_accum
        loss.backward()
        accum_loss += loss.item()

        # ── Optimizer step ───────────────────────────────────────────────────
        if (step + 1) % args.grad_accum == 0:
            lr = cosine_lr(step, args.warmup_steps, args.total_steps,
                           args.lr, args.lr * 0.01)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                msg = (f'step={step+1:6d}  loss={accum_loss:.4f}'
                       f'  lr={lr:.2e}  t={elapsed:.0f}s')
                print(msg, flush=True)
                with open(log_path, 'a') as fh:
                    fh.write(msg + '\n')
                accum_loss = 0.0
                t0 = time.time()

            if (step + 1) % args.save_every == 0:
                save_ckpt(args.output_dir, step + 1, student, optimizer, loss.item())

        step += 1
        gc.collect()

    save_ckpt(args.output_dir, step, student, optimizer, 0.0)
    print(f'\nDone at step {step}.')


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',       default=CHECKPOINT)
    p.add_argument('--video_root',       default=VIDEO_ROOT)
    p.add_argument('--ego4d_json',       default=EGO4D_JSON)
    p.add_argument('--output_dir',       default=OUTPUT_DIR)
    p.add_argument('--resume',           default=None)
    p.add_argument('--num_frames',       type=int,   default=NUM_FRAMES)
    p.add_argument('--fps',              type=float, default=SAMPLE_FPS)
    p.add_argument('--image_resolution', type=int,   default=IMAGE_RES)
    p.add_argument('--mlp_dim',          type=int,   default=1024)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--weight_decay',     type=float, default=1e-2)
    p.add_argument('--grad_clip',        type=float, default=1.0)
    p.add_argument('--grad_accum',       type=int,   default=8)
    p.add_argument('--warmup_steps',     type=int,   default=500)
    p.add_argument('--total_steps',      type=int,   default=50_000)
    p.add_argument('--save_every',       type=int,   default=2_000)
    p.add_argument('--log_every',        type=int,   default=50)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
