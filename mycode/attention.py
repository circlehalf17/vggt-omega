#!/usr/bin/env python3
"""
Attention map visualization for hand-selected stress/control clips.

Outputs per clip (under OUTPUT_BASE/<type>/<name>/):
  inter_frame_attn_cam_xx.png  – camera token inter-frame attention  (S×S, linear)
  frame_attn_xx.png            – full frame self-attention            (N×N, log scale)
  inter_frame_attn_reg_xx.png  – register token inter-frame attention (S*16×S*16, log scale)
  Total: 72 files per clip (3 types × 24 layers)

Sources:
  frame_attn       → FullAttnCapture.frame_blocks  (same as attention_maps2/frame_attn_xx.png)
  inter_attn_cam   → camera token block from inter_frame_blocks  (cf. attention_maps/inter_frame_attn_xx.png)
  inter_attn_reg   → register token block from inter_frame_blocks (cf. attention_maps2/inter_frame_attn_xx.png)
"""
import gc
import glob
import os
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.utils.load_fn import load_and_preprocess_images


# ── Clip list ─────────────────────────────────────────────────────────────────
CLIPS = [
    ('control',   '0b9ee926-00f1-4b22-9e83-b664c5e465e4', 1492, 1503, ''),
    ('control',   '1cdc92fa-50cd-4461-adf2-ece8cb2a5d31',  940,  951, ''),
    ('control',   '39011e23-5fe4-41cd-b146-f3f9e3f3941a', 3776, 3787, ''),
    ('dynamic',   '1eb500cd-cdb4-415e-8bf6-cc1289edf0ee',    0,   11, ''),
    ('dynamic',   '3c826d8a-22b9-4083-a315-2aeee7bde095',   51,   61, ''),
    ('dynamic',   '04fe8f4d-081e-437e-a56a-2d53b6233fc9',   13,   26, ''),
    ('egomotion', '1b1acfa6-ee3b-483c-9c0a-245d70d7125f',   52,   65, 'control'),
    ('egomotion', '1b1acfa6-ee3b-483c-9c0a-245d70d7125f',  902,  916, 'stress'),
    ('egomotion', '04fe8f4d-081e-437e-a56a-2d53b6233fc9', 3240, 3254, 'control'),
    ('egomotion', '04fe8f4d-081e-437e-a56a-2d53b6233fc9',  476,  489, 'stress'),
    ('egomotion', 'cfa96e76-d909-4ba5-bccb-77402fde6be7',  316,  327, 'control'),
    ('egomotion', 'cfa96e76-d909-4ba5-bccb-77402fde6be7', 1001, 1011, 'stress'),
]

VIDEO_ROOT  = '/workspace/data/Ego4D/v2/full_scale'
OUTPUT_BASE = '/workspace/outputs/renders/vggt-omega/ego4d/attention'
CHECKPOINT  = '/workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt'
IMAGE_RES   = 512
SAMPLE_FPS  = 6.0

_LOG_NORM = LogNorm(vmin=1e-4, vmax=1.0)


# ── Attention capture ─────────────────────────────────────────────────────────

class CombinedAttnCapture:
    """
    Single-pass capture of all three attention types:
      frame_attn[i]      (N, N)         full frame self-attention, 1st frame, head-averaged
      inter_attn[i]      (S*p0, S*p0)   cam+reg inter-frame attention, head-averaged
      inter_is_reg[i]    bool           True when inter block uses register-only stride
    """

    def __init__(self):
        self._handles = []
        self.frame_attn: list[torch.Tensor] = []
        self.inter_attn: list[torch.Tensor] = []
        self.inter_is_reg: list[bool] = []
        self.patch_token_start: int = 17
        self.num_frames: int = 1

    def register(self, model: VGGTOmega):
        agg = model.aggregator
        for block in agg.frame_blocks:
            self._handles.append(block.register_forward_hook(
                self._frame_hook, with_kwargs=True))
        for block in agg.inter_frame_blocks:
            self._handles.append(block.register_forward_hook(
                self._inter_hook, with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self):
        self.frame_attn.clear()
        self.inter_attn.clear()
        self.inter_is_reg.clear()

    def _frame_hook(self, module: SelfAttentionBlock, args, kwargs, output):
        x = args[0].detach()
        rope = args[1] if len(args) > 1 else None
        x0 = module.norm1(x[0:1])
        attn_mod = module.attn
        _, N, C = x0.shape
        n_h = attn_mod.num_heads
        d_h = C // n_h
        scale = d_h ** -0.5
        with torch.no_grad():
            qkv = attn_mod.qkv(x0).reshape(1, N, 3, n_h, d_h)
            q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
            if attn_mod.use_qk_norm:
                q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
            if rope is not None:
                q, k = attn_mod.apply_rope(q, k, rope)
            attn_w = F.softmax(
                q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
            self.frame_attn.append(attn_w.mean(dim=[0, 1]).cpu())

    def _inter_hook(self, module: SelfAttentionBlock, args, kwargs, output):
        x = args[0].detach()
        B, N_total, C = x.shape
        S = self.num_frames
        p0 = self.patch_token_start
        n_h = module.attn.num_heads
        d_h = C // n_h
        scale = d_h ** -0.5
        if N_total % S != 0:
            self.inter_attn.append(torch.zeros(S * p0, S * p0))
            self.inter_is_reg.append(False)
            return
        stride = N_total // S
        is_register = (stride == p0)
        x_norm = module.norm1(x)
        attn_mod = module.attn
        with torch.no_grad():
            if is_register:
                # All tokens are cam+reg; capture full S*p0 × S*p0
                qkv = attn_mod.qkv(x_norm).reshape(B, N_total, 3, n_h, d_h)
                q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                if attn_mod.use_qk_norm:
                    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                attn_w = F.softmax(
                    q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
                self.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())
                self.inter_is_reg.append(True)
            else:
                # Extract cam+reg subset: first p0 tokens of each frame group
                cam_reg_idx = torch.cat([
                    torch.arange(p0, device=x_norm.device) + s * stride
                    for s in range(S)
                ])
                x_cr = x_norm[:, cam_reg_idx, :]
                T = x_cr.shape[1]
                qkv = attn_mod.qkv(x_cr).reshape(B, T, 3, n_h, d_h)
                q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                if attn_mod.use_qk_norm:
                    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                attn_w = F.softmax(
                    q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
                self.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())
                self.inter_is_reg.append(False)


# ── Visualization ─────────────────────────────────────────────────────────────

def _save_frame_attn(attn: torch.Tensor, patch_token_start: int,
                     out_path: str, layer_idx: int):
    mat = attn.numpy().clip(1e-6, 1.0)
    N = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(mat, cmap='inferno', aspect='auto', norm=_LOG_NORM,
                   interpolation='nearest')
    ax.set_title(
        f'Frame Attention  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n'
        f'Shape: {N}×{N}  [cam(1) | reg(16) | patch({N - patch_token_start})]',
        fontsize=11)
    ax.set_xlabel('Key token index')
    ax.set_ylabel('Query token index')
    for pos in [0, 1, patch_token_start]:
        ax.axvline(x=pos - 0.5, color='cyan', linewidth=0.6, alpha=0.7)
        ax.axhline(y=pos - 0.5, color='cyan', linewidth=0.6, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_inter_attn_cam(attn: torch.Tensor, num_frames: int,
                          patch_token_start: int, out_path: str, layer_idx: int):
    """S×S camera token inter-frame attention (linear colormap)."""
    raw = attn.numpy()
    S, p0 = num_frames, patch_token_start
    # Camera token: first token of each frame group → indices [0, p0, 2*p0, ...]
    cam_pos = np.array([s * p0 for s in range(S)])
    cam_attn = raw[np.ix_(cam_pos, cam_pos)]
    cam_attn = (cam_attn - cam_attn.min()) / (cam_attn.max() - cam_attn.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    im = ax.imshow(cam_attn, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Inter-Frame Camera Attention  |  Layer {layer_idx:02d}', fontsize=13)
    ax.set_xlabel('Key frame index')
    ax.set_ylabel('Query frame index')
    if S <= 30:
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_inter_attn_reg(attn: torch.Tensor, num_frames: int,
                          patch_token_start: int, out_path: str, layer_idx: int):
    """S*(p0-1) × S*(p0-1) register token inter-frame attention (log colormap)."""
    raw = attn.numpy()
    S, p0, n_reg = num_frames, patch_token_start, patch_token_start - 1
    # Register tokens: positions 1..p0-1 of each frame group
    reg_pos = np.array([s * p0 + r for s in range(S) for r in range(1, p0)])
    reg_attn = raw[np.ix_(reg_pos, reg_pos)].clip(1e-4, 1.0)
    T = reg_attn.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(reg_attn, cmap='viridis', aspect='auto', norm=_LOG_NORM,
                   interpolation='nearest')
    ax.set_title(
        f'Inter-Frame Register Attention  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n'
        f'Shape: {T}×{T}  [S={S}, {n_reg} registers/frame]',
        fontsize=11)
    ax.set_xlabel('Key register index')
    ax.set_ylabel('Query register index')
    for s in range(1, S):
        pos = s * n_reg - 0.5
        ax.axvline(x=pos, color='cyan', linewidth=0.5, alpha=0.6)
        ax.axhline(y=pos, color='cyan', linewidth=0.5, alpha=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Frame extraction & original clip ─────────────────────────────────────────

def save_original_clip(video_path: str, start_sec: float, end_sec: float,
                       out_path: str, fps: float = 30.0):
    """Extract original clip using imageio_ffmpeg (bundled libx264)."""
    tmp_dir = tempfile.mkdtemp()
    try:
        frame_glob = os.path.join(tmp_dir, 'frame_%06d.jpg')
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', str(start_sec), '-i', video_path,
            '-t', str(end_sec - start_sec),
            '-vf', f'fps={fps}', '-q:v', '2', frame_glob,
        ], check=True)
        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.jpg')))
        if not frame_paths:
            print('  WARN: no frames for original clip')
            return
        first = cv2.imread(frame_paths[0])
        h, w = first.shape[:2]
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1
        writer = imageio_ffmpeg.write_frames(
            out_path, size=(w, h), fps=fps,
            codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
        )
        writer.send(None)
        for p in frame_paths:
            f = cv2.imread(p)
            if f is not None:
                writer.send(f[:h, :w].tobytes())
        writer.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_frames(video_path: str, out_dir: str,
                   start_sec: float, end_sec: float, fps: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start_sec), '-i', video_path,
        '-t', str(end_sec - start_sec),
        '-vf', f'fps={fps}', '-q:v', '2',
        os.path.join(out_dir, 'frame_%06d.jpg'),
    ]
    subprocess.run(cmd, check=True)
    return len(glob.glob(os.path.join(out_dir, 'frame_*.jpg')))


# ── Main ──────────────────────────────────────────────────────────────────────

def clip_out_name(uid: str, start: int, end: int, role: str) -> str:
    role_tag = f'_{role}' if role else ''
    return f'{uid}_{start}-{end}s{role_tag}'


def main():
    print(f'Loading model from {CHECKPOINT}')
    model = VGGTOmega().eval()
    state_dict = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to('cuda')
    print('Model loaded.\n')

    num_layers = model.aggregator.depth
    patch_token_start = model.aggregator.patch_token_start
    print(f'Aggregator depth={num_layers}, patch_token_start={patch_token_start}')
    print(f'Expected output: {num_layers * 3} files per clip\n')

    capture = CombinedAttnCapture()
    capture.register(model)
    capture.patch_token_start = patch_token_start

    for i, (clip_type, uid, start, end, role) in enumerate(CLIPS):
        name = clip_out_name(uid, start, end, role)
        out_dir = os.path.join(OUTPUT_BASE, clip_type, name)
        print(f'[{i+1}/{len(CLIPS)}] {clip_type}/{name}')

        # Skip if all 72 PNG files + original.mp4 already exist
        existing = glob.glob(os.path.join(out_dir, '*.png'))
        mp4_exists = os.path.exists(os.path.join(out_dir, 'original.mp4'))
        if len(existing) >= num_layers * 3 and mp4_exists:
            print(f'  Already done ({len(existing)} files), skipping.')
            continue

        video_path = os.path.join(VIDEO_ROOT, uid + '.mp4')
        if not os.path.exists(video_path):
            print(f'  SKIP: video not found')
            continue

        os.makedirs(out_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f'  Extracting frames at {SAMPLE_FPS}fps ...')
            n = extract_frames(video_path, tmp_dir, start, end, SAMPLE_FPS)
            print(f'  {n} frames extracted')

            if n < 2:
                print('  SKIP: too few frames')
                continue

            image_paths = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.jpg')))
            images = load_and_preprocess_images(
                image_paths, image_resolution=IMAGE_RES).to('cuda')

            patch_size = model.aggregator.patch_size
            capture.num_frames = images.shape[0]
            capture.reset()

            print('  Running inference ...')
            try:
                with torch.inference_mode():
                    _ = model(images)
            except Exception as e:
                print(f'  ERROR: {e}')
                gc.collect(); torch.cuda.empty_cache()
                continue

        n_frame = len(capture.frame_attn)
        n_inter = len(capture.inter_attn)
        print(f'  Captured: {n_frame} frame maps, {n_inter} inter maps')

        if n_frame != num_layers or n_inter != num_layers:
            print(f'  WARNING: expected {num_layers} layers each, '
                  f'got {n_frame} frame / {n_inter} inter')

        print('  Saving outputs ...')
        S = capture.num_frames

        for layer_idx in range(num_layers):
            cam_out  = os.path.join(out_dir, f'inter_frame_attn_cam_{layer_idx:02d}.png')
            frm_out  = os.path.join(out_dir, f'frame_attn_{layer_idx:02d}.png')
            reg_out  = os.path.join(out_dir, f'inter_frame_attn_reg_{layer_idx:02d}.png')

            if layer_idx < n_frame:
                _save_frame_attn(capture.frame_attn[layer_idx],
                                 patch_token_start, frm_out, layer_idx)

            if layer_idx < n_inter:
                _save_inter_attn_cam(capture.inter_attn[layer_idx],
                                     S, patch_token_start, cam_out, layer_idx)
                _save_inter_attn_reg(capture.inter_attn[layer_idx],
                                     S, patch_token_start, reg_out, layer_idx)

        mp4_path = os.path.join(out_dir, 'original.mp4')
        if not os.path.exists(mp4_path):
            print('  Saving original.mp4 ...')
            save_original_clip(video_path, start, end, mp4_path)
        print(f'  Saved: {mp4_path}')

        saved = len(glob.glob(os.path.join(out_dir, '*.png')))
        print(f'  Saved {saved} PNG files → {out_dir}\n')

        gc.collect()
        torch.cuda.empty_cache()

    capture.remove()
    print('All done.')


if __name__ == '__main__':
    main()
