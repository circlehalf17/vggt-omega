#!/usr/bin/env python3
"""
VGGT-Omega attention map visualization.

Modes:
  alt    AltAttnCapture: simplified frame heatmap + inter-frame camera token attention
  full   FullAttnCapture: full attention matrices with token group detail (LogNorm)
"""
import argparse
import gc
import glob
import json
import os
import shutil

import cv2
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def extract_frames(video_path: str, out_dir: str, start_sec: float,
                   end_sec: float, sample_fps: float = 1.0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps / sample_fps)), 1)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved, frame_idx = 0, start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.png"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


# ── AltAttnCapture ────────────────────────────────────────────────────────────

class AltAttnCapture:
    """
    Simplified attention maps per layer:
      frame_attn[i]  (N_patch,)        — mean patch attention, averaged over frames
      inter_attn[i]  (S, S)            — camera token inter-frame attention
    """

    def __init__(self):
        self._handles = []
        self.frame_attn: list[torch.Tensor] = []
        self.inter_attn: list[torch.Tensor] = []
        self.patch_token_start: int = 17
        self.num_frames: int = 1
        self.patch_grid: tuple = (28, 37)

    def register(self, model: VGGTOmega):
        agg = model.aggregator
        for i, block in enumerate(agg.frame_blocks):
            self._handles.append(block.register_forward_hook(
                self._make_frame_hook(i), with_kwargs=True))
        for i, block in enumerate(agg.inter_frame_blocks):
            self._handles.append(block.register_forward_hook(
                self._make_inter_hook(i), with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self):
        self.frame_attn.clear()
        self.inter_attn.clear()

    def _make_frame_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            x = args[0].detach()
            rope = args[1] if len(args) > 1 else None
            x_norm = module.norm1(x)
            attn_mod = module.attn
            B_S, N, C = x_norm.shape
            n_h = attn_mod.num_heads
            d_h = C // n_h
            p0 = capture.patch_token_start
            scale = d_h ** -0.5
            per_frame = []
            with torch.no_grad():
                for b in range(B_S):
                    xb = x_norm[b:b+1]
                    qkv = attn_mod.qkv(xb).reshape(1, N, 3, n_h, d_h)
                    q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                    if attn_mod.use_qk_norm:
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    if rope is not None:
                        q, k = attn_mod.apply_rope(q, k, rope)
                    attn_w = F.softmax(
                        q[:, :, p0:, :].float() @ k.float().transpose(-2, -1) * scale,
                        dim=-1)
                    per_frame.append(attn_w.mean(dim=[0, 1, 2])[p0:].cpu())
            capture.frame_attn.append(torch.stack(per_frame).mean(0))

        return hook

    def _make_inter_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            x = args[0].detach()
            x_norm = module.norm1(x)
            attn_mod = module.attn
            B, N_total, C = x_norm.shape
            n_h = attn_mod.num_heads
            d_h = C // n_h
            S = capture.num_frames
            scale = d_h ** -0.5
            if N_total % S != 0:
                capture.inter_attn.append(torch.zeros(S, S)); return
            stride = N_total // S
            cam_idx = torch.arange(S, device=x_norm.device) * stride
            with torch.no_grad():
                x_cam = x_norm[:, cam_idx, :]
                qkv = attn_mod.qkv(x_cam).reshape(B, S, 3, n_h, d_h)
                q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                if attn_mod.use_qk_norm:
                    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                attn_w = F.softmax(
                    q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
                capture.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())

        return hook


# ── FullAttnCapture ───────────────────────────────────────────────────────────

class FullAttnCapture:
    """
    Full attention matrices per layer:
      frame_attn[i]  (N_tokens, N_tokens)  — first frame, head-averaged
      inter_attn[i]  (S*17, S*17)          — camera+register subset, head-averaged
    """

    def __init__(self):
        self._handles = []
        self.frame_attn: list[torch.Tensor] = []
        self.inter_attn: list[torch.Tensor] = []
        self.patch_token_start: int = 17
        self.num_frames: int = 1

    def register(self, model: VGGTOmega):
        agg = model.aggregator
        for i, block in enumerate(agg.frame_blocks):
            self._handles.append(block.register_forward_hook(
                self._make_frame_hook(i), with_kwargs=True))
        for i, block in enumerate(agg.inter_frame_blocks):
            self._handles.append(block.register_forward_hook(
                self._make_inter_hook(i), with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self):
        self.frame_attn.clear()
        self.inter_attn.clear()

    def _make_frame_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
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
                capture.frame_attn.append(attn_w.mean(dim=[0, 1]).cpu())

        return hook

    def _make_inter_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            x = args[0].detach()
            B, N_total, C = x.shape
            S = capture.num_frames
            p0 = capture.patch_token_start
            n_h = module.attn.num_heads
            d_h = C // n_h
            scale = d_h ** -0.5
            if N_total % S != 0:
                capture.inter_attn.append(torch.zeros(S * p0, S * p0)); return
            stride = N_total // S
            is_register = (stride == p0)
            x_norm = module.norm1(x)
            attn_mod = module.attn
            with torch.no_grad():
                if is_register:
                    qkv = attn_mod.qkv(x_norm).reshape(B, N_total, 3, n_h, d_h)
                    q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                    if attn_mod.use_qk_norm:
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    attn_w = F.softmax(
                        q.float() @ k.float().transpose(-2, -1) * scale, dim=-1)
                    capture.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())
                else:
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
                    capture.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())

        return hook


# ── Visualization: alt mode ───────────────────────────────────────────────────

def _save_frame_attn_alt(attn: torch.Tensor, patch_grid: tuple,
                         out_path: str, layer_idx: int):
    H, W = patch_grid
    heatmap = attn.numpy().reshape(H, W)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    im = ax.imshow(heatmap, cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Frame Attention  |  Layer {layer_idx:02d}", fontsize=13)
    ax.set_xlabel("Width patches"); ax.set_ylabel("Height patches")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_inter_attn_alt(attn: torch.Tensor, out_path: str, layer_idx: int):
    mat = attn.numpy()
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)
    S = mat.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Inter-Frame Attention  |  Layer {layer_idx:02d}", fontsize=13)
    ax.set_xlabel("Key frame index"); ax.set_ylabel("Query frame index")
    if S <= 30:
        ax.set_xticks(range(S)); ax.set_yticks(range(S))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Visualization: full mode ──────────────────────────────────────────────────

_LOG_NORM = LogNorm(vmin=1e-4, vmax=1.0)


def _save_frame_attn_full(attn: torch.Tensor, patch_token_start: int,
                           out_path: str, layer_idx: int):
    mat = attn.numpy().clip(1e-6, 1.0)
    N = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(mat, cmap="inferno", aspect="auto", norm=_LOG_NORM,
                   interpolation="nearest")
    ax.set_title(
        f"Frame Attention (1st frame)  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n"
        f"Shape: {N}×{N}  [cam(1) | reg(16) | patch({N - patch_token_start})]",
        fontsize=11)
    ax.set_xlabel("Key token index"); ax.set_ylabel("Query token index")
    for pos in [0, 1, patch_token_start]:
        ax.axvline(x=pos - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
        ax.axhline(y=pos - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_inter_attn_full(attn: torch.Tensor, num_frames: int,
                           patch_token_start: int, attn_type: str,
                           out_path: str, layer_idx: int):
    S, p0, n_reg = num_frames, patch_token_start, patch_token_start - 1
    cam_idx = np.array([s * p0 for s in range(S)])
    reg_idx = np.array([s * p0 + r for s in range(S) for r in range(1, p0)])
    perm = np.concatenate([cam_idx, reg_idx])
    raw = attn.numpy()
    mat = raw[np.ix_(perm, perm)].clip(1e-4, 1.0)
    T = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", norm=_LOG_NORM,
                   interpolation="nearest")
    ax.set_title(
        f"Inter-Frame Attention ({attn_type})  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n"
        f"Shape: {T}×{T}  [cam(×{S}) | reg(×{S}×{n_reg})]",
        fontsize=11)
    ax.set_xlabel("Key token index"); ax.set_ylabel("Query token index")
    ax.axvline(x=S - 0.5, color="cyan", linewidth=1.0, alpha=0.8)
    ax.axhline(y=S - 0.5, color="cyan", linewidth=1.0, alpha=0.8)
    for s in range(1, S):
        pos = S + s * n_reg - 0.5
        ax.axvline(x=pos, color="red", linewidth=0.5, alpha=0.6)
        ax.axhline(y=pos, color="red", linewidth=0.5, alpha=0.6)
    if S <= 60:
        cam_labels = [str(s) for s in range(S)]
        reg_labels = [f"r{s}" for s in range(S)]
        all_ticks = list(range(S)) + [S + s * n_reg for s in range(S)]
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(cam_labels + reg_labels, fontsize=4, rotation=90)
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(cam_labels + reg_labels, fontsize=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VGGT-Omega attention visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True, choices=["alt", "full"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ego4d-root", required=True)
    parser.add_argument("--ego4d-json", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--tmp-dir", default="/tmp/vggt_ego4d_attn_frames")
    parser.add_argument("--clip-uid", default=None,
                        help="Process only this clip UID")
    args = parser.parse_args()

    with open(args.ego4d_json) as f:
        ego4d = json.load(f)
    clip_map = {c["clip_uid"]: c for c in ego4d["clips"]}
    eval_uids = open(args.eval_list).read().strip().split()
    if args.clip_uid is not None:
        eval_uids = [u for u in eval_uids if u == args.clip_uid]
        if not eval_uids:
            raise ValueError(f"--clip-uid {args.clip_uid!r} not found in eval list")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}  [mode={args.mode}]")
    model = load_model(args.checkpoint)

    if args.mode == "alt":
        capture = AltAttnCapture()
        attn_subdir = "attention_maps"
    else:
        capture = FullAttnCapture()
        attn_subdir = "attention_maps2"

    capture.register(model)
    num_layers = model.aggregator.depth
    patch_token_start = model.aggregator.patch_token_start
    patch_size = model.aggregator.patch_size
    capture.patch_token_start = patch_token_start
    inter_attn_types = (model.aggregator.inter_frame_attention_types
                        if args.mode == "full" else None)

    skipped = []

    for i, clip_uid in enumerate(eval_uids):
        attn_dir = os.path.join(args.output_dir, clip_uid, attn_subdir)

        if args.mode == "alt" and os.path.isdir(attn_dir) and len(os.listdir(attn_dir)) >= 48:
            print(f"[{i+1}/{len(eval_uids)}] SKIP (exists): {clip_uid}"); continue

        clip = clip_map.get(clip_uid)
        if clip is None:
            print(f"[{i+1}/{len(eval_uids)}] SKIP (not in ego4d.json): {clip_uid}")
            skipped.append(clip_uid); continue

        video_path = os.path.join(args.ego4d_root, "full_scale", f"{clip['video_uid']}.mp4")
        if not os.path.exists(video_path):
            print(f"[{i+1}/{len(eval_uids)}] SKIP (video missing): {clip_uid}")
            skipped.append(clip_uid); continue

        print(f"[{i+1}/{len(eval_uids)}] Processing: {clip_uid}")
        tmp_dir = os.path.join(args.tmp_dir, clip_uid)
        images_dir = os.path.join(tmp_dir, "images")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        start_sec = clip["video_start_sec"]
        end_sec = min(clip["video_end_sec"], start_sec + args.max_duration)
        n_frames = extract_frames(video_path, images_dir, start_sec, end_sec, args.sample_fps)

        if n_frames == 0:
            print("  SKIP: no frames extracted")
            skipped.append(clip_uid)
            shutil.rmtree(tmp_dir, ignore_errors=True); continue

        print(f"  {n_frames} frames extracted")
        image_names = sorted(glob.glob(os.path.join(images_dir, "*")))
        images = load_and_preprocess_images(
            image_names, image_resolution=args.image_resolution).to("cuda")

        H_patches = images.shape[-2] // patch_size
        W_patches = images.shape[-1] // patch_size
        patch_grid = (H_patches, W_patches)
        capture.num_frames = images.shape[0]
        if args.mode == "alt":
            capture.patch_grid = patch_grid
        capture.reset()

        try:
            with torch.inference_mode():
                _ = model(images)
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(clip_uid)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect(); torch.cuda.empty_cache(); continue

        if len(capture.frame_attn) != num_layers or len(capture.inter_attn) != num_layers:
            print(f"  WARNING: captured {len(capture.frame_attn)} frame / "
                  f"{len(capture.inter_attn)} inter maps (expected {num_layers})")

        if args.mode == "full":
            print("  [diag] Frame attention stats per layer:")
            for li, fa in enumerate(capture.frame_attn):
                mat = fa.numpy()
                row_sums = mat.sum(axis=1)
                pct_nonzero = (mat > 0.001).mean() * 100
                print(f"    layer {li:02d}: min={mat.min():.6f}  max={mat.max():.6f}"
                      f"  mean={mat.mean():.6f}"
                      f"  row_sum=[{row_sums.min():.4f},{row_sums.max():.4f}]"
                      f"  >0.001: {pct_nonzero:.2f}%")

        os.makedirs(attn_dir, exist_ok=True)
        S = capture.num_frames

        for layer_idx in range(num_layers):
            frame_out = os.path.join(attn_dir, f"frame_attn_{layer_idx:02d}.png")
            inter_out = os.path.join(attn_dir, f"inter_frame_attn_{layer_idx:02d}.png")

            if layer_idx < len(capture.frame_attn):
                if args.mode == "alt":
                    _save_frame_attn_alt(capture.frame_attn[layer_idx],
                                         patch_grid, frame_out, layer_idx)
                else:
                    _save_frame_attn_full(capture.frame_attn[layer_idx],
                                          patch_token_start, frame_out, layer_idx)

            if layer_idx < len(capture.inter_attn):
                if args.mode == "alt":
                    _save_inter_attn_alt(capture.inter_attn[layer_idx],
                                         inter_out, layer_idx)
                else:
                    _save_inter_attn_full(capture.inter_attn[layer_idx], S,
                                          patch_token_start,
                                          inter_attn_types[layer_idx],
                                          inter_out, layer_idx)

        print(f"  Saved 48 maps → {attn_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect(); torch.cuda.empty_cache()

    capture.remove()
    print(f"\nDone. Skipped {len(skipped)}/{len(eval_uids)} sequences:")
    for uid in skipped:
        print(f"  {uid}")


if __name__ == "__main__":
    main()
