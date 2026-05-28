#!/usr/bin/env python3
"""
Visualize full attention matrices from VGGT-Omega Alt.Attn blocks.

Saves per layer (24 layers × 2 types = 48 images) into attention_maps2/:

  frame_attn_{layer:02d}.png
    Shape: (N_tokens × N_tokens) = (patch_token_start + N_patch)²
    First frame only, head-averaged full self-attention matrix.
    Token order: [camera | register×16 | patches×N_patch]

  inter_frame_attn_{layer:02d}.png
    Global layers  → (S×17) × (S×17)  camera+register token attention
                     (full S×N_total matrix is O(tens of GB); camera+register subset used)
    Register layers → (S×17) × (S×17)  full register-type attention matrix
"""

import argparse
import gc
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.utils.load_fn import load_and_preprocess_images
from infer_ego4d import extract_frames, load_model


# ── Attention capture ──────────────────────────────────────────────────────

class FullAttnCapture:
    """
    Captures:
      self.frame_attn[layer]  : (N_tokens, N_tokens) float32, first frame, head-avg
      self.inter_attn[layer]  : (S*17, S*17) float32, head-avg
                                 global layers  → camera+register subset
                                 register layers → full register-type attention
    """

    def __init__(self) -> None:
        self._handles: list = []
        self.frame_attn: list[torch.Tensor] = []
        self.inter_attn: list[torch.Tensor] = []

        self.patch_token_start: int = 17   # 1 camera + 16 register
        self.num_frames: int = 1

    def register(self, model: VGGTOmega) -> None:
        agg = model.aggregator
        for i, block in enumerate(agg.frame_blocks):
            h = block.register_forward_hook(
                self._make_frame_hook(i), with_kwargs=True
            )
            self._handles.append(h)
        for i, block in enumerate(agg.inter_frame_blocks):
            h = block.register_forward_hook(
                self._make_inter_hook(i), with_kwargs=True
            )
            self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.frame_attn.clear()
        self.inter_attn.clear()

    # ── frame hook ──────────────────────────────────────────────────────────

    def _make_frame_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            # args[0]: (B*S, N_tokens, C)  B=1 always
            # args[1]: (rope_sin, rope_cos)
            x = args[0].detach()
            rope = args[1] if len(args) > 1 else None

            # Use only the first frame (index 0)
            x0 = module.norm1(x[0:1])          # (1, N, C)
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

                # Full (N, N) attention, head-averaged
                attn_w = F.softmax(
                    q.float() @ k.float().transpose(-2, -1) * scale, dim=-1
                )  # (1, n_h, N, N)
                attn_mean = attn_w.mean(dim=[0, 1]).cpu()  # (N, N)

            capture.frame_attn.append(attn_mean)

        return hook

    # ── inter-frame hook ────────────────────────────────────────────────────

    def _make_inter_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            # args[0]: (B, N_total, C)  — no rope
            x = args[0].detach()
            B, N_total, C = x.shape
            S = capture.num_frames
            p0 = capture.patch_token_start  # 17
            n_h = module.attn.num_heads
            d_h = C // n_h
            scale = d_h ** -0.5

            # Detect attention type by stride
            # register layers: N_total = S * 17  → stride = 17
            # global layers:   N_total = S * N_tokens (>> 17) → stride = N_tokens
            if N_total % S != 0:
                capture.inter_attn.append(torch.zeros(S * p0, S * p0))
                return
            stride = N_total // S
            is_register = (stride == p0)

            x_norm = module.norm1(x)
            attn_mod = module.attn

            with torch.no_grad():
                if is_register:
                    # Full (S*17, S*17) attention — input already contains only
                    # camera+register tokens
                    qkv = attn_mod.qkv(x_norm).reshape(B, N_total, 3, n_h, d_h)
                    q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                    if attn_mod.use_qk_norm:
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    attn_w = F.softmax(
                        q.float() @ k.float().transpose(-2, -1) * scale, dim=-1
                    )  # (B, n_h, S*17, S*17)
                    attn_mean = attn_w.mean(dim=[0, 1]).cpu()  # (S*17, S*17)

                else:
                    # Global layer: full N_total = S*N_tokens is tens of GB.
                    # Extract camera+register tokens (first 17 tokens per frame).
                    cam_reg_idx = torch.cat([
                        torch.arange(p0, device=x_norm.device) + s * stride
                        for s in range(S)
                    ])  # (S*17,)
                    x_cr = x_norm[:, cam_reg_idx, :]  # (B, S*17, C)
                    T = x_cr.shape[1]
                    qkv = attn_mod.qkv(x_cr).reshape(B, T, 3, n_h, d_h)
                    q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                    if attn_mod.use_qk_norm:
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    attn_w = F.softmax(
                        q.float() @ k.float().transpose(-2, -1) * scale, dim=-1
                    )  # (B, n_h, S*17, S*17)
                    attn_mean = attn_w.mean(dim=[0, 1]).cpu()  # (S*17, S*17)

            capture.inter_attn.append(attn_mean)

        return hook


# ── Visualization ──────────────────────────────────────────────────────────

def _token_tick_labels(n_tokens: int, patch_token_start: int, patch_grid: tuple[int, int]):
    """Return sparse tick positions and labels marking token group boundaries."""
    H, W = patch_grid
    ticks = [0, 1, patch_token_start, n_tokens - 1]
    labels = ["cam", "reg↑", f"patch↑\n(0,0)", f"({H-1},{W-1})"]
    return ticks, labels


_LOG_NORM = LogNorm(vmin=1e-4, vmax=1.0)


def save_frame_attn(
    attn: torch.Tensor,
    patch_token_start: int,
    patch_grid: tuple[int, int],
    out_path: str,
    layer_idx: int,
) -> None:
    mat = attn.numpy().clip(1e-6, 1.0)
    N = mat.shape[0]

    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(mat, cmap="inferno", aspect="auto", norm=_LOG_NORM,
                   interpolation="nearest")
    ax.set_title(f"Frame Attention (1st frame)  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n"
                 f"Shape: {N}×{N}  [cam(1) | reg(16) | patch({N-patch_token_start})]",
                 fontsize=11)
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")

    # Mark token group boundaries
    for pos in [0, 1, patch_token_start]:
        ax.axvline(x=pos - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
        ax.axhline(y=pos - 0.5, color="cyan", linewidth=0.6, alpha=0.7)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_inter_attn(
    attn: torch.Tensor,
    num_frames: int,
    patch_token_start: int,
    attn_type: str,
    out_path: str,
    layer_idx: int,
) -> None:
    S = num_frames
    p0 = patch_token_start  # 17 (1 cam + 16 reg)
    n_reg = p0 - 1          # 16

    # Reorder: [cam0,reg0×16, cam1,reg1×16, ...] → [cam0..camS-1, reg0×16..regS-1×16]
    cam_idx = np.array([s * p0 for s in range(S)])
    reg_idx = np.array([s * p0 + r for s in range(S) for r in range(1, p0)])
    perm = np.concatenate([cam_idx, reg_idx])

    raw = attn.numpy()
    mat = raw[np.ix_(perm, perm)].clip(1e-4, 1.0)
    T = mat.shape[0]   # S*17

    fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", norm=_LOG_NORM,
                   interpolation="nearest")
    ax.set_title(
        f"Inter-Frame Attention ({attn_type})  |  Layer {layer_idx:02d}  [log scale 1e-4~1]\n"
        f"Shape: {T}×{T}  [cam(×{S}) | reg(×{S}×{n_reg})]",
        fontsize=11,
    )
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")

    # Boundary between cam section and register section
    ax.axvline(x=S - 0.5, color="cyan", linewidth=1.0, alpha=0.8)
    ax.axhline(y=S - 0.5, color="cyan", linewidth=1.0, alpha=0.8)

    # Frame boundaries within register section
    for s in range(1, S):
        pos = S + s * n_reg - 0.5
        ax.axvline(x=pos, color="red", linewidth=0.5, alpha=0.6)
        ax.axhline(y=pos, color="red", linewidth=0.5, alpha=0.6)

    # Ticks: cam section (one per frame), register section (one per frame group)
    if S <= 60:
        cam_ticks = list(range(S))
        reg_ticks = [S + s * n_reg for s in range(S)]
        all_ticks = cam_ticks + reg_ticks
        cam_labels = [str(s) for s in range(S)]
        reg_labels = [f"r{s}" for s in range(S)]
        all_labels = cam_labels + reg_labels
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=4, rotation=90)
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(all_labels, fontsize=4)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full attention matrix visualization for VGGT-Omega"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ego4d-root", required=True)
    parser.add_argument("--ego4d-json", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--tmp-dir", default="/tmp/vggt_ego4d_attn2_frames")
    parser.add_argument("--clip-uid", default=None, help="Process only this clip UID")
    args = parser.parse_args()

    with open(args.ego4d_json) as f:
        ego4d = json.load(f)
    clip_map = {c["clip_uid"]: c for c in ego4d["clips"]}
    eval_uids = open(args.eval_list).read().strip().split()
    if args.clip_uid is not None:
        eval_uids = [uid for uid in eval_uids if uid == args.clip_uid]
        if not eval_uids:
            raise ValueError(f"--clip-uid {args.clip_uid!r} not found in eval list")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)

    capture = FullAttnCapture()
    capture.register(model)
    num_layers = model.aggregator.depth          # 24
    patch_token_start = model.aggregator.patch_token_start  # 17
    patch_size = model.aggregator.patch_size     # 16
    capture.patch_token_start = patch_token_start

    # Register attention type per layer
    inter_attn_types = model.aggregator.inter_frame_attention_types  # list of 24 strings

    skipped: list[str] = []

    for i, clip_uid in enumerate(eval_uids):
        attn_dir = os.path.join(args.output_dir, clip_uid, "attention_maps2")

        clip = clip_map.get(clip_uid)
        if clip is None:
            print(f"[{i+1}/{len(eval_uids)}] SKIP (not in ego4d.json): {clip_uid}")
            skipped.append(clip_uid)
            continue

        video_uid = clip["video_uid"]
        video_path = os.path.join(args.ego4d_root, "full_scale", f"{video_uid}.mp4")
        if not os.path.exists(video_path):
            print(f"[{i+1}/{len(eval_uids)}] SKIP (video missing): {clip_uid}")
            skipped.append(clip_uid)
            continue

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
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        print(f"  {n_frames} frames extracted")

        image_names = sorted(glob.glob(os.path.join(images_dir, "*")))
        images = load_and_preprocess_images(
            image_names, image_resolution=args.image_resolution
        ).to("cuda")

        H_img, W_img = images.shape[-2], images.shape[-1]
        H_patches = H_img // patch_size
        W_patches = W_img // patch_size
        patch_grid = (H_patches, W_patches)

        capture.num_frames = images.shape[0]   # S
        capture.reset()

        try:
            with torch.inference_mode():
                _ = model(images)
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(clip_uid)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if len(capture.frame_attn) != num_layers or len(capture.inter_attn) != num_layers:
            print(
                f"  WARNING: captured {len(capture.frame_attn)} frame / "
                f"{len(capture.inter_attn)} inter maps (expected {num_layers})"
            )

        # ── Attention distribution diagnostics ──────────────────────────────
        print("  [diag] Frame attention stats per layer:")
        for li, fa in enumerate(capture.frame_attn):
            mat = fa.numpy()
            row_sums = mat.sum(axis=1)
            pct_nonzero = (mat > 0.001).mean() * 100
            print(
                f"    layer {li:02d}: min={mat.min():.6f}  max={mat.max():.6f}"
                f"  mean={mat.mean():.6f}"
                f"  row_sum=[{row_sums.min():.4f},{row_sums.max():.4f}]"
                f"  >0.001: {pct_nonzero:.2f}%"
            )
        # ────────────────────────────────────────────────────────────────────

        os.makedirs(attn_dir, exist_ok=True)
        S = capture.num_frames

        for layer_idx in range(num_layers):
            if layer_idx < len(capture.frame_attn):
                save_frame_attn(
                    capture.frame_attn[layer_idx],
                    patch_token_start,
                    patch_grid,
                    os.path.join(attn_dir, f"frame_attn_{layer_idx:02d}.png"),
                    layer_idx,
                )

            if layer_idx < len(capture.inter_attn):
                atype = inter_attn_types[layer_idx]
                label = atype  # "global" or "register"
                save_inter_attn(
                    capture.inter_attn[layer_idx],
                    S,
                    patch_token_start,
                    label,
                    os.path.join(attn_dir, f"inter_frame_attn_{layer_idx:02d}.png"),
                    layer_idx,
                )

        print(f"  Saved 48 maps → {attn_dir}")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()

    capture.remove()

    print(f"\nDone. Skipped {len(skipped)}/{len(eval_uids)} sequences:")
    for uid in skipped:
        print(f"  {uid}")


if __name__ == "__main__":
    main()
