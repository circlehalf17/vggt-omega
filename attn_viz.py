#!/usr/bin/env python3
"""
Visualize Alt.Attn attention maps from VGGT-Omega.

For each of 24 transformer layers, saves:
  frame_attn_{layer:02d}.png      — intra-frame spatial attention (H×W heatmap)
  inter_frame_attn_{layer:02d}.png — inter-frame camera-token attention (S×S matrix)

Output directory per clip:
  <output-dir>/<clip_uid>/attention_maps/   (48 PNGs)
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
import numpy as np
import torch
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.utils.load_fn import load_and_preprocess_images
from infer_ego4d import extract_frames, load_model


# ── Attention capture via forward hooks ────────────────────────────────────

class AltAttnCapture:
    """
    Registers forward hooks on every frame_block and inter_frame_block in the
    Aggregator.  After one forward pass the captured maps are available in
    self.frame_attn  (list of 24 tensors, each (N_patch,))
    self.inter_attn  (list of 24 tensors, each (S, S))
    """

    def __init__(self) -> None:
        self._handles: list = []
        self.frame_attn: list[torch.Tensor] = []
        self.inter_attn: list[torch.Tensor] = []

        # Set before each forward pass
        self.patch_token_start: int = 17
        self.num_frames: int = 1
        self.patch_grid: tuple[int, int] = (28, 37)

    # ── public ──────────────────────────────────────────────────────────────

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

    # ── hooks ────────────────────────────────────────────────────────────────

    def _make_frame_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            # args[0]: (B*S, N, C) pre-LayerNorm tokens
            # args[1]: (rope_sin, rope_cos) or None
            x = args[0].detach()
            rope = args[1] if len(args) > 1 else None

            x_norm = module.norm1(x)
            attn_mod = module.attn
            B_S, N, C = x_norm.shape
            n_h = attn_mod.num_heads
            d_h = C // n_h
            p0 = capture.patch_token_start
            scale = d_h ** -0.5

            per_frame: list[torch.Tensor] = []
            with torch.no_grad():
                for b in range(B_S):
                    xb = x_norm[b : b + 1]  # (1, N, C)
                    qkv = attn_mod.qkv(xb).reshape(1, N, 3, n_h, d_h)
                    q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                    if attn_mod.use_qk_norm:
                        q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                    if rope is not None:
                        q, k = attn_mod.apply_rope(q, k, rope)

                    # patch queries × all keys → (1, n_h, N_patch, N)
                    attn_w = F.softmax(
                        q[:, :, p0:, :].float() @ k.float().transpose(-2, -1) * scale,
                        dim=-1,
                    )
                    # mean over heads and query positions → (N,) → keep patch portion
                    per_frame.append(attn_w.mean(dim=[0, 1, 2])[p0:].cpu())

            capture.frame_attn.append(torch.stack(per_frame).mean(0))

        return hook

    def _make_inter_hook(self, layer_idx: int):
        capture = self

        def hook(module: SelfAttentionBlock, args, kwargs, output):
            # args[0]: (B, N_total, C)  — no rope for inter-frame blocks
            x = args[0].detach()
            x_norm = module.norm1(x)
            attn_mod = module.attn
            B, N_total, C = x_norm.shape
            n_h = attn_mod.num_heads
            d_h = C // n_h
            S = capture.num_frames
            scale = d_h ** -0.5

            if N_total % S != 0:
                # Unexpected layout — store zeros and skip
                capture.inter_attn.append(torch.zeros(S, S))
                return

            # Camera token is first token in each frame's stride
            stride = N_total // S
            cam_idx = torch.arange(S, device=x_norm.device) * stride

            with torch.no_grad():
                x_cam = x_norm[:, cam_idx, :]  # (B, S, C)
                qkv = attn_mod.qkv(x_cam).reshape(B, S, 3, n_h, d_h)
                q, k, _ = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
                if attn_mod.use_qk_norm:
                    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

                attn_w = F.softmax(
                    q.float() @ k.float().transpose(-2, -1) * scale, dim=-1
                )  # (B, n_h, S, S)
                capture.inter_attn.append(attn_w.mean(dim=[0, 1]).cpu())

        return hook


# ── Visualization ──────────────────────────────────────────────────────────

def save_frame_attn(
    attn: torch.Tensor,
    patch_grid: tuple[int, int],
    out_path: str,
    layer_idx: int,
) -> None:
    H, W = patch_grid
    heatmap = attn.numpy().reshape(H, W)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    im = ax.imshow(heatmap, cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Frame Attention  |  Layer {layer_idx:02d}", fontsize=13)
    ax.set_xlabel("Width patches")
    ax.set_ylabel("Height patches")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_inter_frame_attn(
    attn: torch.Tensor,
    out_path: str,
    layer_idx: int,
) -> None:
    mat = attn.numpy()
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)
    S = mat.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Inter-Frame Attention  |  Layer {layer_idx:02d}", fontsize=13)
    ax.set_xlabel("Key frame index")
    ax.set_ylabel("Query frame index")
    if S <= 30:
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump Alt.Attn attention maps for VGGT-Omega"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ego4d-root", required=True)
    parser.add_argument("--ego4d-json", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--tmp-dir", default="/tmp/vggt_ego4d_attn_frames")
    args = parser.parse_args()

    with open(args.ego4d_json) as f:
        ego4d = json.load(f)
    clip_map = {c["clip_uid"]: c for c in ego4d["clips"]}
    eval_uids = open(args.eval_list).read().strip().split()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)

    capture = AltAttnCapture()
    capture.register(model)
    num_layers = model.aggregator.depth  # 24
    capture.patch_token_start = model.aggregator.patch_token_start  # 17
    patch_size = model.aggregator.patch_size  # 16

    skipped: list[str] = []

    for i, clip_uid in enumerate(eval_uids):
        attn_dir = os.path.join(args.output_dir, clip_uid, "attention_maps")
        if os.path.isdir(attn_dir) and len(os.listdir(attn_dir)) >= 48:
            print(f"[{i+1}/{len(eval_uids)}] SKIP (exists): {clip_uid}")
            continue

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

        # Derive spatial layout for patch grid
        H_img, W_img = images.shape[-2], images.shape[-1]
        H_patches, W_patches = H_img // patch_size, W_img // patch_size

        capture.num_frames = images.shape[0]  # S (no batch dim yet)
        capture.patch_grid = (H_patches, W_patches)
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
                f"  WARNING: got {len(capture.frame_attn)} frame / "
                f"{len(capture.inter_attn)} inter maps (expected {num_layers})"
            )

        os.makedirs(attn_dir, exist_ok=True)
        for layer_idx in range(num_layers):
            if layer_idx < len(capture.frame_attn):
                save_frame_attn(
                    capture.frame_attn[layer_idx],
                    (H_patches, W_patches),
                    os.path.join(attn_dir, f"frame_attn_{layer_idx:02d}.png"),
                    layer_idx,
                )
            if layer_idx < len(capture.inter_attn):
                save_inter_frame_attn(
                    capture.inter_attn[layer_idx],
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
