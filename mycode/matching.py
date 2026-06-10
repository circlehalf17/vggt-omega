#!/usr/bin/env python3
"""
Cross-frame attention visualization for VGGT-Omega's global inter_frame layers.

For each frame pair (i, j) and a fixed query patch in frame i (and j):
  i→j : attention from frame-i's query patch → all image patches in frame j
  j→i : attention from frame-j's query patch → all image patches in frame i

Efficiency: only the Q row of the query token and the full K matrix are computed
per layer. The full (N·T)² attention matrix is never materialized.

Output per (pair, layer, head, direction):
  pair<i>-<j>_layer<ll>_head<h>_i2j.png  — 2-panel: source (query box) | target (attn overlay)
  pair<i>-<j>_layer<ll>_head<h>_j2i.png
  config.json                             — all args + derived geometry
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile
import types
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


# ── Constants (mirror infer_selected.py) ─────────────────────────────────────
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
OUTPUT_BASE = '/workspace/outputs/renders/vggt-omega/ego4d/matching'
CHECKPOINT  = '/workspace/outputs/checkpoints/vggt-omega/vggt_omega_1b_512.pt'
IMAGE_RES   = 512
SAMPLE_FPS  = 6.0
PATCH_SIZE  = 16
PATCH_TOKEN_START = 17          # 1 camera + 16 register tokens per frame
REGISTER_ATTN_LAYERS = frozenset([2, 6, 9, 14, 20])  # excluded from global


def clip_out_name(uid: str, start: int, end: int, role: str) -> str:
    return f'{uid}_{start}-{end}s{"_"+role if role else ""}'


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> VGGTOmega:
    """Load aggregator-only model (skip depth/camera heads to save memory)."""
    model = VGGTOmega(enable_depth=False, enable_camera=False).eval()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Heads present in ckpt but not in this model → expected as "unexpected keys"
    real_missing = [k for k in missing
                    if not any(k.startswith(h) for h in ('dense_head', 'camera_head',
                                                          'text_alignment_head'))]
    if real_missing:
        print(f'  WARN: truly missing keys: {real_missing[:5]}')
    return model.to('cuda')


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, out_dir: str,
                   start_sec: float, end_sec: float, fps: float) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start_sec), '-i', video_path,
        '-t', str(end_sec - start_sec),
        '-vf', f'fps={fps}', '-q:v', '2',
        os.path.join(out_dir, 'frame_%06d.jpg'),
    ], check=True)
    return sorted(glob.glob(os.path.join(out_dir, 'frame_*.jpg')))


# ── Q/K capture context manager ───────────────────────────────────────────────

@contextmanager
def capture_qk(model: VGGTOmega, layer_indices: List[int]):
    """
    For each layer in layer_indices, temporarily replace compute_attention in the
    corresponding inter_frame_block's SelfAttention to capture Q and K tensors.

    Yields a dict  {layer_idx: {'q': Tensor, 'k': Tensor, 'scale': float}}
    where Q/K are float32 CPU tensors of shape [1, num_heads, N_total, head_dim].

    The actual SDPA still runs on GPU so model output is unaffected.
    Originals are restored on exit.
    """
    store: Dict[int, Dict] = {}
    originals: Dict[int, object] = {}

    def make_capturing_fn(li: int):
        def capturing_compute_attention(self_attn, qkv: torch.Tensor,
                                        attn_bias=None, rope=None) -> torch.Tensor:
            B, N, _ = qkv.shape
            C = self_attn.qkv.in_features
            qkv_r = qkv.reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads)
            q, k, v = torch.unbind(qkv_r, 2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            if self_attn.use_qk_norm:
                q = self_attn.q_norm(q)
                k = self_attn.k_norm(k)
            if rope is not None:
                q, k = self_attn.apply_rope(q, k, rope)
            # Store Q, K on CPU as float32 (avoids holding GPU memory)
            store[li] = {
                'q': q.detach().float().cpu(),   # [B, H, N_total, head_dim]
                'k': k.detach().float().cpu(),
                'scale': self_attn.scale,
            }
            # Use SDPA for the actual output (no change to model behaviour)
            x = F.scaled_dot_product_attention(q, k, v)
            return x.transpose(1, 2).reshape(B, N, C)
        return capturing_compute_attention

    for li in layer_indices:
        attn_mod = model.aggregator.inter_frame_blocks[li].attn
        originals[li] = attn_mod.compute_attention
        attn_mod.compute_attention = types.MethodType(make_capturing_fn(li), attn_mod)

    try:
        yield store
    finally:
        for li in layer_indices:
            model.aggregator.inter_frame_blocks[li].attn.compute_attention = originals[li]


# ── Attention computation ─────────────────────────────────────────────────────

def compute_pair_attn(
    q: torch.Tensor,        # [1, num_heads, N_total, head_dim]
    k: torch.Tensor,
    scale: float,
    src_frame: int,
    src_patch_idx: int,     # linear patch index within src frame
    tgt_frame: int,
    num_tokens: int,         # tokens per frame = PATCH_TOKEN_START + n_patches
    patch_token_start: int,
    head_idx: int,
) -> np.ndarray:
    """
    Returns attention weights for one query patch in src_frame over all image
    patches in tgt_frame.

    Uses full-row softmax (over ALL keys in the sequence) so that magnitudes are
    comparable between i→j and j→i directions for the same query.

    Returns: float32 ndarray of shape [n_patches_tgt]
    """
    q_global  = src_frame * num_tokens + patch_token_start + src_patch_idx
    k_tgt_start = tgt_frame * num_tokens + patch_token_start
    k_tgt_end   = (tgt_frame + 1) * num_tokens

    q_row = q[0, head_idx, q_global, :]   # [head_dim]
    k_all = k[0, head_idx]                 # [N_total, head_dim]

    logits   = (q_row @ k_all.T) * scale  # [N_total]  — proper full-row softmax
    attn_all = torch.softmax(logits.float(), dim=-1)
    return attn_all[k_tgt_start:k_tgt_end].numpy()  # [n_patches_tgt]


# ── Auto query selection (texture/edge saliency) ─────────────────────────────

def select_query_patch(frame_bgr: np.ndarray, patch_h: int, patch_w: int,
                       patch_size: int, margin: int = 2) -> Tuple[int, int]:
    """
    Return the patch (row, col) with the highest texture saliency.

    Uses the sum of Shi-Tomasi corner minimum-eigenvalue responses inside each
    patch.  Border patches within `margin` patches of the edge are excluded so
    the query stays away from hard borders where attention is unreliable.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Corner response map — same pixel resolution as the image
    corner_resp = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)  # [H, W]

    # Sum response within each patch cell
    ch, cw = patch_h * patch_size, patch_w * patch_size
    resp_crop = corner_resp[:ch, :cw]
    score_map = (resp_crop
                 .reshape(patch_h, patch_size, patch_w, patch_size)
                 .sum(axis=(1, 3)))   # [patch_h, patch_w]

    # Zero out border
    if margin > 0:
        score_map[:margin, :]  = 0
        score_map[-margin:, :] = 0
        score_map[:, :margin]  = 0
        score_map[:, -margin:] = 0

    idx = int(np.argmax(score_map))
    return divmod(idx, patch_w)          # (row, col)


# ── Visualization helpers ─────────────────────────────────────────────────────

_JET = plt.get_cmap('jet')


def attn_overlay(frame_bgr: np.ndarray,
                 attn_flat: np.ndarray,   # [n_patches] float32
                 patch_h: int, patch_w: int,
                 alpha: float = 0.55) -> np.ndarray:
    """Bilinear-upsample attn map onto frame_bgr with jet colormap."""
    img_h, img_w = frame_bgr.shape[:2]
    attn_2d = attn_flat.reshape(patch_h, patch_w).astype(np.float32)

    # Normalise to [0, 1] for colormap
    a_min, a_max = attn_2d.min(), attn_2d.max()
    norm = (attn_2d - a_min) / (a_max - a_min + 1e-10)

    # Upsample
    t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(img_h, img_w), mode='bilinear',
                       align_corners=False)[0, 0].numpy()

    heat_rgb = (_JET(up)[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)

    frame_f = frame_bgr.astype(np.float32)
    heat_f  = heat_bgr.astype(np.float32)
    return np.clip(frame_f * (1 - alpha) + heat_f * alpha, 0, 255).astype(np.uint8)


def draw_query_box(frame_bgr: np.ndarray,
                   patch_r: int, patch_c: int,
                   patch_h: int, patch_w: int) -> np.ndarray:
    """Highlight the query patch with a red rectangle."""
    out = frame_bgr.copy()
    ih, iw = out.shape[:2]
    y0 = int(patch_r * ih / patch_h)
    x0 = int(patch_c * iw / patch_w)
    y1 = int((patch_r + 1) * ih / patch_h)
    x1 = int((patch_c + 1) * iw / patch_w)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.circle(out, ((x0 + x1) // 2, (y0 + y1) // 2), 3, (0, 0, 255), -1)
    return out


def make_panel(src_frame: np.ndarray, tgt_frame: np.ndarray,
               attn_flat: np.ndarray, patch_h: int, patch_w: int,
               qr: int, qc: int, alpha: float,
               label_src: str, label_tgt: str) -> np.ndarray:
    """Two-column panel: [src with query box] | [tgt with attention overlay]."""
    src_vis = draw_query_box(src_frame, qr, qc, patch_h, patch_w)
    tgt_vis = attn_overlay(tgt_frame, attn_flat, patch_h, patch_w, alpha)

    for img, lbl in ((src_vis, label_src), (tgt_vis, label_tgt)):
        cv2.putText(img, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return np.concatenate([src_vis, tgt_vis], axis=1)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='VGGT-Omega global-attention cross-frame visualization'
    )
    p.add_argument('--clip_idx', type=int, default=8,
                   help='Index into CLIPS (0-11). Default: 8 = egomotion/04fe8f4d_3240-3254s_control')
    p.add_argument('--pairs', type=str, default='0,10 5,30 15,60',
                   help='Space-separated frame-index pairs, e.g. "3,17 8,42"')
    p.add_argument('--query', type=str, default=None,
                   help='Query patch for frame-i: "row,col" in patch grid. '
                        'Default: centre of patch grid')
    p.add_argument('--query_j', type=str, default=None,
                   help='Query patch for frame-j in j→i direction. '
                        'Default: same as --query')
    p.add_argument('--head', type=int, default=9,
                   help='Attention head index (0-indexed). Default: 9')
    p.add_argument('--layers', type=str, default=None,
                   help='Comma-separated global layer indices to render. '
                        'Default: all global layers (0-23 excluding register layers 2,6,9,14,20)')
    p.add_argument('--alpha', type=float, default=0.55,
                   help='Attention overlay opacity. Default: 0.55')
    p.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    p.add_argument('--video_root', type=str, default=VIDEO_ROOT)
    p.add_argument('--output_dir', type=str, default=OUTPUT_BASE)
    p.add_argument('--fps', type=float, default=SAMPLE_FPS)
    p.add_argument('--image_resolution', type=int, default=IMAGE_RES)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Clip ─────────────────────────────────────────────────────────────────
    if not (0 <= args.clip_idx < len(CLIPS)):
        raise ValueError(f'clip_idx must be 0-{len(CLIPS)-1}')
    clip_type, uid, t_start, t_end, role = CLIPS[args.clip_idx]
    clip_name = clip_out_name(uid, t_start, t_end, role)
    print(f'Clip [{args.clip_idx}]: {clip_type}/{clip_name}')

    video_path = os.path.join(args.video_root, uid + '.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')

    # ── Frame pairs ───────────────────────────────────────────────────────────
    pairs: List[Tuple[int, int]] = []
    for tok in args.pairs.strip().split():
        a, b = tok.split(',')
        pairs.append((int(a), int(b)))

    # ── Layers ────────────────────────────────────────────────────────────────
    all_global = [i for i in range(24) if i not in REGISTER_ATTN_LAYERS]
    if args.layers:
        global_layers = [int(x) for x in args.layers.split(',')]
        bad = [l for l in global_layers if l in REGISTER_ATTN_LAYERS]
        if bad:
            raise ValueError(f'Layer(s) {bad} are register-attention type, not global. '
                             f'Choose from: {all_global}')
        bad2 = [l for l in global_layers if not (0 <= l < 24)]
        if bad2:
            raise ValueError(f'Layer indices out of range: {bad2}')
    else:
        global_layers = all_global
    print(f'Global layers ({len(global_layers)}): {global_layers}')

    # ── Extract frames ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f'Extracting frames at {args.fps} fps …')
        frame_paths = extract_frames(video_path, tmp_dir, t_start, t_end, args.fps)
        n_frames = len(frame_paths)
        print(f'  {n_frames} frames extracted')

        for fi, fj in pairs:
            if max(fi, fj) >= n_frames:
                raise ValueError(f'Pair ({fi},{fj}) out of range — clip has {n_frames} frames')

        # ── Preprocess ────────────────────────────────────────────────────────
        print('Preprocessing images …')
        images = load_and_preprocess_images(
            frame_paths, image_resolution=args.image_resolution
        ).unsqueeze(0).to('cuda')   # [1, S, 3, H, W]

        _, S, _, img_h, img_w = images.shape
        patch_h      = img_h // PATCH_SIZE
        patch_w      = img_w // PATCH_SIZE
        n_patches    = patch_h * patch_w
        num_tokens   = PATCH_TOKEN_START + n_patches
        print(f'  {img_w}×{img_h} px  |  patch grid {patch_w}×{patch_h}={n_patches}  '
              f'|  {num_tokens} tokens/frame  |  N_total={S * num_tokens}')

        # ── Query mode ────────────────────────────────────────────────────────
        # Fixed query overrides auto-selection for both directions.
        fixed_qi = tuple(int(x) for x in args.query.split(','))   if args.query   else None
        fixed_qj = tuple(int(x) for x in args.query_j.split(',')) if args.query_j else None

        # Per-frame saliency cache  frame_idx → (qr, qc)
        _auto_query_cache: Dict[int, Tuple[int, int]] = {}

        def auto_query(frame_idx: int) -> Tuple[int, int]:
            if frame_idx not in _auto_query_cache:
                qr, qc = select_query_patch(
                    orig_frames[frame_idx], patch_h, patch_w, PATCH_SIZE)
                _auto_query_cache[frame_idx] = (qr, qc)
                print(f'  auto-query f{frame_idx}: patch ({qr},{qc})')
            return _auto_query_cache[frame_idx]

        auto_mode = fixed_qi is None
        query_tag = f'qi{fixed_qi[0]}-{fixed_qi[1]}' if fixed_qi else 'autoq'
        print(f'  query mode: {"fixed "+str(fixed_qi) if fixed_qi else "auto (texture saliency per frame)"}')
        print(f'  head: {args.head}')

        # ── Load original frames for visualization (resized to model input size)
        orig_frames = []
        for p in frame_paths:
            f = cv2.imread(p)
            orig_frames.append(cv2.resize(f, (img_w, img_h)))

        # ── Load model ────────────────────────────────────────────────────────
        print(f'\nLoading model from {args.checkpoint} …')
        model = load_model(args.checkpoint)
        print('Model loaded.\n')

        # ── Single forward pass with Q/K capture ──────────────────────────────
        print(f'Running forward with Q/K capture on {len(global_layers)} layers …')
        with capture_qk(model, global_layers) as qk_store:
            with torch.inference_mode():
                model(images)
        torch.cuda.empty_cache()
        print(f'  Captured Q/K for {len(qk_store)} layers.\n')

        # ── Output dir ────────────────────────────────────────────────────────
        out_dir = os.path.join(args.output_dir, clip_name, f'head{args.head}_{query_tag}')
        os.makedirs(out_dir, exist_ok=True)

        # ── Save config ───────────────────────────────────────────────────────
        config = {
            'clip_idx': args.clip_idx,
            'clip_name': clip_name,
            'clip_type': clip_type,
            'pairs': [[fi, fj] for fi, fj in pairs],
            'query_mode': 'fixed' if fixed_qi else 'auto_texture',
            'fixed_query_i': list(fixed_qi) if fixed_qi else None,
            'fixed_query_j': list(fixed_qj) if fixed_qj else None,
            'head': args.head,
            'global_layers': global_layers,
            'register_attn_layers': sorted(REGISTER_ATTN_LAYERS),
            'image_size_wh': [img_w, img_h],
            'patch_grid_wh': [patch_w, patch_h],
            'n_patches': n_patches,
            'num_tokens_per_frame': num_tokens,
            'total_sequence_len': S * num_tokens,
            'alpha': args.alpha,
        }
        cfg_path = os.path.join(out_dir, 'config.json')
        with open(cfg_path, 'w') as fh:
            json.dump(config, fh, indent=2)
        print(f'Config → {cfg_path}\n')

        # ── Render ────────────────────────────────────────────────────────────
        n_saved = 0
        for fi, fj in pairs:
            # Resolve per-pair query patches
            qr_i, qc_i = fixed_qi if fixed_qi else auto_query(fi)
            qr_j, qc_j = fixed_qj if fixed_qj else auto_query(fj)
            q_patch_i = qr_i * patch_w + qc_i
            q_patch_j = qr_j * patch_w + qc_j

            for li in global_layers:
                if li not in qk_store:
                    print(f'  WARN: no Q/K captured for layer {li}, skipping')
                    continue

                qk = qk_store[li]
                q_t, k_t, sc = qk['q'], qk['k'], qk['scale']

                # i → j
                attn_i2j = compute_pair_attn(
                    q_t, k_t, sc,
                    src_frame=fi, src_patch_idx=q_patch_i,
                    tgt_frame=fj,
                    num_tokens=num_tokens,
                    patch_token_start=PATCH_TOKEN_START,
                    head_idx=args.head,
                )
                panel_i2j = make_panel(
                    orig_frames[fi], orig_frames[fj],
                    attn_i2j, patch_h, patch_w, qr_i, qc_i, args.alpha,
                    label_src=f'f{fi} query ({qr_i},{qc_i})',
                    label_tgt=f'f{fj}  i→j  layer{li:02d} head{args.head}',
                )
                fname_i2j = os.path.join(
                    out_dir, f'pair{fi}-{fj}_layer{li:02d}_head{args.head}_i2j.png'
                )
                cv2.imwrite(fname_i2j, panel_i2j)

                # j → i
                attn_j2i = compute_pair_attn(
                    q_t, k_t, sc,
                    src_frame=fj, src_patch_idx=q_patch_j,
                    tgt_frame=fi,
                    num_tokens=num_tokens,
                    patch_token_start=PATCH_TOKEN_START,
                    head_idx=args.head,
                )
                panel_j2i = make_panel(
                    orig_frames[fj], orig_frames[fi],
                    attn_j2i, patch_h, patch_w, qr_j, qc_j, args.alpha,
                    label_src=f'f{fj} query ({qr_j},{qc_j})',
                    label_tgt=f'f{fi}  j→i  layer{li:02d} head{args.head}',
                )
                fname_j2i = os.path.join(
                    out_dir, f'pair{fi}-{fj}_layer{li:02d}_head{args.head}_j2i.png'
                )
                cv2.imwrite(fname_j2i, panel_j2i)

                n_saved += 2

            print(f'  pair ({fi},{fj}): {len(global_layers)*2} images saved')

    print(f'\nDone — {n_saved} images → {out_dir}')


if __name__ == '__main__':
    main()
