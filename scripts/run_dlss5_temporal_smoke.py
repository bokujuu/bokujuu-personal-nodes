from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(COMFYUI_ROOT / "custom_nodes"))

DEPTH = importlib.import_module("bokujuu-personal-nodes.dlss5.depth")
ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
PERSONAL_NODES = importlib.import_module("bokujuu-personal-nodes.nodes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short DA3 + DLSS-SR temporal smoke test.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start_seconds * fps)))
    frames = []
    for _ in range(args.frames):
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb.copy()).float().div(255.0))
    capture.release()
    if len(frames) != args.frames:
        raise RuntimeError(f"Requested {args.frames} frames but decoded {len(frames)}")
    images = torch.stack(frames)

    started = time.perf_counter()
    da3_model = PERSONAL_NODES.BokujuuLoadDepthAnything3.execute(
        PERSONAL_NODES.DA3_REPO_ID, "fp16"
    ).result[0]
    guidance = DEPTH.estimate_da3_guidance(
        da3_model,
        images,
        resolution=504,
        resize_method="upper_bound_resize",
        normalization="v2_style",
    )
    depth_seconds = time.perf_counter() - started
    depth_images = DEPTH.render_guidance_depth(guidance)

    import comfy.model_management as model_management

    model_management.unload_all_models()
    started = time.perf_counter()
    output = ENGINE.DLSS5Engine().enhance_sequence(
        images,
        guidance,
        scale=args.scale,
        quality="auto",
        output_mix=1.0,
        scene_cut_threshold=0.35,
    )
    dlss_seconds = time.perf_counter() - started

    target_height, target_width = output.shape[1:3]
    writer = cv2.VideoWriter(
        str(args.output_dir / "temporal_dlss_sr.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width, target_height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create temporal smoke-test MP4")
    for index, frame in enumerate(output):
        rgb = frame.mul(255).round().byte().cpu().numpy()
        Image.fromarray(rgb).save(args.output_dir / f"dlss_frame_{index:02d}.png")
        depth = depth_images[index].mul(255).round().byte().cpu().numpy()
        Image.fromarray(depth).save(args.output_dir / f"depth_frame_{index:02d}.png")
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    summary = {
        "input": str(args.input.resolve()),
        "frame_count": len(frames),
        "input_size": [args.width, args.height],
        "output_size": [target_width, target_height],
        "fps": fps,
        "depth_anything_v3_seconds": depth_seconds,
        "dlss_sr_temporal_seconds": dlss_seconds,
        "finite": bool(torch.isfinite(output).all()),
        "range": [float(output.min()), float(output.max())],
        "temporal_note": "One DLSS session was reused for all frames; reset is sent for frame 0 and detected scene cuts. Motion vectors are zero because ordinary RGB video has no engine vectors.",
    }
    (args.output_dir / "temporal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
