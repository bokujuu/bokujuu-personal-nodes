from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(COMFYUI_ROOT / "custom_nodes"))

DEPTH = importlib.import_module("bokujuu-personal-nodes.dlss5.depth")
ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
PERSONAL_NODES = importlib.import_module("bokujuu-personal-nodes.nodes")


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).float().div(255.0).unsqueeze(0)


def pil_from_tensor(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).mul(255).round().byte().numpy()
    return Image.fromarray(array)


def sharpness(image: torch.Tensor) -> float:
    image = image[..., :3].float()
    horizontal = (image[:, 1:] - image[:, :-1]).abs().mean()
    vertical = (image[1:] - image[:-1]).abs().mean()
    return float(horizontal + vertical)


def rtx_vsr(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import nvvfx

    quality = nvvfx.effects.QualityLevel.ULTRA
    outputs = []
    with nvvfx.VideoSuperRes(quality) as super_resolution:
        super_resolution.output_width = width
        super_resolution.output_height = height
        super_resolution.load()
        for frame in images:
            cuda_frame = frame[..., :3].cuda().movedim(-1, 0).float().contiguous()
            result = torch.from_dlpack(super_resolution.run(cuda_frame).image)
            outputs.append(result.movedim(0, -1).cpu())
    return torch.stack(outputs)


def esrgan_if_available(images: torch.Tensor, width: int, height: int):
    import folder_paths
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

    candidates = [
        name
        for name in folder_paths.get_filename_list("upscale_models")
        if Path(name).suffix.lower() in {".pth", ".pt", ".safetensors"}
    ]
    if not candidates:
        return None, None
    preferred = next((name for name in candidates if "esrgan" in name.lower()), candidates[0])
    model = UpscaleModelLoader.execute(preferred).result[0]
    output = ImageUpscaleWithModel.execute(model, images[..., :3]).result[0]
    if tuple(output.shape[1:3]) != (height, width):
        resized = [ENGINE._resize_lanczos(frame, width, height) for frame in output]
        output = torch.stack(resized)
    return preferred, output


def make_contact_sheet(outputs: dict[str, Image.Image]) -> Image.Image:
    tile_width, tile_height = 440, 586
    crop_size = 320
    label_height = 32
    names = list(outputs)
    sheet = Image.new("RGB", (tile_width * len(names), tile_height + crop_size + label_height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, name in enumerate(names):
        image = outputs[name]
        preview = image.copy()
        preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x0 = index * tile_width
        sheet.paste(preview, (x0 + (tile_width - preview.width) // 2, label_height))
        draw.text((x0 + 8, 8), name, fill="black")

        crop_width = min(image.width, image.height, crop_size * 2)
        left = max(0, image.width // 2 - crop_width // 2)
        top = max(0, image.height // 3 - crop_width // 2)
        crop = image.crop((left, top, left + crop_width, top + crop_width))
        crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
        crop_y = tile_height + label_height * 2
        sheet.paste(crop, (x0 + (tile_width - crop_size) // 2, crop_y))
        draw.text((x0 + 8, tile_height + label_height), "detail crop", fill="black")
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DLSS-SR, Lanczos, RTX VSR, and an installed ESRGAN model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMFYUI_ROOT / "output" / "bokujuu_dlss5_test",
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--depth-resolution", type=int, default=504)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_pil = Image.open(args.input).convert("RGB")
    width = source_pil.width - source_pil.width % 4
    height = source_pil.height - source_pil.height % 4
    source_pil = source_pil.crop((0, 0, width, height))
    images = tensor_from_pil(source_pil)
    target_width = int(round(width * args.scale))
    target_height = int(round(height * args.scale))

    timings = {}
    started = time.perf_counter()
    da3_model = PERSONAL_NODES.BokujuuLoadDepthAnything3.execute(
        PERSONAL_NODES.DA3_REPO_ID, "fp16"
    ).result[0]
    guidance = DEPTH.estimate_da3_guidance(
        da3_model,
        images,
        resolution=args.depth_resolution,
        resize_method="upper_bound_resize",
        normalization="v2_style",
    )
    timings["depth_anything_v3_seconds"] = time.perf_counter() - started
    pil_from_tensor(DEPTH.render_guidance_depth(guidance)[0]).save(args.output_dir / "depth_da3.png")

    import comfy.model_management as model_management

    model_management.unload_all_models()
    started = time.perf_counter()
    dlss = ENGINE.DLSS5Engine().enhance_image(
        images, guidance, scale=args.scale, quality="auto", output_mix=1.0
    )
    timings["dlss_sr_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    lanczos = torch.stack(
        [ENGINE._resize_lanczos(frame, target_width, target_height) for frame in images]
    )
    timings["lanczos_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rtx = rtx_vsr(images, target_width, target_height)
    timings["rtx_vsr_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    esrgan_name, esrgan = esrgan_if_available(images, target_width, target_height)
    timings["esrgan_seconds"] = time.perf_counter() - started if esrgan is not None else None

    outputs = {
        "Lanczos": pil_from_tensor(lanczos[0]),
        "RTX Video Super Resolution": pil_from_tensor(rtx[0]),
        "DLSS Super Resolution + DA3": pil_from_tensor(dlss[0]),
    }
    tensors = {"lanczos": lanczos[0], "rtx_vsr": rtx[0], "dlss_sr_da3": dlss[0]}
    if esrgan is not None:
        outputs[f"ESRGAN ({esrgan_name})"] = pil_from_tensor(esrgan[0])
        tensors["esrgan"] = esrgan[0]

    for key, image in outputs.items():
        safe_name = key.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        image.save(args.output_dir / f"{safe_name}.png")
    make_contact_sheet(outputs).save(args.output_dir / "comparison_contact_sheet.png")

    metrics = {
        key: {
            "sharpness_gradient": sharpness(value),
            "mae_vs_lanczos": float((value[..., :3] - lanczos[0, ..., :3]).abs().mean()),
        }
        for key, value in tensors.items()
    }
    summary = {
        "input": str(args.input.resolve()),
        "input_size": [width, height],
        "output_size": [target_width, target_height],
        "scale": args.scale,
        "runtime": str(importlib.import_module("bokujuu-personal-nodes.dlss5.runtime").resolve_runtime()),
        "esrgan_model": esrgan_name,
        "timings": timings,
        "metrics": metrics,
        "metric_note": "No high-resolution ground truth was available; sharpness and difference from Lanczos are descriptive, not quality scores.",
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
