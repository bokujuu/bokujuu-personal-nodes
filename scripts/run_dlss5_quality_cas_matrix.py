from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(COMFYUI_ROOT / "custom_nodes"))

ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
POST = importlib.import_module("bokujuu-personal-nodes.dlss5.postprocess")
TYPES = importlib.import_module("bokujuu-personal-nodes.dlss5.types")


QUALITY_MODES = (
    "ultra_performance",
    "performance",
    "balanced",
    "quality",
    "ultra_quality",
)
CAS_AMOUNTS = (0.0, 0.35, 0.7, 1.0, 2.0, 4.0)
INTERNAL_CONSTANT_DEPTH = 0.5


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).float().div(255.0).unsqueeze(0)


def pil_from_tensor(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).mul(255).round().byte().numpy()
    return Image.fromarray(array[..., :3])


def sharpness_metrics(image: torch.Tensor) -> dict[str, float]:
    rgb = image[..., :3].float().cpu()
    horizontal = (rgb[:, 1:] - rgb[:, :-1]).abs().mean()
    vertical = (rgb[1:] - rgb[:-1]).abs().mean()
    luminance = (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    )[None, None]
    laplacian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )[None, None]
    response = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="reflect"), laplacian)
    return {
        "gradient_mean": float(horizontal + vertical),
        "laplacian_variance": float(response.var()),
        "clipped_fraction": float(((rgb <= 0.0) | (rgb >= 1.0)).float().mean()),
    }


def _font(size: int):
    candidates = (
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    preview = image.copy()
    preview.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(preview, ((width - preview.width) // 2, (height - preview.height) // 2))
    return canvas


def _detail_crop(image: Image.Image, size: int = 800) -> Image.Image:
    crop_size = min(image.width, image.height, size)
    center_x = int(image.width * 0.52)
    center_y = int(image.height * 0.32)
    left = min(max(0, center_x - crop_size // 2), image.width - crop_size)
    top = min(max(0, center_y - crop_size // 2), image.height - crop_size)
    return image.crop((left, top, left + crop_size, top + crop_size))


def make_grid(rows, output_path: Path) -> None:
    label_width = 210
    cell_width = 320
    cell_height = 320
    header_height = 48
    row_gap = 6
    columns = max(len(items) for _, items in rows)
    sheet = Image.new(
        "RGB",
        (label_width + columns * cell_width, header_height + len(rows) * (cell_height + row_gap)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header_font = _font(18)
    label_font = _font(16)
    for column, (column_label, _) in enumerate(rows[0][1]):
        draw.text(
            (label_width + column * cell_width + 8, 12),
            column_label,
            fill="black",
            font=header_font,
        )
    for row_index, (row_label, items) in enumerate(rows):
        y = header_height + row_index * (cell_height + row_gap)
        draw.multiline_text((8, y + 12), row_label, fill="black", font=label_font, spacing=4)
        for column, (_, image) in enumerate(items):
            preview = _fit(_detail_crop(image), cell_width - 8, cell_height - 8)
            sheet.paste(preview, (label_width + column * cell_width + 4, y + 4))
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a 2x DLSS quality x CAS amount comparison."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = args.output_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    source_pil = Image.open(args.input).convert("RGB")
    width = source_pil.width - source_pil.width % 4
    height = source_pil.height - source_pil.height % 4
    source_pil = source_pil.crop((0, 0, width, height))
    images = tensor_from_pil(source_pil)
    guidance = TYPES.DLSSGuidance(
        depth=torch.full(tuple(images.shape[:3]), INTERNAL_CONSTANT_DEPTH, dtype=torch.float32)
    ).validate_for(images)

    timings = {}
    metrics = {}
    result_images = {}
    raw_tensors = {}

    for quality in QUALITY_MODES:
        started = time.perf_counter()
        try:
            raw = ENGINE.DLSS5Engine(gpu_index=args.gpu_index).enhance_image(
                images, guidance, scale=2.0, quality=quality, output_mix=1.0
            )
        except Exception as error:
            timings[f"dlss_{quality}"] = time.perf_counter() - started
            metrics[f"{quality}__error"] = {"error": str(error)}
            print(f"{quality} failed: {error}", flush=True)
            continue
        timings[f"dlss_{quality}"] = time.perf_counter() - started
        raw_tensors[quality] = raw.detach().float().cpu()
        for amount in CAS_AMOUNTS:
            key = f"{quality}__cas_{amount:g}"
            started = time.perf_counter()
            processed = POST.cas_sharpen(raw, amount=amount)
            timings[f"cas_{key}"] = time.perf_counter() - started
            image = pil_from_tensor(processed[0])
            image.save(outputs_dir / f"{key}.png")
            result_images[key] = image
            values = sharpness_metrics(processed[0])
            values["mae_vs_raw"] = float((processed[0, ..., :3] - raw[0, ..., :3]).abs().mean())
            metrics[key] = values

    baseline_raw = raw_tensors.get("balanced")
    raw_equivalence = {}
    if baseline_raw is not None:
        for quality, value in raw_tensors.items():
            difference = (value - baseline_raw).abs()
            raw_equivalence[quality] = {
                "max_abs_vs_balanced": float(difference.max()),
                "mae_vs_balanced": float(difference.mean()),
                "byte_identical_to_balanced": bool(
                    value.mul(255).round().byte().equal(baseline_raw.mul(255).round().byte())
                ),
            }

    cas_from_balanced = {}
    if baseline_raw is not None:
        previous = None
        for amount in CAS_AMOUNTS:
            current = POST.cas_sharpen(baseline_raw, amount=amount)
            quantized = current.mul(255).round().byte()
            item = {
                "mae_vs_raw": float((current - baseline_raw).abs().mean()),
                "byte_identical_to_raw": bool(
                    quantized.equal(baseline_raw.mul(255).round().byte())
                ),
            }
            if previous is not None:
                prev_amount, prev_tensor = previous
                item[f"mae_vs_cas_{prev_amount:g}"] = float((current - prev_tensor).abs().mean())
                item[f"byte_identical_to_cas_{prev_amount:g}"] = bool(
                    quantized.equal(prev_tensor.mul(255).round().byte())
                )
            cas_from_balanced[str(amount)] = item
            previous = (amount, current)

    rows = []
    for quality in QUALITY_MODES:
        if quality not in raw_tensors:
            continue
        rows.append(
            (
                quality.replace("_", " "),
                [
                    (f"CAS {amount:g}", result_images[f"{quality}__cas_{amount:g}"])
                    for amount in CAS_AMOUNTS
                ],
            )
        )
    if rows:
        make_grid(rows, args.output_dir / "quality_cas_detail_grid.png")

    summary = {
        "input_resolved": str(args.input.resolve()),
        "input_size": [width, height],
        "output_size": [width * 2, height * 2],
        "scale": 2.0,
        "quality_modes": list(QUALITY_MODES),
        "cas_amounts": list(CAS_AMOUNTS),
        "internal_constant_depth": INTERNAL_CONSTANT_DEPTH,
        "timings_seconds": timings,
        "metrics": metrics,
        "raw_quality_analysis": {
            "baseline": "balanced",
            "conditions": raw_equivalence,
            "all_raw_byte_identical_to_balanced": bool(raw_equivalence) and all(
                item["byte_identical_to_balanced"] for item in raw_equivalence.values()
            ),
        },
        "cas_amount_analysis_on_balanced": cas_from_balanced,
        "metric_note": (
            "No high-resolution ground truth was available. Gradient values measure local contrast, "
            "including detail, halos, and noise; they are descriptive rather than quality scores."
        ),
    }
    (args.output_dir / "quality_cas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
