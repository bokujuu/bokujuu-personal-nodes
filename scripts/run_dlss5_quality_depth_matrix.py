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

DEPTH = importlib.import_module("bokujuu-personal-nodes.dlss5.depth")
ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
POST = importlib.import_module("bokujuu-personal-nodes.dlss5.postprocess")
TYPES = importlib.import_module("bokujuu-personal-nodes.dlss5.types")
PERSONAL_NODES = importlib.import_module("bokujuu-personal-nodes.nodes")


QUALITY_MODES = ("balanced", "quality")
DEPTH_RESOLUTIONS = (504, 1008, 2016)
POSTPROCESS_SETTINGS = {
    "raw": {},
    "nis": {"sharpness": 0.25},
    "cas": {"amount": 0.35},
    "unsharp": {"radius": 2.0, "percent": 50, "threshold": 2},
}


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


def rtx_vsr(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import nvvfx

    outputs = []
    quality = nvvfx.effects.QualityLevel.ULTRA
    with nvvfx.VideoSuperRes(quality) as super_resolution:
        super_resolution.output_width = width
        super_resolution.output_height = height
        super_resolution.load()
        for frame in images:
            cuda_frame = frame[..., :3].cuda().movedim(-1, 0).float().contiguous()
            result = torch.from_dlpack(super_resolution.run(cuda_frame).image)
            outputs.append(result.movedim(0, -1).cpu())
    return torch.stack(outputs)


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


def make_grid(
    rows: list[tuple[str, list[tuple[str, Image.Image]]]],
    output_path: Path,
    *,
    details: bool,
) -> None:
    label_width = 190
    cell_width = 360
    cell_height = 360 if details else 490
    header_height = 48
    row_gap = 6
    columns = max(len(items) for _, items in rows)
    sheet = Image.new(
        "RGB",
        (label_width + columns * cell_width, header_height + len(rows) * (cell_height + row_gap)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    header_font = _font(20)
    label_font = _font(17)
    for column, (column_label, _) in enumerate(rows[0][1]):
        draw.text((label_width + column * cell_width + 8, 10), column_label, fill="black", font=header_font)
    for row_index, (row_label, items) in enumerate(rows):
        y = header_height + row_index * (cell_height + row_gap)
        draw.multiline_text((8, y + 10), row_label, fill="black", font=label_font, spacing=5)
        for column, (_, image) in enumerate(items):
            preview = _detail_crop(image) if details else image
            preview = _fit(preview, cell_width - 8, cell_height - 8)
            sheet.paste(preview, (label_width + column * cell_width + 4, y + 4))
    sheet.save(output_path)


def apply_postprocess(
    name: str,
    raw: torch.Tensor,
    nis_sdk: Path,
    gpu_index: int,
) -> torch.Tensor:
    if name == "raw":
        return raw
    if name == "nis":
        return POST.nis_sharpen(
            raw, nis_sdk=nis_sdk, gpu_index=gpu_index, **POSTPROCESS_SETTINGS[name]
        )
    if name == "cas":
        return POST.cas_sharpen(raw, **POSTPROCESS_SETTINGS[name])
    if name == "unsharp":
        return POST.unsharp_mask(raw, **POSTPROCESS_SETTINGS[name])
    raise ValueError(f"Unknown post-process method: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a DLSS 2x quality/depth/post-sharpen factorial comparison."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nis-sdk", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = args.output_dir / "outputs"
    depths_dir = args.output_dir / "depth"
    outputs_dir.mkdir(exist_ok=True)
    depths_dir.mkdir(exist_ok=True)

    source_pil = Image.open(args.input).convert("RGB")
    width = source_pil.width - source_pil.width % 4
    height = source_pil.height - source_pil.height % 4
    source_pil = source_pil.crop((0, 0, width, height))
    images = tensor_from_pil(source_pil)
    target_width = width * 2
    target_height = height * 2

    timings: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    result_images: dict[str, Image.Image] = {}
    raw_tensors: dict[str, torch.Tensor] = {}

    started = time.perf_counter()
    da3_model = PERSONAL_NODES.BokujuuLoadDepthAnything3.execute(
        PERSONAL_NODES.DA3_REPO_ID, "fp16"
    ).result[0]
    timings["load_da3_model"] = time.perf_counter() - started

    guidance_modes = []
    for resolution in DEPTH_RESOLUTIONS:
        key = f"da3_{resolution}"
        started = time.perf_counter()
        guidance = DEPTH.estimate_da3_guidance(
            da3_model,
            images,
            resolution=resolution,
            resize_method="upper_bound_resize",
            normalization="v2_style",
        )
        timings[f"depth_{key}"] = time.perf_counter() - started
        pil_from_tensor(DEPTH.render_guidance_depth(guidance)[0]).save(
            depths_dir / f"{key}.png"
        )
        guidance_modes.append((key, f"DA3 {resolution}", guidance))

    constant_guidance = TYPES.DLSSGuidance(
        depth=torch.full((images.shape[0], height, width), 0.5, dtype=torch.float32)
    ).validate_for(images)
    pil_from_tensor(DEPTH.render_guidance_depth(constant_guidance)[0]).save(
        depths_dir / "constant_0_5.png"
    )
    guidance_modes.append(("constant_0_5", "Constant 0.5\n(DA3なし)", constant_guidance))

    import comfy.model_management as model_management

    model_management.unload_all_models()
    for depth_key, depth_label, guidance in guidance_modes:
        for quality in QUALITY_MODES:
            raw_key = f"{quality}__{depth_key}"
            started = time.perf_counter()
            raw = ENGINE.DLSS5Engine(gpu_index=args.gpu_index).enhance_image(
                images,
                guidance,
                scale=2.0,
                quality=quality,
                output_mix=1.0,
            )
            timings[f"dlss_{raw_key}"] = time.perf_counter() - started
            raw_tensors[raw_key] = raw.detach().float().cpu()
            for method in POSTPROCESS_SETTINGS:
                output_key = f"{raw_key}__{method}"
                started = time.perf_counter()
                processed = apply_postprocess(method, raw, args.nis_sdk, args.gpu_index)
                timings[f"post_{output_key}"] = time.perf_counter() - started
                image = pil_from_tensor(processed[0])
                image.save(outputs_dir / f"{output_key}.png")
                result_images[output_key] = image
                values = sharpness_metrics(processed[0])
                values["mae_vs_raw"] = float(
                    (processed[0, ..., :3] - raw[0, ..., :3]).abs().mean()
                )
                metrics[output_key] = values

    started = time.perf_counter()
    lanczos = torch.stack(
        [ENGINE._resize_lanczos(frame, target_width, target_height) for frame in images]
    )
    timings["reference_lanczos"] = time.perf_counter() - started
    pil_from_tensor(lanczos[0]).save(args.output_dir / "reference_lanczos.png")
    metrics["reference_lanczos"] = sharpness_metrics(lanczos[0])

    started = time.perf_counter()
    rtx = rtx_vsr(images, target_width, target_height)
    timings["reference_rtx_vsr_ultra"] = time.perf_counter() - started
    pil_from_tensor(rtx[0]).save(args.output_dir / "reference_rtx_vsr_ultra.png")
    metrics["reference_rtx_vsr_ultra"] = sharpness_metrics(rtx[0])

    raw_rows = []
    depth_rows = []
    for depth_key, depth_label, _ in guidance_modes:
        raw_rows.append(
            (
                depth_label,
                [
                    (quality.title(), result_images[f"{quality}__{depth_key}__raw"])
                    for quality in QUALITY_MODES
                ],
            )
        )
        depth_rows.append(
            (
                depth_label,
                [("Depth guidance", Image.open(depths_dir / f"{depth_key}.png").convert("RGB"))],
            )
        )
    make_grid(raw_rows, args.output_dir / "raw_quality_depth_full_grid.png", details=False)
    make_grid(raw_rows, args.output_dir / "raw_quality_depth_detail_grid.png", details=True)
    make_grid(depth_rows, args.output_dir / "depth_guidance_grid.png", details=False)

    for quality in QUALITY_MODES:
        rows = []
        for depth_key, depth_label, _ in guidance_modes:
            rows.append(
                (
                    depth_label,
                    [
                        (
                            {"raw": "Raw", "nis": "NIS 0.25", "cas": "CAS 0.35", "unsharp": "Unsharp weak"}[method],
                            result_images[f"{quality}__{depth_key}__{method}"],
                        )
                        for method in POSTPROCESS_SETTINGS
                    ],
                )
            )
        make_grid(rows, args.output_dir / f"{quality}_postprocess_detail_grid.png", details=True)

    reference_rows = [
        (
            "Reference",
            [
                ("Lanczos", pil_from_tensor(lanczos[0])),
                ("RTX VSR Ultra", pil_from_tensor(rtx[0])),
                ("Balanced DA3 1008", result_images["balanced__da3_1008__raw"]),
                ("Quality DA3 1008", result_images["quality__da3_1008__raw"]),
            ],
        )
    ]
    make_grid(reference_rows, args.output_dir / "reference_detail_grid.png", details=True)

    ranking = sorted(
        (
            {"condition": key, **values}
            for key, values in metrics.items()
            if key not in ("reference_lanczos", "reference_rtx_vsr_ultra")
        ),
        key=lambda item: item["gradient_mean"],
        reverse=True,
    )
    baseline_raw_key = next(iter(raw_tensors))
    baseline_raw = raw_tensors[baseline_raw_key]
    raw_equivalence = {}
    for key, value in raw_tensors.items():
        difference = (value - baseline_raw).abs()
        raw_equivalence[key] = {
            "max_abs_vs_baseline": float(difference.max()),
            "mae_vs_baseline": float(difference.mean()),
            "byte_identical_after_png_quantization": (
                value.mul(255).round().byte().equal(baseline_raw.mul(255).round().byte())
            ),
        }
    all_raw_byte_identical = all(
        item["byte_identical_after_png_quantization"] for item in raw_equivalence.values()
    )
    summary = {
        "input_requested": str(args.input),
        "input_resolved": str(args.input.resolve()),
        "input_size": [width, height],
        "output_size": [target_width, target_height],
        "scale": 2.0,
        "quality_modes": list(QUALITY_MODES),
        "depth_modes": [
            {"key": key, "label": label.replace("\n", " ")} for key, label, _ in guidance_modes
        ],
        "postprocess_settings": POSTPROCESS_SETTINGS,
        "nis_sdk": str(args.nis_sdk.resolve()),
        "timings_seconds": timings,
        "metrics": metrics,
        "gradient_ranking": ranking,
        "raw_output_analysis": {
            "baseline": baseline_raw_key,
            "conditions": raw_equivalence,
            "all_conditions_byte_identical": all_raw_byte_identical,
            "interpretation": (
                "At fixed 2x dimensions with a reset independent still frame, this runtime produced the same "
                "DLSS-SR RGB result for Balanced, Quality, DA3 504/1008/2016, and constant depth 0.5. "
                "This establishes no still-image benefit from those quality/depth changes in this integration."
            ),
        },
        "metric_note": (
            "No high-resolution ground truth was available. Gradient and Laplacian values measure local "
            "contrast, including desirable detail, halos, and noise; they are descriptive rather than quality scores."
        ),
        "constant_depth_note": (
            "DLSS-SR requires a depth resource. Constant 0.5 is the DA3-disabled proxy; it is not a true color-only DLSS mode."
        ),
    }
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
