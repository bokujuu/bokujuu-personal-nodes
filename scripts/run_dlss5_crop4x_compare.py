from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(COMFYUI_ROOT / "custom_nodes"))

ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
POST = importlib.import_module("bokujuu-personal-nodes.dlss5.postprocess")
RUNTIME = importlib.import_module("bokujuu-personal-nodes.dlss5.runtime")
TYPES = importlib.import_module("bokujuu-personal-nodes.dlss5.types")


INTERNAL_CONSTANT_DEPTH = 0.5
CAS_AMOUNT = 0.7
SCALE = 4.0


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image).copy()).float().div(255.0).unsqueeze(0)


def pil_from_tensor(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().cpu().clamp(0.0, 1.0).mul(255).round().byte().numpy()
    return Image.fromarray(array[..., :3])


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sharpness_metrics(image: torch.Tensor) -> dict[str, float]:
    rgb = image[..., :3].float().cpu()
    horizontal = (rgb[:, 1:] - rgb[:, :-1]).abs().mean()
    vertical = (rgb[1:] - rgb[:-1]).abs().mean()
    return {
        "gradient_mean": float(horizontal + vertical),
        "clipped_fraction": float(((rgb <= 0.0) | (rgb >= 1.0)).float().mean()),
    }


def detail_crop_box(width: int, height: int, size: int = 800) -> tuple[int, int, int, int]:
    crop_size = min(width, height, size)
    center_x = int(width * 0.52)
    center_y = int(height * 0.32)
    left = min(max(0, center_x - crop_size // 2), width - crop_size)
    top = min(max(0, center_y - crop_size // 2), height - crop_size)
    return left, top, left + crop_size, top + crop_size


def source_crop_from_2x_preview(image: Image.Image) -> Image.Image:
    left, top, right, bottom = detail_crop_box(image.width * 2, image.height * 2, 800)
    return image.crop((left // 2, top // 2, right // 2, bottom // 2))


def rtx_vsr(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import nvvfx

    outputs = []
    with nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA) as super_resolution:
        super_resolution.output_width = width
        super_resolution.output_height = height
        super_resolution.load()
        for frame in images:
            cuda_frame = frame[..., :3].cuda().movedim(-1, 0).float().contiguous()
            result = torch.from_dlpack(super_resolution.run(cuda_frame).image)
            outputs.append(result.movedim(0, -1).cpu())
    return torch.stack(outputs)


def dlss_at_scale(images: torch.Tensor, scale: float, quality: str, gpu_index: int) -> torch.Tensor:
    height, width = int(images.shape[1]), int(images.shape[2])
    output_width = int(round(width * scale))
    output_height = int(round(height * scale))
    guidance = TYPES.DLSSGuidance(
        depth=torch.full(tuple(images.shape[:3]), INTERNAL_CONSTANT_DEPTH, dtype=torch.float32)
    ).validate_for(images)
    frames = []
    with RUNTIME.NativeDLSSSession(
        width,
        height,
        output_width,
        output_height,
        ENGINE.QUALITY_VALUES[quality],
        gpu_index,
        None,
    ) as session:
        for index in range(images.shape[0]):
            processed = session.process(
                ENGINE._to_rgba8(images[index]),
                guidance.depth[index].detach().float().cpu().numpy(),
                reset=True,
            )
            frames.append(torch.from_numpy(processed[..., :3].copy()).float().div(255.0))
    return torch.stack(frames).to(device=images.device, dtype=images.dtype)


def _font(size: int):
    for candidate in (Path(r"C:\Windows\Fonts\meiryo.ttc"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_contact_sheet(labeled: list[tuple[str, Image.Image]], path: Path) -> None:
    cell = 420
    header = 36
    sheet = Image.new("RGB", (cell * len(labeled), cell + header), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for index, (label, image) in enumerate(labeled):
        preview = image.copy()
        preview.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
        x = index * cell
        draw.text((x + 8, 8), label, fill="black", font=font)
        sheet.paste(
            preview,
            (x + (cell - preview.width) // 2, header + (cell - 8 - preview.height) // 2),
        )
    sheet.save(path)


def emit_html(methods: list[str], identical_pairs: dict[str, bool]) -> str:
    options = "".join(f'<option value="{name}">{name}</option>' for name in methods)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crop 4x: Lanczos / RTX VSR / DLSS</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: system-ui, sans-serif; margin: 1rem 1.25rem; line-height: 1.45; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.4rem; }}
    .hint {{ opacity: 0.85; margin: 0 0 1rem; }}
    .row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }}
    figure {{ margin: 0; }}
    figure img {{ display: block; width: min(280px, 100%); height: auto; }}
    figcaption {{ font-size: 0.85rem; margin-top: 0.25rem; }}
    .controls {{ margin: 0.5rem 0; display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
    .stage {{
      position: relative;
      max-width: min(100%, 1100px);
      user-select: none;
      cursor: ew-resize;
    }}
    .stage img {{ display: block; width: 100%; height: auto; pointer-events: none; }}
    .left {{
      position: absolute;
      inset: 0;
      clip-path: inset(0 calc(100% - var(--pos, 50%)) 0 0);
    }}
    .handle {{
      position: absolute; top: 0; bottom: 0; left: var(--pos, 50%);
      width: 2px; margin-left: -1px; background: CanvasText; pointer-events: none;
    }}
    .tag {{
      position: absolute; top: 0.6rem; padding: 0.15rem 0.45rem;
      font-size: 0.8rem; background: Canvas; color: CanvasText; pointer-events: none;
    }}
    .tag.left-tag {{ left: 0.6rem; }}
    .tag.right-tag {{ right: 0.6rem; }}
    input[type="range"] {{ display: block; width: min(100%, 1100px); margin: 0.75rem 0 0; }}
  </style>
</head>
<body>
  <h1>同じクロップの 4 倍比較</h1>
  <p class="hint">元画像の顔まわり（2倍比較で見ていた 800px 相当）を切り出し、4倍しています。DLSS は CAS 0.7 です。</p>
  <div class="row" id="thumbs"></div>
  <div class="controls">
    <label>左 <select id="left-method">{options}</select></label>
    <label>右 <select id="right-method">{options}</select></label>
    <span id="match"></span>
  </div>
  <div class="stage" id="stage" style="--pos: 50%">
    <img id="right-img" alt="" />
    <img id="left-img" class="left" alt="" />
    <div class="handle"></div>
    <span class="tag left-tag" id="left-tag"></span>
    <span class="tag right-tag" id="right-tag"></span>
  </div>
  <input id="split" type="range" min="0" max="100" value="50" />
  <script>
    const files = {{
      "crop": "crop_source.png",
      "Lanczos": "lanczos_4x.png",
      "RTX VSR": "rtx_vsr_4x.png",
      "DLSS": "dlss_4x_cas07.png"
    }};
    const identical = {json.dumps(identical_pairs, ensure_ascii=False)};
    const thumbs = document.getElementById("thumbs");
    for (const [label, file] of Object.entries(files)) {{
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = file;
      img.alt = label;
      const cap = document.createElement("figcaption");
      cap.textContent = label;
      figure.append(img, cap);
      thumbs.appendChild(figure);
    }}
    const leftSelect = document.getElementById("left-method");
    const rightSelect = document.getElementById("right-method");
    leftSelect.value = "DLSS";
    rightSelect.value = "RTX VSR";
    const leftImg = document.getElementById("left-img");
    const rightImg = document.getElementById("right-img");
    const leftTag = document.getElementById("left-tag");
    const rightTag = document.getElementById("right-tag");
    const match = document.getElementById("match");
    function pairKey(a, b) {{
      return a < b ? a + " vs " + b : b + " vs " + a;
    }}
    function refresh() {{
      const left = leftSelect.value;
      const right = rightSelect.value;
      leftImg.src = files[left];
      rightImg.src = files[right];
      leftImg.alt = left;
      rightImg.alt = right;
      leftTag.textContent = left;
      rightTag.textContent = right;
      if (left === right) {{
        match.textContent = "同じ方式です。";
      }} else if (identical[pairKey(left, right)]) {{
        match.textContent = "この2枚は画素が一致しています。";
      }} else {{
        match.textContent = "画素は異なります。スライダーで境を動かしてください。";
      }}
    }}
    leftSelect.addEventListener("change", refresh);
    rightSelect.addEventListener("change", refresh);
    refresh();
    const stage = document.getElementById("stage");
    const split = document.getElementById("split");
    function setPos(percent) {{
      const value = Math.min(100, Math.max(0, percent));
      stage.style.setProperty("--pos", value + "%");
      split.value = String(Math.round(value));
    }}
    function posFromEvent(event) {{
      const box = stage.getBoundingClientRect();
      return ((event.clientX - box.left) / box.width) * 100;
    }}
    split.addEventListener("input", () => setPos(Number(split.value)));
    stage.addEventListener("pointerdown", (event) => {{
      stage.setPointerCapture(event.pointerId);
      setPos(posFromEvent(event));
    }});
    stage.addEventListener("pointermove", (event) => {{
      if (event.buttons) setPos(posFromEvent(event));
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Lanczos, RTX VSR, and DLSS at 4x on the detail crop."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMFYUI_ROOT / "output" / "bokujuu_dlss5_test" / "crop_4x",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.input).convert("RGB")
    crop = source_crop_from_2x_preview(source)
    width = crop.width - crop.width % 4
    height = crop.height - crop.height % 4
    crop = crop.crop((0, 0, width, height))
    crop.save(args.output_dir / "crop_source.png")
    images = tensor_from_pil(crop)
    target_width = int(round(width * SCALE))
    target_height = int(round(height * SCALE))
    print(f"crop {width}x{height} -> {target_width}x{target_height}", flush=True)

    timings = {}
    metrics = {}
    errors = {}
    result_pils = {"crop": crop}

    started = time.perf_counter()
    lanczos = torch.stack(
        [ENGINE._resize_lanczos(frame, target_width, target_height) for frame in images]
    )
    timings["lanczos"] = time.perf_counter() - started
    result_pils["Lanczos"] = pil_from_tensor(lanczos[0])
    result_pils["Lanczos"].save(args.output_dir / "lanczos_4x.png")
    metrics["lanczos"] = sharpness_metrics(lanczos[0])
    print(f"lanczos {timings['lanczos']:.3f}s", flush=True)

    started = time.perf_counter()
    rtx = rtx_vsr(images, target_width, target_height)
    timings["rtx_vsr"] = time.perf_counter() - started
    result_pils["RTX VSR"] = pil_from_tensor(rtx[0])
    result_pils["RTX VSR"].save(args.output_dir / "rtx_vsr_4x.png")
    metrics["rtx_vsr"] = sharpness_metrics(rtx[0])
    metrics["rtx_vsr"]["mae_vs_lanczos"] = float((rtx[0] - lanczos[0]).abs().mean())
    print(f"rtx_vsr {timings['rtx_vsr']:.3f}s", flush=True)

    dlss_quality_used = None
    dlss = None
    for quality in ("ultra_performance", "balanced", "performance", "quality"):
        started = time.perf_counter()
        try:
            raw = dlss_at_scale(images, SCALE, quality, args.gpu_index)
        except Exception as error:
            errors[f"dlss_{quality}"] = str(error)
            timings[f"dlss_{quality}"] = time.perf_counter() - started
            print(f"dlss {quality} failed: {error}", flush=True)
            continue
        timings[f"dlss_{quality}"] = time.perf_counter() - started
        dlss = POST.cas_sharpen(raw, amount=CAS_AMOUNT)
        dlss_quality_used = quality
        print(f"dlss {quality} {timings[f'dlss_{quality}']:.3f}s", flush=True)
        break
    if dlss is None:
        raise RuntimeError(f"DLSS 4x failed for all quality modes: {errors}")

    result_pils["DLSS"] = pil_from_tensor(dlss[0])
    result_pils["DLSS"].save(args.output_dir / "dlss_4x_cas07.png")
    metrics["dlss"] = sharpness_metrics(dlss[0])
    metrics["dlss"]["mae_vs_lanczos"] = float((dlss[0] - lanczos[0]).abs().mean())
    metrics["dlss"]["mae_vs_rtx"] = float((dlss[0] - rtx[0]).abs().mean())
    metrics["rtx_vsr"]["mae_vs_dlss"] = metrics["dlss"]["mae_vs_rtx"]

    make_contact_sheet(
        [
            ("Source crop", result_pils["crop"]),
            ("Lanczos 4x", result_pils["Lanczos"]),
            ("RTX VSR 4x", result_pils["RTX VSR"]),
            ("DLSS 4x CAS 0.7", result_pils["DLSS"]),
        ],
        args.output_dir / "contact_sheet.png",
    )

    names = {
        "Lanczos": "lanczos_4x.png",
        "RTX VSR": "rtx_vsr_4x.png",
        "DLSS": "dlss_4x_cas07.png",
    }
    identical_pairs = {}
    for left, right in (("DLSS", "RTX VSR"), ("DLSS", "Lanczos"), ("RTX VSR", "Lanczos")):
        key = " vs ".join(sorted((left, right)))
        identical_pairs[key] = file_digest(args.output_dir / names[left]) == file_digest(
            args.output_dir / names[right]
        )

    (args.output_dir / "compare.html").write_text(
        emit_html(["Lanczos", "RTX VSR", "DLSS"], identical_pairs),
        encoding="utf-8",
    )
    summary = {
        "input": str(args.input),
        "crop_size": [width, height],
        "output_size": [target_width, target_height],
        "scale": SCALE,
        "cas_amount": CAS_AMOUNT,
        "dlss_quality": dlss_quality_used,
        "timings_seconds": timings,
        "metrics": metrics,
        "errors": errors,
        "identical_pairs": identical_pairs,
        "note": (
            "Crop matches the 800px window used on the 2x detail grid, mapped back to the source. "
            "DLSS 4x bypasses the node scale cap of 3.0 and calls the native SuperSampling feature directly."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
