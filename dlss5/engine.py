from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .runtime import NativeDLSSSession
from .temporal import scene_cut_detected
from .types import DLSSGuidance


QUALITY_VALUES = {
    "performance": 0,
    "balanced": 1,
    "quality": 2,
    "ultra_performance": 3,
    "ultra_quality": 4,
    "dlaa": 5,
}

QUALITY_OPTIONS = (
    "ultra_performance",
    "performance",
    "balanced",
    "quality",
    "ultra_quality",
    "dlaa",
)


def _quality_for(scale: float, mode: str) -> int:
    if mode != "auto":
        return QUALITY_VALUES[mode]
    if scale <= 1.01:
        return QUALITY_VALUES["dlaa"]
    if scale <= 1.55:
        return QUALITY_VALUES["quality"]
    if scale <= 2.25:
        return QUALITY_VALUES["balanced"]
    return QUALITY_VALUES["ultra_performance"]


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"ComfyUI IMAGE must be [B,H,W,C] with 1, 3, or 4 channels; got {tuple(images.shape)}")
    if images.shape[1] < 32 or images.shape[2] < 64:
        raise ValueError("DLSS input must be at least 64 pixels wide and 32 pixels high")
    if not torch.isfinite(images).all():
        raise ValueError("Input images contain NaN or infinite values")


def _to_rgba8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        rgb = image.expand(*image.shape[:-1], 3)
    else:
        rgb = image[..., :3]
    alpha = torch.ones((*rgb.shape[:2], 1), dtype=rgb.dtype)
    return torch.cat((rgb, alpha), dim=-1).mul(255.0).round().byte().numpy()


def _resize_lanczos(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    array = image.detach().float().cpu().clamp(0.0, 1.0).mul(255.0).round().byte().numpy()
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[array.shape[-1]]
    if mode == "L":
        array = array[..., 0]
    resized = Image.fromarray(array, mode=mode).resize((width, height), Image.Resampling.LANCZOS)
    output = np.asarray(resized).copy()
    if output.ndim == 2:
        output = output[..., None]
    return torch.from_numpy(output).float().div(255.0)


class DLSS5Engine:
    """Experimental still/sequence DLSS-SR wrapper for ordinary RGB media."""

    def __init__(self, runtime_path=None, gpu_index: int = 0):
        self.runtime_path = runtime_path
        self.gpu_index = int(gpu_index)

    def _run(
        self,
        images: torch.Tensor,
        guidance: DLSSGuidance,
        scale: float,
        quality: str,
        output_mix: float,
        temporal: bool,
        scene_cut_threshold: float,
    ) -> torch.Tensor:
        _validate_images(images)
        guidance.validate_for(images)
        if guidance.depth is None:
            raise ValueError("DLSS Super Resolution requires estimated depth guidance")
        if quality not in ("auto", *QUALITY_VALUES):
            raise ValueError(f"Unsupported DLSS quality mode: {quality}")
        scale = float(scale)
        if not 1.0 <= scale <= 3.0:
            raise ValueError("DLSS scale must be between 1.0 and 3.0")
        if quality == "dlaa" and scale > 1.01:
            raise ValueError("DLAA is a 1.0x mode; use another quality mode when upscaling")
        output_mix = float(output_mix)
        if not 0.0 <= output_mix <= 1.0:
            raise ValueError("output_mix must be between 0.0 and 1.0")

        batch, height, width, channels = images.shape
        output_width = max(64, int(round(width * scale)))
        output_height = max(32, int(round(height * scale)))
        output_frames = []
        previous = None
        with NativeDLSSSession(
            width,
            height,
            output_width,
            output_height,
            _quality_for(scale, quality),
            self.gpu_index,
            self.runtime_path,
        ) as session:
            for index in range(batch):
                frame = images[index]
                reset = not temporal or index == 0 or scene_cut_detected(
                    previous, frame, scene_cut_threshold
                )
                processed = session.process(
                    _to_rgba8(frame),
                    guidance.depth[index].detach().float().cpu().numpy(),
                    reset=reset,
                )
                dlss_rgb = torch.from_numpy(processed[..., :3].copy()).float().div(255.0)
                baseline = _resize_lanczos(frame, output_width, output_height)
                baseline_rgb = baseline[..., :3].expand(output_height, output_width, 3)
                result_rgb = torch.lerp(baseline_rgb, dlss_rgb, output_mix).clamp(0.0, 1.0)
                if channels == 4:
                    result = torch.cat((result_rgb, baseline[..., 3:4]), dim=-1)
                elif channels == 1:
                    result = result_rgb.mean(dim=-1, keepdim=True)
                else:
                    result = result_rgb
                output_frames.append(result)
                previous = frame

        return torch.stack(output_frames).to(device=images.device, dtype=images.dtype)

    def enhance_image(
        self,
        images: torch.Tensor,
        guidance: DLSSGuidance,
        scale: float = 2.0,
        quality: str = "auto",
        output_mix: float = 1.0,
    ) -> torch.Tensor:
        return self._run(images, guidance, scale, quality, output_mix, False, 0.0)

    def enhance_sequence(
        self,
        images: torch.Tensor,
        guidance: DLSSGuidance,
        scale: float = 2.0,
        quality: str = "auto",
        output_mix: float = 1.0,
        scene_cut_threshold: float = 0.35,
    ) -> torch.Tensor:
        return self._run(
            images,
            guidance,
            scale,
            quality,
            output_mix,
            True,
            scene_cut_threshold,
        )
