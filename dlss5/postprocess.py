from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
NIS_BRIDGE_PATH = REPO_ROOT / "native" / "bin" / "bokujuu_nis_sharpen_bridge.dll"


class NISSharpenError(RuntimeError):
    pass


def resolve_nis_sdk(configured: str | Path | None = None) -> Path:
    if configured:
        candidates = [Path(configured)]
    else:
        candidates = []
        if os.environ.get("NVIDIA_NIS_SDK"):
            candidates.append(Path(os.environ["NVIDIA_NIS_SDK"]))
    for candidate in candidates:
        shader = candidate / "NIS" / "NIS_Main.hlsl"
        if shader.is_file():
            return candidate.resolve()
    raise NISSharpenError(
        "NVIDIA Image Scaling SDK was not found. Pass --nis-sdk or set NVIDIA_NIS_SDK."
    )


def _load_nis_bridge():
    if not NIS_BRIDGE_PATH.is_file():
        raise NISSharpenError(
            f"NIS sharpen bridge was not built: {NIS_BRIDGE_PATH}\n"
            "Run native/build_nis_sharpen_bridge.ps1 with an NVIDIA Image Scaling SDK checkout."
        )
    library = ctypes.WinDLL(str(NIS_BRIDGE_PATH))
    function = library.bokujuu_nis_sharpen
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    return library, function


def nis_sharpen(
    images: torch.Tensor,
    sharpness: float = 0.25,
    nis_sdk: str | Path | None = None,
    gpu_index: int = 0,
) -> torch.Tensor:
    sdk_root = resolve_nis_sdk(nis_sdk)
    shader_path = sdk_root / "NIS" / "NIS_Main.hlsl"
    library, function = _load_nis_bridge()
    outputs = []
    try:
        for image in images:
            rgb = image.detach().float().cpu().clamp(0.0, 1.0)[..., :3]
            alpha = torch.ones((*rgb.shape[:2], 1), dtype=rgb.dtype)
            rgba = (
                torch.cat((rgb, alpha), dim=-1)
                .mul(255.0)
                .round()
                .byte()
                .numpy()
            )
            rgba = np.ascontiguousarray(rgba)
            output = np.empty_like(rgba)
            error = ctypes.create_string_buffer(8192)
            succeeded = function(
                rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                rgba.strides[0],
                rgba.shape[1],
                rgba.shape[0],
                float(sharpness),
                str(shader_path),
                int(gpu_index),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                output.strides[0],
                error,
                len(error),
            )
            if not succeeded:
                raise NISSharpenError(error.value.decode("utf-8", errors="replace"))
            outputs.append(torch.from_numpy(output[..., :3].copy()).float().div(255.0))
    finally:
        del library
    return torch.stack(outputs).to(device=images.device, dtype=images.dtype)


def cas_sharpen(images: torch.Tensor, amount: float = 0.7) -> torch.Tensor:
    """Apply the same PyTorch CAS formulation used by ComfyUI Essentials.

    Amount 0 skips the filter. Amounts above 1.0 remain valid for stronger
    experimental sharpening; they are outside the usual 0-1 CAS range.
    """
    amount = float(amount)
    if amount <= 0.0:
        return images
    amount = min(amount, 4.0)
    source = images.detach().float().cpu().clamp(0.0, 1.0)
    channels = source.shape[-1]
    if channels == 1:
        rgb = source.expand(*source.shape[:-1], 3)
    else:
        rgb = source[..., :3]

    padded = F.pad(rgb.permute(0, 3, 1, 2), pad=(1, 1, 1, 1))
    a = padded[..., :-2, :-2]
    b = padded[..., :-2, 1:-1]
    c = padded[..., :-2, 2:]
    d = padded[..., 1:-1, :-2]
    e = padded[..., 1:-1, 1:-1]
    f = padded[..., 1:-1, 2:]
    g = padded[..., 2:, :-2]
    h = padded[..., 2:, 1:-1]
    i = padded[..., 2:, 2:]

    cross = torch.stack((b, d, e, f, h))
    diagonal = torch.stack((a, c, g, i))
    minimum = cross.amin(dim=0) + diagonal.amin(dim=0)
    maximum = cross.amax(dim=0) + diagonal.amax(dim=0)
    amplitude = torch.sqrt(
        torch.reciprocal(maximum + 1e-5) * torch.minimum(minimum, 2.0 - maximum)
    )
    weight = -amplitude * (amount * (1.0 / 5.0 - 1.0 / 8.0) + 1.0 / 8.0)
    output_rgb = (((b + d + f + h) * weight + e) / (1.0 + 4.0 * weight)).clamp(0.0, 1.0)
    output_rgb = output_rgb.permute(0, 2, 3, 1)

    if channels == 4:
        output = torch.cat((output_rgb, source[..., 3:4]), dim=-1)
    elif channels == 1:
        output = output_rgb.mean(dim=-1, keepdim=True)
    else:
        output = output_rgb
    return output.to(device=images.device, dtype=images.dtype)


def unsharp_mask(
    images: torch.Tensor,
    radius: float = 2.0,
    percent: int = 50,
    threshold: int = 2,
) -> torch.Tensor:
    outputs = []
    for image in images:
        array = (
            image.detach()
            .float()
            .cpu()
            .clamp(0.0, 1.0)[..., :3]
            .mul(255.0)
            .round()
            .byte()
            .numpy()
        )
        filtered = Image.fromarray(array).filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        outputs.append(torch.from_numpy(np.asarray(filtered).copy()).float().div(255.0))
    return torch.stack(outputs).to(device=images.device, dtype=images.dtype)
