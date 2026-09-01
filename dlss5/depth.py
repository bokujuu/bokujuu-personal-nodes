import torch

from comfy_extras.nodes_depth_anything_3 import DA3Inference, DA3Render

from ..depth_utils import adjust_depth, apply_color_theme
from .types import DLSSGuidance


def estimate_da3_guidance(
    da3_model,
    images: torch.Tensor,
    resolution: int = 504,
    resize_method: str = "upper_bound_resize",
    normalization: str = "v2_style",
) -> DLSSGuidance:
    """Estimate relative DA3 depth and normalize it exactly once for DLSS input."""
    geometry = DA3Inference.execute(
        da3_model,
        images[..., :3],
        int(resolution),
        resize_method,
        {"mode": "mono"},
    )[0]
    depth_image = DA3Render.execute(
        geometry,
        {
            "output": "depth",
            "normalization": normalization,
            "apply_sky_clip": False,
        },
    )[0]
    depth = depth_image[..., 0].float().cpu().clamp(0.0, 1.0).contiguous()
    return DLSSGuidance(depth=depth).validate_for(images)


def render_guidance_depth(
    guidance: DLSSGuidance,
    color_theme: str = "grayscale",
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    if guidance.depth is None:
        raise ValueError("DLSS guidance does not contain depth")
    values = adjust_depth(guidance.depth, contrast=contrast, gamma=gamma)
    return apply_color_theme(values, color_theme).float()
