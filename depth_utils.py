import torch

from comfy.ldm.colormap import turbo


COLOR_THEMES = {
    "viridis": (
        (0.267, 0.005, 0.329),
        (0.230, 0.322, 0.546),
        (0.128, 0.567, 0.551),
        (0.369, 0.789, 0.383),
        (0.993, 0.906, 0.144),
    ),
    "plasma": (
        (0.050, 0.030, 0.528),
        (0.494, 0.012, 0.658),
        (0.798, 0.280, 0.470),
        (0.973, 0.586, 0.252),
        (0.940, 0.975, 0.131),
    ),
    "inferno": (
        (0.001, 0.000, 0.014),
        (0.258, 0.039, 0.406),
        (0.578, 0.148, 0.404),
        (0.865, 0.317, 0.226),
        (0.988, 0.998, 0.645),
    ),
}


def adjust_depth(values, contrast=1.0, gamma=1.0):
    values = ((values.float() - 0.5) * float(contrast) + 0.5).clamp(0.0, 1.0)
    return values.pow(1.0 / float(gamma))


def apply_color_theme(values, theme):
    values = values.clamp(0.0, 1.0)
    if theme == "grayscale":
        return values.unsqueeze(-1).expand(*values.shape, 3).contiguous()
    if theme in ("grayscale_inverted", "inverted_grayscale"):
        inverted = 1.0 - values
        return inverted.unsqueeze(-1).expand(*values.shape, 3).contiguous()
    if theme == "turbo":
        return turbo(values)

    if theme not in COLOR_THEMES:
        raise ValueError(f"Unknown depth color theme: {theme}")
    colors = values.new_tensor(COLOR_THEMES[theme])
    position = values * (len(colors) - 1)
    lower = position.floor().long().clamp(max=len(colors) - 2)
    fraction = (position - lower).unsqueeze(-1)
    return torch.lerp(colors[lower], colors[lower + 1], fraction)
