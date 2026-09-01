import torch


def scene_cut_detected(previous: torch.Tensor | None, current: torch.Tensor, threshold: float) -> bool:
    if previous is None or threshold <= 0.0:
        return False
    difference = (current[..., :3].float() - previous[..., :3].float()).abs().mean().item()
    return difference >= float(threshold)
