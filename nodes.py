import math
import os
import random

import folder_paths
import torch
import comfy.sd
import comfy.utils
from comfy.ldm.colormap import turbo
from comfy_extras.nodes_depth_anything_3 import DA3Inference, DA3ModelType, DA3Render
from comfy_api.latest import ComfyExtension, io
from huggingface_hub import hf_hub_download
from typing_extensions import override


DA3_REPO_ID = "depth-anything/DA3-LARGE-1.1"
DA3_REPO_FILENAME = "model.safetensors"
DA3_MODEL_FILENAME = "depth_anything_3_large_1.1.safetensors"

_COLOR_THEMES = {
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


def _geometry_model_path(filename=DA3_MODEL_FILENAME):
    return os.path.join(folder_paths.models_dir, "geometry_estimation", filename)


def _strip_model_prefix(state_dict):
    if "model.backbone.pretrained.patch_embed.proj.weight" not in state_dict:
        return state_dict
    return {key[6:] if key.startswith("model.") else key: value for key, value in state_dict.items()}


def _expand_shared_aux_weights(state_dict):
    for key in [key for key in state_dict if "output_conv2_aux.0." in key]:
        prefix, suffix = key.split("output_conv2_aux.0.", 1)
        for index in range(1, 4):
            target = f"{prefix}output_conv2_aux.{index}.{suffix}"
            if target not in state_dict:
                state_dict[target] = state_dict[key].clone()
    return state_dict


def convert_da3_state_dict(state_dict):
    state_dict = _strip_model_prefix(state_dict)
    prefix = "backbone."
    source_prefix = prefix + "pretrained."
    if prefix + "embeddings.patch_embeddings.projection.weight" in state_dict:
        return _expand_shared_aux_weights(state_dict)
    if source_prefix + "patch_embed.proj.weight" not in state_dict:
        raise ValueError(f"{DA3_REPO_ID} checkpoint has an unsupported state-dict layout")

    for key in list(state_dict):
        if key.startswith(("gs_head.", "gs_adapter.")):
            state_dict.pop(key)

    static_renames = {
        source_prefix + "patch_embed.proj.weight": prefix + "embeddings.patch_embeddings.projection.weight",
        source_prefix + "patch_embed.proj.bias": prefix + "embeddings.patch_embeddings.projection.bias",
        source_prefix + "pos_embed": prefix + "embeddings.position_embeddings",
        source_prefix + "cls_token": prefix + "embeddings.cls_token",
        source_prefix + "camera_token": prefix + "embeddings.camera_token",
        source_prefix + "norm.weight": prefix + "layernorm.weight",
        source_prefix + "norm.bias": prefix + "layernorm.bias",
    }
    for source, target in static_renames.items():
        if source in state_dict:
            state_dict[target] = state_dict.pop(source)

    block_prefix = source_prefix + "blocks."
    for key in [key for key in state_dict if key.startswith(block_prefix)]:
        rest = key[len(block_prefix):]
        index, _, suffix = rest.partition(".")
        target_prefix = f"{prefix}encoder.layer.{index}."

        if suffix in ("attn.qkv.weight", "attn.qkv.bias"):
            qkv = state_dict.pop(key)
            query, key_tensor, value = qkv.chunk(3, dim=0)
            parameter = "weight" if suffix.endswith("weight") else "bias"
            state_dict[target_prefix + f"attention.attention.query.{parameter}"] = query.contiguous()
            state_dict[target_prefix + f"attention.attention.key.{parameter}"] = key_tensor.contiguous()
            state_dict[target_prefix + f"attention.attention.value.{parameter}"] = value.contiguous()
            continue

        replacements = (
            ("attn.proj.", "attention.output.dense."),
            ("attn.q_norm.", "attention.q_norm."),
            ("attn.k_norm.", "attention.k_norm."),
            ("mlp.w12.", "mlp.weights_in."),
            ("mlp.w3.", "mlp.weights_out."),
        )
        target_suffix = None
        for source, target in replacements:
            if suffix.startswith(source):
                target_suffix = target + suffix[len(source):]
                break
        if suffix == "ls1.gamma":
            target_suffix = "layer_scale1.lambda1"
        elif suffix == "ls2.gamma":
            target_suffix = "layer_scale2.lambda1"
        elif suffix.startswith(("norm1.", "norm2.", "mlp.fc1.", "mlp.fc2.")):
            target_suffix = suffix

        if target_suffix is not None:
            state_dict[target_prefix + target_suffix] = state_dict.pop(key)

    return _expand_shared_aux_weights(state_dict)


def ensure_da3_large_model():
    model_path = _geometry_model_path()
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    source_path = hf_hub_download(repo_id=DA3_REPO_ID, filename=DA3_REPO_FILENAME)
    state_dict, metadata = comfy.utils.load_torch_file(source_path, return_metadata=True)
    state_dict = convert_da3_state_dict(state_dict)
    metadata = dict(metadata or {})
    metadata.update({"source_repo": DA3_REPO_ID, "source_file": DA3_REPO_FILENAME})

    temporary_path = model_path + ".tmp"
    try:
        comfy.utils.save_torch_file(state_dict, temporary_path, metadata=metadata)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return model_path


def apply_color_theme(values, theme):
    values = values.clamp(0.0, 1.0)
    if theme == "grayscale":
        return values.unsqueeze(-1).expand(*values.shape, 3).contiguous()
    if theme == "grayscale_inverted":
        inverted = 1.0 - values
        return inverted.unsqueeze(-1).expand(*values.shape, 3).contiguous()
    if theme == "turbo":
        return turbo(values)

    colors = values.new_tensor(_COLOR_THEMES[theme])
    position = values * (len(colors) - 1)
    lower = position.floor().long().clamp(max=len(colors) - 2)
    fraction = (position - lower).unsqueeze(-1)
    return torch.lerp(colors[lower], colors[lower + 1], fraction)


def _normalize_stack_row(row):
    name = row[0]
    model_strength = float(row[1])
    clip_strength = float(row[2]) if len(row) > 2 else model_strength
    return [name, model_strength, clip_strength]


def _merge_lora_stack(stack):
    result = []
    positions = {}
    for row in stack:
        name, model_strength, clip_strength = _normalize_stack_row(row)
        if name in positions:
            existing = result[positions[name]]
            existing[1] = round(existing[1] + model_strength, 2)
            existing[2] = round(existing[2] + clip_strength, 2)
        else:
            positions[name] = len(result)
            result.append([name, round(model_strength, 2), round(clip_strength, 2)])
    return result


def _weighted_allocation(total_cents, capacity_cents, rng):
    if total_cents <= 0 or not capacity_cents:
        return [0] * len(capacity_cents)

    weights = [rng.random() + 1e-12 for _ in capacity_cents]
    raw = [0.0] * len(capacity_cents)
    active = set(range(len(capacity_cents)))
    remaining = float(total_cents)

    while active and remaining > 0:
        weight_total = sum(weights[i] for i in active)
        saturated = []
        for i in active:
            share = remaining * weights[i] / weight_total
            if share >= capacity_cents[i]:
                raw[i] = float(capacity_cents[i])
                saturated.append(i)

        if not saturated:
            for i in active:
                raw[i] = remaining * weights[i] / weight_total
            break

        for i in saturated:
            remaining -= raw[i]
            active.remove(i)

    allocation = [min(math.floor(value), capacity) for value, capacity in zip(raw, capacity_cents)]
    missing = total_cents - sum(allocation)
    order = list(range(len(allocation)))
    rng.shuffle(order)
    order.sort(key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in order:
        if missing == 0:
            break
        if allocation[i] < capacity_cents[i]:
            allocation[i] += 1
            missing -= 1
    return allocation


def random_strengths(count, total_strength, minimum_strength, maximum_strength, randomize_total, seed):
    if count == 0:
        return []

    rng = random.Random(int(seed))
    target = rng.uniform(0.0, float(total_strength)) if randomize_total else float(total_strength)

    minimum_cents = round(float(minimum_strength) * 100)
    maximum_cents = max(round(float(maximum_strength) * 100), minimum_cents)
    lower_total = count * minimum_cents
    upper_total = count * maximum_cents
    target_cents = min(max(round(target * 100), lower_total), upper_total)

    capacity = maximum_cents - minimum_cents
    increments = _weighted_allocation(
        target_cents - lower_total,
        [capacity] * count,
        rng,
    )
    return [round((minimum_cents + value) / 100, 2) for value in increments]


def format_stack_report(stack):
    lines = [f"Loaded LoRAs: {len(stack)}"]
    for index, (name, model_strength, clip_strength) in enumerate(stack, 1):
        lines.append(f"{index:02d}. {name} | model={model_strength:.2f} | clip={clip_strength:.2f}")
    return "\n".join(lines)


class BokujuuLoraWeightRandomizer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lora_input = io.MultiCombo.Input(
            "loras",
            options=[""] + folder_paths.get_filename_list("loras"),
            default=[],
            placeholder="Select LoRAs",
            chip=True,
            socketless=True,
        )
        return io.Schema(
            node_id="BokujuuLoraWeightRandomizer",
            display_name="Bokujuu LoRA Weight Randomizer",
            category="Bokujuu/LoRA",
            description="Assigns reproducible random weights to every selected LoRA and reports the complete stack.",
            inputs=[
                io.Float.Input("total_strength", default=1.0, min=0.0, max=100.0, step=0.01),
                io.Float.Input("minimum_strength", default=-1.0, min=-10.0, max=10.0, step=0.01),
                io.Float.Input("maximum_strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Boolean.Input("randomize_total_strength", default=False),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                lora_input,
                io.Custom("LORA_STACK").Input("input_lora_stack", optional=True),
            ],
            outputs=[
                io.Custom("LORA_STACK").Output(display_name="lora_stack"),
                io.String.Output(display_name="loaded_loras"),
                io.Int.Output(display_name="lora_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        total_strength,
        minimum_strength,
        maximum_strength,
        randomize_total_strength,
        seed,
        loras=None,
        input_lora_stack=None,
    ):
        base = [_normalize_stack_row(row) for row in (input_lora_stack or [])]
        candidates = [name for name in (loras or []) if name]

        strengths = random_strengths(
            len(candidates),
            total_strength,
            minimum_strength,
            maximum_strength,
            randomize_total_strength,
            seed,
        )
        generated = [[name, strength, strength] for name, strength in zip(candidates, strengths)]
        stack = _merge_lora_stack(base + generated)
        return io.NodeOutput(stack, format_stack_report(stack), len(stack))


class BokujuuLoadDepthAnything3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BokujuuLoadDepthAnything3",
            display_name="Bokujuu Load Depth Anything 3",
            category="Bokujuu/Depth",
            description="Downloads DA3-LARGE-1.1 when needed, converts it to ComfyUI's native format, and loads it.",
            inputs=[
                io.Combo.Input("model", options=[DA3_REPO_ID], default=DA3_REPO_ID),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp16", "bf16", "fp32"],
                    default="default",
                ),
            ],
            outputs=[DA3ModelType.Output(display_name="da3_model")],
        )

    @classmethod
    def execute(cls, model, weight_dtype):
        if model != DA3_REPO_ID:
            raise ValueError(f"Unsupported model: {model}")

        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        return io.NodeOutput(comfy.sd.load_diffusion_model(ensure_da3_large_model(), model_options=model_options))


class BokujuuDepthAnything3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BokujuuDepthAnything3",
            display_name="Bokujuu Depth Anything 3",
            category="Bokujuu/Depth",
            description="Creates an editable depth image from an input image with DA3-LARGE-1.1.",
            inputs=[
                DA3ModelType.Input("da3_model"),
                io.Image.Input("image"),
                io.Int.Input("resolution", default=504, min=140, max=2520, step=14),
                io.Combo.Input(
                    "resize_method",
                    options=["upper_bound_resize", "lower_bound_resize"],
                    default="upper_bound_resize",
                ),
                io.Combo.Input(
                    "normalization",
                    options=["v2_style", "min_max"],
                    default="v2_style",
                ),
                io.Combo.Input(
                    "color_theme",
                    options=["grayscale", "grayscale_inverted", "turbo", "viridis", "plasma", "inferno"],
                    default="grayscale",
                ),
                io.Float.Input("contrast", default=1.0, min=0.0, max=3.0, step=0.05),
                io.Float.Input("gamma", default=1.0, min=0.1, max=5.0, step=0.05),
            ],
            outputs=[io.Image.Output(display_name="depth_image")],
        )

    @classmethod
    def execute(cls, da3_model, image, resolution, resize_method, normalization, color_theme, contrast, gamma):
        geometry = DA3Inference.execute(
            da3_model,
            image,
            resolution,
            resize_method,
            {"mode": "mono"},
        )[0]
        grey = DA3Render.execute(
            geometry,
            {
                "output": "depth",
                "normalization": normalization,
                "apply_sky_clip": False,
            },
        )[0][..., 0]
        grey = ((grey - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
        grey = grey.pow(1.0 / gamma)
        return io.NodeOutput(apply_color_theme(grey, color_theme).float())


class BokujuuPersonalNodes(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            BokujuuLoraWeightRandomizer,
            BokujuuLoadDepthAnything3,
            BokujuuDepthAnything3,
        ]
