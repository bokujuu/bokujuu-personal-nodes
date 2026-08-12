import math
import random

import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


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


class BokujuuPersonalNodes(ComfyExtension):
    @override
    async def get_node_list(self):
        return [BokujuuLoraWeightRandomizer]
