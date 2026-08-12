# bokujuu-personal-nodes

Personal [ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes maintained by bokujuu.

## Bokujuu LoRA Weight Randomizer

Assigns reproducible random weights to every selected LoRA and returns an EasyUse-compatible `LORA_STACK`.

- Select any number of LoRAs with one searchable multi-select input.
- `minimum_strength` accepts negative and positive values.
- Every non-empty LoRA input is included in the output stack.
- `loaded_loras` reports the complete current stack without relying on a separate Show Any node.
- An optional input stack is preserved and merged by LoRA name.
- Uses a local seeded RNG and does not modify Python or PyTorch global random state.

`total_strength` is the target sum for the LoRAs configured on this node. When the target is outside the feasible range for the selected count and strength bounds, it is clamped to that range. If `minimum_strength` is greater than `maximum_strength`, the minimum is used as the effective maximum and no error is raised.

## Installation

```powershell
cd D:\ComfyUI\custom_nodes
git clone https://github.com/bokujuu/bokujuu-personal-nodes.git
```

Restart ComfyUI after installation.

## Development

```powershell
python -m unittest discover -s tests -v
```

## License

MIT
