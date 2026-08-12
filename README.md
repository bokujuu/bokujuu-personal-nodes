# bokujuu-personal-nodes

Personal [ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes maintained by bokujuu.

## Bokujuu Depth Anything 3

Two nodes provide a compact DA3-LARGE-1.1 image-to-depth workflow:

- `Bokujuu Load Depth Anything 3` loads `depth-anything/DA3-LARGE-1.1`. On first use it downloads the official checkpoint, converts it to ComfyUI's native DA3 layout, and stores it in `ComfyUI/models/geometry_estimation/depth_anything_3_large_1.1.safetensors`.
- `Bokujuu Depth Anything 3` accepts an `IMAGE` and returns an editable depth `IMAGE` at the original size.

The processing node exposes inference resolution, resize method, normalization, contrast, gamma, and color theme. `grayscale` is the default. Other themes are inverted grayscale, Turbo, Viridis, Plasma, and Inferno.

The model download only occurs when the loader node is executed and the converted local model is missing.

The editable test workflow is `workflows/da3_large_1.1_test.json`. It uses ComfyUI's bundled `input/example.png` and writes its result under `output/bokujuu_da3_test/`. The companion `da3_large_1.1_test_api.json` can be submitted directly to the `/prompt` API for automated testing.

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
