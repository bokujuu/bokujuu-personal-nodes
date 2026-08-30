# bokujuu-personal-nodes

Personal [ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes maintained by bokujuu.

## Bokujuu Depth Anything 3

Two nodes provide a compact DA3-LARGE-1.1 image-to-depth workflow:

- `Bokujuu Load Depth Anything 3` loads `depth-anything/DA3-LARGE-1.1`. On first use it downloads the official checkpoint, converts it to ComfyUI's native DA3 layout, and stores it in `ComfyUI/models/geometry_estimation/depth_anything_3_large_1.1.safetensors`.
- `Bokujuu Depth Anything 3` accepts an `IMAGE` and returns an editable depth `IMAGE` at the original size.

The processing node exposes inference resolution, resize method, normalization, contrast, gamma, and color theme. `grayscale` is the default. Other themes are inverted grayscale, Turbo, Viridis, Plasma, and Inferno.

The model download only occurs when the loader node is executed and the converted local model is missing.

The editable test workflow is `workflows/da3_large_1.1_test.json`. It uses ComfyUI's bundled `input/example.png` and writes its result under `output/bokujuu_da3_test/`. The companion `da3_large_1.1_test_api.json` can be submitted directly to the `/prompt` API for automated testing.

### Usage

1. Add `Bokujuu Load Depth Anything 3` and `Bokujuu Depth Anything 3`.
2. Connect the loader's `da3_model` output to the processing node.
3. Connect an `IMAGE` to the processing node and route `depth_image` to `Preview Image`, `Save Image`, or a ControlNet workflow.
4. Keep `color_theme` at `grayscale` for the usual depth-map appearance, or select another palette for visualization.

### Processing settings

- `resolution`: DA3 inference resolution. Higher values retain more detail and require more VRAM.
- `resize_method`: `upper_bound_resize` limits the longest side; `lower_bound_resize` preserves more detail on wide or tall images at higher memory cost.
- `normalization`: `v2_style` gives a perceptually balanced result; `min_max` stretches each depth map across the full output range.
- `contrast` and `gamma`: adjust the normalized image before applying the selected color theme.

### Notes

- The first loader execution downloads the upstream checkpoint (about 1.64 GB) and writes a converted ComfyUI checkpoint. Later runs reuse that local file.
- This node currently uses the monocular inference path and returns one relative-depth image for each input image. Camera poses, metric depth, meshes, and point clouds are intentionally outside its scope.
- Review and follow the license published with the upstream [`depth-anything/DA3-LARGE-1.1`](https://huggingface.co/depth-anything/DA3-LARGE-1.1) model before distribution or commercial use.

## Bokujuu LoRA Weight Randomizer

Assigns reproducible random weights to every selected LoRA and returns an EasyUse-compatible `LORA_STACK`.

- Click `Select LoRAs` to choose any number of LoRAs in a searchable multi-select dialog. The node shows the current count and selected filenames.
- `minimum_strength` accepts negative and positive values.
- Every non-empty LoRA input is included in the output stack.
- `loaded_loras` reports the complete current stack without relying on a separate Show Any node.
- An optional input stack is preserved and merged by LoRA name.
- Uses a local seeded RNG and does not modify Python or PyTorch global random state.

`total_strength` is the target sum for the LoRAs configured on this node. When the target is outside the feasible range for the selected count and strength bounds, it is clamped to that range. If `minimum_strength` is greater than `maximum_strength`, the minimum is used as the effective maximum and no error is raised.

## Bokujuu Random LoRA Selector

Builds a lottery pool from explicitly selected LoRAs, randomly selects a count between `minimum_count` and `maximum_count`, then draws that many entries without replacement. Each selected LoRA receives an independent random strength between `minimum_strength` and `maximum_strength`.

- Use the searchable `Select LoRAs` dialog to define the pool; no folder-wide selection is required.
- `seed` reproduces both the selected LoRAs and their strengths without changing the global Python or PyTorch random state.
- The selected count and LoRA choices are both reproduced by `seed`.
- If either count is larger than the pool, it is limited to the pool size. If `maximum_count` is smaller than `minimum_count`, the minimum is used.
- Existing workflows without `maximum_count` keep the previous fixed `selection_count` behavior.
- MODEL and CLIP receive the same random strength for each selected LoRA.
- An optional incoming `LORA_STACK` is preserved and merged by LoRA name.

If `minimum_strength` is greater than `maximum_strength`, every selected LoRA uses the minimum value, matching the existing weight randomizer behavior.

Connect `lora_stack` to EasyUse's `easy loraStackApply` node. The editable example is `workflows/random_lora_selector_example.json`; replace its example LoRA pool with files installed in your own `models/loras` directory.

## Personal-LoRA

`Personal-LoRA` builds the same `LORA_STACK` format without randomization. Choose one or more LoRAs with the searchable `Select LoRAs` dialog, then set fixed `model_strength` and `clip_strength` values for every selected LoRA. An incoming `LORA_STACK` is preserved and merged by LoRA name, so the node can be placed between stack-producing nodes inside a subgraph.

## Bokujuu Save WebP

`Bokujuu Save WebP` saves each input image as a lossy WebP while embedding the same prompt and workflow metadata format used by ComfyUI's built-in WebP saver. Saved files can be viewed as ordinary images and loaded back into ComfyUI to restore their workflow.

- `quality` controls lossy WebP quality from 1 to 100. The default is 85.
- `method` controls the WebP encoder effort from 0 (fastest) to 6 (slowest). The default is 4.
- Lossless mode is intentionally not exposed.
- ComfyUI's `--disable-metadata` option is respected.

The editable example is `workflows/webp_save_example.json`. It saves ComfyUI's bundled `input/example.png` under `output/bokujuu_webp/`.

## Bokujuu Seed Control

`Bokujuu Seed Control` is a frontend controller for seed widgets in the current graph. It detects nodes that use ComfyUI's `control_after_generate` setting and lets each one remain unchanged, keep its current value fixed, or randomize after every run.

- Use `すべて固定`, `すべてランダム`, and `すべて解除` for bulk changes.
- Each row shows the node id, title, seed value, and effective mode.
- `↗` centers the canvas on the selected node.
- Connected seed inputs are shown but disabled because their values are owned by the upstream connection.
- If EasyUse's Global Seed node is present, the panel shows a conflict warning.

The settings are stored in the workflow. The editable example is `workflows/seed_control_example.json`.

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
