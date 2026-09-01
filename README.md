# bokujuu-personal-nodes

Personal [ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes maintained by bokujuu.

## Bokujuu Anima Stream Loop

`Bokujuu Anima Stream Loop` is the live Anima StreamDiffusion node. Independent frames sit at different sigma stages; each tick injects new noise, steps the mixed-timestep batch with Euler, and shows the latest completed Light TAE frame on the loop node. Queue the graph once and press Cancel to stop.

The canonical graph is `workflows/anima_stream_loop.json`, with API graph `workflows/anima_stream_loop_api.json`. It uses `fnMomentAnimaTurbo_v40NoTurbo.safetensors` at 768×768, CFG 1, 4 steps, official `anima-turbo-lora-v0.2.safetensors`, `frames_per_tick=1`, and `TorchCompileModelAdvanced` with `dynamic=false`. Four-step Anima graphs always include that Turbo LoRA.

- Images appear on the loop node as a single live preview. CLIP text is sent as you type; a prompt change updates conditioning on the next tick while in-flight slots keep their remaining steps.
- `Empty Latent Image` supplies width and height. Its batch size is not the run length. Pipeline width is `frames_per_tick * steps`. The default `frames_per_tick=1` keeps 4 mixed-timestep latents in one static DiT forward at 4 steps.
- Connect `CLIP` to pick up `CLIPTextEncode` edits while the loop is running.
- Two steps on `BasicScheduler` is faster and softer. LCM is not compatible with Anima.
- On first use the node downloads `lightx2v/Autoencoders/lighttaew2_1.safetensors` into `ComfyUI/models/vae_approx/` if that file is missing.

## Bokujuu Anima Stream Batch Sampler

`Bokujuu Anima Stream Batch Sampler` is the finite-batch variant of the same mixed-timestep Euler pipeline. Connect it to `SamplerCustomAdvanced` with `RandomNoise`, `BasicGuider`, and `BasicScheduler`.

- `microbatch_size` limits the largest mixed-timestep forward without changing output order.
- `frames_per_tick` injects that many new latents each stream tick.
- `live_preview` decodes each completed frame with Light TAE and pushes it to every `Preview Image` node in the current prompt.
- The batch sampler accepts a finite latent batch and supports txt2img only. Masks and img2img are outside that node.
- Unit-test graphs: `workflows/anima_stream_baseline_test_api.json`, `workflows/anima_stream_batch_test_api.json`, and `workflows/anima_stream_realtime_test.json`.

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

## Experimental DLSS 5 / DLSS Super Resolution

Three nodes expose an experimental NVIDIA DLSS Super Resolution path for generated images and ordinary video frames:

- `Bokujuu DLSS Guidance Depth (DA3)` estimates a normalized DA3 relative-depth tensor and a colorized preview. The upscale nodes do not consume this; they keep an internal constant depth of 0.5.
- `Bokujuu DLSS 5 Neural Upscale` applies true DLSS Super Resolution to independent images or an IMAGE batch. Each image resets DLSS history. The default is Balanced plus CAS 0.7.
- `Bokujuu DLSS 5 Temporal Upscale` reuses one DLSS feature session for an IMAGE sequence, retaining history until a scene cut is detected. It uses the same still-image defaults.

The implementation calls the public DLSS-SR `SuperSampling` feature with different input and output dimensions. It does not call same-resolution Feature 18 Neural Rendering and does not pre-resize with Lanczos while labeling that resize as DLSS.

### Runtime setup

This repository does not include NVIDIA SDK headers or proprietary runtime DLLs.

1. Obtain the [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) and review its license.
2. Copy a compatible production `nvngx_dlss.dll` to `ComfyUI/models/dlss5/nvngx_dlss.dll`.
3. Build the project-owned bridge locally:

```powershell
cd D:\ComfyUI_20260315\custom_nodes\bokujuu-personal-nodes
.\native\build_bridge.ps1 -SdkPath C:\path\to\NVIDIA-DLSS
```

The generated `native/bin/bokujuu_dlss_sr_bridge.dll` is ignored by git. Missing bridge/runtime errors are deferred until a DLSS node executes, so other Bokujuu nodes continue to load normally.

### Usage and limitations

Connect an IMAGE to `Bokujuu DLSS 5 Neural Upscale`. Depth mode is not exposed; the node always supplies a constant normalized depth of 0.5 internally.

- `quality` is ordered from fastest/lowest quality to slowest/highest quality: Ultra Performance, Performance, Balanced (default), Quality, Ultra Quality, DLAA. DLAA is 1.0x only. Ultra Quality was rejected at 2.0x on this runtime. On a reset still frame, Ultra Performance through Quality produced byte-identical RGB.
- `cas_amount` is a spatial post-filter after DLSS-SR, not a DLSS quality mode. Default is 0.7. `0` disables it. Values above 1.0 are experimental.
- `scale` accepts 1.0x through 3.0x.
- `output_mix` blends DLSS output with an independently generated Lanczos target-resolution baseline.
- RGBA alpha bypasses DLSS and is resized separately before recombination.
- SDR/sRGB input in `[0,1]` is assumed. HDR, frame generation, reactive masks, optical flow, and engine motion-vector inputs are outside this implementation.
- Ordinary video has no accurate engine motion vectors or sub-pixel jitter. The temporal node sends zero motion vectors and can therefore ghost on motion; it is not equivalent to in-game DLSS integration.
- These nodes and their bridge are unofficial and experimental. A compatible NVIDIA RTX GPU, current driver, and user-supplied NVIDIA runtime are required.

The editable still-image example is `workflows/dlss5_image_test.json`. It uses ComfyUI's bundled `input/example.png` and writes under `output/bokujuu_dlss5/`. A DA3 depth-preview graph is `workflows/dlss5_depth_test.json`. API graphs are `workflows/dlss5_image_test_api.json`, `workflows/dlss5_depth_test_api.json`, and `workflows/dlss5_temporal_test_api.json`. The temporal graph expects [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite). Investigation details and measured results are in `docs/dlss5-investigation.md`.

For local comparisons:

```powershell
D:\ComfyUI_20260315\.venv\Scripts\python.exe scripts\run_dlss5_comparison.py --input C:\path\to\image.png
D:\ComfyUI_20260315\.venv\Scripts\python.exe scripts\run_dlss5_temporal_smoke.py --input C:\path\to\video.mp4 --output-dir D:\ComfyUI_20260315\output\bokujuu_dlss5_test\video
```

The comparison script tests Lanczos and the installed RTX Video Super Resolution custom-node backend. It also tests an installed ESRGAN-compatible model when one exists under `ComfyUI/models/upscale_models/`.

Quality against CAS amount:

```powershell
D:\ComfyUI_20260315\.venv\Scripts\python.exe scripts\run_dlss5_quality_cas_matrix.py --input C:\path\to\image.png --output-dir D:\ComfyUI_20260315\output\bokujuu_dlss5_test\quality_cas
D:\ComfyUI_20260315\.venv\Scripts\python.exe scripts\emit_dlss5_split_compare.py --left C:\path\to\left.png --right C:\path\to\right.png --output-dir D:\ComfyUI_20260315\output\bokujuu_dlss5_test\split
```

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
- `filename_prefix` accepts the same tokens as ComfyUI's Save Image node, including `%date:yyyy-MM-dd%` and `%Empty Latent Image.width%`.
- Lossless mode is intentionally not exposed.
- ComfyUI's `--disable-metadata` option is respected.

The editable example is `workflows/webp_save_example.json`. It saves ComfyUI's bundled `input/example.png` under `output/bokujuu_webp/`.

## Bokujuu Seed Control

`Bokujuu Seed Control` is a frontend controller for seed widgets in the main workflow and every nested subgraph. It detects nodes that use ComfyUI's `control_after_generate` setting and lets each one remain unchanged, keep its current value fixed, or randomize after every run.

- Use `Fix All`, `Randomize All`, and `Clear All` for bulk changes.
- Each row shows the node id, title, seed value, effective mode, and its full path from the main workflow through nested subgraphs.
- `↗` opens the containing graph and centers the canvas on the selected node.
- Connected seed inputs are shown but disabled because their values are owned by the upstream connection.
- If EasyUse's Global Seed node is present, the panel shows a conflict warning.
- Labels follow ComfyUI's locale setting for English and Japanese; English is used for other locales.

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

Run the native DLSS integration test explicitly on a configured RTX machine:

```powershell
$env:BOKUJUU_RUN_DLSS_INTEGRATION="1"
D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## License

MIT
