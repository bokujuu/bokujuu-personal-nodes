# Observations

## Summary

Implement a ComfyUI V3 custom node that downloads, converts, loads, and runs official Depth Anything 3 checkpoints through ComfyUI core.

## Adoption Reason

Reuse ComfyUI's `DA3Inference`, `DA3Render`, model patcher, preprocessing, device placement, and offloading paths instead of maintaining another model implementation. Convert the official checkpoint once because current ComfyUI detects its native DINOv2 key layout, while `depth-anything/DA3-LARGE-1.1` uses the upstream fused-QKV layout.

## Avoid

- Do not perform network access at import or startup; download only when the loader executes and the local model is absent.
- Do not pass the original Hugging Face checkpoint directly to `comfy.sd.load_diffusion_model`; convert backbone keys and split fused QKV first.
- Expand shared `output_conv2_aux.0` weights to the four expected auxiliary head entries before saving, otherwise loading warns about missing keys.

## Experiment Notes

- `DA3-LARGE-1.1` converted to 739 tensor keys and was detected as `DepthAnything3` with `vitl`, `dualdpt`, camera encoder, and camera decoder.
- A CUDA smoke test at 140x196 produced a finite float32 IMAGE with the original input dimensions.

## Verification Commands

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
