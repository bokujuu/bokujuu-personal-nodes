# Observations

## Summary

Implement an experimental ComfyUI DLSS Super Resolution node with a locally built bridge, user-supplied NVIDIA runtime, and DA3 relative-depth guidance.

## Adoption Reason

Use public DLSS-SR Feature SuperSampling for real input/output resolution separation; keep proprietary NVIDIA binaries outside git and build the application bridge locally.

## Avoid

- Do not label same-resolution Feature 18 neural rendering or a Lanczos pre-resize as DLSS upscaling.
- Do not claim DA3 relative depth, zero motion vectors, and zero jitter are equivalent to game-engine DLSS inputs.

## Experiment Notes

- NVIDIA DLSS SDK 310.7.0 on RTX 5080/driver 610.88 produced a finite 2x D3D11 DLSS-SR output.
- A three-frame IMAGE batch reused one native feature session and reset history only for frame zero or a detected scene cut.
- On an independent 880x1200 still frame upscaled 2x, Balanced and Quality plus DA3 guidance at 504, 1008, and 2016 and constant depth 0.5 all produced byte-identical RGB output. Treat PerfQuality as a render-resolution/profile selector rather than a sharpening control, and do not claim DA3 improves reset still frames without a measured difference.
- The same still frame also produced byte-identical RGB for constant depth 0.0, 0.5, and 1.0. Keep 0.5 as a required depth-resource placeholder, not as mix, focus, or sharpness.
- Upscale nodes hide depth_mode and constant_depth. They always supply constant 0.5 internally.
- A 2x quality x CAS matrix on the same 880x1200 still frame found Ultra Performance, Performance, Balanced, and Quality raw outputs byte-identical. Ultra Quality failed feature creation at 2x (`0xBAD00010`). CAS 0/0.35/0.7/1/2/4 all differed; 2 and 4 oversharpened and increased clipping.
- A 4x crop of eyes, jaw contour, and chest gem found RTX VSR more angular and over-sharpened; DLSS Super Resolution plus CAS 0.7 looked smoother. That is a subjective still-frame judgment, not a claim that this path matches in-game DLSS.

## Verification Commands

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- `$env:BOKUJUU_RUN_DLSS_INTEGRATION=\"1\"; D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
