# DLSS 5 / Super Resolution investigation

Date: 2026-09-01

## Decision

The custom node uses NVIDIA DLSS Super Resolution (`NVSDK_NGX_Feature_SuperSampling`) for actual input/output resolution separation. It does not present DLSS Neural Rendering Feature 18 as an upscaler.

Still-image nodes default to a constant normalized depth of 0.5 because DA3, Quality, and constant 0.0/0.5/1.0 did not change RGB on reset independent frames. Depth Anything V3 remains an optional relative-depth path. The same `DLSSGuidance` object is returned by the guidance-depth node and consumed by the DLSS nodes, so the preview and native input use the same depth tensor before visualization-only contrast, gamma, and colormap adjustments.

## Reference implementation boundary

The inspected `codon2001/comfyui-dlssnr-test` main revision was `1484ec33d8d13f4dff6dbd85692bf43794051c86` (2026-09-01). Its repository root had no LICENSE file and its own source had no license header. No source or binary from that repository is included here.

Its Feature 18 bridge explicitly configured:

- input width equal to output width;
- input height equal to output height;
- `DLSSNR.Upscaling = 0`;
- scale and scaling ratio equal to `1.0`.

That path is same-resolution neural rendering, not DLSS Super Resolution. The implementation in this repository was written independently against the public [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) headers and programming guide.

## NVIDIA API findings

The investigated NVIDIA SDK revision was DLSS 310.7.0 (`a291cc7d2cc642a51566f3dfd5376f635cd1b284`, 2026-06-23).

The official API distinguishes render and target dimensions in `NVSDK_NGX_DLSS_Create_Params`. DLSS-SR evaluation requires a color buffer, depth buffer, motion vectors, output buffer, jitter values, and reset state. The public guide also states that accurate per-pixel motion vectors, a renderer depth buffer, and sub-pixel jitter are required for normal high-quality engine integration. See the [NVIDIA DLSS repository](https://github.com/NVIDIA/DLSS) and the [Streamline DLSS-SR integration framework](https://github.com/NVIDIA-RTX/Streamline).

Ordinary generated images and decoded videos do not contain those renderer signals. This node therefore uses:

- DA3 normalized relative depth instead of projection-space engine depth;
- zero motion vectors;
- zero jitter;
- auto exposure;
- history reset for every independent still image;
- one retained DLSS feature session for a temporal IMAGE batch, reset on frame zero or a detected scene cut.

This is an experimental media reconstruction path, not equivalent to a game-engine DLSS integration. In particular, video can ghost because real object/camera motion vectors are unavailable.

## Runtime and licensing

- NVIDIA runtime DLLs and SDK headers are not committed to this repository.
- The user places a legally obtained `nvngx_dlss.dll` in `ComfyUI/models/dlss5/`.
- `native/build_bridge.ps1` builds the project-owned bridge from source against a local NVIDIA DLSS SDK checkout.
- The generated bridge DLL is ignored by git.
- ComfyUI still imports all other Bokujuu nodes when the bridge or runtime is absent; a targeted error is raised only when a DLSS node executes.
- Users must review and comply with NVIDIA's SDK and runtime license before distributing a build.

## Verification on this machine

Hardware and software:

- NVIDIA GeForce RTX 5080, 16 GB;
- NVIDIA driver 610.88;
- DLSS SDK/runtime 310.7.0;
- Windows D3D11 bridge.

Results:

- Native integration: 1.0x, 1.5x, 2.0x, and 3.0x all produced finite tensors in `[0,1]` at their requested output dimensions.
- Portrait comparison: 880x1168 to 1760x2336 completed with DA3 guidance. DLSS-SR took 2.47 seconds after depth estimation; RTX Video Super Resolution took 2.32 seconds; Lanczos took 0.12 seconds.
- The portrait had no high-resolution ground truth. Gradient sharpness was 0.00761 for DLSS-SR, 0.01339 for RTX VSR, and 0.00789 for Lanczos; these are descriptive values, not quality scores.
- ESRGAN was skipped because `ComfyUI/models/upscale_models/` contained no model file.
- Temporal smoke test: three consecutive frames from a C-drive MP4 were processed in one native session from 320x180 to 640x360. Frame count was preserved, all values were finite, and the output range was within `[0,1]`.
- Still-image factorial test: one 880x1200 source was upscaled to 1760x2400 with Balanced/Quality and DA3 at 504/1008/2016 plus constant depth 0.5. All eight raw DLSS outputs were byte-identical. On a reset independent still frame, neither the tested quality enum nor depth input changed this runtime's RGB result; post-sharpening did change it.
- The same source at constant depth 0.0, 0.5, and 1.0 was also byte-identical. Constant 0.5 is therefore a required depth-resource placeholder, not a blend or focus control.
- The node default path (`constant` 0.5, Balanced at 2x, CAS 0.7) matched CAS applied to that raw 0.5 output.
- Quality x CAS matrix on the same source: Ultra Performance, Performance, Balanced, and Quality raw outputs were byte-identical. Ultra Quality failed DLSS feature creation at 2.0x (`0xBAD00010`). CAS amounts 0, 0.35, 0.7, 1, 2, and 4 all produced different RGB. Gradient means on Balanced were 0.01755, 0.02291, 0.02531, 0.02859, 0.08388, and 0.05542; clipped fractions stayed near 0.63% through CAS 1.0, then rose to 2.80% at 2.0 and 2.02% at 4.0. CAS 2 and 4 are past the useful range.
- Crop 4x comparison on the same 400x400 face window (the 800px region from the 2x grid, mapped back to source) to 1600x1600: Lanczos 0.054s / gradient 0.0168, RTX VSR Ultra 1.45s / 0.0259, DLSS-SR Ultra Performance plus CAS 0.7 2.41s / 0.0228. All three RGB results differed. DLSS 4x succeeded through the native SuperSampling path; the node UI still caps scale at 3.0. Subjectively, RTX VSR looked over-sharpened on eyelashes, jaw contours, and the chest gem; DLSS plus CAS 0.7 looked smoother. This is a visual judgment, not a metric ranking.
- Official NVIDIA Image Scaling `NVSharpen` 0.25, ComfyUI Essentials CAS 0.35, and a weak Unsharp Mask were tested after DLSS. Their gradient means were 0.02116, 0.02291, and 0.02350 versus 0.01755 raw; Unsharp also increased clipped pixels substantially, so the gradient values are not quality rankings.

Generated local results are under `ComfyUI/output/bokujuu_dlss5_test/`.
