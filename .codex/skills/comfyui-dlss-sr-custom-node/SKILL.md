---
name: comfyui-dlss-sr-custom-node
description: "Repo-local implementation practice for DLSS Super Resolution, DLSS 5 custom node. Use when working in this repository on Implement an experimental ComfyUI DLSS Super Resolution node with a locally built bridge, user-supplied NVIDIA runtime, and DA3 relative-depth guidance.."
---

# Comfyui Dlss Sr Custom Node

Use this repo-local skill when working on Implement an experimental ComfyUI DLSS Super Resolution node with a locally built bridge, user-supplied NVIDIA runtime, and DA3 relative-depth guidance. in this repository.

## Workflow

1. Read `references/observations.md` to understand the adopted approach and known pitfalls.
2. Review `references/sources.md` before changing the pattern.
3. Run the listed verification commands before claiming completion.

## Trigger Keywords

- `DLSS Super Resolution`
- `DLSS 5 custom node`

## Verification

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- `$env:BOKUJUU_RUN_DLSS_INTEGRATION=\"1\"; D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
