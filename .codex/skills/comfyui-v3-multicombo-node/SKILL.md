---
name: comfyui-v3-multicombo-node
description: "Repo-local practice for ComfyUI V3 nodes that need a subgraph-safe LoRA multi-selector and deterministic local RNG."
---

# ComfyUI V3 LoRA Multi-Selector

Use this repo-local skill when working on a ComfyUI V3 node with a searchable LoRA multi-selector and deterministic local RNG.

## Workflow

1. Read `references/observations.md` to understand the adopted approach and known pitfalls.
2. Review `references/sources.md` before changing the pattern.
3. Run the listed verification commands before claiming completion.

## Trigger Keywords

- `ComfyUI V3 node`
- `MultiCombo`
- `LoRA selector`

## Verification

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
