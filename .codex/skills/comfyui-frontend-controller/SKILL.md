---
name: comfyui-frontend-controller
description: "Repo-local practice for ComfyUI frontend dashboard nodes that control existing workflow widgets without server-side prompt mutation."
---

# ComfyUI Frontend Controller

Use this practice for input-less dashboard nodes that inspect and control other nodes in the current graph.

1. Read `references/observations.md` and `references/sources.md`.
2. Keep execution state on the widgets that already own it; store only override choices on the controller node.
3. Verify the UI in a running ComfyUI instance as well as with the Python test suite.

## Verification

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- Browser test: mix fixed and random modes, run twice, and confirm only random seeds change.
