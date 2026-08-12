# Observations

## Summary

Implement ComfyUI V3 custom nodes with a subgraph-safe searchable LoRA selector and deterministic local RNG.

## Adoption Reason

Use a socketless V3 string input with a dedicated DOM widget. Store the selection as a JSON string and accept legacy list values when workflows are loaded.

## Avoid

- Do not rely on `io.MultiCombo` for this node while the Vue-only component can disappear inside subgraphs.
- Do not mutate global RNG state.

## Experiment Notes

- ComfyUI frontend 1.48.6 exposes MultiCombo metadata but its component widget is not mounted reliably inside subgraphs.
- A custom DOM widget registered through `getCustomWidgets` remains visible and serializes selected filenames as JSON.
- The Python parser accepts both the JSON string and old list-shaped workflow values.
- `io.NodeOutput` cleanly returns LORA_STACK, report STRING, and count INT.

## Verification Commands

- `D:\ComfyUI_20260315\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- Verify the selector in a browser: open the dialog, search, select, apply, reload the workflow, and confirm the count.
