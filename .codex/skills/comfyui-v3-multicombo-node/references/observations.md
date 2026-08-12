# Observations

## Summary

Implement ComfyUI V3 custom nodes with a MultiCombo list widget and deterministic local RNG.

## Adoption Reason

Use the V3 schema API and MultiCombo so one searchable widget can select any number of LoRAs.

## Avoid

- Do not pass a raw Python list in API prompts; wrap MultiCombo values as {"__value__": [...]}, and do not mutate global RNG state.

## Experiment Notes

- ComfyUI treats top-level two-item lists as graph links. MultiCombo prompt values serialize through the __value__ wrapper. io.NodeOutput cleanly returns LORA_STACK, report STRING, and count INT.

## Verification Commands

- `python -m unittest discover -s tests -v`
