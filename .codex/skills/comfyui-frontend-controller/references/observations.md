# Observations

- A no-input/no-output V3 server node provides stable menu and workflow registration while all behavior remains in the frontend extension.
- Reuse each seed widget's linked `control_after_generate` combo. Setting it to `fixed` or `randomize` preserves ComfyUI's normal queue lifecycle and avoids prompt interception.
- Persist override choices in the controller node's `properties`; DOM widgets themselves should use `serialize: false`.
- Treat a linked seed input as read-only because the upstream node owns its effective value.
- Bulk actions should update all targets first and render once so large workflows stay responsive.
- Browser verification must include two consecutive queues: fixed values stay unchanged and randomized values change both times.
