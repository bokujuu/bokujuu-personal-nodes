from dataclasses import dataclass

import torch


@dataclass
class StreamSlot:
    latent: torch.Tensor
    stage_index: int
    frame_id: int


@dataclass(frozen=True)
class StreamBatchStats:
    ticks: int
    forward_calls: int
    max_active_slots: int
    max_forward_batch: int


def nearest_complete_slot(active, stage_count):
    if not active:
        return None
    target = max(stage_count - 1, 0)
    chosen = [slot for slot in active if slot.stage_index == target]
    if not chosen:
        max_stage = max(slot.stage_index for slot in active)
        chosen = [slot for slot in active if slot.stage_index == max_stage]
    return min(chosen, key=lambda slot: slot.frame_id)


def flow_euler_step(x, denoised, sigma, sigma_next):
    sigma = sigma.reshape([sigma.shape[0]] + [1] * (x.ndim - 1))
    sigma_next = sigma_next.reshape([sigma_next.shape[0]] + [1] * (x.ndim - 1))
    return torch.addcmul(x, x - denoised, (sigma_next - sigma) / sigma)


def _sigma_levels(sigmas, ref):
    return sigmas.to(device=ref.device, dtype=ref.dtype)


def _step_active(model, active, sigma_levels, microbatch_size):
    forward_calls = 0
    max_batch = 0
    for start in range(0, len(active), microbatch_size):
        chunk = active[start:start + microbatch_size]
        x = torch.cat([slot.latent for slot in chunk], dim=0)
        stage = torch.tensor([slot.stage_index for slot in chunk], device=x.device, dtype=torch.long)
        denoised = model(x, sigma_levels[stage])
        x_next = flow_euler_step(x, denoised, sigma_levels[stage], sigma_levels[stage + 1])
        for i, slot in enumerate(chunk):
            slot.latent = x_next[i:i + 1].clone()
        forward_calls += 1
        max_batch = max(max_batch, len(chunk))
    return forward_calls, max_batch


def _step_packed(model, latents, stages, sigma_levels, microbatch_size, count, capacity):
    if count < capacity:
        latents[count:capacity].copy_(latents[0].unsqueeze(0).expand(capacity - count, *latents.shape[1:]))
        stages[count:capacity] = 0
    forward_calls = 0
    max_batch = 0
    for start in range(0, capacity, microbatch_size):
        end = min(start + microbatch_size, capacity)
        x = latents[start:end]
        stage = stages[start:end]
        denoised = model(x, sigma_levels[stage])
        x_next = flow_euler_step(x, denoised, sigma_levels[stage], sigma_levels[stage + 1])
        write = min(end, count) - start
        if write > 0:
            latents[start:start + write].copy_(x_next[:write])
        forward_calls += 1
        max_batch = max(max_batch, end - start)
    stages[:count] += 1
    return forward_calls, max_batch


def _packed_active(latents, stages, frame_ids, count):
    return [
        StreamSlot(latents[i:i + 1], int(stages[i].item()), frame_ids[i])
        for i in range(count)
    ]


def sample_stream_batch(model, initial_latents, sigmas, microbatch_size=4, frames_per_tick=1, frame_callback=None, tick_callback=None):
    if initial_latents.shape[0] == 0:
        return initial_latents, StreamBatchStats(0, 0, 0, 0)
    if len(sigmas) < 2:
        raise ValueError("Stream Batch requires at least one sigma transition")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be at least 1")
    if frames_per_tick < 1:
        raise ValueError("frames_per_tick must be at least 1")

    stage_count = len(sigmas) - 1
    frame_count = initial_latents.shape[0]
    outputs = [None] * frame_count
    active = []
    next_frame_id = 0
    ticks = 0
    forward_calls = 0
    max_active_slots = 0
    max_forward_batch = 0
    sigma_levels = _sigma_levels(sigmas, initial_latents)

    while next_frame_id < frame_count or active:
        injected = 0
        while injected < frames_per_tick and next_frame_id < frame_count:
            active.append(StreamSlot(initial_latents[next_frame_id:next_frame_id + 1], 0, next_frame_id))
            next_frame_id += 1
            injected += 1
        if not active:
            break

        max_active_slots = max(max_active_slots, len(active))
        calls, batch = _step_active(model, active, sigma_levels, microbatch_size)
        forward_calls += calls
        max_forward_batch = max(max_forward_batch, batch)

        remaining = []
        for slot in active:
            slot.stage_index += 1
            if slot.stage_index == stage_count:
                outputs[slot.frame_id] = slot.latent
                if frame_callback is not None:
                    frame_callback(slot.frame_id, slot.latent)
            else:
                remaining.append(slot)
        active = remaining
        ticks += 1
        if tick_callback is not None:
            tick_callback(active)

    return torch.cat(outputs, dim=0), StreamBatchStats(
        ticks=ticks,
        forward_calls=forward_calls,
        max_active_slots=max_active_slots,
        max_forward_batch=max_forward_batch,
    )


def sample_stream_loop(model, make_latent, sigmas, microbatch_size=4, frames_per_tick=2, frame_callback=None, should_stop=None, tick_callback=None, reset_pipeline=None):
    if len(sigmas) < 2:
        raise ValueError("Stream Batch requires at least one sigma transition")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be at least 1")
    if frames_per_tick < 1:
        raise ValueError("frames_per_tick must be at least 1")
    if should_stop is None:
        should_stop = lambda: False

    stage_count = len(sigmas) - 1
    capacity = stage_count * frames_per_tick
    latents = None
    stages = None
    frame_ids = None
    sigma_levels = None
    count = 0
    next_frame_id = 0
    ticks = 0
    forward_calls = 0
    max_active_slots = 0
    max_forward_batch = 0

    def inject():
        nonlocal latents, stages, frame_ids, sigma_levels, count, next_frame_id
        injected = 0
        while injected < frames_per_tick and count < capacity:
            z = make_latent(next_frame_id)
            if latents is None:
                latents = z.new_empty((capacity,) + z.shape[1:])
                stages = torch.zeros(capacity, dtype=torch.long, device=z.device)
                frame_ids = [0] * capacity
                sigma_levels = _sigma_levels(sigmas, z)
            latents[count].copy_(z[0])
            stages[count] = 0
            frame_ids[count] = next_frame_id
            count += 1
            next_frame_id += 1
            injected += 1

    def compact():
        nonlocal count
        next_count = 0
        for i in range(count):
            if stages[i] == stage_count:
                if frame_callback is not None:
                    frame_callback(frame_ids[i], latents[i:i + 1].clone())
                continue
            if next_count != i:
                latents[next_count].copy_(latents[i])
                stages[next_count] = stages[i]
                frame_ids[next_count] = frame_ids[i]
            next_count += 1
        count = next_count

    while True:
        stop = should_stop()
        if reset_pipeline is not None and reset_pipeline():
            count = 0
        if not stop:
            inject()
        elif count == 0:
            break

        max_active_slots = max(max_active_slots, count)
        calls, batch = _step_packed(model, latents, stages, sigma_levels, microbatch_size, count, capacity)
        forward_calls += calls
        max_forward_batch = max(max_forward_batch, batch)
        compact()
        ticks += 1
        if tick_callback is not None:
            tick_callback(_packed_active(latents, stages, frame_ids, count))

    return StreamBatchStats(
        ticks=ticks,
        forward_calls=forward_calls,
        max_active_slots=max_active_slots,
        max_forward_batch=max_forward_batch,
    )
