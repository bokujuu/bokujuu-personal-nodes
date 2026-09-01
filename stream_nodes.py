import logging

import comfy.samplers
from comfy_api.latest import io

from .anima_stream import sample_stream_batch
from .stream_preview import StreamPreviewSession


def sample_anima_stream_batch(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    microbatch_size=4,
    live_preview=True,
    frames_per_tick=2,
):
    extra_args = {} if extra_args is None else extra_args
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("Anima Stream Batch currently supports txt2img only")

    preview = [None]
    if live_preview:
        try:
            preview[0] = StreamPreviewSession.start()
        except Exception as exc:
            logging.warning("Anima stream Preview Image updates disabled: %s", exc)

    def frame_callback(frame_id, latent):
        if callback is not None and live_preview:
            callback({
                "x": latent,
                "i": frame_id,
                "sigma": sigmas[-2],
                "sigma_hat": sigmas[-2],
                "denoised": latent,
            })
        session = preview[0]
        if session is None:
            return
        try:
            session.add_frame(model, latent)
        except Exception as exc:
            logging.warning("Anima stream Preview Image update failed: %s", exc)
            preview[0] = None

    def tick_callback(active=None):
        session = preview[0]
        on_tick = getattr(session, "on_tick", None)
        if on_tick is not None:
            on_tick()

    try:
        samples, _ = sample_stream_batch(
            lambda latents, sigma: model(latents, sigma, **extra_args),
            x,
            sigmas,
            microbatch_size=microbatch_size,
            frames_per_tick=frames_per_tick,
            frame_callback=frame_callback,
            tick_callback=tick_callback,
        )
    finally:
        session = preview[0]
        close = getattr(session, "close", None)
        if close is not None:
            close()
    if callback is not None:
        callback({
            "x": samples,
            "i": len(sigmas) - 2,
            "sigma": sigmas[-2],
            "sigma_hat": sigmas[-2],
            "denoised": samples,
        })
    return samples


class BokujuuAnimaStreamBatchSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BokujuuAnimaStreamBatchSampler",
            display_name="Bokujuu Anima Stream Batch Sampler",
            category="Bokujuu/Anima",
            description="Experimental txt2img Euler sampler that pipelines samples at different Anima timesteps. Use CFG 1 and Anima Turbo.",
            is_experimental=True,
            inputs=[
                io.Int.Input("microbatch_size", default=4, min=1, max=32),
                io.Boolean.Input(
                    "live_preview",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Decode each completed frame with Light TAE and push it to Preview Image immediately.",
                ),
                io.Int.Input(
                    "frames_per_tick",
                    default=2,
                    min=1,
                    max=16,
                    optional=True,
                    tooltip="How many new latents to inject each stream tick. 2 is the StreamDiffusion-style parallel width.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, microbatch_size, live_preview=True, frames_per_tick=2):
        sampler = comfy.samplers.KSAMPLER(
            sample_anima_stream_batch,
            extra_options={
                "microbatch_size": microbatch_size,
                "live_preview": live_preview,
                "frames_per_tick": frames_per_tick,
            },
        )
        return io.NodeOutput(sampler)
