import inspect
import logging
import threading

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import latent_preview
from comfy_api.latest import io

from .anima_stream import sample_stream_loop
from .stream_preview import StreamPreviewSession


_live_prompt_text = None
_live_prompt_lock = threading.Lock()
_prompt_route_registered = False


def set_live_prompt_text(text):
    global _live_prompt_text
    with _live_prompt_lock:
        _live_prompt_text = text


def get_live_prompt_text():
    with _live_prompt_lock:
        return _live_prompt_text


def register_live_prompt_route():
    global _prompt_route_registered
    if _prompt_route_registered:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return

    @server.routes.post("/bokujuu/anima_stream_prompt")
    async def anima_stream_prompt(request):
        data = await request.json()
        set_live_prompt_text(data.get("text"))
        return web.json_response({"ok": True})

    _prompt_route_registered = True


def inner_model(model):
    inner = model
    while hasattr(inner, "inner_model"):
        inner = inner.inner_model
    return inner


def apply_live_prompt(guider, clip, text, noise):
    params = inspect.signature(guider.set_conds).parameters
    if len(params) >= 2:
        return False
    tokens = clip.tokenize(text)
    positive = clip.encode_from_tokens_scheduled(tokens)
    guider.set_conds(positive)
    model = getattr(guider, "inner_model", None)
    if model is None:
        return False
    conds = {key: list(map(lambda item: item.copy(), value)) for key, value in guider.original_conds.items()}
    guider.conds = comfy.samplers.process_conds(model, noise, conds, noise.device)
    return True


def sample_anima_stream_loop(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    microbatch_size=4,
    live_preview=True,
    frames_per_tick=2,
    preview=None,
    clip=None,
    max_frames=None,
):
    extra_args = {} if extra_args is None else extra_args
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("Anima Stream Loop currently supports txt2img only")

    sampling = inner_model(model).model_sampling
    seed = extra_args.get("seed", 0)
    first = [True]
    completed = [0]
    applied_text = [None]
    guider = getattr(model, "inner_model", None)
    preview_holder = [preview]

    def torch_zeros_like_x():
        return x.new_zeros((1,) + tuple(x.shape[1:]))

    def make_latent(frame_id):
        if first[0]:
            first[0] = False
            return x[:1]
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed + frame_id) & 0xFFFFFFFF)
        raw = torch.randn((1,) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype, generator=generator)
        return sampling.noise_scaling(sigmas[0], raw, torch_zeros_like_x(), True)

    def maybe_refresh_prompt():
        if clip is None or guider is None:
            return
        text = get_live_prompt_text()
        if text is None or text == applied_text[0]:
            return
        if not apply_live_prompt(guider, clip, text, x):
            applied_text[0] = text
            return
        first_sync = applied_text[0] is None
        applied_text[0] = text
        if first_sync:
            return
        session = preview_holder[0]
        drop = getattr(session, "drop_stale", None)
        if drop is not None:
            drop()

    def should_stop():
        comfy.model_management.throw_exception_if_processing_interrupted()
        maybe_refresh_prompt()
        return max_frames is not None and completed[0] >= max_frames

    def frame_callback(frame_id, latent):
        completed[0] += 1
        if callback is not None and live_preview:
            callback({
                "x": latent,
                "i": frame_id,
                "sigma": sigmas[-2],
                "sigma_hat": sigmas[-2],
                "denoised": latent,
            })
        session = preview_holder[0]
        if session is None:
            return
        try:
            session.add_frame(model, latent, frame_id=frame_id)
        except Exception as exc:
            logging.warning("Anima stream Preview Image update failed: %s", exc)
            preview_holder[0] = None

    def tick_callback(active=None):
        session = preview_holder[0]
        on_tick = getattr(session, "on_tick", None)
        if on_tick is None:
            return
        try:
            on_tick()
        except Exception as exc:
            logging.warning("Anima stream Preview Image update failed: %s", exc)
            preview_holder[0] = None
            close = getattr(session, "close", None)
            if close is not None:
                close()

    sample_stream_loop(
        lambda latents, sigma: model(latents, sigma, **extra_args),
        make_latent,
        sigmas,
        microbatch_size=microbatch_size,
        frames_per_tick=frames_per_tick,
        frame_callback=frame_callback,
        should_stop=should_stop,
        tick_callback=tick_callback,
    )
    return x


class BokujuuAnimaStreamLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BokujuuAnimaStreamLoop",
            display_name="Bokujuu Anima Stream Loop",
            category="Bokujuu/Anima",
            description="Run Anima Stream until Cancel. The live view shows the latest completed Light TAE frame.",
            is_output_node=True,
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Connect CLIP to pick up CLIPTextEncode edits while the loop is running.",
                ),
                io.Int.Input("microbatch_size", default=16, min=1, max=32),
                io.Boolean.Input("live_preview", default=True, optional=True, advanced=True),
                io.Int.Input(
                    "frames_per_tick",
                    default=1,
                    min=1,
                    max=16,
                    optional=True,
                    tooltip="Parallel streams per tick. 1 is lowest latency for a single live view.",
                ),
                io.Int.Input(
                    "preview_window",
                    default=1,
                    min=1,
                    max=64,
                    optional=True,
                    tooltip="How many latest frames to keep on Preview Image.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sigmas,
        latent_image,
        microbatch_size=16,
        live_preview=True,
        frames_per_tick=1,
        preview_window=1,
        clip=None,
    ):
        register_live_prompt_route()
        hidden = cls.hidden
        unique_id = None if hidden is None else hidden.unique_id
        prompt = None if hidden is None else hidden.prompt

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples[:1]

        preview = None
        try:
            if live_preview:
                try:
                    preview = StreamPreviewSession.start(
                        extra_node_ids=[unique_id],
                        window=preview_window,
                        prompt=prompt,
                    )
                except Exception as exc:
                    logging.warning("Anima stream Preview Image updates disabled: %s", exc)

            sampler = comfy.samplers.KSAMPLER(
                sample_anima_stream_loop,
                extra_options={
                    "microbatch_size": microbatch_size,
                    "live_preview": live_preview,
                    "frames_per_tick": frames_per_tick,
                    "preview": preview,
                    "clip": clip,
                },
            )
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
            guider.sample(
                noise.generate_noise(latent),
                latent["samples"],
                sampler,
                sigmas,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=noise.seed,
            )
        finally:
            if preview is not None:
                preview.close()
        return io.NodeOutput()
