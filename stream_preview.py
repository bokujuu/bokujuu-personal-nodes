import logging
import os
import shutil
import threading
import time
from collections import deque

import folder_paths
import numpy as np
import torch
from PIL import Image


TINY_VAE_REPO_ID = "lightx2v/Autoencoders"
TINY_VAE_FILENAME = "lighttaew2_1.safetensors"

_tiny_vae = None


def preview_image_node_ids(prompt):
    return [
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == "PreviewImage"
    ]


def video_latent(latent):
    if latent.ndim == 4:
        return latent.unsqueeze(2)
    return latent


def vae_space_latent(model, latent):
    inner = model
    while hasattr(inner, "inner_model"):
        inner = inner.inner_model
    process_latent_out = getattr(inner, "process_latent_out", None)
    if process_latent_out is None:
        return latent
    return process_latent_out(latent.to(torch.float32))


def current_running_job():
    try:
        from server import PromptServer
    except ImportError:
        return None
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(server, "prompt_queue", None) is None:
        return None
    running, _ = server.prompt_queue.get_current_queue()
    if not running:
        return None
    item = running[0]
    return item[1], item[2]


def current_preview_targets():
    job = current_running_job()
    if job is None:
        return None
    prompt_id, prompt = job
    return prompt_id, preview_image_node_ids(prompt)


def ensure_tiny_vae_path():
    approx_dirs = folder_paths.get_folder_paths("vae_approx")
    existing = next(
        (name for name in folder_paths.get_filename_list("vae_approx") if name.startswith("lighttaew2_1")),
        None,
    )
    if existing:
        return folder_paths.get_full_path("vae_approx", existing)

    dest_dir = approx_dirs[0]
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, TINY_VAE_FILENAME)
    if os.path.isfile(dest_path):
        return dest_path

    from huggingface_hub import hf_hub_download
    source_path = hf_hub_download(repo_id=TINY_VAE_REPO_ID, filename=TINY_VAE_FILENAME)
    temporary_path = dest_path + ".tmp"
    try:
        shutil.copyfile(source_path, temporary_path)
        os.replace(temporary_path, dest_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return dest_path


def load_tiny_vae():
    global _tiny_vae
    if _tiny_vae is not None:
        return _tiny_vae
    import comfy.utils
    from comfy.sd import VAE

    vae = VAE(comfy.utils.load_torch_file(ensure_tiny_vae_path()))
    vae.first_stage_model.show_progress_bar = False
    _tiny_vae = vae
    return vae


def decode_preview_images(model, latent):
    images = load_tiny_vae().decode(video_latent(vae_space_latent(model, latent)))
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images.clamp(0.0, 1.0)


def save_temp_images(images):
    output_dir = folder_paths.get_temp_directory()
    prefix = "bokujuu_anima_stream_temp"
    full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
        prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )
    results = []
    for image in images:
        pixels = 255.0 * image.detach().to("cpu").numpy()
        img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), compress_level=1)
        results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
        counter += 1
    return results


def send_preview_images(prompt_id, node_ids, results, extra=None):
    from server import PromptServer
    server = PromptServer.instance
    output = {"images": results}
    if extra:
        output.update(extra)
    for node_id in node_ids:
        server.send_sync(
            "executed",
            {
                "node": str(node_id),
                "display_node": str(node_id),
                "output": output,
                "prompt_id": prompt_id,
            },
            server.client_id,
        )


class StreamPreviewSession:
    def __init__(self, prompt_id, node_ids, window=8, paced=True, start_pacer=False):
        self.prompt_id = prompt_id
        self.node_ids = node_ids
        self.window = window
        self.results = []
        self._paced = paced
        self._lock = threading.Lock()
        self._pending_latents = []
        self._pending_ids = []
        self._pending_model = None
        self._display_q = deque()
        self._stop = threading.Event()
        self._tick_t0 = time.perf_counter()
        self._ema = None
        self._thread = None
        if paced and start_pacer:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    @classmethod
    def start(cls, extra_node_ids=None, window=8, prompt=None):
        node_ids = []
        prompt_id = None
        targets = current_preview_targets()
        if targets is not None:
            prompt_id, preview_ids = targets
            node_ids.extend(preview_ids)
        if prompt is not None:
            for node_id in preview_image_node_ids(prompt):
                if node_id not in node_ids:
                    node_ids.append(node_id)
        if extra_node_ids is not None:
            for node_id in extra_node_ids:
                if node_id is None:
                    continue
                node_id = str(node_id)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        if not node_ids:
            return None
        if prompt_id is None:
            prompt_id = ""
        load_tiny_vae()
        logging.info("Anima stream updating Preview Image nodes %s", node_ids)
        return cls(prompt_id, node_ids, window=window)

    def add_frame(self, model, latent, frame_id=None):
        with self._lock:
            self._pending_latents = [latent.detach().clone()]
            self._pending_ids = [frame_id]
            self._pending_model = model

    def on_tick(self):
        now = time.perf_counter()
        dt = now - self._tick_t0
        self._tick_t0 = now
        n = self._flush_decode()
        if n > 0 and dt > 1e-4:
            interval = (dt / n) * 1.08
            if self._ema is None:
                self._ema = interval
            else:
                self._ema = 0.6 * self._ema + 0.4 * interval
        self._emit_one()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._flush_decode()
        while self._emit_one():
            pass

    def _interval(self):
        if self._ema is None:
            return 0.08
        return max(1.0 / 30.0, self._ema)

    def _flush_decode(self):
        with self._lock:
            latents = self._pending_latents
            frame_ids = self._pending_ids
            model = self._pending_model
            self._pending_latents = []
            self._pending_ids = []
            self._pending_model = None
        if not latents:
            return 0
        images = decode_preview_images(model, torch.cat(latents, dim=0))
        saved = save_temp_images(images)
        for item, frame_id in zip(saved, frame_ids):
            if frame_id is not None:
                item["frame_id"] = int(frame_id)
        with self._lock:
            self._display_q.clear()
            if saved:
                self._display_q.append(saved[-1])
        return len(saved)

    def drop_stale(self):
        with self._lock:
            self._pending_latents = []
            self._pending_ids = []
            self._pending_model = None
            self._display_q.clear()

    def _emit_one(self):
        with self._lock:
            if not self._display_q:
                return False
            self.results.append(self._display_q.popleft())
            if self.window > 0:
                self.results = self.results[-self.window:]
            snapshot = list(self.results)
            interval = self._interval()
            visible = snapshot[-1:]
        send_preview_images(
            self.prompt_id,
            self.node_ids,
            visible,
            extra={"anima_stream": {"kind": "completed", "interval": interval}},
        )
        return True

    def _run(self):
        while not self._stop.is_set():
            emitted = self._emit_one()
            interval = self._interval()
            self._stop.wait(interval if emitted else min(0.02, interval))
