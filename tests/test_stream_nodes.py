import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
PKG = "bokujuu_personal_nodes_stream_test"
sys.path.insert(0, str(COMFYUI_ROOT))


def load_package_module(name, filename):
    if PKG not in sys.modules:
        package = types.ModuleType(PKG)
        package.__path__ = [str(ROOT)]
        sys.modules[PKG] = package
    spec = importlib.util.spec_from_file_location(f"{PKG}.{name}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{PKG}.{name}"] = module
    spec.loader.exec_module(module)
    return module


STREAM_PREVIEW = load_package_module("stream_preview", "stream_preview.py")
STREAM_NODES = load_package_module("stream_nodes", "stream_nodes.py")
STREAM_LOOP = load_package_module("stream_loop", "stream_loop.py")


class StreamNodeTests(unittest.TestCase):
    def test_sampler_is_registered_on_the_extension(self):
        init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("BokujuuAnimaStreamBatchSampler", init_source)
        self.assertIn("BokujuuAnimaStreamLoop", init_source)

        schema = STREAM_NODES.BokujuuAnimaStreamBatchSampler.define_schema()
        self.assertEqual(schema.node_id, "BokujuuAnimaStreamBatchSampler")
        self.assertTrue(schema.is_experimental)
        self.assertEqual(
            [node_input.id for node_input in schema.inputs],
            ["microbatch_size", "live_preview", "frames_per_tick"],
        )

    def test_sampler_passes_live_preview_to_ksampler(self):
        output = STREAM_NODES.BokujuuAnimaStreamBatchSampler.execute(4, True)
        sampler = output.result[0]
        self.assertEqual(sampler.extra_options["microbatch_size"], 4)
        self.assertTrue(sampler.extra_options["live_preview"])
        self.assertEqual(sampler.extra_options["frames_per_tick"], 2)

        output = STREAM_NODES.BokujuuAnimaStreamBatchSampler.execute(8, False, 2)
        sampler = output.result[0]
        self.assertEqual(sampler.extra_options["microbatch_size"], 8)
        self.assertFalse(sampler.extra_options["live_preview"])
        self.assertEqual(sampler.extra_options["frames_per_tick"], 2)

    def test_live_preview_emits_each_completed_frame(self):
        completed = []

        def callback(payload):
            completed.append(payload["denoised"].detach().clone())

        x = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
        sigmas = torch.tensor([1.0, 0.66, 0.33, 0.0])
        samples = STREAM_NODES.sample_anima_stream_batch(
            lambda latents, sigma, **extra_args: latents,
            x,
            sigmas,
            callback=callback,
            microbatch_size=3,
            live_preview=True,
            frames_per_tick=2,
        )

        self.assertEqual(len(completed), 9)
        for frame_id in range(8):
            torch.testing.assert_close(completed[frame_id], x[frame_id:frame_id + 1])
        torch.testing.assert_close(completed[-1], x)
        torch.testing.assert_close(samples, x)

    def test_disabled_live_preview_emits_only_the_final_batch(self):
        calls = []
        x = torch.zeros((4, 1, 1))
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        STREAM_NODES.sample_anima_stream_batch(
            lambda latents, sigma, **extra_args: latents,
            x,
            sigmas,
            callback=lambda payload: calls.append(payload["denoised"].shape[0]),
            live_preview=False,
        )
        self.assertEqual(calls, [4])

    def test_live_preview_pushes_each_frame_to_preview_image(self):
        published = []

        class FakeSession:
            def add_frame(self, model, latent, frame_id=None):
                published.append(latent.detach().clone())

        x = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        with patch.object(STREAM_NODES.StreamPreviewSession, "start", return_value=FakeSession()):
            STREAM_NODES.sample_anima_stream_batch(
                lambda latents, sigma, **extra_args: latents,
                x,
                sigmas,
                live_preview=True,
                frames_per_tick=2,
            )

        self.assertEqual(len(published), 4)
        torch.testing.assert_close(torch.cat(published), x)

    def test_realtime_workflow_uses_four_steps_and_live_preview(self):
        workflow = json.loads((ROOT / "workflows" / "anima_stream_realtime_test.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "workflows" / "anima_stream_realtime_test_api.json").read_text(encoding="utf-8"))
        widgets = {node["id"]: node["widgets_values"] for node in workflow["nodes"] if "widgets_values" in node}

        self.assertEqual(widgets[2][0], r"Anima\anima-turbo-lora-v0.2.safetensors")
        self.assertEqual(widgets[3][0], "inductor")
        self.assertTrue(widgets[3][-1])
        self.assertEqual(widgets[10][1], 4)
        self.assertEqual(widgets[11], [8, True, 2])
        self.assertEqual(api["10"]["inputs"]["steps"], 4)
        self.assertTrue(api["11"]["inputs"]["live_preview"])
        self.assertEqual(api["11"]["inputs"]["frames_per_tick"], 2)
        self.assertEqual(api["11"]["inputs"]["microbatch_size"], 8)
        self.assertEqual(api["3"]["inputs"]["backend"], "inductor")
        self.assertTrue(api["3"]["inputs"]["disable_dynamic_vram"])


class StreamLoopTests(unittest.TestCase):
    def test_loop_node_is_an_output_node(self):
        schema = STREAM_LOOP.BokujuuAnimaStreamLoop.define_schema()
        self.assertEqual(schema.node_id, "BokujuuAnimaStreamLoop")
        self.assertTrue(schema.is_output_node)
        self.assertFalse(schema.is_experimental)
        loop_source = (ROOT / "stream_loop.py").read_text(encoding="utf-8")
        self.assertIn("drop_stale", loop_source)
        self.assertNotIn("reset_pipeline=", loop_source)
        self.assertEqual(
            [node_input.id for node_input in schema.inputs],
            [
                "noise",
                "guider",
                "sigmas",
                "latent_image",
                "clip",
                "microbatch_size",
                "live_preview",
                "frames_per_tick",
                "preview_window",
            ],
        )

    def test_loop_sampler_stops_after_max_frames_without_emptying_early(self):
        class FakeSampling:
            def noise_scaling(self, sigma, noise, latent_image, max_denoise):
                return noise

        class FakeInner:
            def __init__(self):
                self.model_sampling = FakeSampling()

        class FakeGuider:
            def __init__(self):
                self.inner_model = FakeInner()

        class FakeModel:
            def __init__(self):
                self.inner_model = FakeGuider()

            def __call__(self, latents, sigma, **kwargs):
                return latents

        published = []

        class FakeSession:
            def add_frame(self, model, latent, frame_id=None):
                published.append(latent.shape[0])

        x = torch.zeros((1, 1, 1))
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        STREAM_LOOP.sample_anima_stream_loop(
            FakeModel(),
            x,
            sigmas,
            extra_args={"seed": 1},
            live_preview=True,
            frames_per_tick=2,
            preview=FakeSession(),
            max_frames=8,
        )
        self.assertGreaterEqual(len(published), 8)

    def test_loop_execute_does_not_catch_interrupt(self):
        source = (ROOT / "stream_loop.py").read_text(encoding="utf-8")
        self.assertNotIn("except comfy.model_management.InterruptProcessingException", source)

    def test_loop_workflow_runs_until_cancel(self):
        workflow = json.loads((ROOT / "workflows" / "anima_stream_loop.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "workflows" / "anima_stream_loop_api.json").read_text(encoding="utf-8"))
        widgets = {node["id"]: node["widgets_values"] for node in workflow["nodes"] if "widgets_values" in node}
        types = {node["id"]: node["type"] for node in workflow["nodes"]}

        self.assertEqual(types[11], "BokujuuAnimaStreamLoop")
        self.assertNotIn("SamplerCustomAdvanced", types.values())
        self.assertEqual(types[2], "LoraLoaderModelOnly")
        self.assertEqual(widgets[2][0], r"Anima\anima-turbo-lora-v0.2.safetensors")
        self.assertEqual(widgets[7][0], 768)
        self.assertEqual(widgets[7][1], 768)
        self.assertEqual(widgets[7][2], 1)
        self.assertEqual(widgets[10][1], 4)
        self.assertEqual(widgets[11], [16, True, 1, 1])
        self.assertEqual(api["11"]["class_type"], "BokujuuAnimaStreamLoop")
        self.assertEqual(api["1"]["inputs"]["unet_name"], "fnMomentAnimaTurbo_v40NoTurbo.safetensors")
        self.assertEqual(api["2"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(api["2"]["inputs"]["lora_name"], r"Anima\anima-turbo-lora-v0.2.safetensors")
        self.assertEqual(api["3"]["inputs"]["model"], ["2", 0])
        self.assertEqual(api["3"]["inputs"]["dynamic"], "false")
        self.assertEqual(api["11"]["inputs"]["preview_window"], 1)
        self.assertEqual(api["11"]["inputs"]["frames_per_tick"], 1)
        self.assertEqual(api["11"]["inputs"]["microbatch_size"], 16)
        self.assertEqual(api["7"]["inputs"]["batch_size"], 1)
        self.assertEqual(api["7"]["inputs"]["width"], 768)


class StreamPreviewTests(unittest.TestCase):
    def test_preview_image_nodes_are_collected(self):
        prompt = {
            "12": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0]}},
            "14": {"class_type": "PreviewImage", "inputs": {"images": ["13", 0]}},
        }
        self.assertEqual(STREAM_PREVIEW.preview_image_node_ids(prompt), ["14"])

    def test_session_start_without_server_returns_none(self):
        self.assertIsNone(STREAM_PREVIEW.StreamPreviewSession.start())

    def test_video_latent_adds_a_time_dimension(self):
        latent = torch.zeros(1, 16, 8, 8)
        self.assertEqual(tuple(STREAM_PREVIEW.video_latent(latent).shape), (1, 16, 1, 8, 8))
        latent5 = torch.zeros(2, 16, 1, 8, 8)
        self.assertEqual(tuple(STREAM_PREVIEW.video_latent(latent5).shape), (2, 16, 1, 8, 8))

    def test_tick_burst_is_paced_instead_of_sent_immediately(self):
        sends = []

        def fake_decode(model, latent):
            return latent

        def fake_save(images):
            return [{"filename": f"{index}.png"} for index in range(images.shape[0])]

        def fake_send(prompt_id, node_ids, results, extra=None):
            sends.append(list(results))

        session = STREAM_PREVIEW.StreamPreviewSession("p", ["11"], window=8, paced=True, start_pacer=False)
        with patch.object(STREAM_PREVIEW, "decode_preview_images", fake_decode), patch.object(
            STREAM_PREVIEW, "save_temp_images", fake_save
        ), patch.object(STREAM_PREVIEW, "send_preview_images", fake_send):
            for _ in range(4):
                session.add_frame(None, torch.zeros(1, 1, 1, 1))
            self.assertEqual(sends, [])
            session.on_tick()
            self.assertEqual(len(sends), 1)
            self.assertEqual(len(session._display_q), 0)
            session.close()
            self.assertEqual(len(sends), 1)

    def test_tick_interval_splits_burst_across_the_tick(self):
        session = STREAM_PREVIEW.StreamPreviewSession("p", ["11"], window=4, paced=True, start_pacer=False)
        session._tick_t0 = 0.0
        with patch.object(STREAM_PREVIEW, "decode_preview_images", lambda model, latent: latent), patch.object(
            STREAM_PREVIEW, "save_temp_images", lambda images: [{"filename": "x.png"}] * images.shape[0]
        ), patch.object(STREAM_PREVIEW, "send_preview_images", lambda *args, **kwargs: None), patch.object(
            STREAM_PREVIEW.time, "perf_counter", return_value=0.40
        ):
            for _ in range(4):
                session.add_frame(None, torch.zeros(1, 1, 1, 1))
            session.on_tick()
            session.close()
        self.assertAlmostEqual(session._ema, 0.432)

    def test_loop_frontend_shows_only_the_hero_preview(self):
        source = (ROOT / "web" / "anima_stream.js").read_text(encoding="utf-8")
        self.assertIn("bokujuu-anima-hero", source)
        self.assertIn("pushPrompt", source)
        self.assertIn("addEventListener(\"input\"", source)
        self.assertNotIn("bokujuu-anima-strip", source)
        self.assertNotIn("forming", source)

    def test_drop_stale_clears_queued_previews(self):
        session = STREAM_PREVIEW.StreamPreviewSession("p", ["11"], window=8, start_pacer=False)
        with patch.object(STREAM_PREVIEW, "decode_preview_images", lambda model, latent: latent), patch.object(
            STREAM_PREVIEW, "save_temp_images", lambda images: [{"filename": "a.png"}] * images.shape[0]
        ), patch.object(STREAM_PREVIEW, "send_preview_images", lambda *args, **kwargs: None):
            session.add_frame(None, torch.zeros(1, 1, 1, 1))
            session.on_tick()
            session.add_frame(None, torch.zeros(1, 1, 1, 1))
            session.drop_stale()
            self.assertEqual(len(session._display_q), 0)
            self.assertEqual(session._pending_latents, [])
            session.close()


if __name__ == "__main__":
    unittest.main()
