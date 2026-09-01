import importlib
import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


COMFYUI_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_NODES = COMFYUI_ROOT / "custom_nodes"
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(CUSTOM_NODES))

PACKAGE = importlib.import_module("bokujuu-personal-nodes")
DEPTH = importlib.import_module("bokujuu-personal-nodes.dlss5.depth")
ENGINE = importlib.import_module("bokujuu-personal-nodes.dlss5.engine")
NODES = importlib.import_module("bokujuu-personal-nodes.dlss5.nodes")
RUNTIME = importlib.import_module("bokujuu-personal-nodes.dlss5.runtime")
TYPES = importlib.import_module("bokujuu-personal-nodes.dlss5.types")


class FakeSession:
    resets = []

    def __init__(self, input_width, input_height, output_width, output_height, *args, **kwargs):
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def process(self, rgba, depth, reset):
        self.resets.append(bool(reset))
        y = np.linspace(0, rgba.shape[0] - 1, self.output_height).round().astype(int)
        x = np.linspace(0, rgba.shape[1] - 1, self.output_width).round().astype(int)
        return rgba[y][:, x]


class DLSS5UnitTests(unittest.TestCase):
    def setUp(self):
        FakeSession.resets = []

    def test_guidance_validates_batch_shape_and_finite_values(self):
        images = torch.zeros((2, 32, 64, 3))
        guidance = TYPES.DLSSGuidance(depth=torch.zeros((2, 32, 64)))
        self.assertIs(guidance.validate_for(images), guidance)
        with self.assertRaisesRegex(ValueError, "shape"):
            TYPES.DLSSGuidance(depth=torch.zeros((1, 32, 64))).validate_for(images)

    def test_depth_rendering_uses_shared_color_themes(self):
        guidance = TYPES.DLSSGuidance(depth=torch.linspace(0, 1, 8).reshape(1, 2, 4))
        for theme in ("grayscale", "grayscale_inverted", "turbo", "viridis", "plasma", "inferno"):
            with self.subTest(theme=theme):
                image = DEPTH.render_guidance_depth(guidance, theme, 1.0, 1.0)
                self.assertEqual(tuple(image.shape), (1, 2, 4, 3))
                self.assertTrue(torch.isfinite(image).all())
                self.assertGreaterEqual(image.min().item(), 0.0)
                self.assertLessEqual(image.max().item(), 1.0)

    def test_image_upscale_preserves_alpha_through_separate_resize(self):
        images = torch.rand((1, 32, 64, 4))
        guidance = TYPES.DLSSGuidance(depth=torch.rand((1, 32, 64)))
        expected_alpha = ENGINE._resize_lanczos(images[0], 128, 64)[..., 3]
        with mock.patch.object(ENGINE, "NativeDLSSSession", FakeSession):
            output = ENGINE.DLSS5Engine().enhance_image(images, guidance, scale=2.0)
        self.assertEqual(tuple(output.shape), (1, 64, 128, 4))
        self.assertTrue(torch.equal(output[0, ..., 3], expected_alpha))
        self.assertEqual(FakeSession.resets, [True])

    def test_output_mix_zero_returns_lanczos_baseline(self):
        images = torch.rand((1, 32, 64, 3))
        guidance = TYPES.DLSSGuidance(depth=torch.rand((1, 32, 64)))
        expected = ENGINE._resize_lanczos(images[0], 128, 64)
        with mock.patch.object(ENGINE, "NativeDLSSSession", FakeSession):
            output = ENGINE.DLSS5Engine().enhance_image(
                images, guidance, scale=2.0, output_mix=0.0
            )
        self.assertTrue(torch.equal(output[0], expected))

    def test_temporal_session_resets_first_frame_and_scene_cut_only(self):
        images = torch.zeros((3, 32, 64, 3))
        images[1] = 0.05
        images[2] = 1.0
        guidance = TYPES.DLSSGuidance(depth=torch.rand((3, 32, 64)))
        with mock.patch.object(ENGINE, "NativeDLSSSession", FakeSession):
            output = ENGINE.DLSS5Engine().enhance_sequence(
                images, guidance, scale=2.0, scene_cut_threshold=0.35
            )
        self.assertEqual(tuple(output.shape), (3, 64, 128, 3))
        self.assertEqual(FakeSession.resets, [True, False, True])

    def test_runtime_missing_error_is_deferred_until_resolution(self):
        missing = Path("Z:/definitely-missing/nvngx_dlss.dll")
        with self.assertRaisesRegex(RUNTIME.DLSSRuntimeError, "was not found"):
            RUNTIME.resolve_runtime(missing)

    def test_node_schemas_hide_depth_and_order_quality(self):
        depth_schema = NODES.BokujuuDLSSGuidanceDepth.define_schema()
        image_schema = NODES.BokujuuDLSS5NeuralUpscale.define_schema()
        temporal_schema = NODES.BokujuuDLSS5TemporalUpscale.define_schema()
        image_inputs = {item.id: item for item in image_schema.inputs}
        self.assertEqual(
            [output.display_name for output in depth_schema.outputs],
            ["depth_image", "guidance"],
        )
        self.assertEqual(
            image_inputs["quality"].options,
            [
                "ultra_performance",
                "performance",
                "balanced",
                "quality",
                "ultra_quality",
                "dlaa",
            ],
        )
        self.assertEqual(image_inputs["quality"].default, "balanced")
        self.assertEqual(image_inputs["cas_amount"].default, 0.7)
        self.assertEqual(image_inputs["cas_amount"].max, 4.0)
        self.assertNotIn("depth_mode", image_inputs)
        self.assertNotIn("constant_depth", image_inputs)
        self.assertNotIn("da3_model", image_inputs)
        self.assertNotIn("post_sharpen", image_inputs)
        self.assertIn("scene_cut_threshold", [item.id for item in temporal_schema.inputs])

    def test_upscale_nodes_use_internal_constant_depth(self):
        images = torch.zeros((1, 32, 64, 3))
        guidance = NODES._constant_guidance(images)
        self.assertEqual(tuple(guidance.depth.shape), (1, 32, 64))
        self.assertTrue(torch.equal(guidance.depth, torch.full((1, 32, 64), 0.5)))

    def test_neural_upscale_applies_cas_after_dlss(self):
        images = torch.rand((1, 32, 64, 3))
        raw = torch.rand((1, 64, 128, 3))
        expected = NODES.cas_sharpen(raw, amount=0.35)
        engine = mock.Mock()
        engine.enhance_image.return_value = raw
        with mock.patch.object(NODES, "DLSS5Engine", return_value=engine):
            with mock.patch.object(NODES.model_management, "unload_all_models"):
                output = NODES.BokujuuDLSS5NeuralUpscale.execute(
                    images, 2.0, "balanced", 0.35, 1.0, 0
                )
        self.assertTrue(torch.allclose(output.result[0], expected))
        engine.enhance_image.assert_called_once()

    def test_dlss_nodes_are_registered_on_root_extension(self):
        registered = asyncio.run(PACKAGE.BokujuuPersonalNodesExtension().get_node_list())
        self.assertIn(NODES.BokujuuDLSSGuidanceDepth, registered)
        self.assertIn(NODES.BokujuuDLSS5NeuralUpscale, registered)
        self.assertIn(NODES.BokujuuDLSS5TemporalUpscale, registered)

    def test_example_api_workflows_use_registered_dlss_nodes(self):
        workflows = Path(__file__).resolve().parents[1] / "workflows"
        image = json.loads((workflows / "dlss5_image_test_api.json").read_text(encoding="utf-8"))
        depth = json.loads((workflows / "dlss5_depth_test_api.json").read_text(encoding="utf-8"))
        temporal = json.loads((workflows / "dlss5_temporal_test_api.json").read_text(encoding="utf-8"))
        ui_image = json.loads((workflows / "dlss5_image_test.json").read_text(encoding="utf-8"))
        ui_depth = json.loads((workflows / "dlss5_depth_test.json").read_text(encoding="utf-8"))
        self.assertEqual(image["3"]["class_type"], "BokujuuDLSS5NeuralUpscale")
        self.assertEqual(image["3"]["inputs"]["quality"], "balanced")
        self.assertEqual(image["3"]["inputs"]["cas_amount"], 0.7)
        self.assertNotIn("depth_mode", image["3"]["inputs"])
        self.assertNotIn("da3_model", image["3"]["inputs"])
        self.assertEqual(depth["3"]["class_type"], "BokujuuDLSSGuidanceDepth")
        self.assertEqual(temporal["3"]["class_type"], "BokujuuDLSS5TemporalUpscale")
        self.assertEqual(temporal["3"]["inputs"]["quality"], "balanced")
        self.assertEqual(temporal["4"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(ui_image["nodes"][1]["type"], "BokujuuDLSS5NeuralUpscale")
        self.assertEqual(ui_image["nodes"][1]["widgets_values"], [2.0, "balanced", 0.7, 1.0, 0])
        self.assertEqual(ui_depth["nodes"][2]["type"], "BokujuuDLSSGuidanceDepth")

    def test_dlaa_rejects_upscale_ratio(self):
        images = torch.rand((1, 32, 64, 3))
        guidance = TYPES.DLSSGuidance(depth=torch.rand((1, 32, 64)))
        with self.assertRaisesRegex(ValueError, "1.0x"):
            ENGINE.DLSS5Engine().enhance_image(images, guidance, scale=2.0, quality="dlaa")


@unittest.skipUnless(
    os.environ.get("BOKUJUU_RUN_DLSS_INTEGRATION") == "1",
    "set BOKUJUU_RUN_DLSS_INTEGRATION=1 to run on an NVIDIA RTX GPU",
)
class DLSS5IntegrationTests(unittest.TestCase):
    def test_native_runtime_supports_configured_scale_modes(self):
        height, width = 64, 64
        image = torch.linspace(0, 1, height * width * 3).reshape(1, height, width, 3)
        depth = torch.linspace(0, 1, height * width).reshape(1, height, width)
        for scale in (1.0, 1.5, 2.0, 3.0):
            with self.subTest(scale=scale):
                output = ENGINE.DLSS5Engine().enhance_image(
                    image,
                    TYPES.DLSSGuidance(depth=depth),
                    scale=scale,
                    quality="auto",
                )
                target = int(round(64 * scale))
                self.assertEqual(tuple(output.shape), (1, target, target, 3))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreaterEqual(output.min().item(), 0.0)
                self.assertLessEqual(output.max().item(), 1.0)

    def test_reset_still_frame_ignores_constant_depth_value(self):
        height, width = 64, 64
        image = torch.linspace(0, 1, height * width * 3).reshape(1, height, width, 3)
        quantized = []
        for value in (0.0, 0.5, 1.0):
            output = ENGINE.DLSS5Engine().enhance_image(
                image,
                TYPES.DLSSGuidance(depth=torch.full((1, height, width), value)),
                scale=2.0,
                quality="balanced",
            )
            quantized.append(output.mul(255).round().byte())
        self.assertTrue(torch.equal(quantized[0], quantized[1]))
        self.assertTrue(torch.equal(quantized[1], quantized[2]))


if __name__ == "__main__":
    unittest.main()
