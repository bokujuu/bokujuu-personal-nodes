import unittest
import json
from pathlib import Path

import torch

from test_randomizer import NODES


class DepthAnything3Tests(unittest.TestCase):
    def test_converts_upstream_backbone_keys(self):
        qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        state_dict = {
            "model.backbone.pretrained.patch_embed.proj.weight": torch.ones(2, 3, 1, 1),
            "model.backbone.pretrained.blocks.0.attn.qkv.weight": qkv,
            "model.backbone.pretrained.blocks.0.attn.proj.bias": torch.ones(2),
            "model.head.output_conv1.0.weight": torch.ones(1),
            "model.head.scratch.output_conv2_aux.0.2.weight": torch.ones(1),
        }

        converted = NODES.convert_da3_state_dict(state_dict)

        self.assertIn("backbone.embeddings.patch_embeddings.projection.weight", converted)
        self.assertIn("backbone.encoder.layer.0.attention.attention.query.weight", converted)
        self.assertIn("backbone.encoder.layer.0.attention.attention.key.weight", converted)
        self.assertIn("backbone.encoder.layer.0.attention.attention.value.weight", converted)
        self.assertIn("backbone.encoder.layer.0.attention.output.dense.bias", converted)
        self.assertIn("head.output_conv1.0.weight", converted)
        self.assertIn("head.scratch.output_conv2_aux.3.2.weight", converted)
        self.assertNotIn("backbone.pretrained.blocks.0.attn.qkv.weight", converted)

    def test_rejects_unknown_checkpoint_layout(self):
        with self.assertRaisesRegex(ValueError, "unsupported state-dict layout"):
            NODES.convert_da3_state_dict({"other.weight": torch.ones(1)})

    def test_color_themes_preserve_image_shape_and_range(self):
        values = torch.linspace(0.0, 1.0, 12).reshape(1, 3, 4)
        for theme in ("grayscale", "grayscale_inverted", "turbo", "viridis", "plasma", "inferno"):
            with self.subTest(theme=theme):
                image = NODES.apply_color_theme(values, theme)
                self.assertEqual(image.shape, (1, 3, 4, 3))
                self.assertGreaterEqual(image.min().item(), 0.0)
                self.assertLessEqual(image.max().item(), 1.0)

    def test_grayscale_is_default_theme(self):
        schema = NODES.BokujuuDepthAnything3.define_schema()
        color_theme = next(item for item in schema.inputs if item.id == "color_theme")
        self.assertEqual(color_theme.default, "grayscale")

    def test_example_workflows_use_registered_nodes(self):
        workflows = Path(__file__).resolve().parents[1] / "workflows"
        ui_workflow = json.loads((workflows / "da3_large_1.1_test.json").read_text(encoding="utf-8"))
        api_workflow = json.loads((workflows / "da3_large_1.1_test_api.json").read_text(encoding="utf-8"))

        self.assertEqual(ui_workflow["version"], 0.4)
        self.assertEqual({node["type"] for node in ui_workflow["nodes"]}, {
            "LoadImage",
            "BokujuuLoadDepthAnything3",
            "BokujuuDepthAnything3",
            "SaveImage",
        })
        self.assertEqual(api_workflow["2"]["class_type"], "BokujuuLoadDepthAnything3")
        self.assertEqual(api_workflow["3"]["inputs"]["color_theme"], "grayscale")
        self.assertEqual(api_workflow["4"]["inputs"]["images"], ["3", 0])


if __name__ == "__main__":
    unittest.main()
