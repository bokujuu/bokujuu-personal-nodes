import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFYUI_ROOT))
SPEC = importlib.util.spec_from_file_location("bokujuu_personal_nodes_webp", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODES
SPEC.loader.exec_module(NODES)


class SaveWebPTests(unittest.TestCase):
    def test_node_is_registered_as_lossy_webp_output(self):
        registered = asyncio.run(NODES.BokujuuPersonalNodes().get_node_list())
        self.assertIn(NODES.BokujuuSaveWebP, registered)

        schema = NODES.BokujuuSaveWebP.define_schema()
        self.assertTrue(schema.is_output_node)
        self.assertNotIn("lossless", [node_input.id for node_input in schema.inputs])

        prefix = next(node_input for node_input in schema.inputs if node_input.id == "filename_prefix")
        self.assertIn("%date:yyyy-MM-dd%", prefix.tooltip)
        self.assertIn("%Empty Latent Image.width%", prefix.tooltip)

    def test_saves_lossy_webp_with_prompt_and_workflow(self):
        prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64}}}
        workflow = json.loads((MODULE_PATH.parent / "workflows" / "webp_save_example.json").read_text(encoding="utf-8"))
        image = torch.rand((1, 64, 64, 3), generator=torch.Generator().manual_seed(7))
        NODES.BokujuuSaveWebP.hidden = SimpleNamespace(
            prompt=prompt,
            extra_pnginfo={"workflow": workflow},
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(NODES.folder_paths, "get_output_directory", return_value=output_dir):
                result = NODES.BokujuuSaveWebP.execute(image, "metadata/test", 80, 4)

            files = list((Path(output_dir) / "metadata").glob("*.webp"))
            self.assertEqual(len(files), 1)
            self.assertIs(result[0], image)

            with Image.open(files[0]) as saved:
                self.assertEqual(saved.format, "WEBP")
                self.assertEqual(saved.size, (64, 64))
                self.assertNotEqual(saved.tobytes(), NODES.ui.ImageSaveHelper._convert_tensor_to_pil(image[0]).tobytes())
                exif = saved.getexif()

            self.assertEqual(json.loads(exif[0x0110].split(":", 1)[1]), prompt)
            self.assertEqual(json.loads(exif[0x010F].split(":", 1)[1]), workflow)

    def test_expands_date_tokens_like_save_image(self):
        when = datetime(2026, 8, 30, 10, 54, 7)
        self.assertEqual(
            NODES.expand_filename_prefix("Anima/%date:yyyy-MM-dd%/upscale/ComfyUI", when),
            "Anima/2026-08-30/upscale/ComfyUI",
        )
        self.assertEqual(
            NODES.expand_filename_prefix("%date:yyyy-MM-dd%_%date:hhmmss%", when),
            "2026-08-30_105407",
        )
        self.assertEqual(NODES.expand_filename_prefix("bokujuu_webp/example", when), "bokujuu_webp/example")

    def test_saves_into_dated_subfolder_from_prefix_tokens(self):
        prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64}}}
        image = torch.rand((1, 64, 64, 3), generator=torch.Generator().manual_seed(7))
        NODES.BokujuuSaveWebP.hidden = SimpleNamespace(prompt=prompt, extra_pnginfo=None)
        when = datetime(2026, 8, 30, 10, 54, 7)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(NODES.folder_paths, "get_output_directory", return_value=output_dir):
                with patch.object(NODES, "datetime") as datetime_module:
                    datetime_module.now.return_value = when
                    NODES.BokujuuSaveWebP.execute(
                        image,
                        "Anima/%date:yyyy-MM-dd%/%width%x%height%/ComfyUI",
                        80,
                        4,
                    )

            files = list((Path(output_dir) / "Anima" / "2026-08-30" / "64x64").glob("ComfyUI_*.webp"))
            self.assertEqual(len(files), 1)

    def test_frontend_hooks_filename_prefix_serialization(self):
        source = (MODULE_PATH.parent / "web" / "save_webp.js").read_text(encoding="utf-8")
        self.assertIn("BokujuuSaveWebP", source)
        self.assertIn("applyTextReplacements", source)
        self.assertIn("filename_prefix", source)
        self.assertIn("serializeValue", source)


if __name__ == "__main__":
    unittest.main()
