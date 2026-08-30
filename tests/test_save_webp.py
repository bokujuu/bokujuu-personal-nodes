import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
