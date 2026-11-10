import asyncio
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFYUI_ROOT))
SPEC = importlib.util.spec_from_file_location("bokujuu_load_images_from_path", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODES
SPEC.loader.exec_module(NODES)


class LoadImagesFromPathTests(unittest.TestCase):
    def test_node_outputs_an_image_list(self):
        registered = asyncio.run(NODES.BokujuuPersonalNodes().get_node_list())
        self.assertIn(NODES.BokujuuLoadImagesFromPath, registered)

        schema = NODES.BokujuuLoadImagesFromPath.define_schema()
        self.assertEqual(schema.node_id, "BokujuuLoadImagesFromPath")
        self.assertTrue(schema.outputs[0].is_output_list)
        self.assertEqual(schema.inputs[0].id, "directory")

    def test_loads_absolute_folder_as_native_size_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (2, 3), (255, 0, 0, 128)).save(root / "b.png")
            Image.new("RGB", (4, 1), (0, 255, 0)).save(root / "a.jpg", quality=95)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 8), (0, 0, 255)).save(root / "nested" / "c.png")
            (root / "notes.txt").write_text("skip", encoding="utf-8")

            images = NODES.BokujuuLoadImagesFromPath.execute(f'"{root}"')[0]

            self.assertEqual(len(images), 2)
            self.assertEqual(tuple(images[0].shape), (1, 1, 4, 3))
            self.assertEqual(tuple(images[1].shape), (1, 3, 2, 3))
            self.assertTrue(torch.allclose(images[1][0, 0, 0], torch.tensor([1.0, 0.0, 0.0])))

    def test_rejects_relative_missing_and_empty_directories(self):
        with self.assertRaises(ValueError):
            NODES.BokujuuLoadImagesFromPath.execute("images")

        missing = Path(tempfile.gettempdir()) / "bokujuu-load-images-missing"
        if missing.exists():
            shutil.rmtree(missing)
        with self.assertRaises(ValueError):
            NODES.BokujuuLoadImagesFromPath.execute(str(missing))

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                NODES.BokujuuLoadImagesFromPath.execute(directory)

    def test_fingerprint_changes_when_the_folder_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), (1, 2, 3)).save(root / "a.png")
            before = NODES.BokujuuLoadImagesFromPath.fingerprint_inputs(str(root))
            Image.new("RGB", (2, 2), (4, 5, 6)).save(root / "b.png")
            after = NODES.BokujuuLoadImagesFromPath.fingerprint_inputs(str(root))

        self.assertNotEqual(before, after)
        self.assertIsInstance(NODES.BokujuuLoadImagesFromPath.fingerprint_inputs("images"), str)
