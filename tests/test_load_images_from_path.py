import asyncio
import importlib.util
import shutil
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
        inputs = {node_input.id: node_input for node_input in schema.inputs}
        self.assertEqual(inputs["mode"].default, "queue")
        self.assertEqual(inputs["index"].default, 0)

    def test_loads_absolute_folder_as_native_size_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (2, 3), (255, 0, 0, 128)).save(root / "b.png")
            Image.new("RGB", (4, 1), (0, 255, 0)).save(root / "a.jpg", quality=95)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 8), (0, 0, 255)).save(root / "nested" / "c.png")
            (root / "notes.txt").write_text("skip", encoding="utf-8")

            images = NODES.BokujuuLoadImagesFromPath.execute(f'"{root}"', "list")[0]

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
            before = NODES.BokujuuLoadImagesFromPath.fingerprint_inputs(str(root), "list")
            Image.new("RGB", (2, 2), (4, 5, 6)).save(root / "b.png")
            after = NODES.BokujuuLoadImagesFromPath.fingerprint_inputs(str(root), "list")

        self.assertNotEqual(before, after)
        self.assertIsInstance(NODES.BokujuuLoadImagesFromPath.fingerprint_inputs("images"), str)

    def test_queue_mode_loads_one_image_and_submits_the_next_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 1), (0, 255, 0)).save(root / "a.jpg", quality=95)
            Image.new("RGBA", (2, 3), (255, 0, 0, 128)).save(root / "b.png")
            prompt = {
                "36": {
                    "class_type": "BokujuuLoadImagesFromPath",
                    "inputs": {"directory": str(root), "mode": "queue", "index": 0},
                }
            }
            queued = []

            class Queue:
                def __init__(self):
                    self.currently_running = {
                        7: (1, "job", prompt, {"client_id": "abc"}, ["36"], {"token": "secret"})
                    }

                def put(self, item):
                    queued.append(item)

            server_instance = SimpleNamespace(number=4, prompt_queue=Queue())
            NODES.BokujuuLoadImagesFromPath.hidden = SimpleNamespace(unique_id="36")
            import server
            with patch.object(server.PromptServer, "instance", server_instance, create=True):
                images = NODES.BokujuuLoadImagesFromPath.execute(str(root), "queue", 0)[0]
                last = NODES.BokujuuLoadImagesFromPath.execute(str(root), "queue", 1)[0]

        self.assertEqual(len(images), 1)
        self.assertEqual(tuple(images[0].shape), (1, 1, 4, 3))
        self.assertEqual(tuple(last[0].shape), (1, 3, 2, 3))
        self.assertEqual(prompt["36"]["inputs"]["index"], 0)
        self.assertEqual(len(queued), 1)
        number, _, next_prompt, extra_data, outputs, sensitive = queued[0]
        self.assertEqual(number, -4)
        self.assertEqual(server_instance.number, 5)
        self.assertEqual(next_prompt["36"]["inputs"]["index"], 1)
        self.assertEqual(extra_data["client_id"], "abc")
        self.assertEqual(outputs, ["36"])
        self.assertEqual(sensitive, {"token": "secret"})
        self.assertNotEqual(id(next_prompt), id(prompt))
