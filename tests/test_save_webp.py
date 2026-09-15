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
        self.assertIn(NODES.BokujuuSaveWebPWithJSON, registered)

        schema = NODES.BokujuuSaveWebP.define_schema()
        self.assertTrue(schema.is_output_node)
        self.assertNotIn("lossless", [node_input.id for node_input in schema.inputs])

        prefix = next(node_input for node_input in schema.inputs if node_input.id == "filename_prefix")
        self.assertIn("%date:yyyy-MM-dd%", prefix.tooltip)
        self.assertIn("%Empty Latent Image.width%", prefix.tooltip)

        audit_schema = NODES.BokujuuSaveWebPWithJSON.define_schema()
        self.assertTrue(audit_schema.is_output_node)
        audit_inputs = {node_input.id: node_input for node_input in audit_schema.inputs}
        self.assertTrue(audit_inputs["positive_prompt"].optional)
        self.assertTrue(audit_inputs["positive_prompt"].force_input)
        self.assertTrue(audit_inputs["negative_prompt"].optional)
        self.assertTrue(audit_inputs["lora_stack"].optional)

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

    def test_saves_same_basename_json_with_resolved_runtime_values(self):
        prompt = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
            "4": {
                "class_type": "ImpactWildcardEncode",
                "inputs": {
                    "model": ["1", 0],
                    "wildcard_text": "portrait, __hair__",
                    "populated_text": "portrait, silver hair, <lora:detail:0.65:0.4>",
                    "mode": "populate",
                },
                "_meta": {"title": "Positive wildcard"},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "seed": 987654321,
                    "steps": 24,
                    "cfg": 5.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                },
            },
        }
        positive = "portrait, silver hair, <lora:detail:0.65:0.4>"
        negative = "low quality"
        lora_stack = [
            ["style.safetensors", 0.72, 0.5],
            ["detail.safetensors", -0.15, -0.1],
        ]
        image = torch.rand((1, 32, 48, 3), generator=torch.Generator().manual_seed(11))
        NODES.BokujuuSaveWebPWithJSON.hidden = SimpleNamespace(
            prompt=prompt,
            extra_pnginfo={"workflow": {"nodes": []}},
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(NODES.folder_paths, "get_output_directory", return_value=output_dir):
                result = NODES.BokujuuSaveWebPWithJSON.execute(
                    image,
                    "audit/test",
                    82,
                    5,
                    positive_prompt=positive,
                    negative_prompt=negative,
                    lora_stack=lora_stack,
                )

            webp_files = list((Path(output_dir) / "audit").glob("*.webp"))
            json_files = list((Path(output_dir) / "audit").glob("*.json"))
            self.assertEqual(len(webp_files), 1)
            self.assertEqual(len(json_files), 1)
            self.assertEqual(webp_files[0].stem, json_files[0].stem)
            self.assertIs(result[0], image)

            audit = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(audit["schema"], "bokujuu_generation_audit")
            self.assertEqual(audit["schema_version"], 1)
            self.assertEqual(audit["image"]["filename"], webp_files[0].name)
            self.assertEqual(audit["image"]["width"], 48)
            self.assertEqual(audit["image"]["height"], 32)
            self.assertEqual(audit["resolved"]["positive_prompt"], positive)
            self.assertEqual(audit["resolved"]["negative_prompt"], negative)
            self.assertEqual(
                audit["resolved"]["loras"],
                [
                    {"name": "style.safetensors", "model_strength": 0.72, "clip_strength": 0.5},
                    {"name": "detail.safetensors", "model_strength": -0.15, "clip_strength": -0.1},
                ],
            )
            self.assertEqual(
                audit["resolved"]["wildcard_expansions"][0]["populated_text"],
                positive,
            )
            self.assertEqual(audit["resolved"]["prompt_lora_tags"][0]["name"], "detail")
            self.assertEqual(
                audit["parameters"]["seeds"],
                [{"node_id": "5", "class_type": "KSampler", "input": "seed", "value": 987654321}],
            )
            sampler = next(node for node in audit["parameters"]["nodes"] if node["node_id"] == "5")
            self.assertNotIn("model", sampler["inputs"])
            self.assertEqual(sampler["inputs"]["cfg"], 5.5)
            self.assertEqual(audit["execution_graph"], prompt)
            self.assertNotIn("runtime_trace", audit)

    def test_frontend_hooks_filename_prefix_serialization(self):
        source = (MODULE_PATH.parent / "web" / "save_webp.js").read_text(encoding="utf-8")
        self.assertIn("BokujuuSaveWebP", source)
        self.assertIn("BokujuuSaveWebPWithJSON", source)
        self.assertIn("applyTextReplacements", source)
        self.assertIn("filename_prefix", source)
        self.assertIn("serializeValue", source)

    def test_full_workflow_uses_direct_json_saver_connections(self):
        workflow = json.loads(
            (MODULE_PATH.parent / "workflows" / "anima_ga_F4_webp_json.json").read_text(
                encoding="utf-8"
            )
        )
        all_nodes = list(workflow["nodes"])
        for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
            all_nodes.extend(subgraph.get("nodes", []))

        self.assertNotIn("BokujuuAuditRecord", {node.get("type") for node in all_nodes})
        saver = next(
            node
            for node in workflow["nodes"]
            if node.get("type") == "BokujuuSaveWebPWithJSON"
        )
        self.assertEqual(
            [node_input.get("name") for node_input in saver["inputs"]],
            ["images", "positive_prompt", "negative_prompt", "lora_stack"],
        )
        self.assertTrue(all(node_input.get("link") is not None for node_input in saver["inputs"]))


if __name__ == "__main__":
    unittest.main()
