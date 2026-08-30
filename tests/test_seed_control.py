import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFYUI_ROOT))
SPEC = importlib.util.spec_from_file_location("bokujuu_personal_nodes_seed_control", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODES
SPEC.loader.exec_module(NODES)


class SeedControlTests(unittest.TestCase):
    def test_seed_control_is_registered_as_frontend_only_node(self):
        node_classes = asyncio.run(NODES.BokujuuPersonalNodes().get_node_list())
        self.assertIn(NODES.BokujuuSeedControl, node_classes)

        schema = NODES.BokujuuSeedControl.define_schema()
        self.assertEqual(schema.node_id, "BokujuuSeedControl")
        self.assertEqual(schema.inputs, [])
        self.assertEqual(schema.outputs, [])

    def test_seed_control_execute_has_no_server_result(self):
        self.assertIsNone(NODES.BokujuuSeedControl.execute().result)

    def test_seed_control_example_has_three_seed_targets(self):
        workflow_path = Path(__file__).resolve().parents[1] / "workflows" / "seed_control_example.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = [node["type"] for node in workflow["nodes"]]

        self.assertEqual(node_types.count("RandomNoise"), 3)
        self.assertIn("BokujuuSeedControl", node_types)
        self.assertIn([1, 5, 0, 6, 0, "IMAGE"], workflow["links"])


if __name__ == "__main__":
    unittest.main()
