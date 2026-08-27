import asyncio
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFYUI_ROOT))
SPEC = importlib.util.spec_from_file_location("bokujuu_personal_nodes", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODES
SPEC.loader.exec_module(NODES)


class RandomStrengthTests(unittest.TestCase):
    def test_is_reproducible_and_preserves_sum(self):
        first = NODES.random_strengths(11, 2.5, -0.2, 1.0, False, 1234)
        second = NODES.random_strengths(11, 2.5, -0.2, 1.0, False, 1234)
        self.assertEqual(first, second)
        self.assertEqual(sum(round(value * 100) for value in first), 250)
        self.assertTrue(all(-0.2 <= value <= 1.0 for value in first))

    def test_positive_minimum_is_supported(self):
        values = NODES.random_strengths(5, 2.0, 0.2, 1.0, False, 7)
        self.assertTrue(all(0.2 <= value <= 1.0 for value in values))
        self.assertEqual(sum(round(value * 100) for value in values), 200)

    def test_minimum_above_maximum_does_not_error(self):
        values = NODES.random_strengths(3, 1.0, 0.4, 0.2, False, 7)
        self.assertEqual(values, [0.4, 0.4, 0.4])

    def test_randomize_total_does_not_change_global_rng(self):
        random.seed(99)
        expected = random.random()
        random.seed(99)
        NODES.random_strengths(4, 2.0, 0.0, 1.0, True, 7)
        self.assertEqual(random.random(), expected)

    def test_merge_and_report_keep_every_name(self):
        stack = NODES._merge_lora_stack([
            ["a.safetensors", 0.1, 0.2],
            ["b.safetensors", 0.3, 0.4],
            ["a.safetensors", 0.5, 0.6],
        ])
        self.assertEqual(stack, [["a.safetensors", 0.6, 0.8], ["b.safetensors", 0.3, 0.4]])
        report = NODES.format_stack_report(stack)
        self.assertIn("Loaded LoRAs: 2", report)
        self.assertIn("a.safetensors", report)
        self.assertIn("b.safetensors", report)

    def test_selection_accepts_widget_json_and_legacy_list(self):
        expected = ["a.safetensors", "folder\\b.safetensors"]
        self.assertEqual(NODES.parse_lora_selection('["a.safetensors", "folder\\\\b.safetensors"]'), expected)
        self.assertEqual(NODES.parse_lora_selection(expected), expected)

    def test_selection_rejects_non_list_json(self):
        with self.assertRaises(ValueError):
            NODES.parse_lora_selection('{"lora": "a.safetensors"}')

    def test_random_lora_selection_is_reproducible_without_duplicates(self):
        candidates = [f"lora_{index}.safetensors" for index in range(8)]
        first = NODES.random_lora_stack_rows(candidates, 3, -0.5, 0.75, 1234)
        second = NODES.random_lora_stack_rows(candidates, 3, -0.5, 0.75, 1234)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({row[0] for row in first}), 3)
        self.assertTrue(all(-0.5 <= row[1] <= 0.75 and row[1] == row[2] for row in first))

    def test_random_lora_selection_clamps_count_and_deduplicates_candidates(self):
        rows = NODES.random_lora_stack_rows(["a.safetensors", "a.safetensors", "b.safetensors"], 10, 0.2, 0.6, 7)
        self.assertEqual({row[0] for row in rows}, {"a.safetensors", "b.safetensors"})
        self.assertEqual(len(rows), 2)

    def test_random_lora_selection_handles_reversed_range_without_global_rng_changes(self):
        random.seed(99)
        expected = random.random()
        random.seed(99)
        rows = NODES.random_lora_stack_rows(["a.safetensors", "b.safetensors"], 2, 0.4, 0.2, 7)

        self.assertEqual([row[1] for row in rows], [0.4, 0.4])
        self.assertEqual(random.random(), expected)

    def test_random_lora_sample_workflow_connects_stack_apply(self):
        workflow_path = Path(__file__).resolve().parents[1] / "workflows" / "random_lora_selector_example.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["id"]: node["type"] for node in workflow["nodes"]}

        self.assertEqual(node_types[1], "BokujuuRandomLoraSelector")
        self.assertEqual(node_types[2], "easy loraStackApply")
        self.assertIn([1, 1, 0, 2, 0, "LORA_STACK"], workflow["links"])

    def test_random_lora_selector_is_registered(self):
        node_classes = asyncio.run(NODES.BokujuuPersonalNodes().get_node_list())
        self.assertIn(NODES.BokujuuRandomLoraSelector, node_classes)

    def test_random_lora_selector_executes_and_merges_input_stack(self):
        output = NODES.BokujuuRandomLoraSelector.execute(
            2,
            0.2,
            0.8,
            42,
            ["a.safetensors", "b.safetensors", "c.safetensors"],
            [["fixed.safetensors", 0.5, 0.6]],
        )
        stack, report, count = output.result

        self.assertEqual(stack[0], ["fixed.safetensors", 0.5, 0.6])
        self.assertEqual(count, 3)
        self.assertIn("Loaded LoRAs: 3", report)


if __name__ == "__main__":
    unittest.main()
