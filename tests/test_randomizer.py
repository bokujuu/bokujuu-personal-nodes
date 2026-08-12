import importlib.util
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


if __name__ == "__main__":
    unittest.main()
