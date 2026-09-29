import importlib.util
import random
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path(__file__).resolve().parents[3]
DANBOT_ROOT = COMFYUI_ROOT / "custom_nodes" / "danbot-comfy-node"
sys.path.insert(0, str(COMFYUI_ROOT))
SPEC = importlib.util.spec_from_file_location("bokujuu_danbot_choice", ROOT / "danbot_choice.py")
CHOICE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHOICE
SPEC.loader.exec_module(CHOICE)


def _map_keys(source, name):
    match = re.search(rf"^{name} = \{{(.*?)^\}}", source, re.M | re.S)
    if match is None:
        raise AssertionError(f"{name} was not found")
    return re.findall(r'"([^"]+)"\s*:', match.group(1))


def _enabled_all():
    enabled = {name: True for name in (*CHOICE.RATING_CHOICES, *CHOICE.LENGTH_CHOICES)}
    weights = {name: 25 for name in enabled}
    return enabled, weights


class DanbotRandomChoiceTests(unittest.TestCase):
    def test_link_types_match_danbot_template_config(self):
        rating_keys = _map_keys((DANBOT_ROOT / "src" / "models" / "v2408.py").read_text(encoding="utf-8"), "RATING_MAP")
        length_keys = _map_keys((DANBOT_ROOT / "src" / "models" / "v2408.py").read_text(encoding="utf-8"), "LENGTH_MAP")
        formatter = (DANBOT_ROOT / "src" / "nodes" / "formatter.py").read_text(encoding="utf-8")

        self.assertIn('["auto"] + list(v2408.RATING_MAP.keys())', formatter)
        self.assertIn("list(v2408.LENGTH_MAP.keys())", formatter)
        self.assertEqual(CHOICE.RATING_LINK_TYPE, ["auto", *rating_keys])
        self.assertEqual(CHOICE.LENGTH_LINK_TYPE, length_keys)

        schema = CHOICE.BokujuuDanbotRandomChoice.GET_SCHEMA()
        self.assertEqual(schema.outputs[0].io_type, CHOICE.RATING_LINK_TYPE)
        self.assertEqual(schema.outputs[1].io_type, CHOICE.LENGTH_LINK_TYPE)
        self.assertEqual(CHOICE.BokujuuDanbotRandomChoice.RETURN_TYPES[0], CHOICE.RATING_LINK_TYPE)
        self.assertEqual(CHOICE.BokujuuDanbotRandomChoice.RETURN_TYPES[1], CHOICE.LENGTH_LINK_TYPE)

        from comfy_execution.validation import validate_node_input

        self.assertTrue(validate_node_input(CHOICE.RATING_LINK_TYPE, ["auto", *rating_keys]))
        self.assertTrue(validate_node_input(CHOICE.LENGTH_LINK_TYPE, list(length_keys)))

    def test_node_is_registered_by_the_extension(self):
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("BokujuuDanbotRandomChoice", source)

    def test_same_seed_is_stable_and_never_emits_auto(self):
        enabled, weights = _enabled_all()
        first = CHOICE.choose_danbot_values(1234, "auto", "auto", enabled, weights)
        second = CHOICE.choose_danbot_values(1234, "auto", "auto", enabled, weights)
        self.assertEqual(first, second)
        self.assertIn(first[0], CHOICE.RATING_CHOICES)
        self.assertIn(first[1], CHOICE.LENGTH_CHOICES)
        self.assertNotEqual(first[0], "auto")

    def test_draw_ignores_global_random_state(self):
        enabled, weights = _enabled_all()
        random.seed(99)
        expected = random.random()
        random.seed(99)
        CHOICE.choose_danbot_values(7, "auto", "auto", enabled, weights)
        self.assertEqual(random.random(), expected)

    def test_unchecked_and_zero_rate_are_excluded(self):
        enabled, weights = _enabled_all()
        enabled["sensitive"] = False
        enabled["explicit"] = False
        weights["questionable"] = 0
        for seed in range(40):
            rating, _, _ = CHOICE.choose_danbot_values(seed, "auto", "auto", enabled, weights)
            self.assertEqual(rating, "general")

    def test_rates_are_relative(self):
        enabled = {name: False for name in (*CHOICE.RATING_CHOICES, *CHOICE.LENGTH_CHOICES)}
        weights = {name: 0 for name in enabled}
        enabled["general"] = True
        enabled["questionable"] = True
        weights["general"] = 75
        weights["questionable"] = 25
        enabled["long"] = True
        weights["long"] = 10
        counts = {"general": 0, "questionable": 0}
        for seed in range(200):
            rating, length, _ = CHOICE.choose_danbot_values(seed, "auto", "auto", enabled, weights)
            self.assertIn(rating, counts)
            self.assertEqual(length, "long")
            counts[rating] += 1
        self.assertGreater(counts["general"], counts["questionable"])
        self.assertGreater(counts["questionable"], 0)

    def test_fixed_mode_ignores_the_pool(self):
        enabled = {name: False for name in (*CHOICE.RATING_CHOICES, *CHOICE.LENGTH_CHOICES)}
        weights = {name: 0 for name in enabled}
        rating, length, summary = CHOICE.choose_danbot_values(
            1,
            "explicit",
            "very_short",
            enabled,
            weights,
        )
        self.assertEqual((rating, length), ("explicit", "very_short"))
        self.assertEqual(summary, "rating: explicit\nlength: very_short")

    def test_empty_pool_raises(self):
        enabled, weights = _enabled_all()
        for name in CHOICE.RATING_CHOICES:
            enabled[name] = False
        with self.assertRaises(ValueError):
            CHOICE.choose_danbot_values(1, "auto", "auto", enabled, weights)

    def test_execute_returns_preview_and_connectable_values(self):
        flags = {}
        for name in CHOICE.RATING_CHOICES:
            flags[f"rating_{name}"] = name in {"general", "questionable"}
            flags[f"rating_{name}_rate"] = 50 if name in {"general", "questionable"} else 0
        for name in CHOICE.LENGTH_CHOICES:
            flags[f"length_{name}"] = name == "short"
            flags[f"length_{name}_rate"] = 100 if name == "short" else 0
        output = CHOICE.BokujuuDanbotRandomChoice.execute(3, "auto", "auto", **flags)
        rating, length, summary = output.result
        self.assertIn(rating, {"general", "questionable"})
        self.assertEqual(length, "short")
        self.assertIn(rating, summary)
        self.assertEqual(output.ui.value, summary)


if __name__ == "__main__":
    unittest.main()
