from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch


CUSTOM_NODES_ROOT = Path(__file__).resolve().parents[2]
COMFYUI_ROOT = CUSTOM_NODES_ROOT.parent
sys.path.insert(0, str(COMFYUI_ROOT))
sys.path.insert(0, str(CUSTOM_NODES_ROOT))

POST = importlib.import_module("bokujuu-personal-nodes.dlss5.postprocess")


class DLSS5PostprocessTests(unittest.TestCase):
    def test_cas_and_unsharp_preserve_shape_range_and_dtype(self):
        images = torch.rand((1, 24, 32, 3), dtype=torch.float32)
        for output in (
            POST.cas_sharpen(images, amount=0.35),
            POST.unsharp_mask(images, radius=1.0, percent=30, threshold=2),
        ):
            self.assertEqual(output.shape, images.shape)
            self.assertEqual(output.dtype, images.dtype)
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_cas_preserves_alpha_and_changes_rgb(self):
        images = torch.rand((1, 24, 32, 4), dtype=torch.float32)
        output = POST.cas_sharpen(images, amount=0.35)
        self.assertEqual(output.shape, images.shape)
        self.assertTrue(torch.equal(output[..., 3], images[..., 3]))
        self.assertFalse(torch.equal(output[..., :3], images[..., :3]))

    def test_cas_zero_is_identity_and_higher_amounts_keep_changing(self):
        images = torch.rand((1, 24, 32, 3), dtype=torch.float32)
        zero = POST.cas_sharpen(images, amount=0.0)
        mild = POST.cas_sharpen(images, amount=1.0)
        strong = POST.cas_sharpen(images, amount=2.0)
        stronger = POST.cas_sharpen(images, amount=4.0)
        self.assertTrue(torch.equal(zero, images))
        self.assertFalse(torch.equal(mild, images))
        self.assertFalse(torch.equal(strong, mild))
        self.assertFalse(torch.equal(stronger, strong))

    def test_nis_sdk_resolution_rejects_missing_checkout(self):
        with self.assertRaisesRegex(POST.NISSharpenError, "was not found"):
            POST.resolve_nis_sdk(Path("Z:/definitely-missing-nis-sdk"))


if __name__ == "__main__":
    unittest.main()
