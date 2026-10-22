import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from defect_generator import Config, Source, generate, save
from defect_generator.cli import main


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((91, 137, 3), 140, dtype=np.uint8)
        self.roi = np.zeros((91, 137), dtype=bool)
        self.roi[3:89, 4:130] = True
        self.roi[30:60, 50:80] = False  # Irregular ROI with a hole.
        self.source = Source("original-1", "train", True, "sim", "v1")

    def test_all_kinds_determinism_mask_roi_immutability(self):
        for kind in ("scratch", "line", "spot"):
            with self.subTest(kind=kind):
                image, roi = self.image.copy(), self.roi.copy()
                a = generate(image, roi, self.source, Config(kind=kind, count=5))
                b = generate(image, roi, self.source, Config(kind=kind, count=5))
                np.testing.assert_array_equal(a.image, b.image)
                self.assertEqual(a.metadata, b.metadata)
                self.assertEqual(a.image.shape, image.shape)
                np.testing.assert_array_equal(a.mask, np.any(a.image != image, axis=2))
                self.assertTrue(a.mask.any())
                self.assertFalse(a.mask[~roi].any())
                np.testing.assert_array_equal(image, self.image)
                np.testing.assert_array_equal(roi, self.roi)
                self.assertFalse(np.shares_memory(a.image, image))
                c = generate(image, roi, self.source, Config(kind=kind, count=5, seed=99))
                self.assertFalse(np.array_equal(a.image, c.image))

    def test_one_pixel_line_and_orientation(self):
        roi = np.ones_like(self.roi)
        for angle in (0, 90):
            r = generate(self.image, roi, self.source,
                         Config(kind="line", width=1, length=10, angle=angle, angle_jitter=0))
            ys, xs = np.where(r.mask)
            self.assertEqual(len(set(ys if angle == 0 else xs)), 1)

    def test_single_pixel_roi_and_saturation(self):
        roi = np.zeros_like(self.roi)
        roi[0, 0] = True
        r = generate(self.image, roi, self.source, Config(kind="spot", width=1))
        self.assertEqual(r.mask.sum(), 1)
        with self.assertRaises(ValueError):
            generate(np.zeros_like(self.image), roi, self.source)

    def test_split_and_clean_gate(self):
        for split in ("val", "validation", "test", "V-select", "V-cal", "E", "unknown", "TRAIN"):
            with self.assertRaises(ValueError):
                generate(self.image, self.roi, Source("a", split, True, "d", "v"))
        with self.assertRaises(ValueError):
            generate(self.image, self.roi, Source("a", "train", False, "d", "v"))

    def test_invalid_inputs(self):
        for cfg in (Config(width=0), Config(intensity=0), Config(kind="bad"),
                    Config(length=float("nan")), Config(seed=-1), Config(count=1.5)):
            with self.assertRaises(ValueError):
                generate(self.image, self.roi, self.source, cfg)
        for roi in (self.roi.astype(np.uint8), self.roi[:3], np.zeros_like(self.roi)):
            with self.assertRaises(ValueError):
                generate(self.image, roi, self.source)

    def test_cli_and_lossless_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            Image.fromarray(self.image).save(p / "normal.png")
            Image.fromarray(self.roi.astype(np.uint8)*255).save(p / "roi.png")
            original = (p / "normal.png").read_bytes()
            manifest = {"image": "normal.png", "roi": "roi.png", "source": self.source.__dict__}
            (p / "manifest.json").write_text(json.dumps(manifest))
            (p / "config.json").write_text('{"seed": 12}')
            main(["--manifest", str(p / "manifest.json"), "--config", str(p / "config.json"),
                  "--output", str(p / "out")])
            out = np.array(Image.open(p / "out/image.png"))
            mask = np.array(Image.open(p / "out/mask.png")) != 0
            np.testing.assert_array_equal(mask, np.any(out != self.image, axis=2))
            self.assertEqual(original, (p / "normal.png").read_bytes())
            meta = json.loads((p / "out/metadata.json").read_text())
            self.assertEqual(meta["source"]["image_id"], "original-1")
            self.assertEqual(meta["output_split"], "train")
            self.assertEqual(len(meta["files"]), 3)
            with self.assertRaises(FileExistsError):
                save(generate(self.image, self.roi, self.source), p / "out")


if __name__ == "__main__":
    unittest.main()
