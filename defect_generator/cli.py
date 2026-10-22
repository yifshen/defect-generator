"""Explicit manifest/config CLI. Paths in manifests are relative to the manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from .core import Config, Source, generate, save


def read_rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError("Source must be RGB; convert explicitly before registering the source")
        if im.getexif().get(274, 1) != 1:
            raise ValueError("Normalize EXIF orientation before registering image/ROI")
        return np.array(im)


def read_roi(path):
    with Image.open(path) as im:
        if im.mode not in {"1", "L"}:
            raise ValueError("ROI must be a binary single-channel image")
        a = np.array(im)
    if not np.isin(a, [0, 1, 255]).all():
        raise ValueError("ROI contains nonbinary values")
    return a != 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train-only native-resolution synthetic defects")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        source = Source(**manifest["source"])
        source.validate()  # Reject forbidden splits before opening source pixels.
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
        config.validate()
        image_path = args.manifest.parent / manifest["image"]
        roi_path = args.manifest.parent / manifest["roi"]
        result = generate(read_rgb(image_path), read_roi(roi_path), source, config)
        result.metadata["input_files"] = {
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "roi_sha256": hashlib.sha256(roi_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        }
        save(result, args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
