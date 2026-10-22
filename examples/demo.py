"""Create simulated inputs and outputs outside the repository: python -m examples.demo /tmp/paint-demo"""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from defect_generator import Config, Source, generate, save


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[:256, :384]
    gray = np.clip(150 + 15*x/384 + 3*np.sin(y/17), 0, 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    roi = ((x-192)/160)**2 + ((y-128)/100)**2 < 1
    image[~roi] = [35, 40, 45]
    Image.fromarray(image).save(root / "normal.png")
    Image.fromarray(roi.astype(np.uint8)*255).save(root / "roi.png")
    source = Source("simulated-001", "train", True, "simulation", "v1")
    manifest = {"image": "normal.png", "roi": "roi.png", "source": source.__dict__}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for kind in ("scratch", "line", "spot"):
        cfg = Config(kind=kind, width=3, count=4, seed=42)
        (root / f"{kind}.json").write_text(json.dumps(cfg.__dict__, indent=2))
        save(generate(image, roi, source, cfg), root / kind)


if __name__ == "__main__":
    main()
