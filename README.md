# defect-generator

First baseline for industrial paint defect synthesis. NumPy + Pillow only; no model downloads or training. This is a geometry and software baseline, **not evidence that generated defects match real paint defect distributions**. No real paint images have been used to validate realism or downstream detection performance.

## Install and run

Python 3.10+:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.demo /tmp/paint-demo
# Equivalent installed CLI using simulated inputs:
defect-generator --manifest /tmp/paint-demo/manifest.json --config /tmp/paint-demo/line.json --output /tmp/paint-cli
```

Use new output directories. Existing directories are rejected to protect sources and prior runs. Generated images stay outside Git. Runtime dependencies have compatibility bounds, not an exact environment lock; each run records Python, NumPy and Pillow versions. For exact replay use the same versions and inputs.

## Input contract and leakage boundary

Images are uint8 RGB at original resolution. ROI is a matching binary mask: true/1/255 pixels are **confirmed clean sample surface**, zero is background, unknown, or excluded real defects. Intersect sample ROI with a separately verified clean-region mask before use if the full sample is not clean. The tool cannot establish cleanliness from pixels.

Only the exact split `train` is accepted, with `clean_confirmed: true`. Validation, test, unknown and all other split names fail before CLI image loading. Split and cleanliness are caller assertions, not automatic proofs. Maintain an authoritative upstream split registry, group duplicate images and derived crops by original identity, and check content hashes across splits. Relabeling a test image as train cannot be detected by this standalone tool. Never use validation/test pixels, even apparently clean areas.

Manifest (paths relative to the manifest):

```json
{
  "image": "normal.png",
  "roi": "roi.png",
  "source": {
    "image_id": "original-001",
    "parent_image_id": null,
    "dataset_id": "paint-v1",
    "annotation_version": "clean-roi-v1",
    "split": "train",
    "clean_confirmed": true
  }
}
```

For a crop, set `parent_image_id` to the original image ID. Count unique original IDs (parent IDs for crops) across runs; ten variants of one original still count as one original, not ten independent training images. The per-output `independent_original_count: 1` must never be summed to estimate a dataset budget.

## Python API

```python
import numpy as np
from PIL import Image
from defect_generator import Config, Source, generate, save

image = np.array(Image.open("normal.png"))  # uint8 RGB, no resizing
roi = np.array(Image.open("roi.png").convert("L")) > 0
source = Source("original-001", "train", True, "paint-v1", "clean-roi-v1")
result = generate(image, roi, source,
                  Config(kind="scratch", width=2, length=60, angle=25,
                         angle_jitter=5, curvature=0.1, intensity=-35, seed=42))
save(result, "new-run")
```

API ROI must have bool dtype. CLI rejects nonbinary masks, non-RGB images and non-normalized EXIF orientation to prevent silent coordinate/color changes. Normalize image and annotations together before source registration. No resizing, cropping, antialiasing or JPEG output is performed. Input arrays and files are not modified.

## Configuration

A config JSON contains any subset of `Config` fields; unknown fields fail. Defaults:

```json
{"kind":"scratch","count":1,"width":2,"length":40.0,"angle":0.0,"angle_jitter":15.0,"curvature":0.15,"aspect":1.0,"intensity":-40,"seed":0}
```

`line` draws a straight polyline. `scratch` draws a sampled sinusoidal bend with displacement `curvature * length`. Both use width and length in native pixels. `spot` is a rotated ellipse with diameters `width` and `width * aspect`; length/curvature are ignored. Angle is degrees clockwise in image coordinates (x right, y down), with uniform ±jitter. Intensity is a signed additive RGB change with uint8 clipping; positive brightens and negative darkens. This deliberately simple photometric model does not simulate paint optics.

Centres are sampled uniformly from valid ROI pixels using a local PCG64 generator. Primitives may overlap and may be clipped by the image or ROI, including holes. Count is the number of successful drawing operations, not connected components. Each operation must change at least one pixel; otherwise placement retries up to 100 times and then fails explicitly. Nominal geometry sizes can be shortened by clipping. One-pixel lines/spots are supported; raster rounding affects projected length and diagonal width.

## Output and provenance

Each bundle contains lossless `image.png`, `mask.png` (0/255), effective `roi.png`, and `metadata.json`. The mask is exactly the set of pixels whose final RGB value differs from the source, excluding saturated unchanged pixels. No background pixels can change. Metadata includes original/parent identity, dataset/split/annotation version, source decoded-pixel hash, ROI/output/mask hashes, full config and seed, sampled geometry, runtime versions, generator version, output hashes, and CLI input file/manifest hashes. Preserve source files and the code commit alongside runs; hashes establish identity but cannot reconstruct missing sources. Record the Git commit used in your experiment registry.

## Validation and limits

Automated tests use simulated normal images and verify deterministic replay, different seeds, exact masks, native shape, one-pixel defects, irregular ROI holes and edges, saturation failure, invalid input rejection, train/clean gates, unchanged input arrays/files, lossless CLI outputs and overwrite protection. CI runs the tests and the example on Python 3.10 and 3.12.

These are code function checks. Real-data validation must separately measure defect size/contrast distributions, visual agreement and downstream segmentation performance using a fixed train/validation/test protocol. Synthetic variants remain training derivatives. No claims of real defect realism, detection gains or cross-background generalization are made.

Design constraint source: [paint few-shot research blueprint](https://github.com/yifshen/few-shot-learning-pipeline/blob/main/thoughts/7_Paint_Defect_Few_Shot_Research_Blueprint.md).
