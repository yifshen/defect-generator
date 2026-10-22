"""Simple raster geometry. No resizing, learned models, or implicit data access."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version


@dataclass(frozen=True)
class Source:
    image_id: str
    split: str
    clean_confirmed: bool
    dataset_id: str
    annotation_version: str
    parent_image_id: str | None = None

    def validate(self):
        if self.split != "train":
            raise ValueError("Only split='train' is allowed; validation/test/unknown are rejected")
        if self.clean_confirmed is not True:
            raise ValueError("Effective ROI must be explicitly confirmed clean")
        for name in ("image_id", "dataset_id", "annotation_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.parent_image_id is not None and not self.parent_image_id.strip():
            raise ValueError("parent_image_id must be nonempty when provided")


@dataclass(frozen=True)
class Config:
    kind: str = "scratch"
    count: int = 1
    width: int = 2
    length: float = 40.0
    angle: float = 0.0
    angle_jitter: float = 15.0
    curvature: float = 0.15
    aspect: float = 1.0
    intensity: int = -40
    seed: int = 0

    def validate(self):
        if self.kind not in {"scratch", "line", "spot"}:
            raise ValueError("kind must be scratch, line, or spot")
        for name in ("count", "width", "intensity", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.count < 1 or self.width < 1 or self.seed < 0:
            raise ValueError("count/width must be positive; seed must be nonnegative")
        if not -255 <= self.intensity <= 255 or self.intensity == 0:
            raise ValueError("intensity must be a nonzero integer in [-255,255]")
        for name in ("length", "angle", "angle_jitter", "curvature", "aspect"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.length <= 0 or self.aspect <= 0 or self.angle_jitter < 0 or self.curvature < 0:
            raise ValueError("Invalid size, aspect, jitter, or curvature")


@dataclass
class Result:
    image: np.ndarray
    mask: np.ndarray
    roi: np.ndarray
    metadata: dict


def digest(array):
    """Hash shape, dtype and decoded C-order pixels, independent of file encoding."""
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def generate(image: np.ndarray, roi: np.ndarray, source: Source,
             config: Config = Config()) -> Result:
    """Return new RGB uint8 pixels and exact changed-pixel bool mask.

    roi must be a bool HxW mask of confirmed clean sample pixels. Holes and
    disconnected regions are supported. Geometry crossing a boundary is clipped.
    Every requested primitive must change >=1 pixel or generation fails.
    """
    source.validate()
    config.validate()
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    if not isinstance(roi, np.ndarray) or roi.dtype != np.bool_ or roi.shape != image.shape[:2] or not roi.any():
        raise ValueError("roi must be a nonempty bool mask matching image HxW")
    h, w = roi.shape
    rng = np.random.Generator(np.random.PCG64(config.seed))
    output = image.copy()
    candidates = np.argwhere(roi)
    records = []
    for _ in range(config.count):
        for attempt in range(100):
            y, x = map(int, candidates[rng.integers(len(candidates))])
            angle = config.angle + rng.uniform(-config.angle_jitter, config.angle_jitter)
            theta = math.radians(angle)
            dx, dy = math.cos(theta), math.sin(theta)
            canvas = Image.new("1", (w, h))
            draw = ImageDraw.Draw(canvas)
            if config.kind == "spot":
                # Rotated ellipse evaluated at native pixel centres.
                a, b = config.width / 2, config.width * config.aspect / 2
                r = math.ceil(max(a, b))
                x0, x1 = max(0, x-r), min(w, x+r+1)
                y0, y1 = max(0, y-r), min(h, y+r+1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                u = (xx-x)*dx + (yy-y)*dy
                v = -(xx-x)*dy + (yy-y)*dx
                shape = np.zeros((h, w), dtype=bool)
                shape[y0:y1, x0:x1] = (u/a)**2 + (v/b)**2 <= 1
                points = None
            else:
                t = np.linspace(-0.5, 0.5, max(2, math.ceil(config.length)+1))
                bend = (config.curvature * config.length * np.sin(np.pi*(t+0.5))
                        if config.kind == "scratch" else np.zeros_like(t))
                points = [(round(x+q*config.length*dx-z*dy),
                           round(y+q*config.length*dy+z*dx)) for q, z in zip(t, bend)]
                draw.line(points, fill=1, width=config.width)
                shape = np.array(canvas, dtype=bool)
            support = shape & roi
            values = np.clip(output[support].astype(np.int16) + config.intensity, 0, 255).astype(np.uint8)
            if np.any(values != output[support]):
                output[support] = values
                records.append({"center_xy": [x, y], "angle_degrees": float(angle),
                                "points_xy": points, "clipped_pixels": int((shape & ~roi).sum())})
                break
        else:
            raise ValueError("No visible defect after 100 attempts; check ROI and intensity/saturation")
    mask = np.any(output != image, axis=2)
    metadata = {
        "schema_version": 1, "generator_version": "0.1.0",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": asdict(source), "source_pixel_sha256": digest(image),
        "roi_sha256": digest(roi), "output_pixel_sha256": digest(output),
        "mask_sha256": digest(mask), "shape_hwc": list(image.shape),
        "config": asdict(config), "primitives": records,
        "changed_pixels": int(mask.sum()), "output_split": "train",
        "independent_original_count": 1,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pillow": pillow_version, "rng": "PCG64"},
    }
    return Result(output, mask, roi.copy(), metadata)


def save(result: Result, directory: str | Path):
    """Write lossless PNGs and metadata into a new directory; never overwrite."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    Image.fromarray(result.image).save(path / "image.png")
    Image.fromarray(result.mask.astype(np.uint8)*255).save(path / "mask.png")
    Image.fromarray(result.roi.astype(np.uint8)*255).save(path / "roi.png")
    metadata = dict(result.metadata)
    metadata["files"] = {}
    for name in ("image.png", "mask.png", "roi.png"):
        metadata["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
