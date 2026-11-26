"""Replayable local cast-shadow and sun-patch transforms, before tensor normalization.

Self-contained NumPy/OpenCV/Pillow module: no detector, dataset, torch or research
imports. A future model package can vendor this module and record its source hash.
The renderer approximates incident illumination; it does not recover scene depth,
reflectance, the original sun, or a true shadow-free image.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

IMPLEMENTATION = "local_illumination_v001"
MODES = ("shadow", "sunpatch", "mixed")
PATTERNS = ("foliage", "band")


@dataclass(frozen=True)
# 物理近似与采样参数；用户可编辑的中文解释见同包配置YAML。
class IlluminationConfig:
    probability: float = 0.5
    strength_range: tuple[float, float] = (0.25, 0.85)
    mode_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    foliage_probability: float = 0.65
    mask_resolution: int = 384
    ambient_floor: float = 0.20
    shadow_depth: float = 0.96
    sun_ev: float = 2.6
    foliage_coverage: float = 0.64
    penumbra_range: tuple[float, float] = (0.003, 0.014)
    patch_count: int = 24
    patch_radius_range: tuple[float, float] = (0.025, 0.085)

    def __post_init__(self):
        for name in ("strength_range", "mode_weights", "penumbra_range", "patch_radius_range"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        bounds = {
            "probability": (0, 1),
            "foliage_probability": (0, 1),
            "ambient_floor": (0.1, 0.6),
            "shadow_depth": (0, 1),
            "sun_ev": (0, 3),
            "foliage_coverage": (0.15, 0.85),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        for name, low, high in (
            ("strength_range", 0, 1),
            ("penumbra_range", 0.001, 0.05),
            ("patch_radius_range", 0.01, 0.2),
        ):
            pair = getattr(self, name)
            if (
                len(pair) != 2
                or any(isinstance(v, bool) or not np.isfinite(v) for v in pair)
                or not low <= pair[0] <= pair[1] <= high
            ):
                raise ValueError(f"invalid {name}")
        if (
            len(self.mode_weights) != 3
            or any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in self.mode_weights)
            or not np.isfinite(sum(self.mode_weights))
            or sum(self.mode_weights) <= 0
        ):
            raise ValueError("mode_weights must contain three nonnegative finite weights")
        if type(self.mask_resolution) is not int or not 64 <= self.mask_resolution <= 1024:
            raise ValueError("mask_resolution must be an integer in [64,1024]")
        if type(self.patch_count) is not int or not 1 <= self.patch_count <= 96:
            raise ValueError("patch_count must be an integer in [1,96]")


@dataclass(frozen=True)
# 将一次随机空间方案完整留档，跨强度比较时复用同一方案。
class IlluminationPlan:
    mode: str
    pattern: str
    shape: tuple[int, int]
    seed: int
    penumbra: float
    config: IlluminationConfig
    implementation: str = IMPLEMENTATION

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if (
            self.implementation != IMPLEMENTATION
            or self.mode not in MODES
            or self.pattern not in PATTERNS
        ):
            raise ValueError("unsupported illumination plan")
        if len(self.shape) != 2 or any(type(v) is not int or v < 8 for v in self.shape):
            raise ValueError("plan requires an image height/width of at least 8")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("invalid plan seed")
        if not np.isfinite(self.penumbra) or not 0.001 <= self.penumbra <= 0.05:
            raise ValueError("invalid plan penumbra")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["config"] = IlluminationConfig(**value["config"])
        return cls(**value)


@dataclass
class IlluminationResult:
    image: np.ndarray
    metadata: dict
    shadow_mask: np.ndarray | None = None
    sun_mask: np.ndarray | None = None


def validate_rgb(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 8
    ):
        raise ValueError("expected HWC uint8 RGB with height/width >= 8, before normalization")


def sample_plan(shape, rng, config=None, *, mode=None, pattern=None):
    """Sample geometry once; reuse the plan for paired strengths or save as JSON."""
    config = config or IlluminationConfig()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("pass an explicit NumPy Generator")
    if mode is None:
        weights = np.asarray(config.mode_weights, dtype=np.float64)
        mode = str(rng.choice(MODES, p=weights / weights.sum()))
    if pattern is None:
        pattern = "foliage" if rng.random() < config.foliage_probability else "band"
    return IlluminationPlan(
        mode,
        pattern,
        tuple(int(v) for v in shape[:2]),
        int(rng.integers(0, 2**63)),
        float(rng.uniform(*config.penumbra_range)),
        config,
    )


def sample_rng(seed: int, epoch: int, sample_id: str, draw: int = 0):
    """Order/rank independent RNG; draw disambiguates repeated views of one sample."""
    if (
        any(type(v) is not int or v < 0 for v in (seed, epoch, draw))
        or not isinstance(sample_id, str)
        or not sample_id
    ):
        raise ValueError("use nonnegative seed/epoch/draw and an explicit nonempty sample_id")
    payload = json.dumps(
        [IMPLEMENTATION, seed, epoch, sample_id, draw], ensure_ascii=False
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def _value_noise(shape, rng):
    result = np.zeros(shape, np.float32)
    for cells, weight in ((5, 0.55), (13, 0.30), (31, 0.15)):
        grid = rng.uniform(0, 1, (cells, cells)).astype(np.float32)
        result += weight * cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return result


def _band(shape, rng):
    height, width = shape
    theta = rng.uniform(-1.2, 1.2)
    axis = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    center = np.array([width, height]) * rng.uniform(0.35, 0.65, 2)
    half_width = min(shape) * rng.uniform(0.14, 0.25)
    half_length = max(shape) * 0.9
    points = np.asarray(
        [
            center + a * half_length * axis + b * half_width * normal
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ],
        dtype=np.int32,
    )
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, points, 1)
    return mask.astype(np.float32)


def _patches(shape, rng, config):
    height, width = shape
    result = np.zeros(shape, np.uint8)
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    for _ in range(config.patch_count):
        center = rng.uniform(0.03, 0.97, 2) * [width, height]
        radii = rng.uniform(*config.patch_radius_range, 2) * min(shape)
        phase = rng.uniform(0, 2 * np.pi, 3)
        irregular = (
            1
            + 0.16 * np.sin(3 * angles + phase[0])
            + 0.11 * np.sin(5 * angles + phase[1])
            + 0.06 * np.sin(8 * angles + phase[2])
        )
        theta = rng.uniform(0, np.pi)
        xy = np.stack([np.cos(angles), np.sin(angles)], 1) * (irregular[:, None] * radii)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = np.rint(xy @ rotation.T + center).astype(np.int32)
        cv2.fillPoly(result, [points], 1)
    return result.astype(np.float32)


def _masks(plan, region):
    height, width = plan.shape
    scale = min(1, plan.config.mask_resolution / max(plan.shape))
    small = (max(8, round(height * scale)), max(8, round(width * scale)))
    shadow_rng = np.random.default_rng(np.random.SeedSequence([plan.seed, 1]))
    sun_rng = np.random.default_rng(np.random.SeedSequence([plan.seed, 2]))
    if plan.pattern == "foliage":
        field = _value_noise(small, shadow_rng)
        shadow = (field < np.quantile(field, plan.config.foliage_coverage)).astype(np.float32)
    else:
        shadow = _band(small, shadow_rng)
    sun = _patches(small, sun_rng, plan.config)
    sigma = max(0.35, min(small) * plan.penumbra)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigma)
    sun = cv2.GaussianBlur(sun, (0, 0), max(0.35, sigma * 0.60))
    shadow = np.clip(cv2.resize(shadow, (width, height)), 0, 1)
    sun = np.clip(cv2.resize(sun, (width, height)), 0, 1)
    if plan.mode == "mixed":
        shadow *= 1 - sun  # Gaps in cast shadows receive local direct illumination.
    if plan.mode == "shadow":
        sun.fill(0)
    elif plan.mode == "sunpatch":
        shadow.fill(0)
    return shadow * region, sun * region


def render_illumination(image, plan, strength, *, region_mask=None, return_masks=False):
    # 只改局部入射光，不读取病害框；暗部保留环境光，亮部采用软肩。
    """Change incident light only; never access, resize, filter or alter annotations."""
    validate_rgb(image)
    if image.shape[:2] != plan.shape:
        raise ValueError("plan and image dimensions differ")
    if isinstance(strength, bool) or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite in [0,1]")
    if region_mask is None:
        region = np.ones(image.shape[:2], np.float32)
    else:
        region = np.asarray(region_mask, dtype=np.float32)
        if (
            region.shape != image.shape[:2]
            or not np.isfinite(region).all()
            or np.any((region < 0) | (region > 1))
        ):
            raise ValueError("region_mask must be finite HxW in [0,1]")
    if strength == 0 or not np.any(region):
        zero = np.zeros(plan.shape, np.float32) if return_masks else None
        return IlluminationResult(
            image.copy(),
            {
                "implementation": IMPLEMENTATION,
                "strength": float(strength),
                "identity": True,
                "plan": plan.to_dict(),
            },
            zero,
            zero.copy() if zero is not None else None,
        )
    shadow, sun = _masks(plan, region)
    rgb = image.astype(np.float32) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # Approximate diffuse ambient + attenuated direct light; blue ambient is retained.
    ambient = plan.config.ambient_floor * np.array([0.88, 0.98, 1.12], np.float32)
    gain = 1 - (1 - ambient) * (strength * plan.config.shadow_depth * shadow[..., None])
    shaded = linear * gain
    direct = (
        linear
        * ((2 ** (plan.config.sun_ev * strength) - 1) * sun[..., None])
        * np.array([1.08, 1.0, 0.90], np.float32)
    )
    # A local highlight shoulder approaches 1 smoothly instead of hard white clipping.
    headroom = np.maximum(1 - shaded, 0)
    illuminated = shaded + headroom * (-np.expm1(-direct / np.maximum(headroom, 1e-7)))
    encoded = np.where(
        illuminated <= 0.0031308, illuminated * 12.92, 1.055 * illuminated ** (1 / 2.4) - 0.055
    )
    result = np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
    inactive = (shadow == 0) & (sun == 0)
    result[inactive] = image[inactive]
    metadata = {
        "implementation": IMPLEMENTATION,
        "strength": float(strength),
        "plan": plan.to_dict(),
        "identity": bool(np.array_equal(image, result)),
        "eligible_fraction": float((region > 0).mean()),
        "shadow_fraction": float((shadow > 0.25).mean()),
        "sunpatch_fraction": float((sun > 0.25).mean()),
        "mean_absolute_rgb_change": float(np.abs(result.astype(float) - image).mean() / 255),
        "black_pixel_fraction": float((result.max(2) <= 2).mean()),
        "white_pixel_fraction": float((result.min(2) >= 253).mean()),
        "geometry_changed": False,
        "annotation_dependent": False,
    }
    return IlluminationResult(
        result, metadata, shadow if return_masks else None, sun if return_masks else None
    )


# 基础光照transform保留；统一的七场景训练接口见pipeline.py。
class LocalIlluminationTransform:
    """PIL/NumPy (image,target)->(image,target), suitable before ToTensor/Normalize.

    Default calls use Python's worker-seeded random stream. For scheduling-independent
    replay, pass rng=sample_rng(base_seed, epoch, dataset_and_image_id, draw).
    No private RNG is copied into DataLoader workers; no target fields are consulted.
    """

    def __init__(self, config=None):
        self.config = config or IlluminationConfig()

    def with_metadata(self, image, target=None, *, rng=None, region_mask=None):
        is_pil = isinstance(image, Image.Image)
        if is_pil and image.mode != "RGB":
            raise ValueError("convert the image to RGB before this transform")
        source = np.asarray(image) if is_pil else image
        validate_rgb(source)
        if self.config.probability == 0 or self.config.strength_range[1] == 0:
            return image.copy(), target, {"implementation": IMPLEMENTATION, "applied": False}
        rng = rng if rng is not None else np.random.default_rng(random.getrandbits(63))
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        if rng.random() >= self.config.probability:
            return image.copy(), target, {"implementation": IMPLEMENTATION, "applied": False}
        plan = sample_plan(source.shape, rng, self.config)
        strength = float(rng.uniform(*self.config.strength_range))
        result = render_illumination(source, plan, strength, region_mask=region_mask)
        augmented = Image.fromarray(result.image) if is_pil else result.image
        return augmented, target, {"applied": True, **result.metadata}

    def __call__(self, image, target=None, *, rng=None, region_mask=None):
        augmented, target, _ = self.with_metadata(image, target, rng=rng, region_mask=region_mask)
        return augmented, target
