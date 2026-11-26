"""Explicit, replayable road-scene recipes for image-only research previews.

Own NumPy/OpenCV implementation of elementary image formation approximations.
Not an implementation of the full ICCV rain renderer, relighting or a generator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from ._illumination import validate_rgb

IMPLEMENTATION = "road_scene_augmentation_v001"
REVISION = "r002_shadow_scope"
SCENES = ("sunny", "overcast", "night", "rain", "fog", "snowfall", "night_rain")
SCENE_NAMES = {
    "sunny": "晴天强光",
    "overcast": "阴天",
    "night": "低照夜景",
    "rain": "雨天",
    "fog": "雾天",
    "snowfall": "飘雪",
    "night_rain": "夜雨",
}
DEFAULT_RECIPES = {
    "sunny": {
        "exposure_ev": 0.9,
        "shadow_attenuation": 0.65,
        "glare": 0.055,
        "shadow_count": 3,
        "shadow_start_y": 0.58,
    },
    "overcast": {
        "contrast_reduction": 0.32,
        "saturation_reduction": 0.30,
        "illumination_flattening": 0.45,
    },
    "night": {
        "exposure_ev": -5.6,
        "headlight_gain": 0.48,
        "full_well_electrons": 900.0,
        "read_noise": 0.002,
        "bloom": 0.12,
    },
    "rain": {
        "particles_per_megapixel": 2100.0,
        "streak_length_at576": 20.0,
        "slant_degrees": 16.0,
        "opacity": 0.44,
        "haze_beta": 0.55,
        "exposure_ev": -0.75,
    },
    "fog": {"beta": 2.9, "airlight_linear": [0.64, 0.68, 0.72]},
    "snowfall": {
        "particles_per_megapixel": 850.0,
        "maximum_radius_at576": 3.1,
        "opacity": 0.85,
        "haze_beta": 0.45,
        "saturation_reduction": 0.24,
    },
}


@dataclass
class SceneResult:
    image: np.ndarray
    metadata: dict
    particle_alpha: np.ndarray


def resolve_recipes(value=None):
    recipes = copy.deepcopy(DEFAULT_RECIPES)
    if value is not None:
        if set(value) != set(recipes):
            raise ValueError(
                "declare exactly the six recipe families; night_rain composes night and rain"
            )
        for family, parameters in value.items():
            if set(parameters) != set(recipes[family]):
                raise ValueError(f"missing or unknown recipe parameter: {family}")
            recipes[family].update(parameters)
    bounds = {
        "exposure_ev": (-8, 2),
        "shadow_attenuation": (0, 0.9),
        "glare": (0, 0.2),
        "contrast_reduction": (0, 0.7),
        "saturation_reduction": (0, 0.8),
        "illumination_flattening": (0, 0.8),
        "headlight_gain": (0, 1),
        "full_well_electrons": (100, 10000),
        "read_noise": (0, 0.02),
        "bloom": (0, 0.5),
        "particles_per_megapixel": (0, 6000),
        "streak_length_at576": (1, 60),
        "slant_degrees": (0, 35),
        "opacity": (0, 1),
        "haze_beta": (0, 1.5),
        "beta": (0, 5),
        "maximum_radius_at576": (0.5, 6),
        "shadow_count": (0, 5),
        "shadow_start_y": (0.4, 0.8),
    }
    for family, parameters in recipes.items():
        for name, number in parameters.items():
            if name == "airlight_linear":
                color = np.asarray(number)
                if (
                    color.shape != (3,)
                    or not np.isfinite(color).all()
                    or np.any((color < 0) | (color > 1))
                ):
                    raise ValueError(
                        "fog airlight must have three finite linear RGB values in [0, 1]"
                    )
                continue
            low, high = bounds[name]
            if isinstance(number, bool) or not np.isfinite(number) or not low <= number <= high:
                raise ValueError(f"invalid {family}.{name}")
            if name == "shadow_count" and not isinstance(number, int):
                raise ValueError("shadow_count must be an integer")
    if recipes["night"]["exposure_ev"] >= 0 or recipes["sunny"]["exposure_ev"] < 0:
        raise ValueError("night exposure must be negative and sunny exposure nonnegative")
    return recipes


def srgb_to_linear(image):
    # sRGB解码为线性光强，照明／散射计算在此空间进行。
    rgb = image.astype(np.float32) / 255
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image):
    value = np.clip(image, 0, 1)
    encoded = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def grids(shape):
    height, width = shape[:2]
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    return xx.astype(np.float32), yy.astype(np.float32)


def distance_proxy(shape, mode):
    if mode not in {"vertical_proxy", "uniform"}:
        raise ValueError("visibility geometry must be vertical_proxy or uniform")
    if mode == "uniform":
        return np.full(shape[:2], 0.55, np.float32)
    _, yy = grids(shape)
    return 0.12 + 0.88 * (1 - yy) ** 1.4


def scatter(image, beta, geometry, color):
    transmission = np.exp(-beta * distance_proxy(image.shape, geometry))
    result = image * transmission[..., None] + np.asarray(color, np.float32) * (
        1 - transmission[..., None]
    )
    return result.astype(np.float32), transmission


def _desaturate(image, amount):
    gray = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return image * (1 - amount) + gray[..., None] * amount


def _sunny(image, severity, rng, parameters, geometry):
    height, width = image.shape[:2]
    xx, yy = grids(image.shape)
    shadows = np.zeros((height, width), np.uint8)
    angle = float(rng.uniform(-0.35, 0.35))
    polygons = []
    for _ in range(parameters["shadow_count"]):
        center = float(rng.uniform(-0.1, 1.1))
        band = float(rng.uniform(0.035, 0.09))
        # Restrict front-view cast shadows to a conservative lower-frame proxy.
        # Uniform mode is for the explicitly ground-facing example, not a sky mask.
        top = (
            float(rng.uniform(parameters["shadow_start_y"], parameters["shadow_start_y"] + 0.15))
            if geometry == "vertical_proxy"
            else float(rng.uniform(-0.1, 0.25))
        )
        points = np.array(
            [
                [(center - band) * width, top * height],
                [(center + band) * width, top * height],
                [(center + angle + 2 * band) * width, height],
                [(center + angle - 2 * band) * width, height],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(shadows, points, 255)
        polygons.append(points.tolist())
    soft = cv2.GaussianBlur(shadows.astype(np.float32) / 255, (0, 0), max(1, width * 0.008))
    irradiance = 2 ** (parameters["exposure_ev"] * severity) * (
        1 - parameters["shadow_attenuation"] * severity * soft
    )
    result = (
        image
        * irradiance[..., None]
        * np.array([1 + 0.10 * severity, 1 + 0.025 * severity, 1 - 0.07 * severity])
    )
    light_x, light_y = float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.02, 0.25))
    glare = np.exp(-((xx - light_x) ** 2 + (yy - light_y) ** 2) / (0.19**2))
    result += (parameters["glare"] * severity * glare)[..., None] * np.array([1.0, 0.93, 0.76])
    # Soft highlight roll-off avoids turning all sunlit pavement into flat white.
    result = result / (1 + 0.16 * severity * result)
    return result, {
        "shadow_polygons_px": polygons,
        "light_center": [light_x, light_y],
        "shadow_fraction": float((soft > 0.5).mean()),
        "shadow_fraction_upper_quarter": float((soft[: max(1, height // 4)] > 0.01).mean()),
        "shadow_scope": "lower_frame_proxy"
        if geometry == "vertical_proxy"
        else "ground_facing_full_frame",
    }


def _overcast(image, severity, parameters):
    y = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    log_y = np.log(y + 0.004)
    illumination = cv2.GaussianBlur(log_y, (0, 0), max(2, image.shape[1] * 0.06))
    flatten = np.exp(
        parameters["illumination_flattening"] * severity * (np.median(illumination) - illumination)
    )
    result = image * flatten[..., None]
    mid = np.array([0.24, 0.255, 0.28], np.float32)
    result = (
        result * (1 - parameters["contrast_reduction"] * severity)
        + mid * parameters["contrast_reduction"] * severity
    )
    result = _desaturate(result, parameters["saturation_reduction"] * severity)
    return result * np.array([1 - 0.06 * severity, 1, 1 + 0.08 * severity]), {
        "shadow_removal": "approximate coarse illumination flattening, not inverse rendering"
    }


def _night(image, severity, rng, parameters, geometry):
    # 负曝光、近场暖光及传感器噪声；不生成新的三维灯具或天空。
    xx, yy = grids(image.shape)
    exposure = 2 ** (parameters["exposure_ev"] * severity)
    center = float(rng.uniform(0.42, 0.58))
    if geometry == "vertical_proxy":
        spread = 0.035 + 0.35 * yy**1.7
        headlights = (
            np.exp(-0.5 * ((xx - center) / spread) ** 2) * np.clip((yy - 0.25) / 0.75, 0, 1) ** 0.7
        )
        upper_attenuation = 1 - 0.6 * severity * (1 - yy)
    else:
        headlights = np.exp(-((xx - center) ** 2 / 0.28**2 + (yy - 0.65) ** 2 / 0.48**2))
        upper_attenuation = np.ones_like(xx)
    ambient = (
        image
        * (exposure * upper_attenuation)[..., None]
        * np.array([1 - 0.12 * severity, 1 - 0.03 * severity, 1 + 0.10 * severity])
    )
    result = ambient + image * (parameters["headlight_gain"] * (1 - exposure) * headlights)[
        ..., None
    ] * np.array([1.15, 1.01, 0.78])
    highlights = np.maximum(result - 0.30, 0)
    glow = cv2.GaussianBlur(highlights, (0, 0), max(1, image.shape[1] * 0.015))
    result += parameters["bloom"] * severity * glow
    full_well = parameters["full_well_electrons"]
    # Blend a shot/read-noise realization; no random global NumPy state is used.
    shot = rng.poisson(np.maximum(result, 0) * full_well) / full_well
    result = (
        result
        + severity * (shot - result)
        + rng.normal(0, parameters["read_noise"] * severity, result.shape)
    )
    return result.astype(np.float32), {
        "exposure_multiplier": exposure,
        "headlight_center_x": center,
        "illumination_model": "multiplicative camera-centred light pool; no added lamp objects",
        "shot_full_well": full_well,
    }


def _precipitation(shape, severity, rng, parameters, snow=False):
    # 三层程序化雨丝／雪粒；相同随机初态可重放，未模拟积水或积雪。
    height, width = shape[:2]
    scale = min(height, width) / 576
    full_count = int(round(height * width / 1e6 * parameters["particles_per_megapixel"]))
    # Draw all latent positions first, keeping weather positions paired across severity.
    positions = rng.uniform(0, 1, (full_count, 2))
    layers = rng.integers(0, 3, full_count)
    values = rng.uniform(0.65, 1, full_count)
    visible = rng.uniform(0, 1, full_count) <= severity
    slant = float(
        rng.uniform(-parameters.get("slant_degrees", 12), parameters.get("slant_degrees", 12))
    )
    tangent = np.tan(np.deg2rad(slant))
    alpha = np.zeros((height, width), np.float32)
    layer_counts = []
    for layer in range(3):
        mask = np.zeros((height, width), np.uint8)
        indices = np.flatnonzero(visible & (layers == layer))
        layer_counts.append(int(len(indices)))
        for index in indices:
            x, y = positions[index] * [width - 1, height - 1]
            if snow:
                radius = max(
                    1,
                    int(
                        round(
                            scale
                            * parameters["maximum_radius_at576"]
                            * (0.22 + 0.32 * layer)
                            * values[index]
                        )
                    ),
                )
                cv2.circle(
                    mask, (int(x), int(y)), radius, int(255 * values[index]), -1, cv2.LINE_AA
                )
            else:
                length = (
                    parameters["streak_length_at576"] * scale * (0.45 + 0.5 * layer) * values[index]
                )
                endpoint = (int(round(x + length * tangent)), int(round(y + length)))
                cv2.line(
                    mask,
                    (int(x), int(y)),
                    endpoint,
                    int(255 * values[index]),
                    max(1, int(round(scale * (0.65 + 0.3 * layer)))),
                    cv2.LINE_AA,
                )
        sigma = max(0.25, scale * ((0.3 + 0.35 * layer) if snow else (0.3 + 0.15 * layer)))
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)
        alpha = np.maximum(alpha, soft)
    alpha *= parameters["opacity"] * (0.65 + 0.35 * severity)
    return alpha, {
        "particle_count": int(visible.sum()),
        "depth_layer_counts": layer_counts,
        "slant_degrees": slant,
        "occluding_alpha_gt_025": float((alpha > 0.25).mean()),
    }


# 七类基础天气引擎；合并策略在pipeline.py中控制。
# 晴天的旧投影和glare由合并配置固定关闭，避免重复光照叠加。
class SceneAugmenter:
    """Named coherent recipes; raw outputs are never silently reverted to originals."""

    def __init__(self, recipes=None):
        self.recipes = resolve_recipes(recipes)

    def __call__(self, image, scene, severity, rng, *, geometry="vertical_proxy"):
        validate_rgb(image)
        if scene not in SCENES:
            raise ValueError("unknown scenario")
        if isinstance(severity, bool) or not np.isfinite(severity) or not 0 <= severity <= 1:
            raise ValueError("severity must be a finite number in [0, 1]")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("provide an explicit NumPy Generator")
        distance_proxy(image.shape, geometry)  # Validate the explicit geometry approximation.
        metadata = {
            "implementation": IMPLEMENTATION,
            "revision": REVISION,
            "scene": scene,
            "severity": float(severity),
            "visibility_geometry": geometry,
            "real_depth_used": False,
            "rng_state_before": copy.deepcopy(rng.bit_generator.state),
            "components": {},
        }
        alpha = np.zeros(image.shape[:2], np.float32)
        if severity == 0:
            metadata["identity"] = True
            return SceneResult(image.copy(), metadata, alpha)
        component_seeds = {name: int(rng.integers(0, 2**63 - 1)) for name in DEFAULT_RECIPES}
        streams = {name: np.random.default_rng(seed) for name, seed in component_seeds.items()}
        metadata["component_seeds"] = component_seeds
        source = srgb_to_linear(image)
        if scene == "sunny":
            result, info = _sunny(
                source, severity, streams["sunny"], self.recipes["sunny"], geometry
            )
            metadata["components"]["sunny"] = info
        elif scene == "overcast":
            result, info = _overcast(source, severity, self.recipes["overcast"])
            metadata["components"]["overcast"] = info
        elif scene in {"night", "night_rain"}:
            result, info = _night(
                source, severity, streams["night"], self.recipes["night"], geometry
            )
            metadata["components"]["night"] = info
        else:
            result = source.copy()
        if scene in {"rain", "night_rain"}:
            recipe = self.recipes["rain"]
            if scene == "rain":
                result *= 2 ** (recipe["exposure_ev"] * severity)
                result, transmission = scatter(
                    result, recipe["haze_beta"] * severity, geometry, [0.28, 0.31, 0.35]
                )
                metadata["components"]["rain_haze"] = {
                    "transmission_min": float(transmission.min())
                }
            else:
                # Night rain scatters light already present, instead of adding white daytime fog.
                veiling = cv2.GaussianBlur(
                    np.maximum(result, 0), (0, 0), max(1, image.shape[1] * 0.035)
                )
                result = 0.96 * result + 0.10 * severity * veiling
            alpha, info = _precipitation(image.shape, severity, streams["rain"], recipe)
            if scene == "night_rain":
                illumination = np.sqrt(np.clip(result.mean(axis=2), 0, 1))
                alpha *= 0.25 + 0.75 * illumination
                rain_color = [0.42, 0.45, 0.48]
            else:
                rain_color = [0.66, 0.69, 0.72]
            result = result * (1 - alpha[..., None]) + np.asarray(rain_color) * (alpha[..., None])
            metadata["components"]["rain"] = info
        elif scene == "fog":
            recipe = self.recipes["fog"]
            result, transmission = scatter(
                result, recipe["beta"] * severity, geometry, recipe["airlight_linear"]
            )
            metadata["components"]["fog"] = {
                "transmission_min": float(transmission.min()),
                "transmission_max": float(transmission.max()),
            }
        elif scene == "snowfall":
            recipe = self.recipes["snowfall"]
            result = _desaturate(source, recipe["saturation_reduction"] * severity) * np.array(
                [1 - 0.04 * severity, 1, 1 + 0.05 * severity]
            )
            result, _ = scatter(
                result, recipe["haze_beta"] * severity, geometry, [0.55, 0.59, 0.64]
            )
            alpha, info = _precipitation(
                image.shape, severity, streams["snowfall"], recipe, snow=True
            )
            result = (
                result * (1 - alpha[..., None]) + np.array([0.91, 0.94, 1.0]) * alpha[..., None]
            )
            metadata["components"]["snowfall"] = {**info, "ground_accumulation": False}
        if not np.isfinite(result).all():
            raise ValueError("nonfinite scene rendering")
        output = linear_to_srgb(result)
        metadata["identity"] = bool(np.array_equal(output, image))
        metadata["particle_alpha_gt_025"] = float((alpha > 0.25).mean())
        metadata["rng_state_after"] = copy.deepcopy(rng.bit_generator.state)
        return SceneResult(output, metadata, alpha)
