"""SCAA图层引擎：先在小画布生成场，再以线性RGB乘加叠加到原图。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from ._illumination import _masks, validate_rgb
from ._weather import SCENES, _precipitation

if TYPE_CHECKING:
    from .pipeline import AugmentationPlan

IMPLEMENTATION = "scaa_layers_v002"
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@lru_cache(maxsize=1)
def transfer_tables():
    """256项精确sRGB解码、65536项逆曲线查表；避免每次对整图求幂。"""
    rgb = np.arange(256, dtype=np.float32) / 255
    decode = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear = np.linspace(0, 1, 65536, dtype=np.float32)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    encode = np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
    decode.flags.writeable = encode.flags.writeable = False
    return decode, encode


@lru_cache(maxsize=8)
def coordinates(height, width):
    """只缓存小画布的确定性坐标，不缓存随机图层或最终增强图。"""
    yy, xx = np.meshgrid(
        np.linspace(0, 1, height, dtype=np.float32),
        np.linspace(0, 1, width, dtype=np.float32),
        indexing="ij",
    )
    xx.flags.writeable = yy.flags.writeable = False
    return xx, yy


@dataclass(frozen=True)
class LayerSettings:
    layer_size: int = 384  # 最长边；仅图层画布，绝不用于缩小传入原图。
    particle_reference_size: int = 736  # 粒子数按此参考视场归一化，避免随4K像素数暴涨。

    def __post_init__(self):
        for name, low, high in (("layer_size", 64, 1024), ("particle_reference_size", 256, 1536)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name}必须为[{low},{high}]内整数")


@dataclass(frozen=True)
class AppearanceLayers:
    shape: tuple[int, int]
    gain: np.ndarray  # 小画布HxWx3，线性RGB乘性照明／透射／粒子遮挡。
    bias: np.ndarray  # 同画布，空气光／降水颜色的加性贡献。
    color_matrix: np.ndarray  # 全局3×3色彩矩阵，可做低饱和／冷暖倾向。
    metadata: dict
    masks: tuple = ()  # 仅显式请求时保存小画布阴影／光斑／粒子掩膜。


def _desaturation(amount):
    return (1 - amount) * np.eye(3, dtype=np.float32) + amount * np.tile(LUMA, (3, 1))


def make_layers(
    plan: AugmentationPlan,
    scene,
    strength,
    settings=None,
    *,
    variant="combined",
    region_mask=None,
    return_masks=False,
):
    """只接受形状、参数和随机方案；不接受原图，不分析亮度／高光／目标框。"""
    # 局部导入避免pipeline与图层引擎循环初始化；不依赖research或模型代码。
    from .pipeline import VARIANTS, AugmentationPlan, region_for_geometry

    if not isinstance(plan, AugmentationPlan):
        raise TypeError("plan必须为AugmentationPlan")
    settings = settings or plan.config.layers
    if variant not in VARIANTS:
        raise ValueError("未知消融variant")
    if not isinstance(plan, AugmentationPlan) or not isinstance(settings, LayerSettings):
        raise TypeError("使用AugmentationPlan和LayerSettings")
    if scene not in SCENES:
        raise ValueError("未知场景")
    if isinstance(strength, bool) or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength必须为[0,1]内有限数值")
    height, width = plan.shape
    scale = min(1.0, settings.layer_size / max(plan.shape))
    small = (max(9, round(height * scale)), max(9, round(width * scale)))
    gain = np.ones((*small, 3), np.float32)
    bias = np.zeros_like(gain)
    matrix = np.eye(3, dtype=np.float32)
    masks = [np.zeros(small, np.float32) for _ in range(3)] if return_masks else []
    custom_region = None
    if region_mask is not None:
        custom_region = np.asarray(region_mask, np.float32)
        if (
            custom_region.shape != plan.shape
            or not np.isfinite(custom_region).all()
            or np.any(custom_region < 0)
            or np.any(custom_region > 1)
        ):
            raise ValueError("region_mask必须为原图HxW、[0,1]有限数值")
        custom_region = cv2.resize(
            custom_region, (small[1], small[0]), interpolation=cv2.INTER_AREA
        )
    metadata = {
        "implementation": IMPLEMENTATION,
        "scene": scene,
        "variant": variant,
        "strength": float(strength),
        "shape": list(plan.shape),
        "layer_shape": list(small),
        "particle_reference_size": settings.particle_reference_size,
        "source_pixels_used_for_layer_generation": False,
        "geometry_changed": False,
        "annotations_used": False,
        "identity": strength == 0,
        "approximations_relative_to_reference": [
            "source_highlight_bloom",
            "source_dependent_illumination_flattening",
            "per_pixel_shot_and_read_noise",
            "nonlinear_local_highlight_shoulder",
        ],
        "particle_resolution_policy": "fixed_reference_field_of_view_not_native_megapixels",
    }
    if strength:
        policy = plan.config.policies[scene]
        region = region_for_geometry(small, plan.geometry, plan.config.front_region_top)
        if variant != "weather_only" and policy.mode != "off" and policy.strength_scale:
            light = replace(
                plan.illumination,
                shape=small,
                mode=policy.mode,
                penumbra=min(0.05, plan.illumination.penumbra * policy.softness_scale),
            )
            shadow, sun = _masks(light, region if custom_region is None else custom_region)
            if return_masks:
                masks[:2] = [shadow, sun]
            s = strength * policy.strength_scale
            ambient = light.config.ambient_floor * np.array([0.88, 0.98, 1.12], np.float32)
            gain -= (1 - ambient) * (s * light.config.shadow_depth * shadow[..., None])
            gain += (
                (2 ** (light.config.sun_ev * s) - 1)
                * sun[..., None]
                * np.array([1.08, 1.0, 0.90], np.float32)
            )
        if variant != "illumination_only":
            xx, yy = coordinates(*small)
            distance = (
                0.12 + 0.88 * (1 - yy) ** 1.4
                if plan.geometry == "vertical_proxy"
                else np.full(small, 0.55, np.float32)
            )
            # 和参考版一样给每种天气独立随机流；分辨率政策改变后不承诺相同粒子。
            root_rng = np.random.default_rng(plan.weather_seed)
            seeds = {name: int(root_rng.integers(0, 2**63 - 1)) for name in plan.config.weather}
            headlight = None
            if scene == "sunny":
                recipe = plan.config.weather["sunny"]
                gain *= 2 ** (recipe["exposure_ev"] * strength)
                matrix = np.diag(
                    [1 + 0.10 * strength, 1 + 0.025 * strength, 1 - 0.07 * strength]
                ).astype(np.float32)
            elif scene == "overcast":
                recipe = plan.config.weather["overcast"]
                amount = recipe["contrast_reduction"] * strength
                gain *= 1 - amount
                bias += np.array([0.24, 0.255, 0.28], np.float32) * amount
                matrix = np.diag([1 - 0.06 * strength, 1, 1 + 0.08 * strength]).astype(
                    np.float32
                ) @ _desaturation(recipe["saturation_reduction"] * strength)
            elif scene in {"night", "night_rain"}:
                recipe = plan.config.weather["night"]
                center = float(np.random.default_rng(seeds["night"]).uniform(0.42, 0.58))
                if plan.geometry == "vertical_proxy":
                    spread = 0.035 + 0.35 * yy**1.7
                    headlight = (
                        np.exp(-0.5 * ((xx - center) / spread) ** 2)
                        * np.clip((yy - 0.25) / 0.75, 0, 1) ** 0.7
                    )
                    upper = 1 - 0.6 * strength * (1 - yy)
                else:
                    headlight = np.exp(-((xx - center) ** 2 / 0.28**2 + (yy - 0.65) ** 2 / 0.48**2))
                    upper = np.ones(small, np.float32)
                exposure = 2 ** (recipe["exposure_ev"] * strength)
                night_gain = (exposure * upper)[..., None] * np.array(
                    [1 - 0.12 * strength, 1 - 0.03 * strength, 1 + 0.10 * strength], np.float32
                ) + (recipe["headlight_gain"] * (1 - exposure) * headlight)[..., None] * np.array(
                    [1.15, 1.01, 0.78], np.float32
                )
                gain *= night_gain
                if scene == "night_rain":
                    gain *= 0.96 + 0.10 * strength  # 均匀场近似，不读取原图高光生成光晕。
            elif scene == "snowfall":
                recipe = plan.config.weather["snowfall"]
                matrix = np.diag([1 - 0.04 * strength, 1, 1 + 0.05 * strength]).astype(
                    np.float32
                ) @ _desaturation(recipe["saturation_reduction"] * strength)

            def haze(beta, color):
                transmission = np.exp(-beta * strength * distance)[..., None]
                gain[:] *= transmission
                bias[:] = bias * transmission + np.array(color, np.float32) * (1 - transmission)

            if scene == "rain":
                recipe = plan.config.weather["rain"]
                gain *= 2 ** (recipe["exposure_ev"] * strength)
                haze(recipe["haze_beta"], [0.28, 0.31, 0.35])
            elif scene == "fog":
                recipe = plan.config.weather["fog"]
                haze(recipe["beta"], recipe["airlight_linear"])
            elif scene == "snowfall":
                haze(plan.config.weather["snowfall"]["haze_beta"], [0.55, 0.59, 0.64])
            if scene in {"rain", "night_rain", "snowfall"}:
                family = "snowfall" if scene == "snowfall" else "rain"
                recipe = dict(plan.config.weather[family])
                normalized = min(1.0, settings.particle_reference_size / max(plan.shape))
                reference_pixels = round(height * normalized) * round(width * normalized)
                recipe["particles_per_megapixel"] *= reference_pixels / (small[0] * small[1])
                alpha, particles = _precipitation(
                    (*small, 3),
                    strength,
                    np.random.default_rng(seeds[family]),
                    recipe,
                    snow=family == "snowfall",
                )
                if scene == "night_rain":
                    alpha *= 0.25 + 0.75 * np.sqrt(np.clip(headlight, 0, 1))
                color = {
                    "rain": [0.66, 0.69, 0.72],
                    "night_rain": [0.42, 0.45, 0.48],
                    "snowfall": [0.91, 0.94, 1.0],
                }[scene]
                gain *= 1 - alpha[..., None]
                bias[:] = (
                    bias * (1 - alpha[..., None]) + np.array(color, np.float32) * alpha[..., None]
                )
                metadata["particles"] = particles
                if return_masks:
                    masks[2] = alpha
    for array in [gain, bias, matrix, *masks]:
        if not np.isfinite(array).all():
            raise ValueError("图层含非有限值")
        array.flags.writeable = False
    return AppearanceLayers(tuple(plan.shape), gain, bias, matrix, metadata, tuple(masks))


def apply_layers(image, layers: AppearanceLayers):
    """一轮线性RGB色彩／乘加，再查表编码；保持原尺寸，不原地改写输入。"""
    is_pil = isinstance(image, Image.Image)
    if is_pil and image.mode != "RGB":
        raise ValueError("PIL输入须为RGB")
    source = np.asarray(image) if is_pil else image
    validate_rgb(source)
    if source.shape[:2] != layers.shape:
        raise ValueError("图片与图层尺寸不一致")
    if layers.metadata["identity"]:
        return image.copy()
    decode, encode = transfer_tables()
    linear = cv2.transform(decode[source], layers.color_matrix)
    height, width = layers.shape
    gain = cv2.resize(layers.gain, (width, height), interpolation=cv2.INTER_LINEAR)
    np.multiply(linear, gain, out=linear)
    del gain
    bias = cv2.resize(layers.bias, (width, height), interpolation=cv2.INTER_LINEAR)
    np.add(linear, bias, out=linear)
    del bias
    np.clip(linear, 0, 1, out=linear)
    np.multiply(linear, 65535, out=linear)
    np.rint(linear, out=linear)
    output = encode[linear.astype(np.uint16)]
    return Image.fromarray(output) if is_pil else output


def transform_with_layers(image, target, layers):
    """纯内存训练调用：标签原对象返回；本函数不负责模型接入或文件保存。"""
    return apply_layers(image, layers), target
