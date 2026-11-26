"""场景条件化的天气／局部光照增强；不包含检测器或数据集路径。"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from PIL import Image

from ._illumination import (
    IlluminationConfig,
    IlluminationPlan,
    render_illumination,
    validate_rgb,
)
from ._illumination import (
    sample_plan as sample_light_plan,
)
from ._illumination import (
    sample_rng as light_rng,
)
from ._layers import LayerSettings, apply_layers, make_layers
from ._weather import DEFAULT_RECIPES, SCENE_NAMES, SCENES, SceneAugmenter, resolve_recipes

REFERENCE_VERSION = "scene_illumination_v001"
METHOD_VERSION = "scene_illumination_v002_layers"
TRAINING_SCENES = SCENES[:6]  # 夜雨仅保留显式兼容接口，不进入默认六场景。
VARIANTS = ("weather_only", "illumination_only", "combined")


def default_weather():
    """关闭旧晴天投影／泛光，由新的局部光照引擎统一负责。"""
    recipes = copy.deepcopy(DEFAULT_RECIPES)
    recipes["sunny"].update(shadow_count=0, glare=0.0)
    return recipes


@dataclass(frozen=True)
class ScenePolicy:
    mode: str
    strength_scale: float
    softness_scale: float

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "sunpatch", "mixed"}:
            raise ValueError("局部光照模式应为off/shadow/sunpatch/mixed")
        for name, lo, hi in (("strength_scale", 0, 1), ("softness_scale", 1, 3.5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"无效场景策略参数: {name}")


def default_policies():
    # 晴天含直射光，允许阴影与亮斑；其余场景只保留弱局部遮光。
    # 这些比例是项目探索值，不是论文最优参数或天气物理标定。
    return {
        "sunny": ScenePolicy("mixed", 1.0, 1.0),
        "overcast": ScenePolicy("shadow", 0.15, 2.4),
        "night": ScenePolicy("shadow", 0.25, 1.8),
        "rain": ScenePolicy("shadow", 0.25, 2.0),
        "fog": ScenePolicy("shadow", 0.15, 2.4),
        "snowfall": ScenePolicy("shadow", 0.18, 2.3),
        "night_rain": ScenePolicy("shadow", 0.20, 2.0),
    }


@dataclass(frozen=True)
class AugmentationConfig:
    probability: float = 0.5
    strength_range: tuple = (0.25, 0.65)
    scene_weights: tuple = (1, 1, 1, 1, 1, 1, 0)
    renderer: str = "layers"
    layers: LayerSettings = field(default_factory=LayerSettings)
    geometry: str = "vertical_proxy"
    front_region_top: float = 0.50
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    policies: dict = field(default_factory=default_policies)
    weather: dict = field(default_factory=default_weather)

    def __post_init__(self):
        object.__setattr__(self, "strength_range", tuple(self.strength_range))
        object.__setattr__(self, "scene_weights", tuple(self.scene_weights))
        object.__setattr__(self, "policies", copy.deepcopy(self.policies))
        object.__setattr__(self, "weather", resolve_recipes(self.weather))
        for name, lo, hi in (("probability", 0, 1), ("front_region_top", 0.25, 0.8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"无效配置: {name}")
        if self.renderer not in {"layers", "reference"}:
            raise ValueError("renderer必须为layers或reference")
        if not isinstance(self.layers, LayerSettings):
            raise TypeError("layers应为LayerSettings；字典配置使用from_dict")
        if self.geometry not in {"vertical_proxy", "uniform"}:
            raise ValueError("geometry必须为vertical_proxy或uniform")
        lohi = self.strength_range
        if (
            len(lohi) != 2
            or any(isinstance(v, bool) or not np.isfinite(v) for v in lohi)
            or not 0 <= lohi[0] <= lohi[1] <= 1
        ):
            raise ValueError("strength_range必须是[0,1]内递增区间")
        weights = self.scene_weights
        if (
            len(weights) != len(SCENES)
            or any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in weights)
            or not np.isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ValueError("scene_weights必须是七个非负有限权重，且总和大于0")
        if set(self.policies) != set(SCENES):
            raise ValueError("必须为七类场景逐一配置局部光照策略")
        for scene, policy in self.policies.items():
            if not isinstance(policy, ScenePolicy):
                raise TypeError("policies应包含ScenePolicy，字典配置请使用from_dict")
            if scene != "sunny" and policy.mode in {"sunpatch", "mixed"}:
                raise ValueError("当前协调策略只允许晴天新增太阳光斑")
        if self.weather["sunny"]["shadow_count"] != 0 or self.weather["sunny"]["glare"] != 0:
            raise ValueError("旧sunny.shadow_count/glare必须为0，避免两套投影重复叠加")

    @classmethod
    def from_dict(cls, value):
        """支持YAML／JSON字典；未知字段直接报错，避免拼写错误被忽略。"""
        value = copy.deepcopy(value)
        if "layers" in value:
            value["layers"] = LayerSettings(**value["layers"])
        if "illumination" in value:
            value["illumination"] = IlluminationConfig(**value["illumination"])
        if "policies" in value:
            value["policies"] = {
                name: ScenePolicy(**row) for name, row in value["policies"].items()
            }
        return cls(**value)


@dataclass(frozen=True)
class AugmentationPlan:
    """一次采样的完整方案；同图各场景、各强度复用，便于配对消融。"""

    shape: tuple
    geometry: str
    illumination: IlluminationPlan
    weather_seed: int
    config: AugmentationConfig
    method_version: str = METHOD_VERSION

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if (
            self.method_version not in {METHOD_VERSION, REFERENCE_VERSION}
            or self.shape != self.illumination.shape
        ):
            raise ValueError("方法版本或方案尺寸不一致")
        if self.geometry not in {"vertical_proxy", "uniform"}:
            raise ValueError("无效几何代理")
        if type(self.weather_seed) is not int or not 0 <= self.weather_seed < 2**63:
            raise ValueError("无效天气随机种子")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value["illumination"] = IlluminationPlan.from_dict(value["illumination"])
        if value.get("method_version") == REFERENCE_VERSION:
            value["config"].setdefault("renderer", "reference")
        value["config"] = AugmentationConfig.from_dict(value["config"])
        return cls(**value)


@dataclass
class AugmentationResult:
    image: np.ndarray | Image.Image
    metadata: dict
    shadow_mask: np.ndarray | None = None
    sun_mask: np.ndarray | None = None
    particle_alpha: np.ndarray | None = None


def sample_rng(seed, epoch, sample_id, draw=0):
    """与rank／worker调度无关；同图重复视图由draw区分。"""
    return light_rng(seed, epoch, METHOD_VERSION + "/" + sample_id, draw)


def _source(image):
    is_pil = isinstance(image, Image.Image)
    if is_pil and image.mode != "RGB":
        raise ValueError("请先将PIL图片转换为RGB")
    array = np.asarray(image) if is_pil else image
    validate_rgb(array)
    if min(array.shape[:2]) < 9:
        raise ValueError("天气引擎要求图像宽高至少9像素")
    return array, is_pil


def region_for_geometry(shape, geometry, front_top=0.50):
    """前视下部位置代理，或俯拍全帧；不等于道路／天空分割。"""
    if geometry == "uniform":
        return np.ones(shape[:2], np.float32)
    if geometry != "vertical_proxy":
        raise ValueError("未知geometry")
    y = np.arange(shape[0], dtype=np.float32) / max(1, shape[0] - 1)
    t = np.clip((y - front_top) / 0.10, 0, 1)
    return np.broadcast_to((t * t * (3 - 2 * t))[:, None], shape[:2]).copy()


class RoadSceneAugmenter:
    """显式渲染接口；内存中运行，不创建文件、不加载检测器。"""

    def __init__(self, config=None):
        self.config = config or AugmentationConfig()
        if not isinstance(self.config, AugmentationConfig):
            raise TypeError("config请使用AugmentationConfig或AugmentationConfig.from_dict")

    def make_plan(self, shape, rng, *, geometry=None):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("请提供NumPy Generator")
        return AugmentationPlan(
            tuple(int(v) for v in shape[:2]),
            geometry or self.config.geometry,
            sample_light_plan(shape, rng, self.config.illumination, mode="mixed"),
            int(rng.integers(0, 2**63)),
            self.config,
        )

    def render(
        self,
        image,
        plan,
        scene,
        strength,
        *,
        variant="combined",
        region_mask=None,
        return_masks=False,
    ):
        """默认生成小图层再叠加；reference仅用于历史重放或受控对照。"""
        if not isinstance(plan, AugmentationPlan):
            raise TypeError("plan必须为AugmentationPlan")
        source, _ = _source(image)
        if source.shape[:2] != plan.shape:
            raise ValueError("原图尺寸与采样方案不同")
        if plan.config.renderer == "reference":
            return self.render_reference(
                image,
                plan,
                scene,
                strength,
                variant=variant,
                region_mask=region_mask,
                return_masks=return_masks,
            )
        fields = make_layers(
            plan,
            scene,
            strength,
            variant=variant,
            region_mask=region_mask,
            return_masks=return_masks,
        )
        output = apply_layers(image, fields)
        # 自定义区域的边界在小画布上会插值；纯局部消融需严格保留零区域。
        if region_mask is not None and variant == "illumination_only":
            values = np.asarray(output).copy()
            values[np.asarray(region_mask) == 0] = source[np.asarray(region_mask) == 0]
            output = Image.fromarray(values) if isinstance(image, Image.Image) else values
        masks = [None, None, None]
        if return_masks:
            import cv2

            masks = [cv2.resize(mask, (plan.shape[1], plan.shape[0])) for mask in fields.masks]
        metadata = {
            **fields.metadata,
            "method_version": METHOD_VERSION,
            "renderer": "layers",
            "scene_name": SCENE_NAMES[scene],
            "label_visibility_certified": False,
            "illumination": {"policy": asdict(plan.config.policies[scene])},
        }
        return AugmentationResult(output, metadata, *masks)

    def iter_scenes(
        self, image, target=None, *, scenes=None, strength=0.65, rng=None, geometry=None
    ):
        """逐张生成六个视图，不同时缓存六份全图；训练引擎自行安排batch／更新预算。"""
        source, _ = _source(image)
        scenes = TRAINING_SCENES if scenes is None else tuple(scenes)
        if not scenes or len(set(scenes)) != len(scenes) or not set(scenes) <= set(SCENES):
            raise ValueError("scenes必须是不重复的合法场景")
        rng = rng if rng is not None else np.random.default_rng(random.getrandbits(63))
        plan = self.make_plan(source.shape, rng, geometry=geometry)
        for scene in scenes:
            result = self.render(image, plan, scene, strength)
            yield scene, result.image, target

    def render_reference(
        self,
        image,
        plan,
        scene,
        strength,
        *,
        variant="combined",
        region_mask=None,
        return_masks=False,
    ):
        source, is_pil = _source(image)
        if scene not in SCENES or variant not in VARIANTS:
            raise ValueError("未知场景或消融variant")
        if isinstance(strength, bool) or not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength必须在[0,1]内")
        if source.shape[:2] != plan.shape:
            raise ValueError("原图尺寸与采样方案不同")
        policy = plan.config.policies[scene]
        metadata = {
            "method_version": REFERENCE_VERSION,
            "renderer": "reference",
            "scene": scene,
            "scene_name": SCENE_NAMES[scene],
            "strength": float(strength),
            "variant": variant,
            "plan": plan.to_dict(),
            "geometry_changed": False,
            "annotations_used": False,
            "label_visibility_certified": False,
            "composition_order": ["local_illumination", "weather"],
        }
        zeros = np.zeros(plan.shape, np.float32) if return_masks else None
        shadow, sun, particles = zeros, zeros, zeros
        # 零强度保证原图逐像素不变；render只使用计划，不消耗外部随机状态。
        if strength == 0:
            return AugmentationResult(
                image.copy(), {**metadata, "identity": True}, shadow, sun, particles
            )
        current = source
        if variant != "weather_only" and policy.mode != "off" and policy.strength_scale > 0:
            region = (
                region_mask
                if region_mask is not None
                else region_for_geometry(source.shape, plan.geometry, plan.config.front_region_top)
            )
            softness = min(0.05, plan.illumination.penumbra * policy.softness_scale)
            light_plan = replace(plan.illumination, mode=policy.mode, penumbra=softness)
            light = render_illumination(
                source,
                light_plan,
                strength * policy.strength_scale,
                region_mask=region,
                return_masks=return_masks,
            )
            current, shadow, sun = light.image, light.shadow_mask, light.sun_mask
            metadata["illumination"] = {
                **light.metadata,
                "policy": asdict(policy),
                "effective_penumbra": softness,
                "penumbra_capped": softness < plan.illumination.penumbra * policy.softness_scale,
            }
        else:
            metadata["illumination"] = {"enabled": False}
        # 天气最后渲染：雾的空气光和雨雪粒子不会被再次当作路面投影压暗。
        if variant != "illumination_only":
            weather = SceneAugmenter(plan.config.weather)(
                current,
                scene,
                strength,
                np.random.default_rng(plan.weather_seed),
                geometry=plan.geometry,
            )
            current = weather.image
            particles = weather.particle_alpha if return_masks else None
            metadata["weather"] = weather.metadata
        metadata["identity"] = bool(np.array_equal(source, current))
        metadata["mean_absolute_rgb_change"] = float(
            np.abs(current.astype(float) - source).mean() / 255
        )
        output = Image.fromarray(current) if is_pil else current.copy()
        return AugmentationResult(output, metadata, shadow, sun, particles)

    def augment(
        self,
        image,
        scene,
        strength,
        *,
        seed=42,
        geometry=None,
        variant="combined",
        region_mask=None,
        return_masks=False,
    ):
        source, _ = _source(image)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed必须是非负整数")
        plan = self.make_plan(source.shape, np.random.default_rng(seed), geometry=geometry)
        return self.render(
            image,
            plan,
            scene,
            strength,
            variant=variant,
            region_mask=region_mask,
            return_masks=return_masks,
        )


class RoadSceneTransform:
    """在线训练(image,target)接口；target原对象返回，不读取任何标注。"""

    def __init__(self, config=None):
        self.config = config or AugmentationConfig()
        self.operation = RoadSceneAugmenter(self.config)

    def with_metadata(self, image, target=None, *, rng=None, geometry=None, region_mask=None):
        source, _ = _source(image)
        if self.config.probability == 0 or self.config.strength_range[1] == 0:
            return image.copy(), target, {"method_version": METHOD_VERSION, "applied": False}
        # 无私有持久RNG，避免DataLoader复制对象后各worker重复相同增强。
        rng = rng if rng is not None else np.random.default_rng(random.getrandbits(63))
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng必须是NumPy Generator")
        if rng.random() >= self.config.probability:
            return image.copy(), target, {"method_version": METHOD_VERSION, "applied": False}
        weights = np.asarray(self.config.scene_weights, np.float64)
        scene = str(rng.choice(SCENES, p=weights / weights.sum()))
        plan = self.operation.make_plan(source.shape, rng, geometry=geometry)
        strength = float(rng.uniform(*self.config.strength_range))
        result = self.operation.render(image, plan, scene, strength, region_mask=region_mask)
        return result.image, target, {"applied": True, **result.metadata}

    def __call__(self, image, target=None, *, rng=None, geometry=None, region_mask=None):
        image, target, _ = self.with_metadata(
            image, target, rng=rng, geometry=geometry, region_mask=region_mask
        )
        return image, target
