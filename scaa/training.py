"""SCAA训练适配：几何后、ToImage前；不写图片、不启动CUDA或读取目标框。"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

SCENES = ("sunny", "overcast", "night", "rain", "fog", "snowfall")
MODULE_VERSION = "scene_illumination_v002_layers"
CONTEXT_KEY = "_scaa_context"
VIEW_POLICY = "original_plus_six_v1"
VIEWS = ("clean", *SCENES)
DEFAULTS = {
    "enabled": False,
    "module_version": MODULE_VERSION,
    "strength_range": [0.25, 0.65],
    "geometry": "uniform",
    "layer_size": 384,
    "particle_reference_size": 736,
}
# 仅供历史版本／检查点读取，当前新配置不再暴露采样模式、概率或场景子集。
LEGACY_DEFAULTS = {
    "enabled": False,
    "module_version": MODULE_VERSION,
    "sampling": "random",
    "scenes": list(SCENES),
    "probability": 1.0,
    "strength_range": [0.25, 0.65],
    "geometry": "uniform",
    "layer_size": 384,
    "particle_reference_size": 736,
}


def resolve_scaa_settings(value=None):
    """显式校验新训练政策；旧配置不增加字段或默认启用SCAA。"""
    if value is not None and not isinstance(value, dict):
        raise ValueError("augmentation.scaa contains unknown settings")
    legacy = bool(value and set(value) & {"sampling", "probability", "scenes"})
    defaults = LEGACY_DEFAULTS if legacy else DEFAULTS
    if value is not None and set(value) - defaults.keys():
        raise ValueError("augmentation.scaa contains unknown settings")
    settings = {**copy.deepcopy(defaults), **copy.deepcopy(value or {})}
    if type(settings["enabled"]) is not bool:
        raise ValueError("scaa.enabled must be boolean")
    if settings["module_version"] != MODULE_VERSION:
        raise ValueError(f"scaa.module_version must be {MODULE_VERSION}")
    if legacy and settings["sampling"] not in {"random", "six_views"}:
        raise ValueError("scaa.sampling must be random or six_views")
    scenes = settings.get("scenes", list(SCENES))
    if (
        not isinstance(scenes, list)
        or not scenes
        or any(not isinstance(name, str) or name not in SCENES for name in scenes)
        or len(set(scenes)) != len(scenes)
    ):
        raise ValueError("scaa.scenes must be a unique nonempty subset of the six scenes")
    probability = settings.get("probability", 1.0)
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("scaa.probability must be finite in [0,1]")
    if settings.get("sampling") == "six_views" and probability != 1:
        raise ValueError(
            "six_views requires probability=1; use enabled=false for its paired control"
        )
    strength = settings["strength_range"]
    if (
        not isinstance(strength, (list, tuple))
        or len(strength) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in strength
        )
        or not 0 <= strength[0] <= strength[1] <= 1
    ):
        raise ValueError("scaa.strength_range must be an increasing finite interval in [0,1]")
    settings["strength_range"] = list(strength)
    if settings["geometry"] not in {"uniform", "vertical_proxy"}:
        raise ValueError("scaa.geometry must be uniform or vertical_proxy")
    for key, low, high in (("layer_size", 64, 1024), ("particle_reference_size", 256, 1536)):
        if type(settings[key]) is not int or not low <= settings[key] <= high:
            raise ValueError(f"scaa.{key} must be an integer in [{low},{high}]")
    return settings


def core_config(settings):
    from .scaa_core import AugmentationConfig, LayerSettings

    return AugmentationConfig(
        probability=settings.get("probability", 1.0),
        strength_range=tuple(settings["strength_range"]),
        scene_weights=tuple(
            int(name in settings.get("scenes", SCENES)) for name in (*SCENES, "night_rain")
        ),
        geometry=settings["geometry"],
        renderer="layers",
        layers=LayerSettings(settings["layer_size"], settings["particle_reference_size"]),
    )


def views_per_image(settings):
    """新策略开关决定7／1份；历史六视图的配对控制保持原有预算。"""
    settings = resolve_scaa_settings(settings)
    if "sampling" not in settings:
        return len(VIEWS) if settings["enabled"] else 1
    return len(settings["scenes"]) if settings["sampling"] == "six_views" else 1


def view_policy(settings):
    """留档实际数据访问政策，不增加用户采样参数。"""
    return settings.get("sampling", VIEW_POLICY)


def scene_for_view(settings, view):
    """虚拟索引0保留无SCAA输入，1—6对应六场景，按原batch打乱训练。"""
    count = views_per_image(settings)
    if type(view) is not int or not 0 <= view < count:
        raise ValueError("SCAA virtual view index is outside the configured view count")
    if "sampling" not in settings:
        return VIEWS[view] if settings["enabled"] else "clean"
    return settings["scenes"][view] if settings["sampling"] == "six_views" else None


def scaa_record(value=None):
    settings = resolve_scaa_settings(value)
    if value is None:
        return {"enabled": False, "sampling": "random", "views_per_source_image": 1}
    provenance = json.loads(
        (Path(__file__).resolve().parents[1] / "provenance/scaa.json").read_text(encoding="utf-8")
    )
    return {
        **settings,
        "view_policy": view_policy(settings),
        "view_names": list(VIEWS) if "sampling" not in settings and settings["enabled"] else None,
        "views_per_source_image": views_per_image(settings),
        "module_files_sha256": provenance["files_sha256"],
        "effective_module_config": asdict(core_config(settings)),
        "placement": "native crop/resize/flip -> SCAA RGB PIL -> ToImage/ToDtype -> Normalize",
        "rng": "sample_rng(seed, epoch, original_image_id, virtual_view_index); fallback Python rank/worker RNG",
        "validation_and_test": "disabled; unique original images, fixed native preprocessing",
        "target_policy": "read only private sampling context; do not inspect or change boxes/classes",
    }


class SCAATransform:
    """CPU PIL输入；复用冻结图层算法，不另维护天气实现。"""

    def __init__(self, settings):
        from .scaa_core import RoadSceneAugmenter

        self.settings = resolve_scaa_settings(settings)
        self.operation = RoadSceneAugmenter(core_config(self.settings))

    def __call__(self, image, target=None):
        if not self.settings["enabled"]:
            return image, target  # 不消耗RNG、量化像素或修改标签。
        import numpy as np
        from PIL import Image

        from .scaa_core import sample_rng

        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("SCAA must run on CPU RGB PIL before ToImage/ToDtype")
        context = target.get(CONTEXT_KEY) if target is not None else None
        if "sampling" not in self.settings:
            if context is None:
                raise ValueError("seven views require the RoadDataset virtual-view context")
            selected = scene_for_view(self.settings, context["view"])
            if context.get("scene") != selected:
                raise ValueError("SCAA scene does not match the fixed seven-view index")
            if selected == "clean":
                # 保留原生几何后的像素；不生成图层，也不消耗SCAA随机数。
                returned = target.copy()
                returned.pop(CONTEXT_KEY, None)
                returned.update(scaa_scene="clean", scaa_strength=0.0)
                return image, returned
        if context is not None:
            rng = sample_rng(
                context["seed"], context["epoch"], context["sample_id"], context["view"]
            )
        else:
            if self.settings.get("sampling") == "six_views":
                raise ValueError("six_views requires the RoadDataset virtual-view sampling context")
            rng = np.random.default_rng(random.getrandbits(63))
        selected = context.get("scene") if context is not None else None
        if selected is not None and selected not in self.settings.get("scenes", SCENES):
            raise ValueError("SCAA sampling context scene differs from configuration")
        # 场景和强度只来自源训练随机流，不根据图像、标签或模型反馈筛选。
        # 历史分支保留原随机调用顺序；新分支的六个场景一律生成，无概率开关。
        apply = rng.random() < self.settings["probability"] if "sampling" in self.settings else True
        scene = selected or str(rng.choice(self.settings.get("scenes", SCENES)))
        strength = float(rng.uniform(*self.settings["strength_range"]))
        if apply:
            plan = self.operation.make_plan(np.asarray(image).shape, rng)
            image = self.operation.render(image, plan, scene, strength).image
        if target is not None:
            target = target.copy()
            target.pop(CONTEXT_KEY, None)
            target["scaa_scene"] = scene if apply else "clean"
            target["scaa_strength"] = strength if apply else 0.0
        return image, target
