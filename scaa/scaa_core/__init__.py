"""场景协调天气与局部光照增强，公开调用接口。"""

from ._layers import AppearanceLayers, LayerSettings, apply_layers, make_layers
from .pipeline import (
    METHOD_VERSION,
    SCENE_NAMES,
    SCENES,
    TRAINING_SCENES,
    VARIANTS,
    AugmentationConfig,
    AugmentationPlan,
    AugmentationResult,
    RoadSceneAugmenter,
    RoadSceneTransform,
    ScenePolicy,
    region_for_geometry,
    sample_rng,
)

__all__ = [
    "AppearanceLayers",
    "LayerSettings",
    "apply_layers",
    "make_layers",
    "METHOD_VERSION",
    "TRAINING_SCENES",
    "SCENE_NAMES",
    "SCENES",
    "VARIANTS",
    "AugmentationConfig",
    "AugmentationPlan",
    "AugmentationResult",
    "RoadSceneAugmenter",
    "RoadSceneTransform",
    "ScenePolicy",
    "region_for_geometry",
    "sample_rng",
]
