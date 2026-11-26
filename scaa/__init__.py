"""Scene-Coordinated Appearance Augmentation for RGB training images."""

from .training import (
    CONTEXT_KEY,
    VIEWS,
    SCAATransform,
    resolve_scaa_settings,
    scene_for_view,
    views_per_image,
)

__all__ = [
    "CONTEXT_KEY",
    "VIEWS",
    "SCAATransform",
    "resolve_scaa_settings",
    "scene_for_view",
    "views_per_image",
]
