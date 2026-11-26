# CASE-DETR Core Modules

本目录提供论文中的场景协同外观增强（Scene-Coordinated Appearance Augmentation, SCAA）与坐标对齐空间证据头（Coordinate-Aligned Spatial Evidence Head, CASE Head）。可从仓库根目录导入代码。完整模型、数据、权重及训练配方需在使用方准备。

| 模块 | 入口 | 依赖 |
|---|---|---|
| SCAA | [`scaa/training.py`](scaa/training.py) 和 [`scaa/scaa_core/`](scaa/scaa_core/) | NumPy、Pillow、OpenCV |
| CASE Head | [`case_head/evidence_head.py`](case_head/evidence_head.py) | PyTorch、`rfdetr==1.11.2` 及其运行依赖 |

## SCAA

SCAA 用于训练。在几何增强后、张量转换与归一化前传入 RGB PIL 图。每张源图每轮对应七个虚拟索引。索引 0 保留当前外观，索引 1–6 分别生成强光与阴影、阴天、弱光、雨、雾和雪。框和类别保持原值。训练数据集按 `divmod(index, views_per_image(settings))` 拆分图像下标与视图下标，在每轮开始时更新 `epoch`。

```python
from scaa import CONTEXT_KEY, SCAATransform, resolve_scaa_settings, scene_for_view

settings = resolve_scaa_settings({"enabled": True})
scaa = SCAATransform(settings)


def apply_training_view(image, target, *, sample_id, view, seed, epoch):
    context = {
        "sample_id": sample_id,
        "view": view,
        "seed": seed,
        "epoch": epoch,
        "scene": scene_for_view(settings, view),
    }
    return scaa(image, {**target, CONTEXT_KEY: context})
```

使用稳定的 `sample_id`（例如 `source_train/example.jpg`），可在同一环境下复现对应图像、轮次和视图的增强。验证和测试使用原始预处理。

## CASE Head

对已构建的 RF-DETR 底层 PyTorch 检测器调用安装函数。论文配置使用单尺度 P4、四层固定参考框解码器、16 个交叉注意力头和每头 2 个采样点。分类张量为 3 个通道，CASE 增量应用于末层前两个类别通道。当前实现使用 PyTorch eager 执行。

```python
import torch

from case_head import install_evidence_head

case_settings = {
    "enabled": True,
    "module_version": "spatial_evidence_v001",
    "history": "coordinates",
    "feedback": True,
    "hidden_dim": 16,
    "retention": 0.5,
}


def attach_case_head(model):
    # The reference detector has just been constructed on CPU.
    with torch.random.fork_rng(devices=[]):
        return install_evidence_head(model, case_settings)
```

安装后再构建优化器。加载完整 CASE 检查点时，先用对应结构安装 CASE，再加载状态字典。CASE Head 在后两层解码器传递坐标对齐的类别证据，并反馈至交叉注意力。框回归沿用原模型的 MLP，读取更新后的查询。

## 源码对应关系

源码取自 `v007_spatial_evidence / evidence_full` 正式训练快照。CASE Head 和 SCAA 渲染层保留原实现，SCAA 训练适配器改用发布目录内的相对导入。[SCAA 文件哈希与版本](provenance/scaa.json)记录渲染源码来源。RF-DETR 上游许可单独列于 [third_party/rfdetr/LICENSE](third_party/rfdetr/LICENSE)。
