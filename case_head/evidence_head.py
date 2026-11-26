"""Model-private class-conditional spatial evidence head.

Ported from the q043 operator, source SHA256 a8a5267b6eef19e063b7eab1851d6a000fb9d72f447b4a3e6c739dcd5b09149b.
No research imports. Native attention tensor names are preserved for S1 loading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _axis_support(coordinates, size):
    pixel = coordinates * size - 0.5  # grid_sample, align_corners=False
    lower = pixel.floor()
    fraction = pixel - lower
    low = (1 - fraction) * ((lower >= 0) & (lower < size))
    high = fraction * ((lower + 1 >= 0) & (lower + 1 < size))
    return lower, low, high


def _axis_overlap(current, previous, size):
    ci, cl, ch = _axis_support(current, size)
    pi, pl, ph = _axis_support(previous, size)
    ci, cl, ch = (v.unsqueeze(-1) for v in (ci, cl, ch))
    pi, pl, ph = (v.unsqueeze(-2) for v in (pi, pl, ph))
    return (
        cl * pl * (ci == pi)
        + cl * ph * (ci == pi + 1)
        + ch * pl * (ci + 1 == pi)
        + ch * ph * (ci + 1 == pi + 1)
    )


def support_mass(locations, shape):
    """Sum of valid bilinear pixel weights, including partial zero padding."""
    height, width = shape
    _, xl, xh = _axis_support(locations[..., 0], width)
    _, yl, yh = _axis_support(locations[..., 1], height)
    return (xl + xh) * (yl + yh)


def transport_state(previous, previous_locations, current_locations, shape):
    """Normalized sparse splat/read, with exact zero at uncovered locations.

    ``previous`` is B,Q,K_old,C; locations are B,Q,K,2 in image coordinates.
    The kernel is the inner product of the two bilinear pixel-support stencils.
    No slot identity, radius expansion, cross-query mixing or density-as-
    probability is used. Repeated old points are averaged, not added as votes.
    """
    if (
        previous.ndim != 4
        or previous_locations.shape != (*previous.shape[:-1], 2)
        or current_locations.ndim != 4
        or current_locations.shape[:2] != previous.shape[:2]
        or current_locations.shape[-1] != 2
        or len(shape) != 2
        or min(shape) <= 0
    ):
        raise ValueError("one B,Q,K,C state and matching image-coordinate arrays required")
    height, width = shape
    kernel = _axis_overlap(current_locations[..., 0], previous_locations[..., 0], width)
    kernel = kernel * _axis_overlap(current_locations[..., 1], previous_locations[..., 1], height)
    density = kernel.sum(-1)
    # Safe on both branches: where(0/0, 0) still produces NaN gradients.
    denominator = torch.where(density > 0, density, torch.ones_like(density))
    transported = (kernel @ previous) / denominator.unsqueeze(-1)
    return transported, density


def sampling_locations(module, query, reference_points, spatial_shapes):
    """Reproduce the installed eager MSDA formulas; retain head/point axes."""
    batch, queries = query.shape[:2]
    if module.n_levels != 1 or module._export or reference_points.shape[2] != 1:
        raise ValueError("spatial_evidence_v001 requires single-P4 eager execution")
    offsets = module.sampling_offsets(query).view(
        batch, queries, module.n_heads, 1, module.n_points, 2
    )
    if reference_points.shape[-1] == 4:
        return (
            reference_points[:, :, None, :, None, :2]
            + offsets / module.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )
    if reference_points.shape[-1] == 2:
        normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        return (
            reference_points[:, :, None, :, None] + offsets / normalizer[None, None, None, :, None]
        )
    raise ValueError("reference points require 2 or 4 coordinates")


def point_values(module, memory, padding_mask, locations, shape):
    """The same projected values and bilinear read as the native single level."""
    from rfdetr.utilities.tensors import _bilinear_grid_sample

    batch, queries, heads, levels, points, _ = locations.shape
    height, width = shape
    if levels != 1 or memory.shape != (batch, height * width, module.d_model):
        raise ValueError("single-level memory geometry changed")
    value = module.value_proj(memory)
    if padding_mask is not None:
        value = value.masked_fill(padding_mask[..., None], 0.0)
    value = (
        value.transpose(1, 2)
        .contiguous()
        .view(batch * heads, module.d_model // heads, height, width)
    )
    grid = (2 * locations[:, :, :, 0] - 1).transpose(1, 2).flatten(0, 1)
    sampled = _bilinear_grid_sample(value, grid, padding_mode="zeros", align_corners=False)
    return sampled.view(batch, heads, -1, queries, points).permute(0, 3, 1, 4, 2)


class PointEvidence(nn.Module):
    """Head-specific value maps enter a common two-class relative-score space."""

    def __init__(self, channels=256, heads=16, hidden=16):
        super().__init__()
        width = channels // heads
        self.query_norm = nn.LayerNorm(channels)
        self.value_norm = nn.LayerNorm(width)
        self.query = nn.Linear(channels, hidden)
        self.value = nn.Parameter(torch.empty(heads, width, hidden))
        self.head_bias = nn.Parameter(torch.zeros(heads, hidden))
        self.position = nn.Linear(2, hidden, bias=False)
        self.output = nn.Linear(hidden, 2)
        for weight in self.value:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, sampled, relative, valid_mass):
        local = torch.einsum("bqhpd,hdf->bqhpf", self.value_norm(sampled), self.value)
        hidden = (
            local
            + self.query(self.query_norm(query))[:, :, None, None]
            + self.head_bias[None, None, :, None]
            + self.position(relative)
        )
        return self.output(F.gelu(hidden)) * valid_mass.unsqueeze(-1)


@dataclass
class EvidenceContext:
    history: str = "coordinates"
    feedback: bool = True
    score: bool = True
    diagnostics: bool = False
    active: bool = False
    previous: torch.Tensor | None = None
    locations: torch.Tensor | None = None
    last: torch.Tensor | None = None
    observations: list = field(default_factory=list)

    def reset(self):
        self.previous = self.locations = self.last = None
        self.observations.clear()


class EvidenceAttention(nn.Module):
    """A native attention plus content/state feedback, without installed-file edits."""

    def __init__(self, native, context, *, hidden=16, retention=0.5):
        super().__init__()
        if native.n_levels != 1 or native.n_heads != 16 or native.n_points != 2:
            raise ValueError("qualify actual 16-head/two-point S1 before extending")
        # Reuse the four original children without a `.native` prefix in state_dict.
        for name in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
            self.add_module(name, getattr(native, name))
        for name in ("d_model", "n_levels", "n_heads", "n_points", "im2col_step", "_export"):
            setattr(self, name, getattr(native, name))
        self.context = context
        self.retention = retention
        self.evidence = PointEvidence(native.d_model, native.n_heads, hidden)
        self.routing = nn.Linear(native.d_model, 3)  # two classes and native fallback
        nn.init.zeros_(self.routing.weight)
        nn.init.zeros_(self.routing.bias)
        if not context.feedback:
            self.routing.requires_grad_(False)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        input_spatial_shapes_hw=None,
    ):
        if not self.context.active:
            raise RuntimeError("evidence attention must run inside the installed model forward")
        if self._export:
            raise ValueError("spatial_evidence_v001 supports eager execution; export not qualified")
        shape = (
            tuple(input_spatial_shapes_hw[0])
            if input_spatial_shapes_hw is not None
            else tuple(int(v) for v in input_spatial_shapes[0])
        )
        locations = sampling_locations(self, query, reference_points, input_spatial_shapes)
        sampled = point_values(self, input_flatten, input_padding_mask, locations, shape)
        flat_locations = locations[:, :, :, 0].flatten(2, 3)
        valid = support_mass(flat_locations, shape)
        reference = reference_points[:, :, 0]
        if reference.shape[-1] != 4:
            raise ValueError("the qualified S1 uses box references")
        relative = (locations[:, :, :, 0] - reference[:, :, None, None, :2]) / reference[
            :, :, None, None, 2:
        ].clamp_min(1e-6)
        increment = self.evidence(
            query,
            sampled,
            relative,
            valid.unflatten(-1, (self.n_heads, self.n_points)),
        ).flatten(2, 3)
        state, density = increment, torch.zeros_like(valid)
        context = self.context
        if context.previous is not None and context.history != "none":
            if context.history == "coordinates":
                # Pure-function checkpointing bounds training activation memory;
                # never recompute a stateful attention/model forward in backward.
                if torch.is_grad_enabled() and (
                    context.previous.requires_grad or flat_locations.requires_grad
                ):
                    carried, density = checkpoint(
                        transport_state,
                        context.previous,
                        context.locations,
                        flat_locations,
                        shape,
                        use_reentrant=False,
                    )
                else:
                    carried, density = transport_state(
                        context.previous, context.locations, flat_locations, shape
                    )
            elif context.history == "slots":
                density = ((support_mass(context.locations, shape) > 0) & (valid > 0)).to(valid)
                carried = context.previous * density.unsqueeze(-1)
            else:
                raise ValueError("unknown spatial-state control")
            # Covered points average old/new scores; no history leaves current
            # evidence unscaled. The binary coverage is not a probability.
            mass = self.retention * (density > 0).to(valid)
            state = (increment + self.retention * carried) / (1 + mass.unsqueeze(-1))
        logits = self.attention_weights(query).view(*query.shape[:2], self.n_heads, self.n_points)
        original = logits.softmax(-1)
        weights = original
        if context.feedback:
            class_state = state.unflatten(-2, (self.n_heads, self.n_points))
            conditional = (logits.unsqueeze(-1) + class_state).softmax(-2)
            routes = self.routing(query).softmax(-1)[..., :2]
            difference = ((conditional - original.unsqueeze(-1)) * routes[:, :, None, None]).sum(-1)
            weights = original + difference
        # Preserve the native reduction/order; no second grid_sample or
        # class-conditioned value copies are needed.
        values = sampled.permute(0, 2, 4, 1, 3).flatten(0, 1)
        flat_weights = weights.transpose(1, 2).flatten(0, 1).unsqueeze(1)
        combined = (values * flat_weights).sum(-1).view(len(query), self.d_model, query.shape[1])
        output = self.output_proj(combined.transpose(1, 2).contiguous())
        context.previous, context.locations, context.last = state, flat_locations, state
        if context.diagnostics:
            context.observations.append(
                dict(
                    locations=flat_locations.detach(),
                    density=density.detach(),
                    native_weights=original.detach().flatten(2, 3),
                    modified_weights=weights.detach().flatten(2, 3),
                    state=state.detach(),
                    increment=increment.detach(),
                    valid_mass=valid.detach(),
                )
            )
        return output


def evidence_scores(state):
    if state is None:
        raise ValueError("the current forward has no evidence state")
    return torch.logsumexp(state, dim=-2) - math.log(state.shape[-2])


# Global callbacks use the receiving module's context. EMA/deepcopy therefore
# gets an independent context, rather than closures referencing the student.
def _begin_evidence_forward(module, args):
    context = module._evidence_context
    if context.active:
        raise RuntimeError("nested evidence model forward is unsupported")
    context.reset()
    context.active = True


def _end_evidence_forward(module, args, output):
    context = module._evidence_context
    context.reset()
    context.active = False


def _classification_evidence(module, args, output):
    context = module._evidence_context
    if not context.active or context.last is None:
        raise RuntimeError("classification needs the current image's evidence")
    delta = evidence_scores(context.last).to(output)
    if output.ndim != 4 or output.shape[-1] != 3 or output.shape[-3:-1] != delta.shape[:2]:
        raise ValueError("expected decoder-layer,B,Q,3 native classification output")
    # Auxiliary layers retain native classification; only the last receives
    # the final state. The third channel is unmapped, never a background class.
    last = torch.cat((output[-1, ..., :2] + delta, output[-1, ..., 2:]), -1)
    return torch.cat((output[:-1], last.unsqueeze(0)), 0)


def evidence_parameter_names(model):
    return {
        name
        for name, _ in model.named_parameters()
        if any(part in name for part in (".cross_attn.evidence.", ".cross_attn.routing."))
    }


def install_evidence_head(model, settings):
    if not settings["enabled"]:
        return {"enabled": False, "additional_parameters": 0}
    decoder = model.transformer.decoder
    if (
        decoder.num_layers != 4
        or decoder.bbox_embed is not None
        or not model.bbox_reparam
        or not model.lite_refpoint_refine
        or model.segmentation_head is not None
        or hasattr(model, "_evidence_context")
    ):
        raise ValueError("evidence head requires the original fixed-reference four-layer detector")
    context = EvidenceContext(history=settings["history"], feedback=settings["feedback"])
    for layer in decoder.layers[-2:]:
        layer.cross_attn = EvidenceAttention(
            layer.cross_attn,
            context,
            hidden=settings["hidden_dim"],
            retention=settings["retention"],
        )
    model._evidence_context = context
    model.class_embed._evidence_context = context
    model.register_forward_pre_hook(_begin_evidence_forward)
    model.register_forward_hook(_end_evidence_forward, always_call=True)
    model.class_embed.register_forward_hook(_classification_evidence)
    return {
        **settings,
        "decoder_layers_zero_based": [2, 3],
        "additional_parameters": sum(
            p.numel() for n, p in model.named_parameters() if n in evidence_parameter_names(model)
        ),
        "classification": "final-layer first two logits plus logmeanexp spatial evidence",
        "box_regression": "original MLP on evidence-updated decoder query",
        "state_lifetime": "one model forward; cleared on success or error; no test adaptation",
        "auxiliary_outputs": "native; query features of modified layers include evidence feedback",
        "transport_checkpointing": "pure coordinate transport only, non-reentrant when gradients enabled",
        "export": "not qualified; eager execution only",
    }
