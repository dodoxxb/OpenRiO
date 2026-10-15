"""Opt-in paper-equation sampling overlay for the public RT-DETRv2-OBB carrier.

This module deliberately does not modify the upstream checkout.  State-dict keys
remain unchanged, so a public checkpoint can be screened under either geometry.
"""

from __future__ import annotations

import functools
import math
import types
from typing import List

import torch
import torch.nn.functional as F

from engine.core import register
from engine.rtv4.rtdetrv2_obb_decoder import (
    MSDeformableAttention,
    RTDETRTransformerv2OBB,
)


def paper_sampling_locations(
    sampling_offsets: torch.Tensor,
    ref_expanded: torch.Tensor,
    num_points_scale: torch.Tensor,
    offset_scale: float,
    num_heads: int,
) -> torch.Tensor:
    """Implement RiO-DETR Eq. (3): center + R(theta)(offset * size)."""

    angle = ref_expanded[..., 4:] * math.pi
    angle = angle.expand(-1, -1, num_heads, -1, -1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.cat([cosine, -sine, sine, cosine], dim=-1)
    rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)
    sizes = (ref_expanded[..., 2:4] * offset_scale).expand(
        -1, -1, num_heads, -1, -1
    )
    local_offsets = sampling_offsets * num_points_scale * sizes
    world_offsets = torch.einsum("...ij,...j->...i", rotation, local_offsets)
    return ref_expanded[..., :2] + world_offsets


def _paper_geometry_forward(
    self: MSDeformableAttention,
    query: torch.Tensor,
    reference_points: torch.Tensor,
    value: torch.Tensor,
    value_spatial_shapes: List[int],
    value_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, query_length = query.shape[:2]
    value_length = value.shape[1]
    value = self.value_proj(value)
    if value_mask is not None:
        value = value * value_mask.to(value.dtype).unsqueeze(-1)
    value = value.reshape(batch_size, value_length, self.num_heads, self.head_dim)

    sampling_offsets = self.sampling_offsets(query).reshape(
        batch_size,
        query_length,
        self.num_heads,
        sum(self.num_points_list),
        2,
    )
    attention_weights = self.attention_weights(query).reshape(
        batch_size, query_length, self.num_heads, sum(self.num_points_list)
    )
    attention_weights = F.softmax(attention_weights, dim=-1)
    reference_points = reference_points.expand(-1, -1, self.num_levels, -1)

    expanded = []
    for level, num_points in enumerate(self.num_points_list):
        reference = reference_points[:, :, level : level + 1, :]
        expanded.append(reference.unsqueeze(2).repeat(1, 1, 1, num_points, 1))
    ref_expanded = torch.cat(expanded, dim=3)
    point_scale = self.num_points_scale.to(
        device=query.device, dtype=query.dtype
    ).view(1, 1, 1, -1, 1)
    sampling_locations = paper_sampling_locations(
        sampling_offsets,
        ref_expanded,
        point_scale,
        self.offset_scale,
        self.num_heads,
    )
    output = self.ms_deformable_attn_core(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
        self.num_points_list,
    )
    return self.output_proj(output)


def apply_paper_sampling_overlay(model: torch.nn.Module) -> int:
    """Patch only OBB cross-attention instances and return the patched count."""

    count = 0
    for module in model.modules():
        if isinstance(module, MSDeformableAttention):
            module.forward = types.MethodType(_paper_geometry_forward, module)
            module.openrio_sampling_geometry = "paper_eq3"
            count += 1
    if count == 0:
        raise ValueError("model has no RT-DETRv2-OBB deformable-attention modules")
    return count


@register()
class OpenRiOPaperGeometryRTDETRTransformerv2OBB(RTDETRTransformerv2OBB):
    """Config-selectable RT-DETRv2-OBB decoder using the paper equation."""

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=True,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        value_shape="reshape",
        mlp_act="relu",
        query_pos_method="default",
        anchor_aspect_ratio=1.0,
    ):
        # Upstream mutates feat_strides when extra levels are requested.
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            value_shape=value_shape,
            mlp_act=mlp_act,
            query_pos_method=query_pos_method,
            anchor_aspect_ratio=anchor_aspect_ratio,
        )
        self.openrio_patched_cross_attention_layers = apply_paper_sampling_overlay(self)
