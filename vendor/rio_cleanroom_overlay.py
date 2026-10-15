"""Independent, equation-driven RiO-DETR reconstruction overlays.

This file is a clean-room implementation from the RiO-DETR paper.  It is not
the authors' withheld implementation.  Every behavior exposed here is either a
direct transcription of a paper equation or an explicit, configurable choice
where the paper leaves implementation details under-specified.

The public RT-DETRv2-OBB checkout is imported but never edited.  The overlay
provides:

* 4-D geometry-decoupled query position encoding (paper Eq. 1),
* symmetric orthogonal rotated deformable sampling (paper Eqs. 2--4),
* bounded periodic decoder refinement (paper Eqs. 5--7),
* shortest-path periodic L1 (paper Eq. 8), and
* independently rotated quadrants for Oriented Dense O2O.
"""

from __future__ import annotations

import functools
import math
import random
from collections import OrderedDict
from typing import Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.data.transforms.mosaic_obb import MosaicOBB
from engine.data.transforms.transforms_obb import get_spatial_size, obb2poly, poly2obb
from engine.rtv4.rtdetrv2_obb_decoder import (
    MLP,
    MSDeformableAttention,
    RTDETRTransformerv2OBB,
    TransformerDecoderLayer,
)
from engine.rtv4.rtdetrv2_obb_decoder import TransformerDecoderOBB as _PublicDecoder
from engine.rtv4.rtdetrv2_obb_decoder import inverse_sigmoid
from engine.rtv4.rtv4_obb_criterion import RTv4OBBCriterion


def wrap_unit_angle(angle: torch.Tensor) -> torch.Tensor:
    """Map a normalized pi-periodic angle to [0, 1)."""

    return torch.remainder(angle, 1.0)


def signed_shortest_unit_delta(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Signed shortest displacement on R/Z, in normalized angle units."""

    return torch.remainder(prediction - target + 0.5, 1.0) - 0.5


def shortest_periodic_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_units: str = "normalized",
) -> torch.Tensor:
    """Elementwise geodesic L1 on the rectangle angle quotient.

    The carrier stores theta/pi in [0, 1).  ``normalized`` keeps the angle term
    on the same numeric scale as normalized cx/cy/w/h.  ``radian`` implements
    the literal units of Eq. (8).  The paper does not state which convention is
    used after dataset normalization, so both are explicit experiment arms.
    """

    distance = signed_shortest_unit_delta(prediction, target).abs()
    if output_units == "normalized":
        return distance
    if output_units == "radian":
        return distance * math.pi
    raise ValueError(f"unknown output_units: {output_units}")


def periodic_refine_angle(
    reference: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    layer_index: int,
    alpha0: float = 1.5,
    step_units: str = "radian",
) -> torch.Tensor:
    """RiO bounded coarse-to-fine update in the carrier's normalized units."""

    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if alpha0 <= 1.0:
        raise ValueError("alpha0 must be greater than one")
    alpha = alpha0 ** (-layer_index)
    step = torch.tanh(raw_delta) * alpha
    if step_units == "radian":
        step = step / math.pi
    elif step_units != "normalized":
        raise ValueError(f"unknown step_units: {step_units}")
    return wrap_unit_angle(reference + step)


def orthogonal_sampling_locations(
    sampling_offsets: torch.Tensor,
    ref_expanded: torch.Tensor,
    num_points_scale: torch.Tensor,
    offset_scale: float,
    num_heads: int,
) -> torch.Tensor:
    """RiO Eqs. (2)--(4), with half the heads rotated by pi/2."""

    if num_heads % 2:
        raise ValueError("symmetric orthogonal splitting requires an even head count")
    if sampling_offsets.shape[2] != num_heads:
        raise ValueError("sampling offset head dimension does not match num_heads")

    base_angle = ref_expanded[..., 4:] * math.pi
    head_shift = torch.zeros(
        (1, 1, num_heads, 1, 1),
        dtype=base_angle.dtype,
        device=base_angle.device,
    )
    head_shift[:, :, num_heads // 2 :] = math.pi / 2
    angle = base_angle.expand(-1, -1, num_heads, -1, -1) + head_shift
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.cat([cosine, -sine, sine, cosine], dim=-1)
    rotation = rotation.reshape(*rotation.shape[:-1], 2, 2)

    size = (ref_expanded[..., 2:4] * offset_scale).expand(
        -1, -1, num_heads, -1, -1
    )
    local_offset = sampling_offsets * num_points_scale * size
    world_offset = torch.einsum("...ij,...j->...i", rotation, local_offset)
    return ref_expanded[..., :2] + world_offset


def _orthogonal_attention_forward(
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
    sampling_locations = orthogonal_sampling_locations(
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


class CleanroomRiODecoder(_PublicDecoder):
    """Decoder with auditable query geometry and periodic theta updates."""

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        eval_idx: int = -1,
        *,
        alpha0: float = 1.5,
        angle_step_units: str = "radian",
        query_geometry_dimensions: int = 4,
        mask_query_angle: bool = False,
        periodic_refinement: bool = True,
    ) -> None:
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        if alpha0 <= 1.0:
            raise ValueError("alpha0 must be greater than one")
        if angle_step_units not in {"radian", "normalized"}:
            raise ValueError("angle_step_units must be radian or normalized")
        self.alpha0 = float(alpha0)
        self.angle_step_units = angle_step_units
        if query_geometry_dimensions not in {4, 5}:
            raise ValueError("query_geometry_dimensions must be four or five")
        self.query_geometry_dimensions = query_geometry_dimensions
        if mask_query_angle and query_geometry_dimensions != 5:
            raise ValueError("mask_query_angle requires a shape-preserving 5-D query input")
        self.mask_query_angle = bool(mask_query_angle)
        self.periodic_refinement = bool(periodic_refinement)

    def _query_position_input(self, reference: torch.Tensor) -> torch.Tensor:
        query_geometry = reference[..., : self.query_geometry_dimensions]
        if self.mask_query_angle:
            # Keep the public 5-D MLP shape, checkpoint and RNG consumption
            # while removing theta from the query-position computation.
            query_geometry = torch.cat(
                [query_geometry[..., :4], torch.zeros_like(query_geometry[..., 4:5])],
                dim=-1,
            )
        return query_geometry

    def _refine(
        self,
        reference: torch.Tensor,
        raw_delta: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if not self.periodic_refinement:
            return torch.sigmoid(raw_delta + inverse_sigmoid(reference))
        spatial = torch.sigmoid(
            raw_delta[..., :4] + inverse_sigmoid(reference[..., :4])
        )
        angle = periodic_refine_angle(
            reference[..., 4:],
            raw_delta[..., 4:],
            layer_index=layer_index,
            alpha0=self.alpha0,
            step_units=self.angle_step_units,
        )
        return torch.cat([spatial, angle], dim=-1)

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        output = target

        for layer_index, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(self._query_position_input(ref_points_detach))
            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )
            raw_delta = bbox_head[layer_index](output)
            inter_ref_bbox = self._refine(
                ref_points_detach, raw_delta, layer_index
            )

            if self.training:
                dec_out_logits.append(score_head[layer_index](output))
                if layer_index == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    # Preserve the public decoder's gradient path through the
                    # previous layer while attention still consumes detached refs.
                    dec_out_bboxes.append(
                        self._refine(ref_points, raw_delta, layer_index)
                    )
            elif layer_index == self.eval_idx:
                dec_out_logits.append(score_head[layer_index](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register()
class OpenRiOCleanroomRTDETRTransformerv2OBB(RTDETRTransformerv2OBB):
    """Config-selectable clean-room reconstruction of the paper's decoder."""

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
        query_pos_method="rio_cleanroom",
        anchor_aspect_ratio=1.0,
        rio_alpha0=1.5,
        rio_angle_step_units="radian",
        rio_geometry_decoupled=True,
        rio_mask_query_angle=False,
        rio_periodic_refinement=True,
        rio_orthogonal_attention=True,
    ):
        del query_pos_method  # this class always uses the paper's 4-D query input
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
            query_pos_method="default",
            anchor_aspect_ratio=anchor_aspect_ratio,
        )
        query_dimensions = 4 if rio_geometry_decoupled else 5
        if rio_geometry_decoupled:
            self.query_pos_method = "rio_cleanroom_4d"
        elif rio_mask_query_angle:
            self.query_pos_method = "rio_masked_theta_5d"
        else:
            self.query_pos_method = "public_5d"
        self.query_pos_head = MLP(
            query_dimensions, 2 * hidden_dim, hidden_dim, 2, act=mlp_act
        )
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)

        self.decoder = CleanroomRiODecoder(
            hidden_dim,
            self.decoder.layers[0],
            num_layers,
            eval_idx,
            alpha0=rio_alpha0,
            angle_step_units=rio_angle_step_units,
            query_geometry_dimensions=query_dimensions,
            mask_query_angle=rio_mask_query_angle,
            periodic_refinement=rio_periodic_refinement,
        )
        self.rio_geometry_decoupled = bool(rio_geometry_decoupled)
        self.rio_mask_query_angle = bool(rio_mask_query_angle)
        self.rio_periodic_refinement = bool(rio_periodic_refinement)
        self.rio_orthogonal_attention = bool(rio_orthogonal_attention)
        if self.rio_orthogonal_attention:
            for module in self.decoder.modules():
                if isinstance(module, MSDeformableAttention):
                    module.forward = functools.partial(
                        _orthogonal_attention_forward, module
                    )
                    module.openrio_sampling_geometry = "rio_symmetric_orthogonal"


def adapt_public_obb_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = "decoder.query_pos_head.layers.0.weight",
) -> OrderedDict[str, torch.Tensor]:
    """Crop the public 5-D query MLP to initialize the 4-D clean-room MLP."""

    adapted = OrderedDict((key, value) for key, value in state_dict.items())
    candidates = [key for key in adapted if key.endswith(prefix)]
    for key in candidates:
        weight = adapted[key]
        if weight.ndim != 2 or weight.shape[1] != 5:
            raise ValueError(f"unexpected public query weight shape for {key}: {weight.shape}")
        adapted[key] = weight[:, :4].clone()
    return adapted


@register()
class OpenRiOPeriodicRTv4OBBCriterion(RTv4OBBCriterion):
    """Public OBB criterion with paper Eq. (8) for only the angle coordinate."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        boxes_weight_format=None,
        use_uni_set=True,
        angle_loss_units="normalized",
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            boxes_weight_format=boxes_weight_format,
            use_uni_set=use_uni_set,
        )
        if angle_loss_units not in {"normalized", "radian"}:
            raise ValueError("angle_loss_units must be normalized or radian")
        self.angle_loss_units = angle_loss_units

    def loss_l1(self, outputs, targets, indices, num_boxes, **kwargs):
        del kwargs
        idx = self._get_src_permutation_idx(indices)
        source = outputs["pred_boxes"][idx]
        target = torch.cat(
            [item["boxes"][selection] for item, (_, selection) in zip(targets, indices)],
            dim=0,
        )
        spatial = F.l1_loss(source[..., :4], target[..., :4], reduction="none")
        angle = shortest_periodic_l1(
            source[..., 4], target[..., 4], output_units=self.angle_loss_units
        ).unsqueeze(-1)
        loss = torch.cat([spatial, angle], dim=-1).sum() / num_boxes
        if target.shape[0] == 0:
            loss = loss + outputs["pred_boxes"].sum() * 0.0
        return {"loss_l1": loss}


def rotate_obb_quadrant(
    image,
    target: Mapping[str, object],
    quarter_turns: int,
):
    """Rotate an image/OBBs CCW by exact multiples of 90 degrees."""

    turns = int(quarter_turns) % 4
    copied = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }
    if turns == 0:
        return image.clone() if isinstance(image, torch.Tensor) else image.copy(), copied

    height, width = [int(value) for value in get_spatial_size(image)]
    if isinstance(image, torch.Tensor):
        rotated_image = torch.rot90(image, turns, dims=(-2, -1))
    else:
        rotated_image = image.rotate(90 * turns, expand=True)

    boxes = copied.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.numel():
        polygons = obb2poly(boxes)
        x = polygons[..., 0].copy()
        y = polygons[..., 1].copy()
        if turns == 1:
            polygons[..., 0], polygons[..., 1] = y, width - x
        elif turns == 2:
            polygons[..., 0], polygons[..., 1] = width - x, height - y
        else:
            polygons[..., 0], polygons[..., 1] = height - y, x
        copied["boxes"] = poly2obb(polygons, device=boxes.device).to(boxes.dtype)
    return rotated_image, copied


@register()
class OpenRiOOrientedDenseO2O(MosaicOBB):
    """Dense-O2O mosaic with an independent discrete rotation per quadrant."""

    def __init__(
        self,
        output_size=320,
        max_size=None,
        rotation_range=0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        probability=1.0,
        fill_value=114,
        use_cache=True,
        max_cached_images=50,
        random_pop=True,
        quadrant_rotations=(0, 90, 180, 270),
    ) -> None:
        super().__init__(
            output_size=output_size,
            max_size=max_size,
            rotation_range=rotation_range,
            translation_range=translation_range,
            scaling_range=scaling_range,
            probability=probability,
            fill_value=fill_value,
            use_cache=use_cache,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
        )
        rotations = tuple(int(value) for value in quadrant_rotations)
        if not rotations or any(value % 90 for value in rotations):
            raise ValueError("quadrant rotations must be multiples of 90 degrees")
        self.quadrant_rotations = rotations
        self.last_quadrant_rotations: tuple[int, ...] = ()

    def _rotate_samples(self, images, targets):
        rotations = tuple(random.choice(self.quadrant_rotations) for _ in images)
        rotated = [
            rotate_obb_quadrant(image, target, angle // 90)
            for image, target, angle in zip(images, targets, rotations)
        ]
        self.last_quadrant_rotations = rotations
        return [item[0] for item in rotated], [item[1] for item in rotated]

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        if self.use_cache:
            samples, _, _ = self.load_samples_from_cache(
                image, target, self.mosaic_cache
            )
            images = [sample["img"] for sample in samples]
            targets = [sample["labels"] for sample in samples]
            images, targets = self._rotate_samples(images, targets)
            samples = [
                {"img": sample_image, "labels": sample_target}
                for sample_image, sample_target in zip(images, targets)
            ]
            sizes = [get_spatial_size(item) for item in images]
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(
                samples,
                max(int(size[0]) for size in sizes),
                max(int(size[1]) for size in sizes),
            )
        else:
            images, targets, _, _ = self.load_samples_from_dataset(
                image, target, dataset
            )
            images, targets = self._rotate_samples(images, targets)
            sizes = [get_spatial_size(item) for item in images]
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(
                images,
                targets,
                max(int(size[0]) for size in sizes),
                max(int(size[1]) for size in sizes),
            )

        result = self.affine_transform(mosaic_image, mosaic_target)
        if isinstance(result, tuple):
            mosaic_image, mosaic_target = result[:2]
        else:
            mosaic_image = result
        return mosaic_image, mosaic_target, dataset
