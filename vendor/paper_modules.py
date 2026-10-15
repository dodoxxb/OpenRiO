"""Paper-described RiO-DETR v2 reconstruction, independent of withheld code.

This opt-in module leaves the 2026-08-30 experiments byte-for-byte unchanged.
Unspecified conventions are configuration fields, not asserted author choices.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys

import torch
from torch import nn

from engine.core import register
from engine.rtv4.rtdetrv2_obb_decoder import MLP
from engine.data.transforms.transforms_obb import get_spatial_size
from rio_cleanroom_overlay import (
    OpenRiOCleanroomRTDETRTransformerv2OBB,
    OpenRiOOrientedDenseO2O,
    CleanroomRiODecoder,
)


def geometry_sine_encoding(coordinates, num_pos_feats=64, temperature=10000.0,
                           coordinate_order='yxwh'):
    """Four coordinates -> 4F channels; DETR-family frequency convention.

    RiO Eq. 1 specifies sine PE, but not F, temperature or coordinate order.
    yxwh follows the public DAB-DETR implementation; xywh is also explicit.
    This is theta independence, NOT invariance to rotating the entire image.
    """
    if coordinates.shape[-1] != 4:
        raise ValueError('GQE sine encoding must receive exactly cx,cy,w,h')
    if num_pos_feats <= 0 or num_pos_feats % 2 or temperature <= 0:
        raise ValueError('positive even feature dimension and positive temperature required')
    if coordinate_order not in ('xywh', 'yxwh'):
        raise ValueError('coordinate_order must be xywh or yxwh')
    x = coordinates.float()
    if coordinate_order == 'yxwh':
        x = x[..., [1, 0, 2, 3]]
    index = torch.arange(num_pos_feats, device=x.device, dtype=torch.float32)
    frequency = temperature ** (2 * torch.div(index, 2, rounding_mode='floor') / num_pos_feats)
    phase = 2 * math.pi * x[..., None] / frequency
    encoding = torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), -1)
    return encoding.flatten(-3).to(coordinates.dtype)


class SineGeometryMLP(nn.Module):
    def __init__(self, hidden_dim, num_pos_feats=None, temperature=10000.0,
                 coordinate_order='yxwh', act='relu'):
        super().__init__()
        self.num_pos_feats = hidden_dim // 2 if num_pos_feats is None else num_pos_feats
        self.temperature = temperature
        self.coordinate_order = coordinate_order
        self.mlp = MLP(4 * self.num_pos_feats, 2 * hidden_dim, hidden_dim, 2, act=act)
        for layer in self.mlp.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, coordinates):
        return self.mlp(geometry_sine_encoding(
            coordinates, self.num_pos_feats, self.temperature, self.coordinate_order))


class PaperPeriodicDecoder(CleanroomRiODecoder):
    def _refine(self, reference, raw_delta, layer_index):
        # Appendix C uses 0..L-1; main Eqs. 5-7 use 1..L.
        return super()._refine(reference, raw_delta, layer_index + self.layer_index_origin)


@register()
class PaperRiOTransformer(OpenRiOCleanroomRTDETRTransformerv2OBB):
    def __init__(self, num_classes=80, hidden_dim=128, num_queries=300,
                 feat_channels=(128, 128), feat_strides=(16, 32), num_levels=2,
                 num_points=(6, 6), nhead=8, num_layers=3, dim_feedforward=512,
                 dropout=0.0, activation='relu', num_denoising=100,
                 label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=True,
                 eval_spatial_size=None, eval_idx=-1, eps=1e-2, aux_loss=True,
                 cross_attn_method='default', query_select_method='default',
                 value_shape='reshape', mlp_act='relu', query_pos_method='paper_sine',
                 anchor_aspect_ratio=1.0, rio_alpha0=1.5,
                 rio_angle_step_units='radian', rio_geometry_decoupled=True,
                 rio_mask_query_angle=False, rio_periodic_refinement=True,
                 rio_orthogonal_attention=True, pe_num_pos_feats=None,
                 pe_temperature=10000.0, pe_coordinate_order='yxwh',
                 layer_index_origin=0):
        if layer_index_origin not in (0, 1):
            raise ValueError('layer_index_origin must be appendix 0 or main-text 1')
        if query_pos_method not in ('paper_sine', 'legacy_direct'):
            raise ValueError('query_pos_method must be paper_sine or legacy_direct')
        if query_pos_method == 'paper_sine' and not rio_geometry_decoupled:
            raise ValueError('paper sine GQE uses four coordinates')
        # Keep initialization of shared modules independent of optional PE size.
        num_points = list(num_points)
        feat_channels = list(feat_channels)
        feat_strides = list(feat_strides)
        super().__init__(
            num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
            feat_channels=feat_channels, feat_strides=feat_strides, num_levels=num_levels,
            num_points=num_points, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            num_denoising=num_denoising, label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale, learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size, eval_idx=eval_idx, eps=eps,
            aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method=query_select_method, value_shape=value_shape,
            mlp_act=mlp_act, query_pos_method=query_pos_method,
            anchor_aspect_ratio=anchor_aspect_ratio,
            rio_alpha0=rio_alpha0, rio_angle_step_units=rio_angle_step_units,
            rio_geometry_decoupled=rio_geometry_decoupled,
            rio_mask_query_angle=rio_mask_query_angle,
            rio_periodic_refinement=rio_periodic_refinement,
            rio_orthogonal_attention=rio_orthogonal_attention)
        # Same layout and unchanged forward; only the index convention changes.
        self.decoder.__class__ = PaperPeriodicDecoder
        self.decoder.layer_index_origin = layer_index_origin
        if query_pos_method == 'paper_sine':
            with torch.random.fork_rng(devices=[]):
                self.query_pos_head = SineGeometryMLP(
                    hidden_dim, pe_num_pos_feats, pe_temperature, pe_coordinate_order, mlp_act)
        self.query_pos_method = query_pos_method


@register()
class PaperRiOPublic5DTransformer(PaperRiOTransformer):
    """RiO equation overlay with the public checkpoint's 5-D query-head shape.

    This arm isolates the query-position/checkpoint-shape hypothesis. It keeps
    the reconstructed periodic decoder, RROA overlay, FFN width, denoising and
    matcher unchanged, but restores a 5-D public-style MLP input and lets the
    decoder query all five normalized box coordinates.
    """

    def __init__(self, num_classes=80, hidden_dim=128, num_queries=300,
                 feat_channels=(128, 128), feat_strides=(16, 32), num_levels=2,
                 num_points=(6, 6), nhead=8, num_layers=3, dim_feedforward=1024,
                 dropout=0.0, activation='relu', num_denoising=100,
                 label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=True,
                 eval_spatial_size=None, eval_idx=-1, eps=1e-2, aux_loss=True,
                 cross_attn_method='default', query_select_method='default',
                 value_shape='reshape', mlp_act='relu', query_pos_method='public_5d',
                 anchor_aspect_ratio=1.0, rio_alpha0=1.5,
                 rio_angle_step_units='radian', rio_geometry_decoupled=True,
                 rio_mask_query_angle=False, rio_periodic_refinement=True,
                 rio_orthogonal_attention=True, pe_num_pos_feats=None,
                 pe_temperature=10000.0, pe_coordinate_order='yxwh',
                 layer_index_origin=0):
        # The parent uses the paper_sine validation path to initialize all
        # shared modules; replace only the query head after construction.
        parent_query_method = 'paper_sine' if rio_geometry_decoupled else 'legacy_direct'
        super().__init__(
            num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries,
            feat_channels=feat_channels, feat_strides=feat_strides, num_levels=num_levels,
            num_points=num_points, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            num_denoising=num_denoising, label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale, learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size, eval_idx=eval_idx, eps=eps,
            aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method=query_select_method, value_shape=value_shape,
            mlp_act=mlp_act, query_pos_method=parent_query_method,
            anchor_aspect_ratio=anchor_aspect_ratio, rio_alpha0=rio_alpha0,
            rio_angle_step_units=rio_angle_step_units,
            rio_geometry_decoupled=rio_geometry_decoupled,
            rio_mask_query_angle=rio_mask_query_angle,
            rio_periodic_refinement=rio_periodic_refinement,
            rio_orthogonal_attention=rio_orthogonal_attention,
            pe_num_pos_feats=pe_num_pos_feats, pe_temperature=pe_temperature,
            pe_coordinate_order=pe_coordinate_order, layer_index_origin=layer_index_origin)
        self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        for layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(layer.weight)
        self.decoder.query_geometry_dimensions = 5
        self.decoder.mask_query_angle = False
        self.query_pos_method = 'public_5d_compat'


@register()
class PaperOrientedDenseO2O(OpenRiOOrientedDenseO2O):
    """RiO Sec. 3.4 literal four REPLICAS, each independently rotated.

    The older overlay sampled three OTHER images using public MosaicOBB.
    This class makes the distinction explicit, with replica provenance tests.
    Subsequent global affine/resizing follows the public carrier; interpolation,
    probabilities and sampling with replacement remain disclosed choices.
    """
    def __init__(self, output_size=800, max_size=None, rotation_range=0,
                 translation_range=(0.1, 0.1), scaling_range=(0.5, 1.5),
                 probability=1.0, fill_value=0, use_cache=False,
                 max_cached_images=50, random_pop=True,
                 quadrant_rotations=(0, 90, 180, 270)):
        if use_cache:
            raise ValueError('same-image replicas do not use a cross-image cache')
        super().__init__(output_size, max_size, rotation_range, translation_range,
                         scaling_range, probability, fill_value, False,
                         max_cached_images, random_pop, quadrant_rotations)

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        if self.probability < 1 and random.random() > self.probability:
            return image, target, dataset
        resized = self.resize(image, target)
        image, target = resized[:2]
        # Rotation helper clones boxes even for the zero-turn case.
        images, targets = self._rotate_samples([image] * 4, [target] * 4)
        sizes = [get_spatial_size(x) for x in images]
        image, target = self.create_mosaic_from_dataset(
            images, targets, max(int(x[0]) for x in sizes), max(int(x[1]) for x in sizes))
        result = self.affine_transform(image, target)
        if isinstance(result, tuple):
            image, target = result[:2]
        else:
            image = result
        return image, target, dataset
