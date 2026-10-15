#!/usr/bin/env python3
"""Opt-in OBB matcher retaining exact costs on GPU for grouped LSAP."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from engine.core import register
from engine.rtv4.obb_matcher import HungarianMatcherOBB
from engine.rtv4.utils_obb import hausdorff_loss, kld_loss

from grouped_torch_assignment import solve_concatenated_costs


@register()
class OpenRiOExactGroupedHungarianMatcherOBB(HungarianMatcherOBB):
    """Objective-identical matcher whose only changed factor is LSAP execution."""

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        if return_topk:
            raise NotImplementedError("exact grouped overlay currently supports the criterion's k=1 path only")
        batch_size, query_count = outputs["pred_logits"].shape[:2]
        if self.use_focal_loss:
            probabilities = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            probabilities = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        boxes = outputs["pred_boxes"].flatten(0, 1)
        target_ids = torch.cat([value["labels"] for value in targets])
        target_boxes = torch.cat([value["boxes"] for value in targets])

        if self.use_focal_loss:
            probabilities = probabilities[:, target_ids]
            negative = (1 - self.alpha) * probabilities.pow(self.gamma) * (-(1 - probabilities + 1e-8).log())
            positive = self.alpha * (1 - probabilities).pow(self.gamma) * (-(probabilities + 1e-8).log())
            class_cost = positive - negative
        else:
            class_cost = -probabilities[:, target_ids]

        box_cost = torch.cdist(boxes, target_boxes, p=1)
        expanded_boxes = boxes.unsqueeze(1).expand(-1, len(target_boxes), -1)
        expanded_targets = target_boxes.unsqueeze(0).expand(len(boxes), -1, -1)
        geometry_cost = kld_loss(expanded_boxes, expanded_targets, fun="log1p", tau=1.0, sqrt=False)
        cost = self.cost_bbox * box_cost + self.cost_class * class_cost + self.cost_giou * geometry_cost
        if self.cost_hausdorff > 0:
            cost = cost + self.cost_hausdorff * hausdorff_loss(boxes, target_boxes)

        sizes = [len(value["boxes"]) for value in targets]
        cost = cost.view(batch_size, query_count, -1)
        return {"indices": solve_concatenated_costs(cost, sizes)}
