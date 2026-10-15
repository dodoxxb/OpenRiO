"""OpenRiO-only criterion overlays; the public RiO snapshot remains untouched."""

from __future__ import annotations

import torch

from engine.core import register
from engine.rtv4.rtv4_obb_criterion import RTv4OBBCriterion
from rio_cleanroom_overlay import OpenRiOPeriodicRTv4OBBCriterion


@register(force=True)
class BudgetedRTv4OBBCriterion(RTv4OBBCriterion):
    """Cross-layer consensus matching with an explicit positive-count budget.

    ``union_budget='all'`` reproduces the released union implementation.
    ``union_budget='local'`` keeps the most frequently recurring union pairs but
    truncates each image to its final-layer Hungarian count.  The latter is the
    preregistered mechanism control for consensus versus additional regression
    positives; it is not launched unless the layer-local screen is promoted.
    """

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
        union_budget="local",
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
        if union_budget not in {"all", "local"}:
            raise ValueError(f"unknown union_budget: {union_budget}")
        self.union_budget = union_budget

    def _get_go_indices(self, indices, indices_aux_list):
        local_counts = [len(src) for src, _ in indices]
        combined = [(src.clone(), tgt.clone()) for src, tgt in indices]
        for auxiliary in indices_aux_list:
            combined = [
                (torch.cat([left[0], right[0]]), torch.cat([left[1], right[1]]))
                for left, right in zip(combined, auxiliary)
            ]

        results = []
        for image_index, (src, tgt) in enumerate(combined):
            pairs = torch.stack([src, tgt], dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            # Stable sorting makes ties deterministic because torch.unique has
            # already placed pairs in lexicographic order.
            order = torch.argsort(counts, descending=True, stable=True)
            ranked = unique[order]

            chosen_by_query = {}
            for query_index, target_index in ranked:
                query = int(query_index.item())
                if query not in chosen_by_query:
                    chosen_by_query[query] = int(target_index.item())

            selected = list(chosen_by_query.items())
            if self.union_budget == "local":
                selected = selected[: local_counts[image_index]]
            if not selected:
                empty = torch.zeros(0, dtype=torch.long, device=pairs.device)
                results.append((empty, empty.clone()))
                continue
            rows, cols = zip(*selected)
            results.append(
                (
                    torch.tensor(rows, dtype=torch.long, device=pairs.device),
                    torch.tensor(cols, dtype=torch.long, device=pairs.device),
                )
            )
        return results


@register(force=True)
class BudgetedOpenRiOPeriodicRTv4OBBCriterion(
    OpenRiOPeriodicRTv4OBBCriterion
):
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
        union_budget="local",
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
            angle_loss_units=angle_loss_units,
        )
        if union_budget not in {"all", "local"}:
            raise ValueError(f"unknown union_budget: {union_budget}")
        self.union_budget = union_budget

    def _get_go_indices(self, indices, indices_aux_list):
        local_counts = [len(src) for src, _ in indices]
        combined = [(src.clone(), tgt.clone()) for src, tgt in indices]
        for auxiliary in indices_aux_list:
            combined = [
                (torch.cat([left[0], right[0]]), torch.cat([left[1], right[1]]))
                for left, right in zip(combined, auxiliary)
            ]

        results = []
        for image_index, (src, tgt) in enumerate(combined):
            pairs = torch.stack([src, tgt], dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            order = torch.argsort(counts, descending=True, stable=True)
            ranked = unique[order]
            chosen_by_query = {}
            for query_index, target_index in ranked:
                query = int(query_index.item())
                if query not in chosen_by_query:
                    chosen_by_query[query] = int(target_index.item())
            selected = list(chosen_by_query.items())
            if self.union_budget == "local":
                selected = selected[: local_counts[image_index]]
            if not selected:
                empty = torch.zeros(0, dtype=torch.long, device=pairs.device)
                results.append((empty, empty.clone()))
                continue
            rows, cols = zip(*selected)
            results.append(
                (
                    torch.tensor(rows, dtype=torch.long, device=pairs.device),
                    torch.tensor(cols, dtype=torch.long, device=pairs.device),
                )
            )
        return results
