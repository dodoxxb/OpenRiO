"""OpenRiO evaluator overlay that exports a deterministic prediction artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from engine.core import register
from engine.data.dataset.dota_evaluator import DotaEvaluator, rbox2qbox
from prediction_contract import (
    ANGLE_UNIT,
    BOX_FORMAT,
    SCHEMA_VERSION,
    write_artifact,
)


def _as_list(array: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in array.tolist()]


@register(force=True)
class OpenRiODotaEvaluator(DotaEvaluator):
    """DOTA metrics plus the cross-carrier OpenRiO OBB prediction contract."""

    def __init__(
        self,
        dataset,
        iou_types=None,
        iou_thrs=None,
        export_path=None,
        contract=None,
        require_complete=True,
    ):
        super().__init__(dataset=dataset, iou_types=iou_types, iou_thrs=iou_thrs)
        if not export_path:
            raise ValueError("OpenRiODotaEvaluator requires export_path")
        if not isinstance(contract, Mapping):
            raise ValueError("OpenRiODotaEvaluator requires a contract mapping")
        self.export_path = Path(export_path)
        self.contract = dict(contract)
        self.require_complete = bool(require_complete)
        self.export_manifest = None

    def _stable_image_id(self, dataset_index: int) -> str:
        ids = getattr(self.dataset, "img_ids", None)
        if ids is None:
            return str(dataset_index)
        return str(ids[dataset_index])

    def synchronize_between_processes(self):
        super().synchronize_between_processes()
        # DistributedSampler may pad a non-divisible validation set.  The
        # upstream evaluator merges prediction keys but leaves padded indices
        # duplicated in img_ids, thereby duplicating both GT and detections.
        # Evaluate each physical image exactly once.
        self.img_ids = sorted((int(index) for index in self.predictions), key=int)

    def _prediction_records(self) -> list[dict[str, Any]]:
        indices = {int(index) for index in self.predictions}
        if self.require_complete:
            expected = set(range(len(self.dataset)))
            if indices != expected:
                missing = sorted(expected.difference(indices))
                extra = sorted(indices.difference(expected))
                raise ValueError(
                    "incomplete prediction set: "
                    f"expected={len(expected)} got={len(indices)} "
                    f"missing_head={missing[:10]} extra_head={extra[:10]}"
                )
        records = []
        for raw_index in sorted(self.predictions, key=int):
            dataset_index = int(raw_index)
            prediction = self.predictions[raw_index]
            boxes = np.asarray(prediction["bboxes"], dtype=np.float64).reshape(-1, 5)
            scores = np.asarray(prediction["scores"], dtype=np.float64).reshape(-1)
            labels = np.asarray(prediction["labels"], dtype=np.int64).reshape(-1)
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"prediction length mismatch for dataset index {dataset_index}")

            # Stable detector-neutral order: descending score, then class and box tuple.
            order = sorted(
                range(len(scores)),
                key=lambda i: (
                    -float(scores[i]),
                    int(labels[i]),
                    *(float(value) for value in boxes[i]),
                ),
            )
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]
            polygons = rbox2qbox(torch.from_numpy(boxes)).numpy()
            records.append(
                {
                    "record_type": "prediction",
                    "dataset_index": dataset_index,
                    "image_id": self._stable_image_id(dataset_index),
                    "boxes_cxcywha_rad": _as_list(boxes),
                    "polygons_xy": _as_list(polygons),
                    "scores": [float(value) for value in scores.tolist()],
                    "labels": [int(value) for value in labels.tolist()],
                }
            )
        return records

    def export_predictions(self) -> dict[str, Any]:
        records = self._prediction_records()
        header = {
            "record_type": "header",
            "schema_version": SCHEMA_VERSION,
            "box_format": BOX_FORMAT,
            "angle_unit": ANGLE_UNIT,
            "polygon_format": "x1y1x2y2x3y3x4y4",
            "num_images": len(records),
            "num_predictions": sum(len(record["scores"]) for record in records),
            "classes": [str(value) for value in self.classes],
            "contract": self.contract,
        }
        self.export_manifest = write_artifact(self.export_path, header, records)
        return self.export_manifest

    def summarize(self):
        # The upstream solver invokes DOTA-style summarize on rank zero after sync.
        self.export_predictions()
        return super().summarize()
