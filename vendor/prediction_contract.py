#!/usr/bin/env python3
"""Deterministic, detector-agnostic OBB prediction artifact contract."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "openrio.obb.predictions.v1"
BOX_FORMAT = "cxcywha"
ANGLE_UNIT = "radian"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_number(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} contains a non-finite value")
    return number


def validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "record_type",
        "dataset_index",
        "image_id",
        "boxes_cxcywha_rad",
        "polygons_xy",
        "scores",
        "labels",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"prediction record missing fields: {sorted(missing)}")
    if record["record_type"] != "prediction":
        raise ValueError("prediction record_type must be 'prediction'")
    if int(record["dataset_index"]) < 0:
        raise ValueError("dataset_index must be non-negative")

    boxes = record["boxes_cxcywha_rad"]
    polygons = record["polygons_xy"]
    scores = record["scores"]
    labels = record["labels"]
    lengths = {len(boxes), len(polygons), len(scores), len(labels)}
    if len(lengths) != 1:
        raise ValueError("boxes, polygons, scores, and labels must have equal length")
    for index, box in enumerate(boxes):
        if len(box) != 5:
            raise ValueError(f"box {index} must contain five values")
        values = [_finite_number(v, f"box {index}") for v in box]
        if values[2] < 0.0 or values[3] < 0.0:
            raise ValueError(f"box {index} has a negative side length")
    for index, polygon in enumerate(polygons):
        if len(polygon) != 8:
            raise ValueError(f"polygon {index} must contain eight values")
        for value in polygon:
            _finite_number(value, f"polygon {index}")
    for index, score in enumerate(scores):
        value = _finite_number(score, f"score {index}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"score {index} is outside [0, 1]")
    for index, label in enumerate(labels):
        if int(label) != label or int(label) < 0:
            raise ValueError(f"label {index} must be a non-negative integer")


def validate_header(header: Mapping[str, Any]) -> None:
    expected = {
        "record_type": "header",
        "schema_version": SCHEMA_VERSION,
        "box_format": BOX_FORMAT,
        "angle_unit": ANGLE_UNIT,
        "polygon_format": "x1y1x2y2x3y3x4y4",
    }
    for field, value in expected.items():
        if header.get(field) != value:
            raise ValueError(f"invalid header {field}: {header.get(field)!r}")
    if int(header.get("num_images", -1)) < 0:
        raise ValueError("header num_images must be non-negative")
    if int(header.get("num_predictions", -1)) < 0:
        raise ValueError("header num_predictions must be non-negative")
    if not isinstance(header.get("contract"), Mapping):
        raise ValueError("header contract must be a mapping")


def encode_jsonl_gzip(header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> bytes:
    validate_header(header)
    for record in records:
        validate_record(record)
    raw = ("\n".join([canonical_json(header), *(canonical_json(r) for r in records)]) + "\n").encode(
        "utf-8"
    )
    sink = io.BytesIO()
    # mtime=0 makes the compressed artifact byte-identical across repeated runs.
    with gzip.GzipFile(fileobj=sink, mode="wb", mtime=0, filename="") as stream:
        stream.write(raw)
    return sink.getvalue()


def write_artifact(
    output_path: Path | str,
    header: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(output_path)
    if path.suffix != ".gz":
        raise ValueError("prediction artifact path must end in .gz")
    payload = encode_jsonl_gzip(header, records)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": digest,
        "bytes": len(payload),
        "num_images": int(header["num_images"]),
        "num_predictions": int(header["num_predictions"]),
    }
    sidecar_path = path.with_name(path.name + ".sha256.json")
    sidecar_tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
    sidecar_tmp.write_text(canonical_json(sidecar) + "\n", encoding="utf-8")
    sidecar_tmp.replace(sidecar_path)
    return sidecar


def read_and_validate(output_path: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(output_path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip()]
    if not lines:
        raise ValueError("empty prediction artifact")
    header = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    validate_header(header)
    for record in records:
        validate_record(record)
    if header["num_images"] != len(records):
        raise ValueError("header num_images does not match record count")
    if header["num_predictions"] != sum(len(r["scores"]) for r in records):
        raise ValueError("header num_predictions does not match record contents")
    indices = [r["dataset_index"] for r in records]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("records must have unique, ascending dataset_index values")
    return header, records

