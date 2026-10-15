#!/usr/bin/env python3
"""Exact, no-padding batching for DETR assignment costs kept on device."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import torch


AssignmentSolver = Callable[[torch.Tensor], torch.Tensor]


def torch_linear_assignment_solver(costs: torch.Tensor) -> torch.Tensor:
    """Load the optional CUDA extension only when the backend is selected."""
    try:
        from torch_linear_assignment import batch_linear_assignment
    except ImportError as error:
        raise RuntimeError(
            "torch-linear-assignment is not installed; build the pinned 0.0.6 source "
            "in a separate CUDA environment with NVCC matching PyTorch"
        ) from error
    return batch_linear_assignment(costs)


def solve_concatenated_costs(
    costs: torch.Tensor,
    target_sizes: Sequence[int],
    solver: AssignmentSolver = torch_linear_assignment_solver,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve upstream ``[B,Q,sum(T_i)]`` costs by exact-size cohorts.

    Each image uses only its own target slice.  Images are batched together
    only when they have the same target count, so no padded/fictitious target
    can enter the optimization problem.  The solver returns one target index
    per query, using -1 for an unmatched query.
    """
    if costs.ndim != 3:
        raise ValueError(f"costs must have shape [B,Q,sum(T_i)], got {tuple(costs.shape)}")
    sizes = [int(value) for value in target_sizes]
    if len(sizes) != costs.shape[0] or any(value < 0 for value in sizes):
        raise ValueError("target_sizes must contain one non-negative value per batch item")
    if sum(sizes) != costs.shape[2]:
        raise ValueError(f"sum(target_sizes)={sum(sizes)} != cost width {costs.shape[2]}")
    if torch.isinf(costs).any():
        raise ValueError("costs contain +/-inf; exact assignment is undefined by this contract")
    costs = torch.nan_to_num(costs, nan=1.0)

    per_image: list[torch.Tensor] = []
    offset = 0
    for image_index, size in enumerate(sizes):
        per_image.append(costs[image_index, :, offset : offset + size])
        offset += size

    grouped: dict[int, list[int]] = defaultdict(list)
    for image_index, size in enumerate(sizes):
        grouped[size].append(image_index)

    output: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(sizes)
    query_count = costs.shape[1]
    for size, image_indices in grouped.items():
        if size == 0 or query_count == 0:
            empty = torch.empty(0, dtype=torch.int64, device=costs.device)
            for image_index in image_indices:
                output[image_index] = (empty.clone(), empty.clone())
            continue
        cohort = torch.stack([per_image[index] for index in image_indices], dim=0).contiguous()
        assignment = solver(cohort)
        if assignment.shape != (len(image_indices), query_count):
            raise ValueError(
                f"solver returned {tuple(assignment.shape)}, expected {(len(image_indices), query_count)}"
            )
        assignment = assignment.to(device=costs.device, dtype=torch.int64)
        for cohort_index, image_index in enumerate(image_indices):
            columns = assignment[cohort_index]
            mask = columns >= 0
            rows = torch.arange(query_count, device=costs.device, dtype=torch.int64)[mask]
            columns = columns[mask]
            expected = min(query_count, size)
            if rows.numel() != expected:
                raise ValueError(f"image {image_index}: solver returned {rows.numel()} of {expected} matches")
            if columns.numel() and (columns.min() < 0 or columns.max() >= size):
                raise ValueError(f"image {image_index}: solver returned a target outside [0,{size})")
            if torch.unique(columns).numel() != columns.numel():
                raise ValueError(f"image {image_index}: solver assignment is not one-to-one")
            output[image_index] = (rows, columns)
    assert all(item is not None for item in output)
    return [item for item in output if item is not None]
