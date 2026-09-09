#!/usr/bin/env python3
"""Qwen3 adapters for the final PhysBrain point-localization protocol."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .final_point_metrics import (
    PARSE_EXPLICIT_NULL,
    PARSE_INVALID,
    PARSE_POINTS,
    aggregate_final_metrics,
    attach_score_fields,
    robospatial_overalls,
    score_sample,
)


@dataclass(frozen=True)
class ParsedOursOutput:
    points: List[List[float]]
    status: str
    answer_text: str


def _strip_markdown_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:json|python)?\s*(.*?)\s*```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def parse_ours_qwen3_output(raw_output: str) -> ParsedOursOutput:
    """Parse only the standard Qwen3-VL final JSON output, never reasoning text."""
    if not isinstance(raw_output, str):
        return ParsedOursOutput([], PARSE_INVALID, "")
    answer = _strip_markdown_fence(raw_output)
    if re.fullmatch(r"(?:no[\s_-]*object|none|null)\s*[.!]?", answer, re.I):
        return ParsedOursOutput([], PARSE_EXPLICIT_NULL, answer)
    try:
        payload = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return ParsedOursOutput([], PARSE_INVALID, answer)
    if payload == []:
        return ParsedOursOutput([], PARSE_EXPLICIT_NULL, answer)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return ParsedOursOutput([], PARSE_INVALID, answer)

    points: List[List[float]] = []
    for item in payload:
        if not isinstance(item, dict) or "point_2d" not in item:
            return ParsedOursOutput([], PARSE_INVALID, answer)
        point = item["point_2d"]
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(value, (int, float)) for value in point)
        ):
            return ParsedOursOutput([], PARSE_INVALID, answer)
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return ParsedOursOutput([], PARSE_INVALID, answer)
        points.append([x, y])
    return ParsedOursOutput(points, PARSE_POINTS, answer)


def qwen3_points_to_pixels(
    points: Sequence[Sequence[float]], width: int, height: int
) -> List[List[int]]:
    """Convert Qwen3 [x,y] coordinates from 0..1000 using floor + clamp."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    pixels: List[List[int]] = []
    for point in points:
        if len(point) != 2:
            continue
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            continue
        pixels.append(
            [
                min(width - 1, int(x / 1000.0 * width)),
                min(height - 1, int(y / 1000.0 * height)),
            ]
        )
    return pixels


def _mask_array(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.max(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {array.shape}")
    return array


def validate_mask_size(mask: Any, width: int, height: int) -> None:
    array = _mask_array(mask)
    if array.shape != (height, width):
        raise ValueError(
            f"mask size {(array.shape[1], array.shape[0])} != image {(width, height)}"
        )


def count_points_in_mask(points: Sequence[Sequence[int]], mask: Any) -> int:
    array = _mask_array(mask)
    height, width = array.shape
    return sum(
        1
        for x, y in points
        if 0 <= int(x) < width
        and 0 <= int(y) < height
        and array[int(y), int(x)] > 0
    )


def count_points_in_any_mask(
    points: Sequence[Sequence[int]], masks: Sequence[Any]
) -> int:
    arrays = [_mask_array(mask) for mask in masks]
    hits = 0
    for x, y in points:
        x, y = int(x), int(y)
        if any(
            0 <= y < array.shape[0]
            and 0 <= x < array.shape[1]
            and array[y, x] > 0
            for array in arrays
        ):
            hits += 1
    return hits


def linear_sum_assignment_compat(cost_matrix: Any):
    """Use SciPy when available, with a deterministic Hungarian fallback."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost_matrix)

    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"expected a 2D cost matrix, got {cost.shape}")
    rows, columns = cost.shape
    if rows == 0 or columns == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    transposed = rows > columns
    if transposed:
        cost = cost.T
        rows, columns = cost.shape

    u = np.zeros(rows + 1, dtype=float)
    v = np.zeros(columns + 1, dtype=float)
    p = np.zeros(columns + 1, dtype=int)
    way = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_value = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    row_indices = []
    column_indices = []
    for column in range(1, columns + 1):
        if p[column] != 0:
            row_indices.append(p[column] - 1)
            column_indices.append(column - 1)
    if transposed:
        row_indices, column_indices = column_indices, row_indices
    order = np.argsort(row_indices)
    return (
        np.asarray(row_indices, dtype=int)[order],
        np.asarray(column_indices, dtype=int)[order],
    )


def distance_first_assigned_mask_matches(
    predicted_points: Sequence[Sequence[int]],
    gt_points: Sequence[Sequence[int]],
    masks: Sequence[Any],
) -> int:
    """PixMo matching: Hungarian distance assignment, then assigned-mask test."""
    if len(gt_points) != len(masks):
        raise ValueError("PixMo requires one instance mask per GT point")
    if not predicted_points or not gt_points:
        return 0
    predicted = np.asarray(predicted_points, dtype=float)
    ground_truth = np.asarray(gt_points, dtype=float)
    distances = np.linalg.norm(
        predicted[:, np.newaxis, :] - ground_truth[np.newaxis, :, :], axis=2
    )
    predicted_indices, gt_indices = linear_sum_assignment_compat(distances)
    arrays = [_mask_array(mask) for mask in masks]
    matched = 0
    for predicted_index, gt_index in zip(predicted_indices, gt_indices):
        x, y = (int(value) for value in predicted_points[int(predicted_index)])
        mask = arrays[int(gt_index)]
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            matched += 1
    return matched


def score_single_mask_output(
    raw_output: str, *, mask: Any, width: int, height: int
) -> Dict[str, Any]:
    """Score a standard Qwen3-VL output against one target mask."""
    parsed = parse_ours_qwen3_output(raw_output)
    pixels = qwen3_points_to_pixels(parsed.points, width, height)
    validate_mask_size(mask, width, height)
    raw_hits = count_points_in_mask(pixels, mask)
    score = score_sample(
        predicted_count=len(parsed.points),
        matched_count=min(raw_hits, 1),
        raw_hit_count=raw_hits,
        gt_count=1,
        parse_status=parsed.status,
    )
    return attach_score_fields(
        {
            "raw_output": raw_output,
            "answer_text": parsed.answer_text,
            "parse_status": parsed.status,
            "parsed_points": parsed.points,
            "processed_points": pixels,
            "points_in_mask": raw_hits,
            "total_points": len(parsed.points),
        },
        score,
    )


def score_pointbench_output(
    raw_output: str,
    *,
    mask: Any,
    width: int,
    height: int,
    category: str,
    expected_count: int,
) -> Dict[str, Any]:
    """Score PointBench, including the counting exact-count gate."""
    parsed = parse_ours_qwen3_output(raw_output)
    pixels = qwen3_points_to_pixels(parsed.points, width, height)
    validate_mask_size(mask, width, height)
    raw_hits = count_points_in_mask(pixels, mask)
    score = score_sample(
        predicted_count=len(parsed.points),
        matched_count=min(raw_hits, int(expected_count)),
        raw_hit_count=raw_hits,
        gt_count=int(expected_count),
        parse_status=parsed.status,
        exact_count_required=category == "counting",
    )
    return attach_score_fields(
        {
            "raw_output": raw_output,
            "answer_text": parsed.answer_text,
            "parse_status": parsed.status,
            "parsed_points": parsed.points,
            "processed_points": pixels,
            "points_in_mask": raw_hits,
            "total_points": len(parsed.points),
            "category": category,
            "expected_count": int(expected_count),
            "count_correct": len(parsed.points) == int(expected_count),
        },
        score,
    )


def score_pixmo_output(
    raw_output: str,
    *,
    masks: Sequence[Any],
    gt_points_pixels: Sequence[Sequence[int]],
    width: int,
    height: int,
    has_no_object: bool,
) -> Dict[str, Any]:
    """Score PixMo with instance recall matching and union-mask precision."""
    parsed = parse_ours_qwen3_output(raw_output)
    pixels = qwen3_points_to_pixels(parsed.points, width, height)
    for mask in masks:
        validate_mask_size(mask, width, height)
    if has_no_object:
        gt_count = 1
        matched = 0
        raw_hits = 0
    else:
        gt_count = len(masks)
        matched = distance_first_assigned_mask_matches(
            pixels, gt_points_pixels, masks
        )
        raw_hits = count_points_in_any_mask(pixels, masks)
    score = score_sample(
        predicted_count=len(parsed.points),
        matched_count=matched,
        raw_hit_count=raw_hits,
        gt_count=gt_count,
        parse_status=parsed.status,
        has_no_object=has_no_object,
    )
    return attach_score_fields(
        {
            "raw_output": raw_output,
            "answer_text": parsed.answer_text,
            "parse_status": parsed.status,
            "parsed_points": parsed.points,
            "processed_points": pixels,
            "masks_with_points": matched,
            "total_masks": gt_count,
            "points_in_any_gt_mask": raw_hits,
            "has_no_object": bool(has_no_object),
            "matching_protocol": "distance_first_hungarian_then_assigned_mask",
        },
        score,
    )


def point_benchmark_statistics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = aggregate_final_metrics(results)
    metrics.update(
        {
            "overall_accuracy": metrics["strict_micro_f1"],
            "average_accuracy": metrics["strict_macro_f1"],
            "strict_overall_score": metrics["strict_micro_f1"],
            "non_strict_overall_score": metrics["non_strict_micro_f1"],
        }
    )
    return metrics


def robospatial_statistics(
    context_results: Sequence[Dict[str, Any]],
    *,
    binary_correct: int,
    binary_total: int,
) -> Dict[str, Any]:
    mixed = robospatial_overalls(
        context_records=context_results,
        binary_correct=binary_correct,
        binary_total=binary_total,
    )
    return {
        **mixed["point_metrics"],
        **mixed,
        "overall_accuracy": mixed["strict_overall_score"],
    }
