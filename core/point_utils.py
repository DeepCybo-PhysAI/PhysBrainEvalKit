import ast
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .final_point_metrics import (
    aggregate_final_metrics,
    attach_score_fields,
    score_sample,
)


PARSE_POINTS = "points"
PARSE_EXPLICIT_NULL = "explicit_null"
PARSE_INVALID = "invalid"
PARSE_TRUNCATED_THINK = "truncated_think"


@dataclass(frozen=True)
class PointParseResult:
    points: List[List[float]]
    status: str
    answer_text: str


def _extract_final_answer(output: str) -> Tuple[str, bool]:
    """Return evaluator-visible answer text and whether an open think was truncated."""
    text = str(output)
    close_matches = list(re.finditer(r"</think\s*>", text, re.IGNORECASE))
    if close_matches:
        text = text[close_matches[-1].end():]
    else:
        open_matches = list(re.finditer(r"<think(?:\s[^>]*)?>", text, re.IGNORECASE))
        if open_matches:
            return "", True

    answer_matches = list(
        re.finditer(r"<answer(?:\s[^>]*)?>(.*?)</answer\s*>", text, re.DOTALL | re.IGNORECASE)
    )
    if answer_matches:
        text = answer_matches[-1].group(1)
    return text.strip(), False


def _markdown_candidates(text: str) -> List[str]:
    pattern = re.compile(
        r"```[ \t]*(?:json|python|javascript|js)?[ \t]*\r?\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    return [match.group(1).strip() for match in pattern.finditer(text)]


def _parse_serialized_points(text: str) -> Optional[List[List[float]]]:
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError, MemoryError):
            continue
        if data == []:
            return []
        points = _parse_structured_data(data)
        if points:
            return points
    return None


def clean_model_answer(output: str) -> Tuple[str, str]:
    """Return final answer text plus its parsing status for non-point consumers."""
    answer_text, truncated = _extract_final_answer(str(output))
    if truncated:
        return "", PARSE_TRUNCATED_THINK
    return answer_text, PARSE_INVALID if not answer_text else "answer"


def parse_point_output(output: str) -> PointParseResult:
    """Parse only the final answer, preserving why an output contained no points."""
    if not isinstance(output, str):
        return PointParseResult([], PARSE_INVALID, "")

    answer_text, truncated = _extract_final_answer(output)
    if truncated:
        return PointParseResult([], PARSE_TRUNCATED_THINK, "")
    if not answer_text:
        return PointParseResult([], PARSE_INVALID, answer_text)

    candidates = list(reversed(_markdown_candidates(answer_text))) + [answer_text]
    for candidate in candidates:
        serialized = _parse_serialized_points(candidate)
        if serialized is not None:
            status = PARSE_POINTS if serialized else PARSE_EXPLICIT_NULL
            return PointParseResult(serialized, status, candidate)

    if re.fullmatch(r"\s*(?:no[\s_-]*object|none|null)\s*[.!]?\s*", answer_text, re.IGNORECASE):
        return PointParseResult([], PARSE_EXPLICIT_NULL, answer_text)

    xml_points = _extract_from_xml_attributes(answer_text)
    if xml_points:
        return PointParseResult(xml_points, PARSE_POINTS, answer_text)

    text = _preprocess_text(answer_text)
    points = _extract_points_by_regex(text)
    if points:
        return PointParseResult(points, PARSE_POINTS, answer_text)
    return PointParseResult([], PARSE_INVALID, answer_text)


def omni_decode_points(output: str) -> List[List[float]]:
    """Backward-compatible point-only decoder."""
    return parse_point_output(output).points

def _preprocess_text(text: str) -> str:
    """Removes markdown wrappers and extracts content from within XML-style tags."""
    # Remove Markdown blocks
    text = re.sub(r'```(?:json|python|html)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    
    # Extract content from <point> or <points> tags if they exist
    tag_match = re.search(r'<(?:point|points)>(.*?)</(?:point|points)>', text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1)
        
    return text.strip()

def _parse_structured_data(data: Any) -> List[List[float]]:
    """Recursively traverses Python objects to find lists/dicts representing points."""
    points = []
    
    if isinstance(data, dict):
        # Handle Qwen/VLM specific keys
        for key in ["point_2d", "points", "point", "coordinates"]:
            if key in data:
                return _parse_structured_data(data[key])
                
    elif isinstance(data, (list, tuple)):
        if not data:
            return []
            
        # Check if it's a flat point: [x, y]
        if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            return [[float(data[0]), float(data[1])]]
        
        # Check if it's a nested structure: [[x, y], ...] or [{"point_2d": [x, y]}, ...]
        for item in data:
            extracted = _parse_structured_data(item)
            if extracted:
                points.extend(extracted)
                
    return points


def _extract_from_xml_attributes(text):
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    return all_points


def _extract_points_by_regex(text: str) -> List[List[float]]:
    """Regex to find coordinate pairs in loosely formatted text."""
    points = []
    # Pattern for [x, y] or (x, y)
    bracket_pattern = r'[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]'
    matches = re.findall(bracket_pattern, text)
    
    if matches:
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    else:
        # Last resort: look for "Number, Number" patterns in the text
        # Only if no bracketed points were found to avoid duplicates
        raw_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        matches = re.findall(raw_pattern, text)
        for m in matches:
            points.append([float(m[0]), float(m[1])])
            
    return points


def convert_points_to_pixels(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
    backbone: str,
) -> List[List[int]]:
    """Convert a backbone's coordinate protocol to bounded absolute pixels."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    converted: List[List[int]] = []
    for point in points:
        if len(point) != 2:
            continue
        try:
            first, second = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(first) or not math.isfinite(second):
            continue

        if backbone in {"gemma4", "qwen3_5", "qwen3", "gemini-2.5"}:
            if not (0.0 <= first <= 1000.0 and 0.0 <= second <= 1000.0):
                continue
            x = min(width - 1, int(first / 1000.0 * width))
            y = min(height - 1, int(second / 1000.0 * height))
        elif backbone == "gemini_robotics":
            if not (0.0 <= first <= 1000.0 and 0.0 <= second <= 1000.0):
                continue
            y = min(height - 1, int(first / 1000.0 * height))
            x = min(width - 1, int(second / 1000.0 * width))
        elif backbone == "molmo":
            if not (0.0 <= first <= 100.0 and 0.0 <= second <= 100.0):
                continue
            x = min(width - 1, int(first / 100.0 * width))
            y = min(height - 1, int(second / 100.0 * height))
        elif backbone in {"gpt", "pelican", "internvl", "magma"}:
            if not (0.0 <= first <= 1.0 and 0.0 <= second <= 1.0):
                continue
            x = min(width - 1, int(first * width))
            y = min(height - 1, int(second * height))
        elif backbone in {"qwen2.5", "qwen2_5", "mimo"}:
            if not (0.0 <= first < width and 0.0 <= second < height):
                continue
            x, y = int(first), int(second)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        converted.append([x, y])
    return converted


def _mask_array(mask: Any) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = np.max(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape}")
    return array


def validate_mask_size(mask: Any, width: int, height: int) -> None:
    array = _mask_array(mask)
    if array.shape != (height, width):
        raise ValueError(
            f"Mask size {(array.shape[1], array.shape[0])} "
            f"does not match image size {(width, height)}"
        )


def distance_first_mask_matching(
    points: Sequence[Sequence[float]],
    gt_points: Sequence[Sequence[float]],
    masks: Sequence[Any],
) -> int:
    """Assign by minimum point distance, then test the assigned instance masks."""
    if len(gt_points) != len(masks):
        raise ValueError(
            f"Expected one mask per GT point, got {len(gt_points)} points and {len(masks)} masks"
        )
    if not points or not gt_points:
        return 0

    from .framework_integration import linear_sum_assignment_compat

    predicted = np.asarray(points, dtype=float)
    ground_truth = np.asarray(gt_points, dtype=float)
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError(f"Expected predicted points with shape (N, 2), got {predicted.shape}")
    if ground_truth.ndim != 2 or ground_truth.shape[1] != 2:
        raise ValueError(f"Expected GT points with shape (N, 2), got {ground_truth.shape}")

    distances = np.linalg.norm(
        predicted[:, np.newaxis, :] - ground_truth[np.newaxis, :, :], axis=2
    )
    predicted_indices, gt_indices = linear_sum_assignment_compat(distances)
    mask_arrays = [_mask_array(mask) for mask in masks]
    matched = 0
    for predicted_idx, gt_idx in zip(predicted_indices, gt_indices):
        x, y = (int(value) for value in points[int(predicted_idx)])
        mask = mask_arrays[int(gt_idx)]
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
            matched += 1
    return matched


def point_counts(
    *,
    predicted_count: int,
    matched_count: int,
    gt_count: int,
    parse_status: str,
    strict_single: bool = False,
    has_no_object: bool = False,
) -> Dict[str, int]:
    """Convert one sample into protocol TP/FP/FN counts."""
    if has_no_object:
        if parse_status == PARSE_EXPLICIT_NULL:
            return {"tp": 1, "fp": 0, "fn": 0}
        return {"tp": 0, "fp": predicted_count, "fn": 1}

    if parse_status == PARSE_EXPLICIT_NULL:
        return {"tp": 0, "fp": 1, "fn": gt_count}
    if strict_single and predicted_count != 1:
        return {"tp": 0, "fp": predicted_count, "fn": gt_count}
    tp = min(matched_count, predicted_count, gt_count)
    return {"tp": tp, "fp": predicted_count - tp, "fn": gt_count - tp}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_point_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate point results, preferring the final 2026-08-17 protocol.

    The legacy branch remains for older saved rows that only contain the former
    non_strict_tp/fp/fn fields. New rows carry the final protocol's separate
    non-strict precision and recall sufficient statistics.
    """
    if not results or "non_strict_precision_hits" in results[0]:
        statistics = aggregate_final_metrics(results)
        statistics.update(
            {
                "overall_accuracy": statistics["strict_micro_f1"],
                "average_accuracy": statistics["strict_macro_f1"],
                "strict_overall_score": statistics["strict_micro_f1"],
                "non_strict_overall_score": statistics["non_strict_micro_f1"],
            }
        )
        return statistics

    statistics: Dict[str, Any] = {}
    for protocol in ("non_strict", "strict"):
        triples = [
            (
                int(result[f"{protocol}_tp"]),
                int(result[f"{protocol}_fp"]),
                int(result[f"{protocol}_fn"]),
            )
            for result in results
        ]
        tp = sum(item[0] for item in triples)
        fp = sum(item[1] for item in triples)
        fn = sum(item[2] for item in triples)
        statistics[f"{protocol}_micro_precision"] = _safe_ratio(tp, tp + fp)
        statistics[f"{protocol}_micro_recall"] = _safe_ratio(tp, tp + fn)
        statistics[f"{protocol}_micro_f1"] = _safe_ratio(2 * tp, 2 * tp + fp + fn)

        sample_precision = [_safe_ratio(a, a + b) for a, b, _ in triples]
        sample_recall = [_safe_ratio(a, a + c) for a, _, c in triples]
        sample_f1 = [_safe_ratio(2 * a, 2 * a + b + c) for a, b, c in triples]
        count = len(triples)
        statistics[f"{protocol}_macro_precision"] = _safe_ratio(sum(sample_precision), count)
        statistics[f"{protocol}_macro_recall"] = _safe_ratio(sum(sample_recall), count)
        statistics[f"{protocol}_macro_f1"] = _safe_ratio(sum(sample_f1), count)
        statistics[f"{protocol}_tp"] = tp
        statistics[f"{protocol}_fp"] = fp
        statistics[f"{protocol}_fn"] = fn
    statistics["total_samples"] = len(results)
    return statistics


def score_single_mask_sample(
    raw_output: str,
    *,
    mask: Any,
    width: int,
    height: int,
    backbone: str,
    strict_single: bool = True,
) -> Dict[str, Any]:
    """Parse, convert and score one target mask under the final protocol.

    strict_single remains in the signature for source compatibility but no
    longer changes scoring. Final strict uses TP/FP/FN target counts, while
    final non-strict precision retains every raw point hit.
    """
    parsed = parse_point_output(str(raw_output))
    pixels = convert_points_to_pixels(parsed.points, width, height, backbone)
    validate_mask_size(mask, width, height)
    hits, _ = check_points_in_mask(pixels, mask)
    matched = min(hits, 1)
    predicted_count = len(parsed.points)
    score = score_sample(
        predicted_count=predicted_count,
        matched_count=matched,
        raw_hit_count=hits,
        gt_count=1,
        parse_status=parsed.status,
    )
    return attach_score_fields(
        {
            "answer_text": parsed.answer_text,
            "parse_status": parsed.status,
            "parsed_points": parsed.points,
            "processed_points": pixels,
            "points_in_mask": hits,
            "total_points": predicted_count,
            "accuracy_score": _safe_ratio(hits, predicted_count),
        },
        score,
    )


def check_points_in_mask(points: List[List[float]], mask: Image.Image) -> Tuple[int, int]:
    """
    Check if points (absolute pixel coordinates) are in the mask foreground.
    
    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        mask: PIL Image mask where foreground pixels have value > 0
        
    Returns:
        Tuple[int, int]: (number of points in mask, total number of points)
    """
    if not points or mask is None:
        return 0, 0

    mask_array = _mask_array(mask)
    height, width = mask_array.shape
    
    points_in_mask = 0
    total_points = len(points)
    
    for point in points:
        x_pixel = int(round(point[0]))
        y_pixel = int(round(point[1]))
        
        # Check if coordinates are within image bounds
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            # mask_array is in [y, x] format
            if mask_array[y_pixel, x_pixel] > 0:
                points_in_mask += 1
    
    return points_in_mask, total_points


def check_masks_coverage(points: List[List[float]], masks: List[Image.Image]) -> Tuple[int, int]:
    """
    Check how many masks have at least one point in them.

    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        masks: List of PIL Image masks where foreground pixels have value > 0

    Returns:
        Tuple[int, int]: (number of masks with at least one point, total number of masks)
    """
    if not points or not masks:
        return 0, len(masks) if masks else 0

    total_masks = len(masks)
    masks_with_points = 0

    for mask in masks:
        # Convert to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')

        width, height = mask.size
        mask_array = np.array(mask)

        # Check if any point is in this mask
        has_point = False
        for point in points:
            x_pixel = int(round(point[0]))
            y_pixel = int(round(point[1]))

            # Check if coordinates are within image bounds
            if 0 <= x_pixel < width and 0 <= y_pixel < height:
                # mask_array is in [y, x] format
                if mask_array[y_pixel, x_pixel] > 0:
                    has_point = True
                    break  # Found at least one point in this mask, move to next mask

        if has_point:
            masks_with_points += 1

    return masks_with_points, total_masks


def check_points_in_bbox(points: List[List[float]], bbox: List[float]) -> Tuple[int, int]:
    """
    Check if points (absolute pixel coordinates) are within a bounding box.
    
    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        bbox: Bounding box in format [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right
        
    Returns:
        Tuple[int, int]: (number of points in bbox, total number of points)
    """
    if not points or bbox is None or len(bbox) < 4:
        return 0, 0
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    points_in_bbox = 0
    total_points = len(points)
    
    for point in points:
        x_pixel = int(round(point[0]))
        y_pixel = int(round(point[1]))
        
        # Check if point is within bounding box
        if x1 <= x_pixel <= x2 and y1 <= y_pixel <= y2:
            points_in_bbox += 1
    
    return points_in_bbox, total_points


if __name__ == "__main__":
    test_cases = [
        '[{"point_2d": [[100, 200]], "label": "eye"}]',        # Qwen-style
        '<point x="63.5" y="44.5">Mountain</point><point x="63.8" y="44.5">Mountain</point>',           # Tag attributes
        '```json\n[[10, 20], [30, 40]]\n```',                  # Markdown
        'The center is at (500, 500) and (789, 1000).',                        # Natural language
        '<points>[123, 456]</points>',
        '<points>[[122, 333], [222, 333]]</points>',           # Custom tags
        'point: 12.5, 13.5',                                   # Lazy labeling
    ]
    
    for case in test_cases:
        print(f"Input: {case}")
        print(f"Parsed: {omni_decode_points(case)}\n")
