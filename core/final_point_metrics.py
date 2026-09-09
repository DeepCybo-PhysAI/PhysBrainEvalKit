#!/usr/bin/env python3
"""Final point-localization metrics for the PhysBrain evaluation kit.

Benchmark adapters first reduce each sample to four counts: predicted targets,
ground-truth targets, matched targets, and raw point hits.  This module turns
that evidence into the sufficient statistics used by the final protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


PROTOCOL_NAME = "physbrain_point_metrics_final_20260817"
PARSE_POINTS = "points"
PARSE_EXPLICIT_NULL = "explicit_null"
PARSE_INVALID = "invalid"
PARSE_TRUNCATED_THINK = "truncated_think"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _harmonic(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class FinalSampleScore:
    """Sufficient statistics for one sample under the final protocol."""

    strict_tp: int
    strict_fp: int
    strict_fn: int
    non_strict_precision_hits: int
    non_strict_precision_predicted_points: int
    non_strict_recall_tp: int
    non_strict_recall_fn: int
    predicted_count: int
    gt_count: int
    matched_count: int
    raw_hit_count: int
    parse_status: str
    has_no_object: bool
    exact_count_required: bool
    exact_count_match: bool | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_sample(
    *,
    predicted_count: int,
    matched_count: int,
    raw_hit_count: int,
    gt_count: int,
    parse_status: str = PARSE_POINTS,
    has_no_object: bool = False,
    exact_count_required: bool = False,
) -> FinalSampleScore:
    """Convert sample evidence into final strict/non-strict statistics.

    ``matched_count`` measures covered GT targets and supplies recall.
    ``raw_hit_count`` counts every prediction in an acceptable region and
    supplies non-strict precision.  PointBench counting enables the exact-count
    gate, which zeros both matched targets and precision hits on a count error.
    """
    predicted_count = int(predicted_count)
    matched_count = int(matched_count)
    raw_hit_count = int(raw_hit_count)
    gt_count = int(gt_count)
    if min(predicted_count, matched_count, raw_hit_count, gt_count) < 0:
        raise ValueError("counts must be non-negative")
    if matched_count > min(predicted_count, gt_count):
        raise ValueError("matched_count cannot exceed predicted_count or gt_count")
    if raw_hit_count > predicted_count:
        raise ValueError("raw_hit_count cannot exceed predicted_count")
    if has_no_object and gt_count != 1:
        raise ValueError("no-object samples use one sentinel GT target")

    exact_count_match = None
    precision_hits = raw_hit_count

    if has_no_object:
        if parse_status == PARSE_EXPLICIT_NULL:
            tp, fp, fn = 1, 0, 0
        else:
            tp, fp, fn = 0, predicted_count, 1
        precision_hits = 0
    elif exact_count_required:
        exact_count_match = predicted_count == gt_count
        if not exact_count_match:
            tp, fp, fn = 0, predicted_count, gt_count
            precision_hits = 0
        else:
            tp = min(matched_count, predicted_count, gt_count)
            fp = predicted_count - tp
            fn = gt_count - tp
    elif parse_status == PARSE_EXPLICIT_NULL:
        tp, fp, fn = 0, 1, gt_count
        precision_hits = 0
    else:
        tp = min(matched_count, predicted_count, gt_count)
        fp = predicted_count - tp
        fn = gt_count - tp

    return FinalSampleScore(
        strict_tp=tp,
        strict_fp=fp,
        strict_fn=fn,
        non_strict_precision_hits=precision_hits,
        non_strict_precision_predicted_points=predicted_count,
        non_strict_recall_tp=tp,
        non_strict_recall_fn=fn,
        predicted_count=predicted_count,
        gt_count=gt_count,
        matched_count=matched_count,
        raw_hit_count=raw_hit_count,
        parse_status=parse_status,
        has_no_object=bool(has_no_object),
        exact_count_required=bool(exact_count_required),
        exact_count_match=exact_count_match,
    )


def _value(record: FinalSampleScore | Mapping[str, Any], key: str) -> Any:
    if isinstance(record, FinalSampleScore):
        return getattr(record, key)
    return record[key]


def aggregate_final_metrics(
    records: Sequence[FinalSampleScore | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute final micro metrics plus diagnostic per-sample macro metrics."""
    strict_tp = sum(int(_value(record, "strict_tp")) for record in records)
    strict_fp = sum(int(_value(record, "strict_fp")) for record in records)
    strict_fn = sum(int(_value(record, "strict_fn")) for record in records)
    raw_hits = sum(
        int(_value(record, "non_strict_precision_hits")) for record in records
    )
    predicted = sum(
        int(_value(record, "non_strict_precision_predicted_points"))
        for record in records
    )

    strict_precision = _ratio(strict_tp, strict_tp + strict_fp)
    strict_recall = _ratio(strict_tp, strict_tp + strict_fn)
    strict_f1 = _ratio(2 * strict_tp, 2 * strict_tp + strict_fp + strict_fn)
    non_strict_precision = _ratio(raw_hits, predicted)
    non_strict_recall = strict_recall
    non_strict_f1 = _harmonic(non_strict_precision, non_strict_recall)

    strict_sample_precision = []
    strict_sample_recall = []
    strict_sample_f1 = []
    non_strict_sample_precision = []
    non_strict_sample_recall = []
    non_strict_sample_f1 = []
    for record in records:
        tp = int(_value(record, "strict_tp"))
        fp = int(_value(record, "strict_fp"))
        fn = int(_value(record, "strict_fn"))
        hits = int(_value(record, "non_strict_precision_hits"))
        pred = int(_value(record, "non_strict_precision_predicted_points"))
        sp = _ratio(tp, tp + fp)
        sr = _ratio(tp, tp + fn)
        sf = _ratio(2 * tp, 2 * tp + fp + fn)
        nsp = _ratio(hits, pred)
        nsr = sr
        nsf = _harmonic(nsp, nsr)
        strict_sample_precision.append(sp)
        strict_sample_recall.append(sr)
        strict_sample_f1.append(sf)
        non_strict_sample_precision.append(nsp)
        non_strict_sample_recall.append(nsr)
        non_strict_sample_f1.append(nsf)

    count = len(records)
    return {
        "protocol": PROTOCOL_NAME,
        "strict_micro_precision": strict_precision,
        "strict_micro_recall": strict_recall,
        "strict_micro_f1": strict_f1,
        "strict_macro_precision": _ratio(sum(strict_sample_precision), count),
        "strict_macro_recall": _ratio(sum(strict_sample_recall), count),
        "strict_macro_f1": _ratio(sum(strict_sample_f1), count),
        "strict_tp": strict_tp,
        "strict_fp": strict_fp,
        "strict_fn": strict_fn,
        "non_strict_micro_precision": non_strict_precision,
        "non_strict_micro_recall": non_strict_recall,
        "non_strict_micro_f1": non_strict_f1,
        "non_strict_macro_precision": _ratio(sum(non_strict_sample_precision), count),
        "non_strict_macro_recall": _ratio(sum(non_strict_sample_recall), count),
        "non_strict_macro_f1": _ratio(sum(non_strict_sample_f1), count),
        "non_strict_precision_hits": raw_hits,
        "non_strict_precision_predicted_points": predicted,
        "non_strict_recall_tp": strict_tp,
        "non_strict_recall_fn": strict_fn,
        "total_samples": count,
    }


def robospatial_overalls(
    *,
    context_records: Sequence[FinalSampleScore | Mapping[str, Any]],
    binary_correct: int,
    binary_total: int,
) -> Dict[str, Any]:
    """Mix context micro F1 with unchanged binary accuracy by sample count."""
    point_metrics = aggregate_final_metrics(context_records)
    context_total = len(context_records)
    total = context_total + int(binary_total)
    if not 0 <= int(binary_correct) <= int(binary_total):
        raise ValueError("binary_correct must be within [0, binary_total]")
    return {
        "strict_overall_score": _ratio(
            context_total * point_metrics["strict_micro_f1"] + binary_correct,
            total,
        ),
        "non_strict_overall_score": _ratio(
            context_total * point_metrics["non_strict_micro_f1"] + binary_correct,
            total,
        ),
        "point_metrics": point_metrics,
        "binary_correct": int(binary_correct),
        "binary_total": int(binary_total),
        "total_samples": total,
    }


def attach_score_fields(result: Dict[str, Any], score: FinalSampleScore) -> Dict[str, Any]:
    """Attach canonical final metric fields to an evaluator result row."""
    result.update(score.to_dict())
    return result
