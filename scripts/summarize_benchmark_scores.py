#!/usr/bin/env python
"""Summarize EmbodiedEvalKit benchmark result bundles.

Expected input layout:
  <base_dir>/<benchmark>/meta_result.json
  <base_dir>/<benchmark>/all_samples.json
  <base_dir>/<benchmark>/run.log
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRIMARY_METRIC_BY_BENCHMARK = {
    "3DSRBench": "overall_accuracy",
    "BLINK": "overall_accuracy",
    "COSMOS": "overall_score",
    "CV-Bench": "overall_accuracy",
    "ERQA": "overall_accuracy",
    "ERQA-PLUS": "overall_accuracy",
    "EgoPlan2": "overall_score",
    "Ego3D-Bench": "overall_accuracy",
    "EmbSpatial": "overall_accuracy",
    "MMSI-Bench": "overall_accuracy",
    "MindCube": "overall_accuracy",
    "OpenEQA": "overall_score",
    "PIOBench": "non_strict_micro_f1",
    "PIO-Bench": "non_strict_micro_f1",
    "Part-Affordance-2K": "non_strict_micro_f1",
    "PIO-S3-Verified": "overall_score",
    "Pixmo-Points": "non_strict_micro_f1",
    "PointBench": "non_strict_micro_f1",
    "Q-Spatial-Bench": "success_rate",
    "RefSpatial-Bench": "non_strict_micro_f1",
    "RoboAfford": "non_strict_micro_f1",
    "RoboRefit": "non_strict_micro_f1",
    "RoboSpatial": "non_strict_overall_score",
    "RoboSpatial_context": "non_strict_micro_f1",
    "RoboVQA": "overall_bleu",
    "SAT": "overall_accuracy",
    "ShareRobot-Trajectory": "normalized_rmse_score",
    "VABench-Point": "non_strict_micro_f1",
    "VABench-Visual-Trace": "normalized_rmse_score",
    "VLABench": "overall_score",
    "VSI-Bench": "overall_score",
    "ViewSpatial": "overall_accuracy",
    "Where2Place": "non_strict_micro_f1",
}

BENCHMARK_ORDER = [
    "3DSRBench",
    "BLINK",
    "COSMOS",
    "CV-Bench",
    "EgoPlan2",
    "Ego3D-Bench",
    "EmbSpatial",
    "ERQA",
    "ERQA-PLUS",
    "MMSI-Bench",
    "MindCube",
    "OpenEQA",
    "Part-Affordance-2K",
    "PIO-S3-Verified",
    "PIOBench",
    "Pixmo-Points",
    "PointBench",
    "Q-Spatial-Bench",
    "RefSpatial-Bench",
    "RoboAfford",
    "RoboRefit",
    "RoboSpatial",
    "RoboVQA",
    "SAT",
    "ShareRobot-Trajectory",
    "VABench-Point",
    "VABench-Visual-Trace",
    "VLABench",
    "VSI-Bench",
    "ViewSpatial",
    "Where2Place",
]

BENCHMARK_ORDER_ALIASES = {
    "Point-Bench": "PointBench",
}

FALLBACK_PRIMARY_KEYS = [
    "overall_accuracy",
    "overall_score",
    "success_rate",
    "overall_bleu",
    "average_accuracy",
    "normalized_rmse_score",
    "normalized_mae_score",
    "normalized_dfd_score",
    "avg_rmse",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_primary_metric(benchmark: str, metrics: dict[str, Any]) -> tuple[str | None, Any]:
    preferred = PRIMARY_METRIC_BY_BENCHMARK.get(benchmark)
    if preferred and preferred in metrics:
        return preferred, metrics.get(preferred)
    for key in FALLBACK_PRIMARY_KEYS:
        if key in metrics:
            return key, metrics.get(key)
    numeric = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric:
        return numeric[0]
    return None, None


def format_value(value: Any, as_percent: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if as_percent:
            return f"{value * 100:.2f}"
        if abs(value) >= 10:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def is_fraction_metric(metric_name: str | None, value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if metric_name in {"overall_bleu", "avg_rmse", "avg_mae", "avg_dfd"}:
        return False
    return 0 <= value <= 1


def collect_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for meta_path in sorted(base_dir.glob("*/meta_result.json")):
        meta = load_json(meta_path)
        benchmark = meta.get("benchmark") or meta_path.parent.name
        metrics = meta.get("metrics") or {}
        metric_name, score = pick_primary_metric(benchmark, metrics)
        strict_overall = metrics.get(
            "strict_overall_score", metrics.get("strict_micro_f1")
        )
        non_strict_overall = metrics.get(
            "non_strict_overall_score", metrics.get("non_strict_micro_f1")
        )
        num_samples = meta.get("num_samples") or metrics.get("total_samples")
        valid_samples = metrics.get("valid_samples")
        note = ""
        if valid_samples == 0:
            note = "valid_samples=0"
        elif score is None:
            note = "primary metric is null"

        rows.append(
            {
                "benchmark": benchmark,
                "score_field": metric_name or "",
                "score": score,
                "score_display": format_value(score, as_percent=is_fraction_metric(metric_name, score)),
                "score_unit": "%" if is_fraction_metric(metric_name, score) else "",
                "strict_overall": strict_overall,
                "strict_overall_display": format_value(
                    strict_overall,
                    as_percent=is_fraction_metric("strict_overall_score", strict_overall),
                ),
                "non_strict_overall": non_strict_overall,
                "non_strict_overall_display": format_value(
                    non_strict_overall,
                    as_percent=is_fraction_metric("non_strict_overall_score", non_strict_overall),
                ),
                "num_samples": num_samples,
                "valid_samples": valid_samples,
                "dataset": meta.get("dataset"),
                "split": meta.get("split"),
                "model_name": meta.get("model_name"),
                "meta_path": str(meta_path),
                "note": note,
            }
        )
    order_index = {name: idx for idx, name in enumerate(BENCHMARK_ORDER)}

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        benchmark = str(row["benchmark"])
        canonical = BENCHMARK_ORDER_ALIASES.get(benchmark, benchmark)
        return (order_index.get(canonical, len(order_index)), benchmark)

    return sorted(rows, key=sort_key)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    headers = ["Benchmark", "Score", "Field", "Strict Overall", "Non-strict Overall", "Samples", "Valid", "Note"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        score = row["score_display"]
        if row["score_unit"]:
            score = f"{score}{row['score_unit']}"
        print(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    score,
                    str(row["score_field"]),
                    (
                        f"{row['strict_overall_display']}%"
                        if row["strict_overall"] is not None
                        else "-"
                    ),
                    (
                        f"{row['non_strict_overall_display']}%"
                        if row["non_strict_overall"] is not None
                        else "-"
                    ),
                    str(row["num_samples"] if row["num_samples"] is not None else "-"),
                    str(row["valid_samples"] if row["valid_samples"] is not None else "-"),
                    str(row["note"]),
                ]
            )
            + " |"
        )


def print_tsv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark", "score_field", "score", "score_display", "score_unit",
        "strict_overall", "non_strict_overall", "num_samples", "valid_samples", "note",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path, help="Directory containing per-benchmark result folders.")
    parser.add_argument("--format", choices=["markdown", "tsv", "json"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(args.base_dir)

    rows = collect_rows(args.base_dir)
    if not rows:
        raise FileNotFoundError(f"No */meta_result.json files found under {args.base_dir}")

    if args.output:
        if args.format == "json":
            args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.format == "tsv":
            with args.output.open("w", encoding="utf-8", newline="") as f:
                fields = [
                    "benchmark", "score_field", "score", "score_display", "score_unit",
                    "strict_overall", "non_strict_overall", "num_samples", "valid_samples", "note",
                ]
                writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        else:
            from io import StringIO

            buf = StringIO()
            headers = ["Benchmark", "Score", "Field", "Strict Overall", "Non-strict Overall", "Samples", "Valid", "Note"]
            buf.write("| " + " | ".join(headers) + " |\n")
            buf.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                score = row["score_display"]
                if row["score_unit"]:
                    score = f"{score}{row['score_unit']}"
                buf.write(
                    "| "
                    + " | ".join(
                        [
                            str(row["benchmark"]),
                            score,
                            str(row["score_field"]),
                            (
                                f"{row['strict_overall_display']}%"
                                if row["strict_overall"] is not None
                                else "-"
                            ),
                            (
                                f"{row['non_strict_overall_display']}%"
                                if row["non_strict_overall"] is not None
                                else "-"
                            ),
                            str(row["num_samples"] if row["num_samples"] is not None else "-"),
                            str(row["valid_samples"] if row["valid_samples"] is not None else "-"),
                            str(row["note"]),
                        ]
                    )
                    + " |\n"
                )
            args.output.write_text(buf.getvalue(), encoding="utf-8")
        return

    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.format == "tsv":
        print_tsv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
