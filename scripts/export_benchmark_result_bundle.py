#!/usr/bin/env python3
"""Export a benchmark result JSON file into the persistent runner bundle format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_result(path: Path) -> tuple[list[Any], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"result JSON not found or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result JSON must contain an object: {path}")

    samples = payload.get("results")
    metrics = payload.get("statistics")
    if not isinstance(samples, list):
        raise ValueError(f"result JSON is missing a list-valued 'results': {path}")
    if not isinstance(metrics, dict):
        raise ValueError(f"result JSON is missing an object-valued 'statistics': {path}")
    return samples, metrics


def parse_inference_params(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--inference_params_json must encode a JSON object")
    return parsed


def export_bundle(args: argparse.Namespace) -> tuple[Path, Path]:
    result_json = args.result_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    samples, metrics = load_result(result_json)
    inference_params = parse_inference_params(args.inference_params_json)

    all_samples_path = output_dir / "all_samples.json"
    meta_result_path = output_dir / "meta_result.json"
    atomic_write_json(all_samples_path, samples)
    atomic_write_json(
        meta_result_path,
        {
            "format": "physbrain_evalkit_result_bundle_v1",
            "benchmark": args.benchmark,
            "model_name": args.model_name,
            "model_path": str(args.model_path.expanduser().resolve()),
            "entry_script": args.entry_script,
            "dataset": args.dataset,
            "split": args.split,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "command": args.command,
            "result_json": str(result_json),
            "log_file": str(args.log_file.expanduser().resolve()),
            "num_samples": len(samples),
            "metrics": metrics,
            "inference_params": inference_params,
        },
    )
    return all_samples_path, meta_result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--result_json", type=Path, required=True)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--entry_script", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start_time", required=True)
    parser.add_argument("--end_time", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--inference_params_json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_bundle(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
