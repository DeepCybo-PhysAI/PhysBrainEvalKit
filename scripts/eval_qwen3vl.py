#!/usr/bin/env python3
"""Persistent, sample-sharded Hugging Face evaluation runner.

Each process owns one resident model. All workers execute one benchmark at a
time, infer disjoint interleaved sample shards, and synchronize through files.
Worker 0 restores original order, computes metrics once, and exports the normal
EmbodiedEvalKit result bundle.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import logging
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from benchmark_registry import (
    POINT_BENCHMARKS,
    POINT_PROTOCOL,
    BenchmarkSpec,
    build_specs,
    common_cli,
    effective_cli_value,
)


LOG = logging.getLogger("eval_qwen3vl")
_BENCHMARK_ARGS = argparse.Namespace(mmsi_temperature=0.7)
ORIGINAL_PROMPT_SOURCE_COMMIT = "5ec3a60e03894ea5b9215127b47afaf39d969dae"
PROMPT_SWITCH_BENCHMARKS = POINT_BENCHMARKS | {"BLINK", "CV-Bench"}
ORIGINAL_INPUT_BENCHMARKS = frozenset(
    {
        "ERQA",
        "RoboSpatial",
        "EgoPlan2",
        "SAT",
        "Where2Place",
        "RefSpatial-Bench",
        "Part-Affordance-2K",
        "ShareRobot-Trajectory",
        "VABench-Visual-Trace",
        "VABench-Point",
        "Pixmo-Points",
        "RoboAfford",
        "PIOBench",
        "RoboRefit",
        "BLINK",
        "CV-Bench",
        "VSI-Bench",
        "EmbSpatial",
        "PointBench",
        "COSMOS",
        "RoboVQA",
        "VLABench",
    }
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def touch_atomic(path: Path, text: str = "ok\n") -> None:
    atomic_write_text(path, text)


def configure_cpu_runtime(cpu_threads: int) -> None:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cpu_threads)
    for name in ("OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ[name] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"

    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    try:
        import cv2

        cv2.setNumThreads(1)
        opencv_threads: int | str = cv2.getNumThreads()
    except ImportError:
        opencv_threads = "unavailable"
    LOG.info(
        "CPU pools: torch_intra=%d torch_interop=%d opencv=%s affinity=%d",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        opencv_threads,
        len(os.sched_getaffinity(0)),
    )


def setup_logging(worker_id: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker {worker_id:02d}] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_backbone(model_path: Path) -> str:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    if model_type.startswith("qwen3_vl"):
        return "qwen3"
    if model_type.startswith("qwen3_5"):
        return "qwen3_5"
    if model_type.startswith("gemma4"):
        return "gemma4"
    raise ValueError(f"unsupported persistent HF model_type: {model_type}")


def split_names(value: str) -> set[str]:
    return {item for item in value.replace(",", " ").split() if item}


def select_specs(specs: list[BenchmarkSpec], only: str, skip: str) -> list[BenchmarkSpec]:
    available = {spec.name for spec in specs}
    only_names = split_names(only)
    skip_names = split_names(skip)
    recognized_exclusions = {"Ego3D-Bench", "OpenEQA", "PIO-S3-Verified"}
    unknown = (only_names | skip_names) - available - recognized_exclusions
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(sorted(unknown))}")
    if only_names & recognized_exclusions:
        names = ", ".join(sorted(only_names & recognized_exclusions))
        raise ValueError(f"persistent non-judge plan does not include: {names}")
    return [
        spec
        for spec in specs
        if (not only_names or spec.name in only_names) and spec.name not in skip_names
    ]


def _replace_cli_option(argv: list[str], option: str, value: str) -> list[str]:
    result: list[str] = []
    replaced = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} is missing its value")
            if not replaced:
                result.extend([option, value])
                replaced = True
            index += 2
            continue
        result.append(token)
        index += 1
    if not replaced:
        result.extend([option, value])
    return result


def _remove_cli_option(argv: list[str], option: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} is missing its value")
            index += 2
            continue
        result.append(token)
        index += 1
    return result


def effective_extra_args(spec: BenchmarkSpec, prompt_policy: str) -> list[str]:
    extra = list(spec.extra_args)
    if prompt_policy == "original" and spec.name == "RoboVQA":
        extra = _replace_cli_option(extra, "--prompt_policy", "raw")
        extra = _remove_cli_option(extra, "--system_prompt")
    return extra


def benchmark_cli(
    spec: BenchmarkSpec,
    model_name: str,
    model_path: Path,
    backbone: str,
    prompt_policy: str,
) -> list[str]:
    argv = [
        *common_cli(model_name, model_path, backbone),
        *effective_extra_args(spec, prompt_policy),
    ]
    if spec.name == "MMSI-Bench":
        argv.extend(["--temperature", str(getattr(_BENCHMARK_ARGS, "mmsi_temperature", 0.7)), "--top_p", "0.8", "--top_k", "20"])
    if spec.name in PROMPT_SWITCH_BENCHMARKS:
        argv.extend(["--prompt-policy", prompt_policy])
    return argv


def effective_input_policy(spec: BenchmarkSpec, prompt_policy: str) -> str:
    if prompt_policy == "internal":
        return "internal"
    if spec.name in ORIGINAL_INPUT_BENCHMARKS:
        return "original"
    return "internal_no_original_reference"


def validate_cli_options(
    project_root: Path,
    spec: BenchmarkSpec,
    model_name: str,
    model_path: Path,
    backbone: str,
    debug: bool,
    prompt_policy: str,
) -> list[str]:
    tree = ast.parse((project_root / spec.script).read_text(encoding="utf-8"))
    declared: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    declared.add(argument.value)
    argv = benchmark_cli(spec, model_name, model_path, backbone, prompt_policy)
    if debug:
        argv.append("--debug")
    return sorted({token for token in argv if token.startswith("--")} - declared)


def build_manifest(args: argparse.Namespace, specs: list[BenchmarkSpec], backbone: str) -> dict[str, Any]:
    return {
        "format": 1,
        "model_path": str(args.model_path),
        "model_name": args.model_name,
        "model_config_sha256": sha256_file(args.model_path / "config.json"),
        "model_index_sha256": sha256_file(args.model_path / "model.safetensors.index.json"),
        "backbone": backbone,
        "backend": "hf",
        "world_size": args.world_size,
        "models_per_gpu": args.models_per_gpu,
        "cpu_threads_per_worker": args.cpu_threads,
        "gpu_layout": args.gpu_layout,
        "point_protocol": POINT_PROTOCOL,
        "prompt_policy": args.prompt_policy,
        "original_prompt_source_commit": ORIGINAL_PROMPT_SOURCE_COMMIT,
        "debug": args.debug,
        "mmsi_num_samples": args.mmsi_num_samples,
        "mmsi_seed": args.mmsi_seed,
        "mmsi_temperature": args.mmsi_temperature,
        "common_cli": common_cli(args.model_name, args.model_path, backbone),
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "registry": sha256_file(Path(__file__).with_name("benchmark_registry.py")),
            "hf_engine": sha256_file(args.project_root / "core" / "hf_engine.py"),
            "point_metrics": sha256_file(args.project_root / "core" / "final_point_metrics.py"),
            "prompt_policy": sha256_file(args.project_root / "benchmark" / "prompt_policy.py"),
        },
        "benchmarks": [
            {
                "name": spec.name,
                "script": spec.script,
                "dataset": spec.dataset,
                "split": spec.split,
                "result_json": spec.result_json,
                "extra_args": effective_extra_args(spec, args.prompt_policy),
                "input_policy": effective_input_policy(spec, args.prompt_policy),
            }
            for spec in specs
        ],
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def failure_files(args: argparse.Namespace) -> list[Path]:
    failure_dir = args.run_state / "attempts" / args.attempt_id / "failures"
    return sorted(failure_dir.glob("*.json")) if failure_dir.exists() else []


def raise_if_failed(args: argparse.Namespace) -> None:
    failures = failure_files(args)
    if not failures:
        return
    details = []
    for path in failures[:4]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            details.append(f"{path.name}: {payload.get('error', 'unknown error')}")
        except Exception:
            details.append(path.name)
    raise RuntimeError("another worker failed: " + "; ".join(details))


def wait_for(
    predicate: Callable[[], bool],
    description: str,
    args: argparse.Namespace,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + args.barrier_timeout
    while not predicate():
        raise_if_failed(args)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(poll_seconds)


def ensure_manifest(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    path = args.run_state / "manifest.json"
    initialized = args.run_state / "manifest.ready"
    if args.worker_id == 0:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise RuntimeError(
                    f"run-id {args.run_id!r} belongs to a different model/configuration; "
                    "use a new --run-id"
                )
        else:
            atomic_write_json(path, manifest)
        touch_atomic(initialized, manifest_digest(manifest) + "\n")
    else:
        wait_for(initialized.is_file, "worker 0 to validate the run manifest", args)
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("persistent run manifest differs across workers")
    return manifest_digest(manifest)


class PersistentShardedEngineMixin:
    def configure_persistence(
        self,
        args: argparse.Namespace,
        manifest_sha256: str,
    ) -> None:
        self.persistent_args = args
        self.worker_id = args.worker_id
        self.world_size = args.world_size
        self.manifest_sha256 = manifest_sha256
        self.benchmark_state = args.run_state
        self.benchmark_name = "startup"
        self.call_index = 0

    def begin_benchmark(self, name: str, benchmark_state: Path) -> None:
        self.benchmark_name = name
        self.benchmark_state = benchmark_state
        self.call_index = 0

    def batch_inference(self, prepared_dataset: list[dict[str, Any]], **kwargs: Any) -> list[str]:
        call_number = self.call_index
        self.call_index += 1
        call_dir = self.benchmark_state / "calls" / f"call-{call_number:04d}"
        shard_dir = call_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        total_samples = len(prepared_dataset)
        indices = list(range(self.worker_id, total_samples, self.world_size))
        shard_path = shard_dir / f"worker-{self.worker_id:02d}.json"
        local_outputs: list[str] | None = None
        if shard_path.is_file():
            try:
                cached = json.loads(shard_path.read_text(encoding="utf-8"))
                candidate = cached.get("outputs")
                if (
                    cached.get("manifest_sha256") == self.manifest_sha256
                    and cached.get("benchmark") == self.benchmark_name
                    and cached.get("call") == call_number
                    and cached.get("worker_id") == self.worker_id
                    and cached.get("world_size") == self.world_size
                    and cached.get("total_samples") == total_samples
                    and cached.get("indices") == indices
                    and isinstance(candidate, list)
                    and len(candidate) == len(indices)
                ):
                    local_outputs = [str(output) for output in candidate]
                    LOG.info(
                        "[%s] call=%d worker=%02d reused=%d",
                        self.benchmark_name,
                        call_number,
                        self.worker_id,
                        len(local_outputs),
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                LOG.warning("ignoring invalid shard %s: %s", shard_path, exc)

        if local_outputs is None:
            local_dataset = [prepared_dataset[index] for index in indices]
            LOG.info(
                "[%s] call=%d worker=%02d samples=%d/%d",
                self.benchmark_name,
                call_number,
                self.worker_id,
                len(local_dataset),
                total_samples,
            )
            if local_dataset:
                sample_count = int(os.environ.get("MMSI_NUM_SAMPLES", "1")) if self.benchmark_name == "MMSI-Bench" else 1
                sample_seed = int(os.environ.get("MMSI_SEED", "3407"))
                local_outputs = []
                for draw in range(sample_count):
                    if sample_count > 1:
                        from accelerate.utils import set_seed
                        set_seed(sample_seed + draw)
                    outputs = super().batch_inference(local_dataset, **kwargs)
                    if len(outputs) != len(indices):
                        raise RuntimeError(f"worker {self.worker_id} produced {len(outputs)} outputs for {len(indices)} assigned samples")
                    if sample_count == 1:
                        local_outputs = [str(output) for output in outputs]
                    else:
                        if not local_outputs:
                            local_outputs = [[] for _ in indices]
                        for row, output in zip(local_outputs, outputs):
                            row.append(str(output))
            else:
                local_outputs = []
            if len(local_outputs) != len(indices):
                raise RuntimeError(
                    f"worker {self.worker_id} produced {len(local_outputs)} outputs "
                    f"for {len(indices)} assigned samples"
                )
            atomic_write_json(
                shard_path,
                {
                    "manifest_sha256": self.manifest_sha256,
                    "benchmark": self.benchmark_name,
                    "call": call_number,
                    "worker_id": self.worker_id,
                    "world_size": self.world_size,
                    "total_samples": total_samples,
                    "indices": indices,
                    "outputs": local_outputs,
                },
            )

        if self.worker_id != 0:
            return local_outputs

        expected = [shard_dir / f"worker-{worker:02d}.json" for worker in range(self.world_size)]
        wait_for(
            lambda: all(path.is_file() for path in expected),
            f"all shards for {self.benchmark_name} call {call_number}",
            self.persistent_args,
        )
        merged: list[Any] = [None] * total_samples
        seen: set[int] = set()
        for path in expected:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("manifest_sha256") != self.manifest_sha256:
                raise RuntimeError(f"manifest mismatch in {path}")
            if payload.get("total_samples") != total_samples:
                raise RuntimeError(f"dataset length mismatch in {path}")
            shard_indices = payload.get("indices", [])
            outputs = payload.get("outputs", [])
            if len(shard_indices) != len(outputs):
                raise RuntimeError(f"index/output length mismatch in {path}")
            for index, output in zip(shard_indices, outputs):
                if index in seen or not 0 <= index < total_samples:
                    raise RuntimeError(f"invalid or duplicate index {index} in {path}")
                seen.add(index)
                merged[index] = output
        missing = [index for index, output in enumerate(merged) if output is None]
        if missing:
            raise RuntimeError(f"missing {len(missing)} outputs; first indices: {missing[:10]}")
        final_outputs = merged
        empty_count = sum((not output) or (isinstance(output, str) and not output.strip()) for output in final_outputs)
        if final_outputs and empty_count == len(final_outputs):
            raise RuntimeError(f"all outputs are empty for {self.benchmark_name} call {call_number}")
        atomic_write_json(
            call_dir / "merged_outputs.json",
            {"total_samples": total_samples, "outputs": final_outputs, "empty_count": empty_count},
        )
        LOG.info(
            "[%s] call=%d merged=%d empty=%d",
            self.benchmark_name,
            call_number,
            len(final_outputs),
            empty_count,
        )
        return final_outputs


def bundle_complete(output_base: Path, spec: BenchmarkSpec) -> bool:
    bundle = output_base / spec.name
    meta = bundle / "meta_result.json"
    samples = bundle / "all_samples.json"
    if not (meta.is_file() and meta.stat().st_size > 0 and samples.is_file() and samples.stat().st_size > 0):
        return False
    if spec.name in POINT_BENCHMARKS:
        return POINT_PROTOCOL in meta.read_text(encoding="utf-8", errors="replace")
    return True


def run_entry(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    backbone: str,
    engine: Any,
    benchmark_state: Path,
) -> None:
    argv = [
        spec.script,
        *benchmark_cli(
            spec,
            args.model_name,
            args.model_path,
            backbone,
            args.prompt_policy,
        ),
    ]
    if args.debug:
        argv.append("--debug")
    old_argv = sys.argv[:]
    engine.begin_benchmark(spec.name, benchmark_state)
    try:
        sys.argv = argv
        runpy.run_path(str(args.project_root / spec.script), run_name="__main__")
    finally:
        sys.argv = old_argv


def export_bundle(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    backbone: str,
    log_file: Path,
    started: str,
    ended: str,
) -> None:
    cli = benchmark_cli(
        spec,
        args.model_name,
        args.model_path,
        backbone,
        args.prompt_policy,
    )
    if args.debug:
        cli.append("--debug")
    result_json = args.project_root / spec.result_json
    if not result_json.is_file() or result_json.stat().st_size == 0:
        raise FileNotFoundError(f"expected result JSON not found: {result_json}")
    inference_params = {
        "backend": "hf",
        "backbone": backbone,
        "dtype": "bfloat16",
        "max_tokens": int(effective_cli_value(cli, "--max_tokens", "1024")),
        "temperature": float(effective_cli_value(cli, "--temperature", "0.0")),
        "top_p": float(effective_cli_value(cli, "--top_p", "1.0")),
        "top_k": int(effective_cli_value(cli, "--top_k", "-1")),
        "repetition_penalty": float(effective_cli_value(cli, "--repetition_penalty", "1.05")),
        "presence_penalty": float(effective_cli_value(cli, "--presence_penalty", "0.0")),
        "persistent_workers": args.world_size,
        "models_per_gpu": args.models_per_gpu,
        "cpu_threads_per_worker": args.cpu_threads,
        "scheduler": "qwen3vl_hf",
        "point_protocol": POINT_PROTOCOL,
        "requested_prompt_policy": args.prompt_policy,
        "input_policy": effective_input_policy(spec, args.prompt_policy),
        "original_prompt_source_commit": ORIGINAL_PROMPT_SOURCE_COMMIT,
    }
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts" / "export_benchmark_result_bundle.py"),
            "--benchmark", spec.name,
            "--output_dir", str(args.output_base / spec.name),
            "--result_json", str(result_json),
            "--log_file", str(log_file),
            "--model_name", args.model_name,
            "--model_path", str(args.model_path),
            "--entry_script", spec.script,
            "--dataset", spec.dataset,
            "--split", spec.split,
            "--start_time", started,
            "--end_time", ended,
            "--command", " ".join([spec.script, *cli]),
            "--inference_params_json", json.dumps(inference_params, ensure_ascii=False),
        ],
        cwd=args.project_root,
        check=True,
    )


def record_failure(args: argparse.Namespace, benchmark: str, error: BaseException) -> None:
    path = args.run_state / "attempts" / args.attempt_id / "failures" / f"worker-{args.worker_id:02d}.json"
    atomic_write_json(
        path,
        {
            "worker_id": args.worker_id,
            "benchmark": benchmark,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "time": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--models-per-gpu", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-layout", default="0")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--run-id", default="dry-run")
    parser.add_argument("--attempt-id", default="attempt-0")
    parser.add_argument("--barrier-timeout", type=int, default=172800)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip", default="Ego3D-Bench")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prompt-policy",
        choices=["internal", "original"],
        default="internal",
        help=(
            "Use original EmbodiedEvalKit inputs where available; internal-only "
            "benchmarks retain their current inputs. Parsing and metrics are unchanged."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mmsi-num-samples", type=int, default=1)
    parser.add_argument("--mmsi-seed", type=int, default=3407)
    parser.add_argument("--mmsi-temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.model_path = args.model_path.resolve()
    args.output_base = args.output_base.resolve()
    args.run_state = args.output_base / ".persistent_scheduler" / "runs" / args.run_id
    return args


def main() -> int:
    args = parse_args()
    global _BENCHMARK_ARGS
    _BENCHMARK_ARGS = args
    os.environ["MMSI_NUM_SAMPLES"] = str(args.mmsi_num_samples)
    os.environ["MMSI_SEED"] = str(args.mmsi_seed)
    backbone = infer_backbone(args.model_path)
    specs = select_specs(build_specs(args.project_root, args.model_name), args.only, args.skip)
    if not specs:
        raise SystemExit("no benchmarks selected")
    invalid = {
        spec.name: validate_cli_options(
            args.project_root,
            spec,
            args.model_name,
            args.model_path,
            backbone,
            args.debug,
            args.prompt_policy,
        )
        for spec in specs
    }
    invalid = {name: options for name, options in invalid.items() if options}
    if invalid:
        for name, options in invalid.items():
            print(f"ERROR {name}: undeclared options: {', '.join(options)}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"model={args.model_path}")
        print(f"backbone={backbone}")
        print(f"worker_count={args.world_size}")
        print(f"models_per_gpu={args.models_per_gpu}")
        print(f"benchmark_count={len(specs)}")
        for index, spec in enumerate(specs, 1):
            print(f"{index:02d} {spec.name}: {spec.script} -> {args.output_base / spec.name}")
        print("cli_option_validation=ok")
        return 0

    if not 0 <= args.worker_id < args.world_size:
        raise SystemExit("worker-id must be in [0, world-size)")
    gpu_layout = [int(item) for item in args.gpu_layout.split(",") if item]
    if len(gpu_layout) != args.world_size or gpu_layout[args.worker_id] != args.gpu:
        raise SystemExit("gpu-layout must map every worker to its declared GPU")
    for path in (args.project_root, args.model_path, args.output_base):
        if not path.exists():
            raise SystemExit(f"required path does not exist: {path}")

    os.chdir(args.project_root)
    sys.path.insert(0, str(args.project_root))
    setup_logging(args.worker_id)
    configure_cpu_runtime(args.cpu_threads)
    manifest = build_manifest(args, specs, backbone)

    try:
        manifest_sha256 = ensure_manifest(args, manifest)
        from core.hf_engine import HFInferenceEngine
        import core.inference as inference_module

        class PersistentHFInferenceEngine(PersistentShardedEngineMixin, HFInferenceEngine):
            pass

        max_model_len = max(
            int(
                effective_cli_value(
                    [*common_cli(args.model_name, args.model_path, backbone), *spec.extra_args],
                    "--max_model_len",
                    "8192",
                )
            )
            for spec in specs
        )
        LOG.info(
            "loading resident model model=%s backbone=%s gpu=%d slot=%d max_model_len=%d",
            args.model_path,
            backbone,
            args.gpu,
            args.slot,
            max_model_len,
        )
        engine = PersistentHFInferenceEngine(
            model_path=str(args.model_path),
            model_name=args.model_name,
            backbone=backbone,
            dtype="bfloat16",
            max_model_len=max_model_len,
            seed=3407,
        )
        engine.configure_persistence(args, manifest_sha256)

        def cached_factory(entry_args: Any, model_path: str, model_name: str) -> tuple[Any, str]:
            if Path(model_path).resolve() != args.model_path:
                raise ValueError(f"persistent runner refuses a different model path: {model_path}")
            if getattr(entry_args, "backbone", backbone) != backbone:
                raise ValueError("entry point requested a different backbone")
            engine.max_model_len = getattr(entry_args, "max_model_len", engine.max_model_len)
            return engine, "hf"

        inference_module.create_inference_engine = cached_factory
        startup = args.run_state / "attempts" / args.attempt_id / "startup"
        ready = startup / f"slot-{args.slot}" / f"worker-{args.worker_id:02d}.ready"
        touch_atomic(
            ready,
            json.dumps(
                {"worker": args.worker_id, "gpu": args.gpu, "slot": args.slot, "time": utc_now()}
            )
            + "\n",
        )
        LOG.info("resident model ready")
        wait_for((startup / "all-workers.ready").is_file, "all resident models", args)

        # Set only after Accelerator/model initialization. Entry points use it
        # solely to suppress duplicate metric computation on nonzero workers.
        os.environ["RANK"] = str(args.worker_id)
        for index, spec in enumerate(specs):
            if args.resume and bundle_complete(args.output_base, spec):
                LOG.info("[%s] resume: complete current-protocol bundle exists", spec.name)
                continue

            stable_state = args.run_state / "benchmarks" / f"{index:02d}-{safe_name(spec.name)}"
            sync_state = (
                args.run_state
                / "attempts"
                / args.attempt_id
                / "benchmarks"
                / f"{index:02d}-{safe_name(spec.name)}"
            )
            initialized = sync_state / "initialized"
            result_json = args.project_root / spec.result_json
            log_file = args.output_base / spec.name / "run.log"
            if args.worker_id == 0:
                stable_state.mkdir(parents=True, exist_ok=True)
                sync_state.mkdir(parents=True, exist_ok=True)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                if result_json.exists():
                    result_json.unlink()
                touch_atomic(sync_state / "start_time", utc_now() + "\n")
                touch_atomic(initialized)
            else:
                wait_for(initialized.is_file, f"worker 0 to initialize {spec.name}", args)

            started = (sync_state / "start_time").read_text(encoding="utf-8").strip()
            handler: logging.Handler | None = None
            if args.worker_id == 0:
                handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
                logging.getLogger().addHandler(handler)
            try:
                LOG.info("[%s] starting", spec.name)
                run_entry(args, spec, backbone, engine, stable_state)
                LOG.info("[%s] local entry finished", spec.name)
                done_dir = sync_state / "done"
                touch_atomic(done_dir / f"worker-{args.worker_id:02d}.done", utc_now() + "\n")
                release = sync_state / "release"
                if args.worker_id == 0:
                    expected = [
                        done_dir / f"worker-{worker:02d}.done" for worker in range(args.world_size)
                    ]
                    wait_for(lambda: all(path.is_file() for path in expected), f"all workers to finish {spec.name}", args)
                    ended = utc_now()
                    export_bundle(args, spec, backbone, log_file, started, ended)
                    touch_atomic(release, ended + "\n")
                else:
                    wait_for(release.is_file, f"worker 0 to export {spec.name}", args)
                LOG.info("[%s] exported and released", spec.name)
            finally:
                if handler is not None:
                    handler.flush()
                    logging.getLogger().removeHandler(handler)
                    handler.close()
            gc.collect()

        if args.worker_id == 0:
            subprocess.run(
                [
                    sys.executable,
                    str(args.project_root / "scripts" / "summarize_benchmark_scores.py"),
                    str(args.output_base),
                    "--output",
                    str(args.output_base / "summary.txt"),
                ],
                cwd=args.project_root,
                check=True,
            )
            touch_atomic(args.run_state / "COMPLETE", utc_now() + "\n")
            LOG.info("all selected benchmarks complete")
        return 0
    except BaseException as error:
        LOG.exception("worker failed")
        record_failure(args, getattr(locals().get("spec", None), "name", "startup"), error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
