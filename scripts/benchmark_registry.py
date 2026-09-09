"""Canonical non-judge benchmark plan for persistent Qwen3-VL evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


POINT_PROTOCOL = "physbrain_point_metrics_final_20260817"
POINT_BENCHMARKS = frozenset(
    {
        "Pixmo-Points",
        "PointBench",
        "Part-Affordance-2K",
        "RoboRefit",
        "RoboSpatial",
        "VABench-Point",
        "PIOBench",
        "Where2Place",
        "RefSpatial-Bench",
        "RoboAfford",
    }
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    script: str
    dataset: str
    split: str
    result_json: str
    extra_args: tuple[str, ...]


def build_specs(project_root: Path, model_name: str) -> list[BenchmarkSpec]:
    """Return the current 28-benchmark plan (non-judge, excluding Ego3D)."""
    data = project_root / "datasets"
    robovqa_root = Path(__import__("os").environ.get("ROBOVQA_DATA_ROOT", str(data / "RoboVQA-16frames")))
    vlabench_root = Path(__import__("os").environ.get("VLABENCH_DATASET_PATH", str(data / "VLABench" / "vlm_evaluation_v1.0")))

    def spec(
        name: str,
        script: str,
        dataset: str,
        split: str,
        result_json: str,
        *extra_args: str,
    ) -> BenchmarkSpec:
        return BenchmarkSpec(name, script, dataset, split, result_json, tuple(extra_args))

    return [
        spec("ERQA", "eval_erqa.py", "FlagEval/ERQA", "test", f"logs/results/ERQA_{model_name}.json", "--max_model_len", "18000"),
        spec("RoboSpatial", "eval_robospatial.py", "chanhee-luke/RoboSpatial-Home", "context+compatibility+configuration", f"logs/results/RoboSpatial_{model_name}.json", "--max_model_len", "10240", "--max_tokens", "4096"),
        spec("EgoPlan2", "eval_egoplan2.py", "IffYuan/ego-plan", "train", f"logs/results/EgoPlan2_{model_name}.json", "--max_model_len", "10240"),
        spec("SAT", "eval_sat.py", "FlagEval/SAT", "default/test", f"logs/results/SAT_{model_name}.json", "--max_model_len", "25000"),
        spec("Where2Place", "eval_where2place.py", "FlagEval/Where2Place", "test", f"logs/results/Where2Place_{model_name}.json", "--max_model_len", "10240", "--max_tokens", "4096"),
        spec("RefSpatial-Bench", "eval_refspatial.py", "BAAI/RefSpatial-Bench", "location+placement+unseen", f"logs/results/RefSpatial-Bench_{model_name}_all.json", "--max_model_len", "10240"),
        spec("Part-Affordance-2K", "eval_partafford.py", "IffYuan/Part-Affordance-2K", "train", f"logs/results/Part-Affordance-2K_{model_name}.json", "--max_model_len", "10240", "--max_tokens", "4096"),
        spec("ShareRobot-Trajectory", "eval_sharerobot_v.py", "IffYuan/sharerobot_trajectory", "train", f"logs/results/sharerobot-v_{model_name}.json", "--max_model_len", "20000"),
        spec("VABench-Visual-Trace", "eval_vabench_v.py", "IffYuan/vabench-v", "train", f"logs/results/VABench-v_{model_name}.json", "--max_model_len", "16000"),
        spec("Q-Spatial-Bench", "eval_qspatial.py", "andrewliao11/Q-Spatial-Bench", "QSpatial_plus", f"logs/results/Q-Spatial-Bench_{model_name}_QSpatial_plus.json", "--max_model_len", "10240", "--split", "QSpatial_plus", "--success_delta", "2.0"),
        spec("VABench-Point", "eval_vabench_point.py", "IffYuan/VABench-P", "test", f"logs/results/VABench-P_{model_name}.json", "--max_model_len", "10240", "--max_tokens", "4096"),
        spec("Pixmo-Points", "eval_pixmo_points.py", "IffYuan/pixmo-points-eval", "train", f"logs/results/PixmoPoints_{model_name}.json", "--max_model_len", "10240", "--max_tokens", "4096"),
        spec("RoboAfford", "eval_roboafford.py", "Zray26/roboafford-eval", "test", f"logs/results/RoboAfford_{model_name}.json", "--max_model_len", "10240"),
        spec("PIOBench", "eval_pio.py", "IffYuan/PIO-Bench", "train", f"logs/results/PIO-Bench_{model_name}.json", "--max_model_len", "8196", "--max_tokens", "4096"),
        spec("RoboRefit", "eval_roborefit.py", "IffYuan/Roborefit", "test", f"logs/results/RoboRefit_{model_name}.json", "--max_model_len", "10240"),
        spec("BLINK", "eval_blink.py", "BLINK-Benchmark/BLINK", "Counting+Relative_Depth+Spatial_Relation/val", f"logs/results/BLINK-Bench_{model_name}.json", "--max_model_len", "10240"),
        spec("CV-Bench", "eval_cvbench.py", "nyu-visionx/CV-Bench", "default/test", f"logs/results/CV-Bench_{model_name}.json", "--max_model_len", "10240"),
        spec("VSI-Bench", "eval_vsi_bench.py", "IffYuan/vsi-bench", "train", f"logs/results/VSI-Bench_{model_name}_results.json", "--max_model_len", "24000"),
        spec("EmbSpatial", "eval_embspatial.py", "FlagEval/EmbSpatial-Bench", "test", f"logs/results/EmbSpatial-Bench_{model_name}.json", "--max_model_len", "10240"),
        spec("PointBench", "eval_pointbench.py", "IffYuan/PointBench", "train", f"logs/results/PointBench_{model_name}.json", "--max_model_len", "20000", "--max_tokens", "4096"),
        spec("COSMOS", "eval_cosmos.py", "IffYuan/COSMOS", "train", f"logs/results/COSMOS_{model_name}.json", "--max_model_len", "20000"),
        spec(
            "RoboVQA", "eval_robovqa.py", str(robovqa_root), "local-16frames/train_explicit_style", f"logs/results/RoboVQA_{model_name}.json",
            "--data_root", str(robovqa_root), "--system_prompt", "You are a helpful assistant.", "--expected_num_frames", "16", "--max_model_len", "10240", "--max_images_per_prompt", "16", "--max_tokens", "128", "--temperature", "0.0", "--top_p", "1.0", "--top_k", "-1", "--repetition_penalty", "1.05", "--presence_penalty", "0.0",
        ),
        spec(
            "VLABench", "eval_vlabench.py", "VLABench/vlm_evaluation_v1.0", "local", f"logs/results/VLABench_{model_name}_results.json",
            "--max_model_len", "10240", "--max_images_per_prompt", "2", "--max_tokens", "512", "--dataset_path", str(vlabench_root), "--chunk_size", "100",
        ),
        spec("ERQA-PLUS", "eval_erqa_plus.py", str(data / "erqa-plus"), "train", f"logs/results/ERQA-PLUS_{model_name}.json", "--dataset_name", str(data / "erqa-plus"), "--split", "train", "--max_model_len", "20000", "--max_images_per_prompt", "16", "--max_tokens", "128"),
        spec("3DSRBench", "eval_3dsrbench.py", str(data / "3DSRBench" / "3DSRBench" / "3dsrbench_v1_vlmevalkit_circular.tsv"), "test", f"logs/results/3DSRBench_{model_name}.json", "--dataset_name", str(data / "3DSRBench" / "3DSRBench" / "3dsrbench_v1_vlmevalkit_circular.tsv"), "--max_model_len", "10240", "--max_images_per_prompt", "1", "--max_tokens", "128"),
        spec("ViewSpatial", "eval_viewspatial.py", str(data / "ViewSpatial_lmmseval"), "test", f"logs/results/ViewSpatial_{model_name}.json", "--dataset_name", str(data / "ViewSpatial_lmmseval"), "--max_model_len", "20000", "--max_images_per_prompt", "16", "--max_tokens", "128"),
        spec("MindCube", "eval_mindcube.py", str(data / "MindCube"), "tinybench", f"logs/results/MindCube_{model_name}.json", "--dataset_name", str(data / "MindCube"), "--split", "tinybench", "--max_model_len", "20000", "--max_images_per_prompt", "4", "--max_tokens", "128"),
        spec("MMSI-Bench", "eval_mmsi_bench.py", str(data / "MMSI-Bench"), "test", f"logs/results/MMSI-Bench_{model_name}.json", "--dataset_name", str(data / "MMSI-Bench"), "--split", "test", "--max_model_len", "20000", "--max_images_per_prompt", "10", "--max_tokens", "128"),
    ]


def common_cli(model_name: str, model_path: Path, backbone: str) -> list[str]:
    return [
        "--model_name", model_name,
        "--model_path", str(model_path),
        "--backbone", backbone,
        "--backend", "hf",
        "--dtype", "bfloat16",
        "--max_tokens", "1024",
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--top_k", "-1",
        "--repetition_penalty", "1.05",
        "--presence_penalty", "0.0",
    ]


def effective_cli_value(argv: list[str], option: str, default: str) -> str:
    value = default
    for index, token in enumerate(argv[:-1]):
        if token == option:
            value = argv[index + 1]
    return value
