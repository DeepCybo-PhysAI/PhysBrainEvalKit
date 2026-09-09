# Agent Instructions for PhysBrainEvalKit

This file is the operational guide for coding agents (including Codex and Claude Code) working in this repository. The goal is to configure a reproducible environment and run the public Qwen3-VL benchmark suite without relying on private project conventions.

## Scope and repository rules

- This repository evaluates Qwen3-VL-compatible Hugging Face models on spatial and embodied-intelligence benchmarks.
- Keep changes limited to evaluation code, documentation, tests, and reproducibility helpers.
- Never add model weights, dataset archives, generated results, runtime logs, API keys, or machine-specific absolute paths to the repository.
- Use environment variables for paths that differ between machines.
- Treat upstream datasets as governed by their own licenses and access requirements.
- Do not rewrite metric definitions or prompt formats unless the task explicitly requests a protocol change. Point-localization metrics are specified in `docs/final_point_metrics_protocol.md`.

## Environment setup

Run commands from the repository root. Use an existing compatible environment when available; otherwise create a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU evaluation requires a CUDA-compatible PyTorch installation. The multi-worker launcher also requires Bash, `taskset`, and `nvidia-smi`:

```bash
command -v python3
python3 --version
command -v bash taskset nvidia-smi
python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__)"
```

For a restricted or offline environment, set `HF_HOME` to a writable cache and pre-populate all required Hugging Face datasets and model files before starting the evaluation.

## Model requirements

The model path must be a local Hugging Face directory containing at least:

- `config.json`
- tokenizer and processor files
- model weight files

The runner recognizes Qwen3-VL model configurations. Verify the model before launching:

```bash
export MODEL_PATH=/path/to/qwen3_vl_model
test -f "$MODEL_PATH/config.json"
python - <<'PY'
import json, os
p = os.environ["MODEL_PATH"]
with open(os.path.join(p, "config.json")) as f:
    print(json.load(f).get("model_type"))
PY
```

Use an official or otherwise publicly redistributable Qwen3-VL checkpoint. Do not assume that a checkpoint from another model family is compatible.

## Dataset configuration

Most benchmarks load from Hugging Face Datasets using the IDs and splits in `scripts/benchmark_registry.py`. Set these variables when local files are required:

```bash
export ROBOVQA_DATA_ROOT=/path/to/RoboVQA-16frames
export VLABENCH_DATASET_PATH=/path/to/VLABench/vlm_evaluation_v1.0
export ROBOREFIT_DATA_ROOT=/path/to/RoboRefit-corrected
export EGO3DBENCH_IMAGE_ROOT=/path/to/Ego3D-Bench/images
```

Before a full run, confirm that every local path needed by the selected benchmarks exists. If only part of the data is available, use `--only` to select supported benchmarks.

## Standard evaluation workflow

1. Inspect the benchmark plan and validate command-line options without loading samples:

   ```bash
   bash scripts/eval_qwen3vl.sh \
     --model-path "$MODEL_PATH" \
     --model-name qwen3-vl-public \
     --output-base /path/to/results/qwen3-vl-public \
     --gpus 0,1 \
     --models-per-gpu 2 \
     --cpu-per-worker 4 \
     --prompt-policy original \
     --dry-run
   ```

2. Review the printed benchmark list. Use `--only NAME1,NAME2` or `--skip NAME` when datasets are unavailable. The default plan contains 28 non-judge benchmarks and does not run API-judge workloads.

3. Start the real evaluation by removing `--dry-run` and adding `--resume`:

   ```bash
   bash scripts/eval_qwen3vl.sh \
     --model-path "$MODEL_PATH" \
     --model-name qwen3-vl-public \
     --output-base /path/to/results/qwen3-vl-public \
     --gpus 0,1 \
     --models-per-gpu 2 \
     --cpu-per-worker 4 \
     --prompt-policy original \
     --resume
   ```

4. Keep the same `--run-id` when restarting an interrupted run and shard reuse is desired. A new run ID starts a new shard namespace.

5. Inspect the output directory for per-benchmark summaries and raw predictions. Aggregate scores with:

   ```bash
   python scripts/summarize_benchmark_scores.py --help
   ```

The launcher keeps one Hugging Face model process resident per worker and divides samples into interleaved shards. Set `--models-per-gpu` conservatively for larger models. Ensure `world_size * cpu_per_worker` fits the process CPU affinity budget.

## Running one benchmark

For focused debugging or a dataset-specific run, invoke an entry point directly from the repository root:

```bash
python eval_pointbench.py \
  --model_path "$MODEL_PATH" \
  --model_name qwen3-vl-public \
  --backend hf \
  --debug
```

Every `eval_*.py` entry point supports `--help`. Prefer a small `--debug` run before a long evaluation when changing code or dataset configuration.

## Validation after code changes

Run syntax checks and the relevant tests after modifying code:

```bash
python -m compileall -q benchmark core eval_*.py scripts
python -m unittest discover -s tests -v
```

If a test or benchmark cannot run because data, GPU, or an optional dependency is unavailable, report the exact missing prerequisite and preserve the generated dry-run output. Do not silently substitute a different dataset, model, metric, or prompt policy.

## Troubleshooting

- `--model-path must contain config.json`: pass the Hugging Face model directory, not a training-run parent directory.
- Dataset or file-not-found errors: set the corresponding environment variable and verify permissions and filenames.
- CUDA out-of-memory: reduce `--models-per-gpu`, use fewer GPUs/workers, or select a smaller Qwen3-VL checkpoint.
- CPU affinity errors: reduce `--cpu-per-worker` or the number of workers.
- Resume does not reuse shards: restart with the original `--run-id` and unchanged model, plan, and worker layout.
- Offline download errors: populate `HF_HOME` and all dataset caches before launching.

When reporting results, include the exact model identifier, benchmark selection, command-line arguments, dataset revision or source, software versions, and output directory. Avoid exposing credentials or private filesystem paths.
