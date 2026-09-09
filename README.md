# PhysBrainEvalKit

PhysBrainEvalKit is an evaluation toolkit for spatial and embodied-intelligence benchmarks targeting vision-language models (VLMs). This clean release contains the benchmark adapters, a unified Hugging Face inference interface, the point-localization metrics protocol, and a resident-model sharded runner for community reproduction.

## 🛠️ Installation

```bash
cd /path/to/PhysBrainEvalKit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You may also use an existing PyTorch/Transformers environment. GPU evaluation requires a PyTorch build compatible with your CUDA version. The sharded launcher also requires Bash, `taskset`, and `nvidia-smi`.

Download a Qwen3-VL checkpoint in Hugging Face format (for example, an official Qwen3-VL-Instruct release) before running evaluation. The path passed to `--model-path` must contain `config.json`, the tokenizer files, and the model weight files.

## 📦 Data

Most benchmarks load automatically from Hugging Face Datasets. The benchmark names, dataset IDs, splits, and default arguments are listed in `scripts/benchmark_registry.py`. Some benchmarks require local files; prepare those datasets before running and set the relevant environment variables:

```bash
export ROBOVQA_DATA_ROOT=/path/to/RoboVQA-16frames
export VLABENCH_DATASET_PATH=/path/to/VLABench/vlm_evaluation_v1.0
export ROBOREFIT_DATA_ROOT=/path/to/RoboRefit-corrected
export EGO3DBENCH_IMAGE_ROOT=/path/to/Ego3D-Bench/images
```

When the environment has network access, Hugging Face data is downloaded on first use. For offline reproduction, populate the cache in advance and set `HF_HOME`. Dataset licenses and access terms are governed by the respective upstream dataset owners.

The full launcher plan contains 28 benchmarks and excludes API-judge workloads. If you have not prepared all local datasets, use `--only` to run the benchmarks whose data is available. A dry run validates the model and prints the selected plan without loading dataset samples.

## 🚀 Evaluation

The model directory must be in Hugging Face format and contain `config.json`. Use a dry run to inspect the benchmark plan first:

```bash
bash scripts/eval_qwen3vl.sh \
  --model-path /path/to/hf_model \
  --model-name my-model \
  --output-base /path/to/results \
  --gpus 0,1 \
  --models-per-gpu 2 \
  --cpu-per-worker 4 \
  --prompt-policy original \
  --dry-run
```

Remove `--dry-run` to start the evaluation. For a first real run, add `--resume` so completed benchmark bundles can be reused after an interruption. `--resume` supports benchmark-level and shard-level resumption; to reuse unfinished shards, pass the same `--run-id` when restarting. Use `--only ERQA,PointBench` to run a subset, or `--skip VLABench` to skip a benchmark.

The runner uses the Hugging Face backend. The model remains loaded throughout the benchmark sequence, and each worker processes an interleaved sample shard. An individual benchmark can also be run directly:

```bash
python eval_pointbench.py \
  --model_path /path/to/hf_model \
  --model_name my-model \
  --backend hf
```

Run `python eval_<benchmark>.py --help` for the complete options of each entry point. Results are written to `--output-base`. Each completed benchmark contains its raw predictions and summary; scheduler state is stored under the output directory. Use `python scripts/summarize_benchmark_scores.py --help` for score aggregation.

The launcher loads one model process per worker. Set `--models-per-gpu` according to available GPU memory; lower it for larger checkpoints. `--cpu-per-worker` must fit the host CPU affinity budget.

## 📊 Benchmarks and Metrics

The benchmark plan is defined in `scripts/benchmark_registry.py`. Point-localization tasks use the unified protocol in `docs/final_point_metrics_protocol.md`. Evaluation outputs include raw predictions, per-sample results, and summaries for auditing and reproduction.

## 📁 Directory Layout

- 🧩 `benchmark/`: dataset adapters and benchmark implementations
- ⚙️ `core/`: inference backends, media processing, shared metrics, and logging
- ▶️ `eval_*.py`: single-benchmark command-line entry points
- 📜 `scripts/`: sharded runner, benchmark registry, and score aggregation
- 📚 `docs/`: metrics protocols

This directory contains no model weights, dataset caches, evaluation results, runtime logs, or API keys.
