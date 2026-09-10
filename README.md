# PhysBrainEvalKit

<p align="center">
  <img src="assets/physbrain-evalkit-banner.png" alt="PhysBrainEvalKit banner" width="100%">
</p>

PhysBrainEvalKit is an evaluation toolkit for spatial and embodied-intelligence benchmarks targeting vision-language models (VLMs). This release contains the benchmark adapters, a unified Hugging Face inference interface, the point-localization metrics protocol, and a resident-model sharded runner for community reproduction.

This project is based on and extends the open-source [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit) framework.

> 🤖 **For coding agents:** Before configuring the environment or running an evaluation, read [`AGENTS.md`](AGENTS.md). It contains the repository workflow, cache configuration, model requirements, evaluation commands, and validation steps for Codex, Claude Code, and other coding agents.

## 🛠️ Installation

Use Python 3.10 or newer. Python 3.11 is recommended and is the version used for validation.

```bash
cd /path/to/PhysBrainEvalKit
python --version  # Python 3.10+ (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You may also use an existing PyTorch/Transformers environment. GPU evaluation requires a PyTorch build compatible with your CUDA version. The sharded launcher also requires Bash, `taskset`, and `nvidia-smi`.

Download a Qwen3-VL checkpoint in Hugging Face format (for example, an official Qwen3-VL-Instruct release) before running evaluation. The path passed to `--model-path` must contain `config.json`, the tokenizer files, and the model weight files.

## 📦 Data

All 28 benchmarks in the default plan load from public Hugging Face repositories. Dataset IDs, splits, and arguments are listed in `scripts/benchmark_registry.py`. Choose writable cache locations before launching Python:

```bash
export HF_HOME=/data/huggingface
export HF_DATASETS_CACHE=/data/huggingface/datasets
```

No benchmark-specific path variables are required for the default plan. The first run downloads the selected datasets; later runs reuse cached files.

### Dataset sources

The following public sources correspond to the benchmark plan. Hugging Face datasets are cached automatically on first use, including repositories with custom archives or directory layouts.

| Benchmark | Dataset source |
|---|---|
| ERQA | [FlagEval/ERQA](https://huggingface.co/datasets/FlagEval/ERQA) |
| RoboSpatial | [chanhee-luke/RoboSpatial-Home](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) |
| EgoPlan2 | [IffYuan/ego-plan](https://huggingface.co/datasets/IffYuan/ego-plan) |
| SAT | [FlagEval/SAT](https://huggingface.co/datasets/FlagEval/SAT) |
| Where2Place | [FlagEval/Where2Place](https://huggingface.co/datasets/FlagEval/Where2Place) |
| RefSpatial-Bench | [BAAI/RefSpatial-Bench](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) |
| Part-Affordance-2K | [IffYuan/Part-Affordance-2K](https://huggingface.co/datasets/IffYuan/Part-Affordance-2K) |
| ShareRobot-Trajectory | [IffYuan/sharerobot_trajectory](https://huggingface.co/datasets/IffYuan/sharerobot_trajectory) |
| VABench-Visual-Trace | [IffYuan/vabench-v](https://huggingface.co/datasets/IffYuan/vabench-v) |
| Q-Spatial-Bench | [andrewliao11/Q-Spatial-Bench](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench) |
| VABench-Point | [IffYuan/VABench-P](https://huggingface.co/datasets/IffYuan/VABench-P) |
| Pixmo-Points | [IffYuan/pixmo-points-eval](https://huggingface.co/datasets/IffYuan/pixmo-points-eval) |
| RoboAfford | [Zray26/roboafford-eval](https://huggingface.co/datasets/Zray26/roboafford-eval) |
| PIOBench | [IffYuan/PIO-Bench](https://huggingface.co/datasets/IffYuan/PIO-Bench) |
| RoboRefit | [IffYuan/RoboRefit](https://huggingface.co/datasets/IffYuan/RoboRefit) |
| BLINK | [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) |
| CV-Bench | [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) |
| VSI-Bench | [IffYuan/vsi-bench](https://huggingface.co/datasets/IffYuan/vsi-bench) |
| EmbSpatial | [FlagEval/EmbSpatial-Bench](https://huggingface.co/datasets/FlagEval/EmbSpatial-Bench) |
| PointBench | [IffYuan/PointBench](https://huggingface.co/datasets/IffYuan/PointBench) |
| COSMOS | [IffYuan/COSMOS](https://huggingface.co/datasets/IffYuan/COSMOS) |
| RoboVQA | [VLyb/RoboVQA-16frames](https://huggingface.co/datasets/VLyb/RoboVQA-16frames) |
| VLABench | [VLyb/VLABench](https://huggingface.co/datasets/VLyb/VLABench) |
| ERQA-PLUS | [huggingdas/erqa-plus](https://huggingface.co/datasets/huggingdas/erqa-plus) |
| 3DSRBench | [VLyb/3DSRBench](https://huggingface.co/datasets/VLyb/3DSRBench) |
| ViewSpatial | [lidingm/ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench) |
| MindCube | [VLyb/MindCube-TinyBench](https://huggingface.co/datasets/VLyb/MindCube-TinyBench) |
| MMSI-Bench | [RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench) |

### Downloading and caching datasets

Set the Hugging Face cache locations in the shell used for evaluation. The framework downloads all selected datasets there automatically on first use:

```bash
export HF_HOME=/data/huggingface
export HF_DATASETS_CACHE=/data/huggingface/datasets
```

No manual dataset download command is required. Running the evaluation launcher downloads the selected datasets into the configured Hugging Face cache and reuses them on later runs.

Use `--only` or `--skip` to select datasets available in your environment. A dry run validates the model configuration and CLI options and prints the plan; it does not download data or check dataset files. Dataset licenses and access terms are governed by their respective owners.

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
  --dry-run
```

### Launcher parameters

| Parameter | Meaning | Example value |
|---|---|---|
| `--model-path PATH` | Local Hugging Face model directory containing `config.json`, tokenizer/processor files, and weights. | `/models/Qwen3-VL-8B-Instruct` |
| `--model-name NAME` | Identifier embedded in result filenames and metadata. | `qwen3-vl-8b-instruct` |
| `--output-base DIR` | Directory for benchmark results, raw predictions, summaries, and scheduler state. | `/results/qwen3-vl-8b` |
| `--backend hf` | Inference backend. The sharded launcher currently supports Hugging Face only. | `hf` |
| `--gpus LIST` | GPU IDs, separated by commas or spaces. | `0,1,2,3` |
| `--models-per-gpu N` | Number of resident model workers placed on each GPU. Lower this for larger checkpoints. | `2` |
| `--cpu-per-worker N` | CPU cores assigned to each worker; must fit the host affinity budget. | `4` |
| `--only LIST` | Run only the listed benchmarks. Names must match `benchmark_registry.py`. | `ERQA,PointBench` |
| `--skip LIST` | Exclude listed benchmarks from the plan. | `VLABench,MMSI-Bench` |
| `--run-id ID` | Persistent shard namespace. Reuse the same ID to resume unfinished shards. | `qwen3-vl-run-01` |
| `--mmsi-num-samples N` | Number of samples generated per MMSI-Bench item. | `1` |
| `--mmsi-seed N` | Random seed for MMSI-Bench sampling. | `3407` |
| `--mmsi-temperature FLOAT` | Sampling temperature used by MMSI-Bench. | `0.7` |
| `--resume` | Skip complete benchmark bundles and reuse valid shard outputs. | (flag) |
| `--debug` | Run the reduced debug mode supported by selected benchmark entry points. | (flag) |
| `--dry-run` | Validate the model and print the selected plan without loading models or samples. | (flag) |
| `--help` | Print command usage. | (flag) |

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
