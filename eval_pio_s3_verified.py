# ✅
import argparse
import logging
import os
import time

from benchmark.pio_s3 import PIOS3Dataset
from core.inference import create_inference_engine
from core.logger import setup_logging

# Set vLLM multiprocessing method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = logging.getLogger(__name__)


def first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def main(
    args,
    task_name,
    model_name,
    model_path,
    instruct_following,
    dataset_name,
    subset,
    split
):
    start_time = time.time()
    is_main_process = int(os.environ.get("RANK", "0")) == 0

    # ============== Part 1: Load and preprocess dataset ==============
    logger.info("=" * 60)
    logger.info("Part 1: Load and preprocess dataset")
    logger.info("=" * 60)

    dataset = PIOS3Dataset(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        instruct_following=instruct_following,
        task_name=task_name,
        model_name=model_name,
        backbone=args.backbone,
        debug=args.debug,
        openai_key=args.openai_key,
        openai_base_url=args.openai_base_url,
        judge_model=args.judge_model,
        judge_seed=args.judge_seed,
        judge_max_tokens=args.judge_max_tokens,
        judge_temperature=args.judge_temperature,
        judge_max_workers=args.judge_max_workers,
        judge_max_concurrent_requests=args.judge_max_concurrent_requests,
        thinking_model=args.thinking_model,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir
    )
    raw_dataset = dataset.load_dataset()
    if args.debug:
        debug_limit = min(args.debug_limit, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(debug_limit))
        logger.info(f"Debug mode: processing first {debug_limit} samples only")

    prepared_dataset = dataset.prepare_dataset(raw_dataset)

    logger.info(f"✓ Preprocessing completed: {len(prepared_dataset)} samples\n")

    # ============== Part 2: Initialize vLLM and batch inference ==============
    logger.info("=" * 60)
    logger.info("Part 2: Initialize vLLM and batch inference")
    logger.info("=" * 60)
    inference_engine, engine = create_inference_engine(args, model_path, model_name)

    raw_outputs = inference_engine.batch_inference(
        prepared_dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    logger.info("✓ Batch inference completed\n")

    if engine != "hf" or is_main_process:
        logger.info("=" * 60)
        logger.info("Part 3: Evaluate results")
        logger.info("=" * 60)

        all_results = dataset.evaluate_results(prepared_dataset, raw_outputs)

        logger.info("=" * 60)
        logger.info("Part 4: Compute statistics")
        logger.info("=" * 60)

        statistics = dataset.compute_statistics(all_results)

        # ============== Save results ==============
        result_file_name = dataset.save_results(all_results, statistics)

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Completed!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {result_file_name}")
        if args.save_visualizations:
            logger.info(f"Visualizations saved to: {args.visualization_dir}")
        logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIO-S3-Verified evaluation")

    # Model config
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=['gemma4', 'qwen3_5', 'qwen3', 'qwen2_5', 'gpt', 'gemini-2.5','molmo','mimo','magma','internvl','pelican','gemini_robotics'],
                        help="Backbone type (qwen3, qwen2_5, gpt, gemini-2.5). If not specified, will auto-detect from model_name")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "hf"],
                        help="Inference backend override. Use 'hf' when vLLM is not compatible with the local CUDA driver.")
    parser.add_argument("--instruct_following", type=str, default=None, help="Instruction following prompt")
    parser.add_argument("--thinking_model", action="store_true", help="Enable thinking model mode (extract answer from <answer> tags)")

    # vLLM engine
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size (default: 2)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=10240, help="Maximum model context length (default: 10240)")
    parser.add_argument("--max_images_per_prompt", type=int, default=16, help="Maximum images per prompt (default: 16)")
    parser.add_argument("--max_videos_per_prompt", type=int, default=1, help="Maximum videos per prompt (default: 1)")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility (default: 3407)")

    # HF engine
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)

    # API engine
    parser.add_argument("--max_concurrent_requests", type=int, default=100, help="Max concurrent requests for API engine (default: 100)")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k (default: 20)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty (default: 1.05)")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty (default: 0.0)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate (default: 4096)")

    # Dataset
    parser.add_argument("--debug", action="store_true", help="Debug mode (process first 20 samples only)")

    # PIO-S3-Verified specific
    parser.add_argument("--debug_limit", type=int, default=20, help="Number of samples to run in debug mode")
    parser.add_argument(
        "--openai_key",
        type=str,
        default=first_env("PIO_S3_OPENAI_KEY", "OPENAI_API_KEY", "API_KEY"),
        help="OpenAI-compatible judge API key. Defaults to PIO_S3_OPENAI_KEY, OPENAI_API_KEY, or API_KEY.",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=first_env("PIO_S3_OPENAI_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"),
        help="OpenAI-compatible judge API base URL. Defaults to PIO_S3_OPENAI_BASE_URL, OPENAI_BASE_URL, or BASE_URL.",
    )
    parser.add_argument("--judge_model", type=str, default="gpt-5.5", help="Judge model used for trajectory scoring")
    parser.add_argument("--judge_seed", type=int, default=1234, help="Judge model seed")
    parser.add_argument("--judge_max_tokens", type=int, default=512, help="Max tokens for judge responses")
    parser.add_argument("--judge_temperature", type=float, default=0.2, help="Judge model temperature")
    parser.add_argument("--judge_max_workers", type=int, default=8, help="Max worker threads for judge scoring")
    parser.add_argument("--judge_max_concurrent_requests", type=int, default=8, help="Max concurrent judge API requests")
    visualization_group = parser.add_mutually_exclusive_group()
    visualization_group.add_argument("--save_visualizations", dest="save_visualizations", action="store_true", default=True, help="Save trajectory visualizations")
    visualization_group.add_argument("--no_save_visualizations", dest="save_visualizations", action="store_false", help="Delete trajectory visualizations after scoring")
    parser.add_argument("--visualization_dir", type=str, default="logs/pio_s3_visualizations", help="Directory to save visualizations")

    args = parser.parse_args()

    # Dataset configuration
    task_name = "PIO-S3-Verified"
    dataset_name = "IffYuan/PIO-S3"
    subset = None
    split = "train"

    model_path = args.model_path
    model_name = args.model_name

    log_file_name = setup_logging(task_name, model_name)


    instruct_following = args.instruct_following

    main(
        args=args,
        task_name=task_name,
        model_name=model_name,
        model_path=model_path,
        instruct_following=instruct_following,
        dataset_name=dataset_name,
        subset=subset,
        split=split
    )
