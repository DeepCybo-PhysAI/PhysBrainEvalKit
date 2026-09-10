import argparse
import logging
import os
import time

from benchmark.mindcube import MindCubeDataset
from core.inference import create_inference_engine
from core.logger import setup_logging

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logger = logging.getLogger(__name__)


def main(args, task_name, model_name, model_path, instruct_following, dataset_name, subset, split):
    start_time = time.time()
    is_main_process = int(os.environ.get("RANK", "0")) == 0

    logger.info("=" * 60)
    logger.info("Part 1: Load and preprocess dataset")
    logger.info("=" * 60)

    dataset = MindCubeDataset(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        instruct_following=instruct_following,
        task_name=task_name,
        model_name=model_name,
        backbone=args.backbone,
        debug=args.debug,
        thinking_model=args.thinking_model,
    )
    raw_dataset = dataset.load_dataset()

    if args.debug:
        raw_dataset = raw_dataset.select(range(min(20, len(raw_dataset)))) if hasattr(raw_dataset, "select") else raw_dataset[:20]
        logger.info("Debug mode: processing first 20 samples only")

    prepared_dataset = dataset.prepare_dataset(raw_dataset)
    logger.info(f"Preprocessing completed: {len(prepared_dataset)} samples\n")

    logger.info("=" * 60)
    logger.info("Part 2: Initialize engine and batch inference")
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

    logger.info("Batch inference completed\n")

    if engine != "hf" or is_main_process:
        logger.info("=" * 60)
        logger.info("Part 3: Evaluate results")
        logger.info("=" * 60)
        all_results = dataset.evaluate_results(prepared_dataset, raw_outputs)

        logger.info("=" * 60)
        logger.info("Part 4: Compute statistics")
        logger.info("=" * 60)
        statistics = dataset.compute_statistics(all_results)
        result_file_name = dataset.save_results(all_results, statistics)

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Completed!")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {result_file_name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindCube evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["gemma4", "qwen3_5", "qwen3", "qwen2_5", "gpt", "gemini-2.5", "molmo", "mimo", "magma", "internvl", "pelican", "gemini_robotics"],
                        help="Backbone type")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "hf"], help="Inference backend override")
    parser.add_argument("--instruct_following", type=str, default=None, help="Instruction following prompt")
    parser.add_argument("--thinking_model", action="store_true", help="Extract final answer from <answer> tags")

    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=20000, help="Maximum model context length")
    parser.add_argument("--max_images_per_prompt", type=int, default=4, help="Maximum images per prompt")
    parser.add_argument("--max_videos_per_prompt", type=int, default=1, help="Maximum videos per prompt")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--max_concurrent_requests", type=int, default=100, help="Max concurrent API requests")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL")

    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate")

    parser.add_argument("--debug", action="store_true", help="Debug mode: process first 20 samples only")
    parser.add_argument("--dataset_name", type=str, default="VLyb/MindCube-TinyBench", help="Dataset directory or jsonl path")
    parser.add_argument("--split", type=str, default="tinybench", help="Dataset split: tinybench, full/default, train, or a jsonl filename")

    args = parser.parse_args()

    task_name = "MindCube"
    setup_logging(task_name, args.model_name)

    main(
        args=args,
        task_name=task_name,
        model_name=args.model_name,
        model_path=args.model_path,
        instruct_following=args.instruct_following,
        dataset_name=args.dataset_name,
        subset=None,
        split=args.split,
    )
