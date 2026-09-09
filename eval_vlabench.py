import argparse
import gc
import logging
import os
import time

from benchmark.vlabench import VLABenchDataset
from core.inference import create_inference_engine
from core.logger import setup_logging

# Set vLLM multiprocessing method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = logging.getLogger(__name__)


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"

def main(
    args,
    task_name,
    model_name,
    model_path,
    instruct_following,
    dataset_path,
    subset,
    split
):
    start_time = time.time()
    is_main_process = int(os.environ.get("RANK", "0")) == 0            
    
    # ============== Part 1: Load and preprocess dataset ==============
    logger.info("=" * 60)
    logger.info("Part 1: Load and preprocess dataset")
    logger.info("=" * 60)
    
    dataset = VLABenchDataset(
        dataset_path=dataset_path,
        subset=subset,
        split=split,
        instruct_following=instruct_following,
        task_name=task_name,
        model_name=model_name,
        backbone=args.backbone,
        debug=args.debug,
        thinking_model=args.thinking_model
    )
    raw_dataset = dataset.load_dataset()
    if args.debug:
        raw_dataset = raw_dataset[:20]
        logger.info("Debug mode: processing first 20 samples only")

    total_samples = len(raw_dataset)
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else total_samples
    logger.info(f"Dataset ready: {total_samples} samples")
    logger.info(f"Chunk size: {chunk_size}")
    
    # ============== Part 2: Initialize vLLM and batch inference ==============
    logger.info("=" * 60)
    logger.info("Part 2: Initialize vLLM and batch inference")
    logger.info("=" * 60)
    
    inference_engine, engine = create_inference_engine(args, model_path, model_name)

    all_results = []
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    chunk_loop_start = time.time()
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_number = start_idx // chunk_size + 1
        chunk_start_time = time.time()
        logger.info("=" * 60)
        logger.info(
            f"Processing chunk {chunk_number}/{total_chunks}: "
            f"{start_idx}-{end_idx - 1} ({end_idx - start_idx} samples)"
        )
        logger.info("=" * 60)

        chunk_raw = raw_dataset[start_idx:end_idx]
        prepared_chunk = dataset.prepare_dataset(chunk_raw, start_index=start_idx)
        logger.info(f"✓ Chunk preprocessing completed: {len(prepared_chunk)} samples")

        raw_outputs = inference_engine.batch_inference(
            prepared_chunk,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
        )

        if engine != "hf" or is_main_process:
            all_results.extend(dataset.evaluate_results(prepared_chunk, raw_outputs))
            completed = len(all_results)
            elapsed = time.time() - chunk_loop_start
            chunk_elapsed = time.time() - chunk_start_time
            samples_per_second = completed / elapsed if elapsed > 0 else 0.0
            remaining_samples = total_samples - completed
            eta_seconds = remaining_samples / samples_per_second if samples_per_second > 0 else 0.0
            eta_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds))

            logger.info(
                "✓ Chunk completed: "
                f"{completed} / {total_samples} results "
                f"({completed / total_samples * 100:.2f}%)"
            )
            logger.info(
                "Progress ETA: "
                f"chunk_time={format_duration(chunk_elapsed)}, "
                f"elapsed={format_duration(elapsed)}, "
                f"speed={samples_per_second:.3f} samples/s, "
                f"remaining={format_duration(eta_seconds)}, "
                f"eta_at={eta_time}"
            )

            partial_statistics = dataset.compute_statistics(all_results, log=False)
            dataset.save_partial_results(
                all_results,
                partial_statistics,
                {
                    "completed_samples": completed,
                    "total_samples": total_samples,
                    "completed_chunks": chunk_number,
                    "total_chunks": total_chunks,
                    "chunk_size": chunk_size,
                    "last_chunk_start_idx": start_idx,
                    "last_chunk_end_idx": end_idx - 1,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": eta_seconds,
                    "eta_at": eta_time,
                    "samples_per_second": samples_per_second,
                },
            )

        del prepared_chunk, raw_outputs, chunk_raw
        gc.collect()
    
    logger.info(f"✓ Batch inference completed\n")
    
    # ============== Part 3: Evaluate results ==============
    if engine != "hf" or is_main_process:
        logger.info("=" * 60)
        logger.info("Part 3: Evaluate results")
        logger.info("=" * 60)

        # ============== Part 4: Compute statistics ==============
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
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLABench evaluation")

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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512).",
    )

    # Dataset
    parser.add_argument("--debug", action="store_true", help="Debug mode (process first 20 samples only)")

    # VLABench specific
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "VLABENCH_DATASET_PATH",
            os.environ.get("VLABENCH_DATASET_PATH", "datasets/VLABench/vlm_evaluation_v1.0"),
        ),
        help="Path to the VLABench vlm_evaluation_v1.0 dataset directory.",
    )
    parser.add_argument("--subset", type=str, nargs='+', default=None,
                        help="Specific dimension(s) to evaluate")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    # Keep full VLABench evaluation memory-bounded: each sample opens an input
    # image plus a visual-prompt mask, and model outputs can be long JSON plans.
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of VLABench samples to preprocess and infer at a time. Use <=0 to disable chunking.",
    )

    args = parser.parse_args()

    task_name = "VLABench"
    model_path = args.model_path
    model_name = args.model_name

    setup_logging(task_name, model_name)

    instruct_following = args.instruct_following

    main(
        args=args,
        task_name=task_name,
        model_name=model_name,
        model_path=model_path,
        instruct_following=instruct_following,
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split
    )
