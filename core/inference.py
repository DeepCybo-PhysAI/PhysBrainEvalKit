import logging
import os

from .api_engine import APIInferenceEngine

logger = logging.getLogger(__name__)


def create_inference_engine(args, model_path, model_name):
    """Create inference engine based on backbone type"""
    engine_name = None
    backend = getattr(args, "backend", "auto")

    if backend == "hf":
        from .hf_engine import HFInferenceEngine

        inference_engine = HFInferenceEngine(
            model_path=model_path,
            model_name=model_name,
            backbone=args.backbone,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        engine_name = "hf"
    elif args.backbone == 'qwen3_5':
        from .hf_engine import HFInferenceEngine

        inference_engine = HFInferenceEngine(
            model_path=model_path,
            model_name=model_name,
            backbone=args.backbone,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        engine_name = "hf"
    elif args.backbone in ['qwen3', 'qwen2_5', 'pelican']:
        from .vllm_engine import VLLMInferenceEngine

        inference_engine = VLLMInferenceEngine(
            model_path=model_path,
            model_name=model_name,
            backbone=args.backbone,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": args.max_images_per_prompt, "video": args.max_videos_per_prompt},
            seed=args.seed
        )
        engine_name = "vllm"
    elif args.backbone in ['gpt', 'gemini-2.5','gemini_robotics']:
        inference_engine = APIInferenceEngine(
            model_path=None,
            model_name=model_name,
            backbone=args.backbone,
            api_key=os.getenv("API_KEY"),
            base_url=args.base_url,
            max_concurrent_requests=args.max_concurrent_requests,
            image_max_side=getattr(args, "image_max_side", 336),
        )
        engine_name = "api"
    elif args.backbone in ['molmo', 'mimo', 'magma', 'internvl']:
        from .hf_engine import HFInferenceEngine

        inference_engine = HFInferenceEngine(
            model_path=model_path,
            model_name=model_name,
            backbone=args.backbone,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        engine_name = "hf"
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    return inference_engine, engine_name
