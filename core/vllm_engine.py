import time
import logging
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def detect_backbone(model_name: str) -> str:
    """
    Auto-detect backbone type from model name
    
    Args:
        model_name: Model name to analyze
        
    Returns:
        Detected backbone type: 'qwen3', 'qwen2_5', or 'intern-vl3.5'
        
    Raises:
        ValueError: If backbone cannot be detected from model name
    """
    model_name_lower = model_name.lower()
    
    if "qwen3" in model_name_lower:
        return "qwen3"
    elif "qwen2.5" in model_name_lower or "qwen2_5" in model_name_lower:
        return "qwen2_5"
    elif "intern" in model_name_lower:
        return "intern-vl3.5"
    else:
        # Raise error if cannot detect
        raise ValueError(
            f"Cannot auto-detect backbone from model name '{model_name}'. "
            f"Supported backbones: qwen3, qwen2_5, intern-vl3.5. "
            f"Please specify backbone explicitly using --backbone parameter."
        )


class VLLMInferenceEngine:
    """vLLM inference engine wrapper for multimodal models (local models only)"""
    
    SUPPORTED_BACKBONES = ['qwen3', 'qwen3_5', 'qwen2_5', 'pelican']
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        backbone: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 10240,
        seed: int = 3407,
        limit_mm_per_prompt: dict = None,
        attention_backend: Optional[str] = None,
    ):
        """
        Initialize vLLM inference engine
        
        Args:
            model_path: Path to the model
            model_name: Name of the model
            backbone: Backbone type ('qwen3', 'qwen2_5', 'intern-vl3.5'). 
                     If None, will auto-detect from model_name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model context length
            seed: Random seed for reproducibility
            limit_mm_per_prompt: Limit for multimodal data per prompt
            attention_backend: Optional vLLM attention backend override
        """
        self.model_path = model_path
        self.model_name = model_name
        
        # Auto-detect or validate backbone
        if backbone is None:
            self.backbone = detect_backbone(model_name)
            logger.info(f"Auto-detected backbone: {self.backbone}")
        else:
            if backbone not in self.SUPPORTED_BACKBONES:
                raise ValueError(
                    f"Unsupported backbone '{backbone}'. "
                    f"Supported backbones: {self.SUPPORTED_BACKBONES}"
                )
            self.backbone = backbone
        
        logger.info(f"Initializing vLLM...")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Backbone: {self.backbone}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  Seed: {seed}")
        
        # Load processor and initialize vLLM
        if self.backbone in ['qwen3', 'qwen3_5', 'qwen2_5', 'pelican']:
            self.processor = AutoProcessor.from_pretrained(model_path)
            logger.info(f"✓ Processor loaded")
            
            # Initialize vLLM
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
                limit_mm_per_prompt=limit_mm_per_prompt,
                attention_config={"backend": attention_backend} if attention_backend else None,
                seed=seed,
            )
            logger.info(f"✓ vLLM initialized\n")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def prepare_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare message format for a single sample, supporting multiple images and videos"""
        question = sample['question']
        
        # Build content list
        content = []
        
        # Handle images (single or multiple)
        if 'image' in sample and sample['image'] is not None:
            images = sample['image']
            # Convert single image to list for uniform processing
            if not isinstance(images, list):
                images = [images]
            
            for img in images:
                if img is not None:
                    content.append({"type": "image", "image": img})
        
        # Handle videos (single or multiple)
        if 'video' in sample and sample['video'] is not None:
            videos = sample['video']
            # Convert single video to list for uniform processing
            if not isinstance(videos, list):
                videos = [videos]
            
            for vid in videos:
                if vid is not None:
                    content.append({"type": "video", "video": vid})
        
        # Add text question
        content.append({"type": "text", "text": question})
        
        messages = [{"role": "user", "content": content}]
        
        return messages
    
    def prepare_vllm_input(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare vLLM input format based on backbone type
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary containing prompt and multimodal data
        """
        # Handle different backbones
        if self.backbone == 'qwen2_5' or self.backbone== 'pelican':
            # Qwen2.5-VL processing
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True
            )
            
            # Build multimodal data
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            
            return {
                'prompt': text,
                'multi_modal_data': mm_data,
                'mm_processor_kwargs': video_kwargs
            }
            
        elif self.backbone in ['qwen3', 'qwen3_5']:
            # Qwen3-VL processing
            template_kwargs = {"enable_thinking": False} if self.backbone == "qwen3_5" else {}
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                **template_kwargs
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            # Build multimodal data
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            
            return {
                'prompt': text,
                'multi_modal_data': mm_data,
                'mm_processor_kwargs': video_kwargs
            }
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def batch_inference(
        self,
        prepared_dataset: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.05,
        presence_penalty: float = 0.0,
    ) -> List[str]:
        """
        Batch inference with automatic vLLM optimization
        
        Args:
            prepared_dataset: Preprocessed dataset
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty
        
        Returns:
            List of generated texts
        """
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop_token_ids=[],
        )
        
        logger.info(f"Sampling parameters:")
        logger.info(f"  max_tokens={max_tokens}")
        logger.info(f"  temperature={temperature}, top_p={top_p}, top_k={top_k}")
        logger.info(f"  repetition_penalty={repetition_penalty}, presence_penalty={presence_penalty}")
        
        # Prepare all inputs
        logger.info(f"\nPreparing {len(prepared_dataset)} inputs...")
        all_inputs = []
        
        for sample in tqdm(prepared_dataset, desc="Preparing inputs"):
            messages = self.prepare_messages(sample)
            vllm_input = self.prepare_vllm_input(messages)
            all_inputs.append(vllm_input)
        
        # Batch inference
        logger.info(f"Starting batch inference ({len(all_inputs)} samples)...")
        start_time = time.time()
        
        outputs = self.llm.generate(all_inputs, sampling_params=sampling_params)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Inference completed in {elapsed:.2f}s")
        logger.info(f"  Average speed: {elapsed/len(prepared_dataset):.2f}s/sample")
        logger.info(f"  Throughput: {len(prepared_dataset)/elapsed:.2f} samples/s\n")
        
        # Extract generated texts
        results = []
        for output in outputs:
            text = output.outputs[0].text
            results.append(text)
        
        return results
