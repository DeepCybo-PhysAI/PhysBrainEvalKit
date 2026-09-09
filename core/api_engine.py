import asyncio
import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

logger = logging.getLogger(__name__)


def detect_api_backbone(model_name: str) -> str:
    """
    Auto-detect API backbone type from model name

    Args:
        model_name: Model name to analyze

    Returns:
        Detected backbone type: 'gpt', 'gemini_robotics', or 'gemini-2.5'

    Raises:
        ValueError: If backbone cannot be detected from model name
    """
    model_name_lower = model_name.lower()

    if "gpt" in model_name_lower:
        return "gpt"
    elif "gemini-2.5" in model_name_lower or "gemini-2-5" in model_name_lower:
        return "gemini-2.5"
    elif "gemini_robotics" in model_name_lower:
        return "gemini_robotics"
    else:
        # Raise error if cannot detect
        raise ValueError(
            f"Cannot auto-detect API backbone from model name '{model_name}'. "
            f"Supported API backbones: gpt, gemini_robotics, gemini-2.5. "
            f"Please specify backbone explicitly using --backbone parameter."
        )


def encode_image_to_base64(image_input: Any, target_size: int = 336) -> str:

    if isinstance(image_input, Image.Image):
        w, h = image_input.size
        if max(w, h) > target_size:
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_input = image_input.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        if image_input.mode in ('RGBA', 'P'):
            image_input = image_input.convert('RGB')
            
        image_input.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def get_image_format(image_input: Any) -> str:


    if isinstance(image_input, Image.Image):
        if image_input.format:
            return image_input.format.lower()
        return 'jpeg'
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def sample_frames_from_video(video_path: str, n_frames: int = 8) -> List[np.ndarray]:
    """
    Sample n frames uniformly from a video file
    
    Args:
        video_path: Path to the video file
        n_frames: Number of frames to sample (default: 8)
        
    Returns:
        List of frame images as numpy arrays (RGB format)
        
    Raises:
        ValueError: If video cannot be opened or has no frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Calculate frame indices to sample uniformly
    if total_frames < n_frames:
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample n_frames
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            logger.warning(f"Failed to read frame {idx} from {video_path}")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")
    
    logger.info(f"Sampled {len(frames)} frames from video: {video_path}")
    return frames


def encode_frame_to_base64(frame: np.ndarray, format: str = 'jpeg', quality: int = 95) -> str:
    """
    Encode a video frame (numpy array) to base64 string
    
    Args:
        frame: Frame image as numpy array (RGB format)
        format: Image format for encoding ('jpeg' or 'png')
        quality: JPEG quality (1-100, only for JPEG)
        
    Returns:
        Base64 encoded string
    """
    # Encode frame to image format
    if format.lower() == 'jpeg':
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param)
    else:  # png
        _, buffer = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Convert to base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str


class APIInferenceEngine:
    """API inference engine for GPT/Gemini models with async parallel inference"""
    
    SUPPORTED_BACKBONES = ['gpt', 'gemini_robotics', 'gemini-2.5']

    def __init__(
        self,
        model_path: str,
        model_name: str,
        backbone: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent_requests: int = 50,
        timeout: int = 120,
        video_frame_sample_n: int = 8,
        image_max_side: int = 336,
    ):
        """
        Initialize API inference engine

        Args:
            model_path: Model identifier for API (e.g., 'gpt-4o', 'gemini-2.5-pro', 'gemini-robotics-er-1.5-preview')
            model_name: Name of the model
            backbone: Backbone type ('gpt', 'gemini_robotics', 'gemini-2.5'). If None, will auto-detect
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            max_concurrent_requests: Maximum number of concurrent API requests
            timeout: Request timeout in seconds
            video_frame_sample_n: Number of frames to sample from each video (default: 8)
            Note: video_frame_format is fixed to 'jpeg' and video_frame_quality is fixed to 95
        """
        self.model_path = model_path
        self.model_name = model_name
        
        # Auto-detect or validate backbone
        if backbone is None:
            self.backbone = detect_api_backbone(model_name)
            logger.info(f"Auto-detected API backbone: {self.backbone}")
        else:
            if backbone not in self.SUPPORTED_BACKBONES:
                raise ValueError(
                    f"Unsupported API backbone '{backbone}'. "
                    f"Supported API backbones: {self.SUPPORTED_BACKBONES}"
                )
            self.backbone = backbone

        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.image_max_side = image_max_side

        # Video processing parameters
        self.video_frame_sample_n = video_frame_sample_n
        self.video_frame_format = 'jpeg'  # Fixed to jpeg
        self.video_frame_quality = 95  # Fixed to 95

        # Initialize gemini client if using gemini backbone
        self.gemini_client = None
        if self.backbone == 'gemini_robotics':
            if not GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai package is required for gemini backbone. "
                    "Install it with: pip install google-genai"
                )
            self.gemini_client = genai.Client(api_key=self.api_key)

        # Set default URLs if not provided
        if self.base_url is None:
            raise ValueError("Error base_url")
        
        logger.info(f"Initializing API Engine...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Backbone: {self.backbone}")
        if self.backbone != 'gemini_robotics':
            logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Max concurrent requests: {max_concurrent_requests}")
        logger.info(f"  Image max side: {image_max_side}")
        logger.info(f"  Timeout: {timeout}s")
        logger.info(f"  Video frame sampling: {video_frame_sample_n} frames, format: {self.video_frame_format}, quality: {self.video_frame_quality}")
        logger.info(f"✓ API Engine initialized\n")
    
    def prepare_messages(self, sample: Dict[str, Any]) -> Any:
        """
        Prepare message format for a single sample, supporting multiple images and videos

        Returns:
            For gpt/gemini-2.5: List[Dict[str, Any]] (OpenAI format)
            For gemini_robotics: List of types.Part objects
        """
        question = sample['question']

        # For gemini backbone, return list of Parts
        if self.backbone == 'gemini_robotics':
            parts = []

            # Handle images (single or multiple)
            if 'image' in sample and sample['image'] is not None:
                images = sample['image']
                if not isinstance(images, list):
                    images = [images]

                for img in images:
                    if img is not None:
                        # Convert PIL Image to bytes
                        buffered = io.BytesIO()
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        img.save(buffered, format="JPEG", quality=95)
                        image_bytes = buffered.getvalue()

                        parts.append(types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg'
                        ))

            # Handle videos
            if 'video' in sample and sample['video'] is not None:
                videos = sample['video']
                if isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], Image.Image):
                    videos = [videos]
                elif not isinstance(videos, list):
                    videos = [videos]

                for vid in videos:
                    if vid is not None:
                        try:
                            frames_to_process = []

                            if isinstance(vid, str):
                                sampled_frames = sample_frames_from_video(
                                    vid,
                                    n_frames=self.video_frame_sample_n
                                )
                                frames_to_process = sampled_frames
                            elif isinstance(vid, list) and len(vid) > 0:
                                total_frames = len(vid)
                                if total_frames > self.video_frame_sample_n:
                                    indices = np.linspace(0, total_frames - 1, self.video_frame_sample_n, dtype=int).tolist()
                                    frames_to_process = [vid[i] for i in indices]
                                else:
                                    frames_to_process = vid

                            for frame in frames_to_process:
                                if isinstance(frame, np.ndarray):
                                    # Convert numpy array to bytes
                                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.video_frame_quality]
                                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param)
                                    frame_bytes = buffer.tobytes()
                                elif isinstance(frame, Image.Image):
                                    # Convert PIL Image to bytes
                                    buffered = io.BytesIO()
                                    if frame.mode in ('RGBA', 'P'):
                                        frame = frame.convert('RGB')
                                    frame.save(buffered, format="JPEG", quality=self.video_frame_quality)
                                    frame_bytes = buffered.getvalue()
                                else:
                                    logger.warning(f"Unexpected frame type: {type(frame)}")
                                    continue

                                parts.append(types.Part.from_bytes(
                                    data=frame_bytes,
                                    mime_type='image/jpeg'
                                ))
                        except Exception as e:
                            logger.error(f"Failed to process video input: {e}")

            # Add text prompt
            parts.append(question)
            return parts

        # For gpt/gemini-2.5 backbones, use OpenAI format
        else:
            content = []

            # Handle images (single or multiple)
            if 'image' in sample and sample['image'] is not None:
                images = sample['image']
                if not isinstance(images, list):
                    images = [images]

                for img in images:
                    if img is not None:
                        image_format = get_image_format(img)
                        base64_image = encode_image_to_base64(img, target_size=self.image_max_side)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        })

            if 'video' in sample and sample['video'] is not None:
                videos = sample['video']
                if not isinstance(videos, list):
                    if isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], Image.Image):
                        videos = [videos]
                    elif not isinstance(videos, list):
                        videos = [videos]
                for vid in videos:
                    if vid is not None:
                        try:
                            base64_frames = []

                            if isinstance(vid, str):
                                sampled_frames = sample_frames_from_video(
                                    vid,
                                    n_frames=self.video_frame_sample_n
                                )
                                for frame in sampled_frames:
                                    b64 = encode_frame_to_base64(
                                        frame,
                                        format=self.video_frame_format,
                                        quality=self.video_frame_quality
                                    )
                                    base64_frames.append(b64)

                            elif isinstance(vid, list) and len(vid) > 0:
                                total_frames = len(vid)
                                if total_frames > self.video_frame_sample_n:
                                    indices = np.linspace(0, total_frames - 1, self.video_frame_sample_n, dtype=int).tolist()
                                    sampled_pil_frames = [vid[i] for i in indices]
                                else:
                                    sampled_pil_frames = vid

                                for frame in sampled_pil_frames:
                                    if isinstance(frame, Image.Image):
                                        b64 = encode_image_to_base64(frame, target_size=self.image_max_side)
                                        base64_frames.append(b64)
                                    else:
                                        logger.warning(f"Unexpected frame type in video list: {type(frame)}")

                            for b64 in base64_frames:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{self.video_frame_format};base64,{b64}"
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Failed to process video input: {e}")

            content.append({"type": "text", "text": question})
            return [{"role": "user", "content": content}]
    
    async def create_completion(
        self,
        session: aiohttp.ClientSession,
        messages: Any,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.05,
        presence_penalty: float = 0.0,
        semaphore: asyncio.Semaphore = None,
    ) -> Optional[str]:
        """
        Create a single completion request

        Args:
            session: aiohttp client session
            messages: Message list for the request (List[Dict] for gpt/gemini-2.5, List[Part] for gemini)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty
            semaphore: Semaphore for controlling concurrency

        Returns:
            Generated text or None if failed
        """
        if semaphore:
            async with semaphore:
                return await self._make_request(
                    session, messages, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty
                )
        else:
            return await self._make_request(
                session, messages, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty
            )
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        messages: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = -1,
        repetition_penalty: float = 1.05,
        presence_penalty: float = 0.0,
    ) -> Optional[str]:
        """
        Make the actual API request

        Args:
            session: aiohttp client session
            messages: Message list for the request (List[Dict] for gpt/gemini-2.5, List[Part] for gemini)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty

        Returns:
            Generated text or None if failed
        """
        # Handle gemini backbone using google.genai SDK
        if self.backbone == 'gemini_robotics':
            try:
                # Run the synchronous SDK call in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.gemini_client.models.generate_content(
                        model=self.model_name,
                        contents=messages,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            top_p=top_p,
                            max_output_tokens=max_tokens,
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini API request exception: {e}")
                return None

        # Handle gpt/gemini-2.5 backbones using OpenAI-compatible API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.backbone == 'gpt' or self.backbone == 'gemini-2.5':
            # GPT API format
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages,
            }
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        try:
            async with session.post(
                url=url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract text based on backbone
                    if self.backbone == 'gpt' or self.backbone == 'gemini-2.5':
                        return result['choices'][0]['message']['content']
                    else:
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Request failed with status {response.status}: {error_text}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Request exception: {e}")
            return None
    
    async def _process_single_sample_with_index(
        self,
        index: int,
        messages: Any,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
    ) -> Tuple[int, Optional[str]]:
        """Helper to return index with result to maintain order"""
        result = await self.create_completion(
            session, messages, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty, semaphore
        )
        return index, result

    async def batch_inference_async(
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
        Async batch inference with parallel requests
        
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
        logger.info(f"Sampling parameters:")
        logger.info(f"  max_tokens={max_tokens}")
        logger.info(f"  temperature={temperature}, top_p={top_p}")
        
        logger.info(f"Starting batch inference: {len(prepared_dataset)} samples")
        # Prepare all messages
        logger.info(f"\nPreparing {len(prepared_dataset)} inputs...")
        all_messages = []
        for sample in tqdm(prepared_dataset, desc="Preparing inputs"):
            messages = self.prepare_messages(sample)
            all_messages.append(messages)

        final_results = [None] * len(all_messages)
        # Create semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Create async tasks
        logger.info(f"Starting async batch inference ({len(all_messages)} samples)...")
        logger.info(f"  Max concurrent requests: {self.max_concurrent_requests}")
        start_time = time.time()
        
        async with aiohttp.ClientSession(trust_env=True) as session:
            tasks = [
                self._process_single_sample_with_index(
                    idx, msg, session, semaphore, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty
                )
                for idx, msg in enumerate(all_messages)
            ]
            
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="API requests"
            ):
                index, result = await coro
                final_results[index] = result
        
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Inference completed in {elapsed:.2f}s")
        logger.info(f"  Average speed: {elapsed/len(prepared_dataset):.2f}s/sample")
        logger.info(f"  Throughput: {len(prepared_dataset)/elapsed:.2f} samples/s\n")
        
        # Handle failed requests
        failed_count = sum(1 for r in final_results if r is None)
        if failed_count > 0:
            logger.warning(f"Warning: {failed_count} requests failed")
        
        # Replace None with empty string
        final_results = [r if r is not None else "" for r in final_results]
        
        return final_results
    
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
        Synchronous wrapper for batch inference
        
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
        return asyncio.run(
            self.batch_inference_async(
                prepared_dataset, max_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty
            )
        )
