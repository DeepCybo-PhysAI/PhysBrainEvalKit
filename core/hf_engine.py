import io
import logging
import re
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

from .internvl_utils import (
    build_transform,
    dynamic_preprocess,
    find_closest_aspect_ratio,
)
from .lazy_media import resolve_lazy_hf_media

logger = logging.getLogger(__name__)


def detect_hf_backbone(model_name: str) -> str:
    name = model_name.lower()
    if "molmo" in name:
        return "molmo"
    elif "intern" in name:
        return "internvl"
    elif "magma" in name:
        return "magma"
    elif "mimo" in name:
        return "mimo"
    elif "qwen3.5" in name or "qwen3_5" in name:
        return "qwen3_5"
    elif "qwen3" in name:
        return "qwen3"
    elif "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    else:
        return "huggingface"


class HFInferenceEngine:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        backbone: Optional[str] = None,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        seed: int = 3407,
        **kwargs
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.accelerator = Accelerator()
        set_seed(seed)

        self.backbone = backbone if backbone else detect_hf_backbone(model_name)

        self.dtype_str = dtype
        self.torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        if self.accelerator.is_main_process:
            logger.info(f"Init Engine | Backbone: {self.backbone} | Dtype: {self.torch_dtype}")

        if self.backbone == "molmo":
            logger.info("[Molmo] Loading AutoProcessor & AutoModelForCausalLM...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self.torch_dtype
            )

        elif "internvl" in self.backbone:
            logger.info("[InternVL] Loading AutoTokenizer & AutoModel...")
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            self.transform = build_transform(input_size=448)
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True, use_flash_attn=True
            )

        elif self.backbone == "magma":
            logger.info("[Magma] Loading AutoProcessor & AutoModelForCausalLM...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self.torch_dtype
            )

        elif self.backbone =='mimo':
            logger.info("[Mimo] Loading AutoProcessor & Qwen2_5_VLForConditionalGeneration...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self.torch_dtype
            )
        elif self.backbone == "qwen3":
            logger.info("[Qwen3-VL] Loading AutoProcessor & Qwen3VLForConditionalGeneration...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        elif self.backbone == "qwen3_5":
            logger.info("[Qwen3.5] Loading AutoProcessor & AutoModelForImageTextToText...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=self.torch_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        elif self.backbone == "hy_v3_vl":
            logger.info("[Hy-V3-VL] Loading remote AutoProcessor & AutoModelForImageTextToText...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
        elif self.backbone == "gemma4":
            if AutoModelForMultimodalLM is None:
                raise ImportError("gemma4 requires a Transformers version with AutoModelForMultimodalLM")
            logger.info("[Gemma 4] Loading AutoProcessor & AutoModelForMultimodalLM...")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = self.processor.tokenizer
            model = AutoModelForMultimodalLM.from_pretrained(
                model_path,
                dtype=self.torch_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        else:
            logger.info("[Else] Loading Generic...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self.torch_dtype
            )

        model.eval()
        self.model = self.accelerator.prepare(model)

    def prepare_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        question = sample['question']
        content = []

        if 'image' in sample and sample['image'] is not None:
            images = sample['image'] if isinstance(sample['image'], list) else [sample['image']]
            pending_images = list(images)
            while pending_images:
                img = pending_images.pop(0)
                if img is not None:
                    is_lazy, resolved_images = resolve_lazy_hf_media(
                        img, "__lazy_hf_images__", "images"
                    )
                    if is_lazy:
                        if isinstance(resolved_images, (list, tuple)):
                            pending_images[0:0] = list(resolved_images)
                        elif resolved_images is not None:
                            pending_images.insert(0, resolved_images)
                        continue
                    if isinstance(img, dict) and img.get("type") == "tar_image":
                        archive_path = img.get("archive")
                        member_name = img.get("member")
                        if not archive_path or not member_name:
                            raise ValueError(f"Invalid tar image reference: {img}")
                        archives = getattr(self, "_image_archives", None)
                        if archives is None:
                            archives = self._image_archives = {}
                        archive = archives.get(archive_path)
                        if archive is None:
                            archive = archives[archive_path] = tarfile.open(archive_path, mode="r:")
                        member_file = archive.extractfile(member_name)
                        if member_file is None:
                            raise ValueError(f"Image member not found: {archive_path}!{member_name}")
                        with member_file, Image.open(member_file) as image:
                            img = image.convert('RGB').copy()
                    elif isinstance(img, dict) and img.get("type") == "zip_image":
                        archive_path = img.get("archive")
                        member_name = img.get("member")
                        if not archive_path or not member_name:
                            raise ValueError(f"Invalid zip image reference: {img}")
                        archives = getattr(self, "_zip_image_archives", None)
                        if archives is None:
                            archives = self._zip_image_archives = {}
                        archive = archives.get(archive_path)
                        if archive is None:
                            archive = archives[archive_path] = zipfile.ZipFile(archive_path, mode="r")
                        try:
                            member_file = archive.open(member_name, mode="r")
                        except KeyError as exc:
                            raise ValueError(
                                f"Image member not found: {archive_path}!{member_name}"
                            ) from exc
                        with member_file, Image.open(member_file) as image:
                            img = image.convert("RGB").copy()
                    elif isinstance(img, bytes):
                        with Image.open(io.BytesIO(img)) as image:
                            img = image.convert('RGB').copy()
                    elif isinstance(img, dict) and img.get("bytes") is not None:
                        with Image.open(io.BytesIO(img["bytes"])) as image:
                            img = image.convert('RGB').copy()
                    elif isinstance(img, dict) and img.get("path"):
                        img = img["path"]
                    if isinstance(img, (str, Path)):
                        try:
                            with Image.open(img) as image:
                                img = image.convert('RGB').copy()
                        except (OSError, ValueError) as exc:
                            raise ValueError(f"Failed to load image: {img}") from exc
                    if not isinstance(img, Image.Image):
                        try: img = Image.fromarray(img)
                        except: continue
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    content.append({"type": "image", "image": img})
        
        # Video support
        if 'video' in sample and sample['video'] is not None:
            videos = sample['video'] if isinstance(sample['video'], list) else [sample['video']]
            for vid in videos:
                is_lazy, resolved_video = resolve_lazy_hf_media(
                    vid, "__lazy_hf_video__", "videos"
                )
                if is_lazy:
                    vid = resolved_video
                    if not isinstance(vid, (list, tuple)) or not vid:
                        raise ValueError(
                            "Lazy video descriptor resolved to an empty or invalid frame list"
                        )
                content.append({"type": "video", "video": vid})

        content.append({"type": "text", "text": question})
        messages = []
        if sample.get("system_prompt"):
            messages.append({"role": "system", "content": sample["system_prompt"]})
        messages.append({"role": "user", "content": content})
        return messages


    def _safe_image(self, img, size=244):
        if img is None: 
            return None
        
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                return None

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((size, size), Image.BICUBIC)

    def _process_video_frames(self, video_input, max_num=1, num_segments=8):
        """Helper to process video frames for InternVL"""
        frames = video_input if isinstance(video_input, list) else []
        
        if not frames: 
            raise ValueError(f"Invalid video input: {video_input}")
        
        pixel_values_list = []
        for frame in frames:
            tiles = dynamic_preprocess(frame, image_size=448, use_thumbnail=True, max_num=max_num)
            tile_tensors = [self.transform(tile) for tile in tiles]
            pixel_values_list.append(torch.stack([self.transform(tile) for tile in tiles]))
        return torch.cat(pixel_values_list) if pixel_values_list else None


    def _video_to_images(self, video, max_frames=8):

        if not isinstance(video, list) or len(video) == 0:
            return []

        if len(video) <= max_frames:
            return video

        idxs = np.linspace(0, len(video) - 1, max_frames).astype(int)
        return [video[i] for i in idxs]
        
    # =================================================================
    # prepare_input
    # =================================================================
    def prepare_input(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        device = self.accelerator.device
        
        # 1. Molmo
        MAX_IMAGES=8
        if self.backbone == "molmo":
            raw_text = ""
            raw_images = []
            for msg in messages:
                if msg['role'] == 'user':
                    for item in msg['content']:
                        if item['type'] == 'text': raw_text += item['text']
                        elif item['type'] == 'image': raw_images.append(item['image'])
                        elif item["type"] == "video":

                            frames = self._video_to_images(item["video"], max_frames=MAX_IMAGES)

                            for frame in frames:
                                img = self._safe_image(frame)
                                if img is None:
                                    continue
                                raw_images.append(img)
            
            inputs = self.processor.process(text=raw_text, images=raw_images or [])
            if inputs is None or "input_ids" not in inputs: return None
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point(): v = v.to(self.torch_dtype)
                    inputs[k] = v
            
            if inputs["input_ids"].dim() == 1:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor): inputs[k] = v.unsqueeze(0)
            return inputs

        elif "intern" in self.backbone:
            pixel_values_list = []
            num_patches_list = []
            question_str = ""

            for msg in messages:
                if msg['role'] == 'user':
                    for item in msg['content']:
                        if item['type'] == 'text':
                            question_str += item['text']
                        elif item['type'] == 'image':
                            # Process Image
                            tiles = dynamic_preprocess(item['image'], image_size=448, use_thumbnail=True, max_num=12)
                            tile_tensors = [self.transform(tile) for tile in tiles]
                            stack_tensor = torch.stack(tile_tensors) 
                            
                            pixel_values_list.append(stack_tensor)
                            num_patches_list.append(stack_tensor.size(0))
                            question_str += "<image>\n"
                        elif item['type'] == 'video':
                            vid_tensor = self._process_video_frames(item['video'], max_num=1)
                            if vid_tensor is not None:
                                pixel_values_list.append(vid_tensor)
                                num_patches_list.append(vid_tensor.size(0))
                                question_str += "<image>\n" 

            if not pixel_values_list:
                return {"question": question_str, "pixel_values": None, "num_patches_list": []}

            pixel_values = torch.cat(pixel_values_list, dim=0).to(device).to(self.torch_dtype)
            
            return {
                "pixel_values": pixel_values,
                "question": question_str,
                "num_patches_list": num_patches_list
            }
        elif self.backbone == "magma":
            raw_text = ""
            raw_images = []
            for msg in messages:
                if msg['role'] == 'user':
                    for item in msg['content']:
                        if item['type'] == 'text':
                            raw_text += item['text']
                        elif item['type'] == 'image':
                            raw_images.append(item['image'])
            if raw_images:
                image_tags = "<image_start><image><image_end>\n" * len(raw_images)
                content_str = image_tags + raw_text
            else:
                content_str = raw_text

            convs = [
                {"role": "system", "content": "You are agent that can see, talk and act."},
                {"role": "user", "content": content_str},
            ]
            
            prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
            
            # Magma processing
            inputs = self.processor(images=raw_images if raw_images else None, texts=prompt, return_tensors="pt")

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if k == 'pixel_values':
                        if v.dim() == 4:
                             v = v.unsqueeze(0)
                        elif v.dim() == 3:
                             v = v.unsqueeze(0).unsqueeze(0)
                    if k == 'image_sizes':
                        if v.dim() == 2:
                             v = v.unsqueeze(0)
                        elif v.dim() == 1:
                             v = v.unsqueeze(0).unsqueeze(0)
                    
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(self.torch_dtype)
                    inputs[k] = v
            
            return inputs

        elif self.backbone == "mimo":
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 3. Create Inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # 4. Move to device & correct dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    
                    if v.is_floating_point():
                        v = v.to(self.torch_dtype)
                    inputs[k] = v
            return inputs
        elif self.backbone in ["qwen3", "qwen3_5"]:
            template_kwargs = {}
            if self.backbone == "qwen3_5":
                template_kwargs["enable_thinking"] = False
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

            image_inputs, video_items, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=16,
            )
            video_inputs = None
            video_metadata = None
            if video_items:
                video_inputs = [item[0] for item in video_items]
                video_metadata = [item[1] for item in video_items]

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(self.torch_dtype)
                    inputs[k] = v
            return inputs
        elif self.backbone == "gemma4":
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    if value.is_floating_point():
                        value = value.to(self.torch_dtype)
                    inputs[key] = value
            return inputs
        else:
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(self.torch_dtype)
                    inputs[k] = v
            
            return inputs


    # =================================================================
    # batch_inference
    # =================================================================
    def batch_inference(self, prepared_dataset: List[Dict[str, Any]], **kwargs) -> List[str]:
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        top_k = kwargs.get('top_k', -1)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        local_results = []
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with self.accelerator.split_between_processes(prepared_dataset) as shard:
            for sample in tqdm(shard, desc=f"Rank {self.accelerator.process_index}", disable=not self.accelerator.is_local_main_process):
                try:
                    msgs = self.prepare_messages(sample)
                    inputs = self.prepare_input(msgs)
                    
                    if inputs is None:
                        local_results.append("")
                        continue

                    if self.backbone == "molmo":
                        with torch.no_grad():
                            gen_cfg = GenerationConfig(
                                max_new_tokens=max_tokens,
                                stop_strings="<|endoftext|>",
                                do_sample=temperature > 0,
                                temperature=max(temperature, 1e-5)
                            )
                            output = unwrapped_model.generate_from_batch(
                                inputs, gen_cfg, tokenizer=self.processor.tokenizer
                            )
                            input_len = inputs["input_ids"].size(1)
                            text = self.processor.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
                            local_results.append(text)
                    
                    elif "intern" in self.backbone:
                        gen_config = dict(
                            max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                        )
                        if temperature > 0:
                            gen_config["temperature"] = temperature

                        with torch.no_grad():
                            response = unwrapped_model.chat(
                                tokenizer=self.tokenizer,
                                pixel_values=inputs['pixel_values'],
                                question=inputs['question'],
                                generation_config=gen_config,
                                num_patches_list=inputs['num_patches_list'],
                                history=None,
                                return_history=False
                            )
                            local_results.append(response)
                    elif self.backbone == "magma":
                        with torch.no_grad():
                            gen_kwargs = { 
                                "max_new_tokens": max_tokens, 
                                "temperature": max(temperature, 1e-5), 
                                "do_sample": temperature > 0, 
                                "use_cache": True,
                                "num_beams": 1,
                            }
                            generate_ids = unwrapped_model.generate(**inputs, **gen_kwargs)
                            
                            # Slice input tokens
                            input_len = inputs["input_ids"].shape[-1]
                            generate_ids = generate_ids[:, input_len:]
                            text = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                            local_results.append(text)
                    elif self.backbone == "mimo":
                        with torch.no_grad():
                            generated_ids = unwrapped_model.generate(**inputs, max_new_tokens=max_tokens)
                            
                            generated_ids_trimmed = [
                                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                            ]
                            
                            text = self.processor.batch_decode(
                                generated_ids_trimmed, 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False
                            )[0]
                            local_results.append(text)
                    elif self.backbone == "qwen3_5":
                        with torch.no_grad():
                            gen_kwargs = {
                                "max_new_tokens": max_tokens,
                                "do_sample": temperature > 0,
                                "repetition_penalty": repetition_penalty,
                            }
                            if temperature > 0:
                                gen_kwargs["temperature"] = temperature
                                gen_kwargs["top_p"] = top_p
                                if top_k is not None and top_k > 0:
                                    gen_kwargs["top_k"] = top_k
                            outputs = unwrapped_model.generate(**inputs, **gen_kwargs)
                            input_len = inputs["input_ids"].size(1)
                            text = self.processor.decode(
                                outputs[0, input_len:], skip_special_tokens=True
                            ).strip()
                            text = re.sub(
                                r"^<think>.*?</think>\s*",
                                "",
                                text,
                                flags=re.DOTALL,
                            )
                            local_results.append(text)
                    elif self.backbone == "gemma4":
                        with torch.no_grad():
                            gen_kwargs = {
                                "max_new_tokens": max_tokens,
                                "do_sample": temperature > 0,
                            }
                            if temperature > 0:
                                gen_kwargs["temperature"] = temperature
                            outputs = unwrapped_model.generate(**inputs, **gen_kwargs)
                            input_len = inputs["input_ids"].size(1)
                            text = self.processor.decode(
                                outputs[0, input_len:], skip_special_tokens=False
                            )
                            try:
                                parsed = self.processor.parse_response(text)
                            except (TypeError, ValueError):
                                parsed = None
                            if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
                                text = parsed["content"]
                            text = re.sub(
                                r"^(?:<\|channel>thought\n.*?<channel\|>|thought\n<channel\|>)",
                                "",
                                text,
                                flags=re.DOTALL,
                            )
                            text = text.replace("<turn|>", "").strip()
                            local_results.append(text)
                    else:
                        with torch.no_grad():
                            gen_kwargs = {
                                "max_new_tokens": max_tokens,
                                "do_sample": temperature > 0,
                                "temperature": max(temperature, 1e-5)
                            }
                            outputs = unwrapped_model.generate(**inputs, **gen_kwargs)
                            input_len = inputs["input_ids"].size(1)
                            text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
                            local_results.append(text)

                except Exception as e:
                    logger.error(f"Error on rank {self.accelerator.process_index}: {e}")
                    import traceback
                    traceback.print_exc()
                    local_results.append("")

        self.accelerator.wait_for_everyone()
        return self.accelerator.gather_for_metrics(local_results)
