import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

class RoboFACDataset(BaseDataset):
    """RoboFAC Benchmark Dataset for robotic arm task evaluation"""

    def __init__(
        self,
        data_root: str = "/mnt/18T/hyt/OmniScope/benchmark/robofac",
        json_files: List[str] = None,
        instruct_following: str = None,
        task_name: str = "RoboFAC",
        model_name: str = None,
        backbone: str = None,
        debug: bool = False,
        thinking_model: bool = False,
        max_frames: int = 16,
        fps: int = 2,
        force_reextract: bool = False,
        image_size: int = 336 
    ):
        """
        Initialize RoboFAC Dataset
        
        Args:
            image_size: The maximum size for the longer edge of the image. 
                        Set to -1 to keep original resolution.
        """
        super().__init__(instruct_following)
        self.data_root = data_root
        
        self.json_files = json_files or ["merged_qa_data_transformed.json"]
        self.task_name = task_name
        self.model_name = model_name
        self.backbone = backbone
        self.debug = debug
        self.thinking_model = thinking_model
        self.max_frames = max_frames
        self.fps = fps
        self.force_reextract = force_reextract
        self.image_size = image_size 
        self.cache_root = os.path.join(self.data_root, "frames_cache")
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root, exist_ok=True)
            logger.info(f"Created frame cache directory at: {self.cache_root}")

    def get_default_instruct(self) -> str:
        """Return default instruction for RoboFAC"""
        return ""

    def _resize_image(self, image: Image.Image) -> Image.Image:

        if self.image_size is None or self.image_size <= 0:
            return image
        
        w, h = image.size
        if max(w, h) <= self.image_size:
            return image
            
        ratio = self.image_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def extract_video_frames(self, video_path: str) -> List[Image.Image]:
        """
        Extract frames from video. 
        Logic: Check cache -> Load if exists -> Else Extract & Save
        """
        video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        relative_dir = os.path.dirname(video_path)
        
        video_cache_dir = os.path.join(self.cache_root, relative_dir, video_name_no_ext)
        
        if not self.force_reextract and os.path.exists(video_cache_dir):
            cached_files = sorted([f for f in os.listdir(video_cache_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if len(cached_files) > 0:
                if self.debug:
                    logger.debug(f"Loading cached frames from: {video_cache_dir}")
                
                frames = []
                try:
                    for img_file in cached_files:
                        img_path = os.path.join(video_cache_dir, img_file)
                        img = Image.open(img_path).convert('RGB')
                        
                        img = self._resize_image(img)
                        frames.append(img)
                    return frames
                except Exception as e:
                    logger.warning(f"Error loading cached frames from {video_cache_dir}, re-extracting... Error: {e}")
            else:
                pass

        full_path = os.path.join(self.data_root, "realworld_data", video_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return []
        
        if os.path.getsize(full_path) < 1024:
            logger.warning(f"File too small (corrupted): {full_path}")
            return []

        extracted_frames = []
        extraction_success = False

        if not extraction_success:
            try:
                
                from decord import VideoReader, cpu
                vr = VideoReader(full_path, ctx=cpu(0))
                total_frames = len(vr)
                if total_frames > 0:
                    if total_frames <= self.max_frames:
                        indices = list(range(total_frames))
                    else:
                        indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()
                    batch = vr.get_batch(indices).asnumpy()
                    
                    extracted_frames = [self._resize_image(Image.fromarray(img)) for img in batch]
                    extraction_success = True
            except Exception as e:
                logger.debug(f"Decord failed for {video_path}: {e}")

        if not extraction_success:
            try:
                import av
                container = av.open(full_path)
                if container.streams.video:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    total_frames = stream.frames
                    temp_frames = []
                    
                    if total_frames > 0:
                        if total_frames <= self.max_frames:
                            indices = set(range(total_frames))
                        else:
                            indices = set(np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist())
                        
                        for i, frame in enumerate(container.decode(stream)):
                            if i in indices:
                                
                                temp_frames.append(self._resize_image(frame.to_image()))
                            if i > max(indices): break
                    else:
                        
                        all_frames = [self._resize_image(frame.to_image()) for frame in container.decode(stream)]
                        total_frames = len(all_frames)
                        if total_frames > 0:
                            if total_frames <= self.max_frames:
                                temp_frames = all_frames
                            else:
                                indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).astype(int)
                                temp_frames = [all_frames[i] for i in indices]
                    
                    if temp_frames:
                        extracted_frames = temp_frames
                        extraction_success = True
            except Exception as e:
                logger.debug(f"PyAV failed for {video_path}: {e}")


        if not extraction_success:
            try:
                cap = cv2.VideoCapture(full_path)
                if cap.isOpened():
                    temp_frames = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        if total_frames <= self.max_frames:
                            indices = range(total_frames)
                        else:
                            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                
                                temp_frames.append(self._resize_image(img))
                    cap.release()
                    if temp_frames:
                        extracted_frames = temp_frames
                        extraction_success = True
            except Exception:
                pass


        if extraction_success and extracted_frames:
            try:
                os.makedirs(video_cache_dir, exist_ok=True)
                for f in os.listdir(video_cache_dir):
                    os.remove(os.path.join(video_cache_dir, f))

                for idx, frame in enumerate(extracted_frames):
                    save_path = os.path.join(video_cache_dir, f"frame_{idx:06d}.jpg")
                    
                    frame.save(save_path, quality=95)
                
                if self.debug:
                    logger.info(f"Cached {len(extracted_frames)} frames to {video_cache_dir}")
                    
            except Exception as e:
                logger.error(f"Failed to save cache for {video_path}: {e}")
        
        if not extraction_success:
            logger.error(f"ALL methods failed for video: {full_path}")
            return []

        return extracted_frames

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load RoboFAC dataset from JSON files"""
        all_data = []

        for json_file in self.json_files:
            json_path = os.path.join(self.data_root, json_file)
            logger.info(f"Loading JSON file: {json_path}")

            if not os.path.exists(json_path):
                logger.warning(f"JSON file not found: {json_path}")
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} samples from {json_file}")
                    all_data.extend(data)
            except Exception as e:
                logger.error(f"Error loading {json_path}: {e}")

        logger.info(f"Total samples loaded: {len(all_data)}")
        return all_data

    def prepare_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess RoboFAC samples"""
        prepared_dataset = []

        logger.info("Preparing dataset and extracting video frames...")
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            sample_id = sample.get("id", "")
            video_path = sample.get("video", "")
            task_type = sample.get("type", "")
            question = sample.get("question", "").strip()
            answer = sample.get("answer", "").strip()

            frames = self.extract_video_frames(video_path)

            if not frames:
                logger.warning(f"No frames extracted for sample {sample_id}, skipping")
                continue

            full_prompt = question
            if self.instruct_following:
                full_prompt = question + "\n" + self.instruct_following

            prepared_sample = {
                'question': full_prompt,
                'answer': answer,
                'video': [frames], 
                'metadata': {
                    'id': sample_id,
                    'video': video_path,
                    'type': task_type,
                    'answer': answer,
                    'num_frames': len(frames)
                }
            }

            prepared_dataset.append(prepared_sample)

        logger.info(f"Prepared {len(prepared_dataset)} samples")
        return prepared_dataset

    @staticmethod
    def extract_category_from_text(text: str) -> Optional[str]:

        text = str(text).strip()
        
        VALID_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        if len(text) == 1 and text.upper() in VALID_CHOICES:
            return text.upper()
        
        patterns = [
            r'<answer>\s*([A-H])', 
            r'(?:Answer|Choice|Option|Result)\s*[:\-\s]\s*([A-H])\b', 
            r'The\s*(?:correct|final)?\s*(?:answer|option|choice)\s*(?:is|matches)\s*[:\s]*\(?([A-H])\)?',
            r'\(([A-H])\)',
            r'\b([A-H])\s+is\s+(?:the\s+)?(?:correct|right|answer)',
            r'(?:Therefore|Thus|So|Hence|Consequently|In conclusion).*?(?:correct|final|best)\s*(?:answer|option|choice)\s*(?:is|matches)\s*[:\s-]*\*?\(?([A-H])\)?',
            
            r'The\s*(?:correct|final)?\s*(?:answer|option|choice)\s*(?:is|matches)\s*[:\s]*\*?\(?([A-H])\)?',
            r'\*\*([A-H])\*\*(?:[.)]|\s|$)',
            r'^([A-H])[.)]',
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
            if matches:
                return matches[-1].group(1).upper()

        if len(text) < 100:
            simple_matches = re.findall(r'\b([A-H])\b', text)
            if simple_matches:
                return simple_matches[-1].upper()

        return None

    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:

        question = processed_sample['question']
        ground_truth = processed_sample['answer'].strip().upper()
        metadata = processed_sample['metadata']
        sample_id = metadata['id']

        if self.thinking_model:
            
            raw_output_text = re.sub(
                r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()
            
            answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        predicted_answer = self.extract_category_from_text(raw_output_text)

        is_correct = False
        if predicted_answer is not None:
            
            gt_letter = ground_truth[0] if len(ground_truth) > 0 else ""
            
            if predicted_answer == gt_letter:
                is_correct = True
        
        return {
            'idx': sample_id,
            'question': question,
            'ground_truth': ground_truth,
            'metadata': metadata,
            'raw_output': raw_output_text,
            'processed_answer': predicted_answer if predicted_answer else "Extraction Failed",
            'is_correct': is_correct,
        }

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_predictions = sum(1 for result in results if result['is_correct'])
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        logger.info("\n" + "=" * 60)
        logger.info("Overall Statistics")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        logger.info("=" * 60)

        type_results = {}
        for result in results:
            task_type = result['metadata']['type']
            if task_type not in type_results:
                type_results[task_type] = {"total": 0, "correct": 0, "accuracy": 0.0}
            type_results[task_type]["total"] += 1
            if result['is_correct']:
                type_results[task_type]["correct"] += 1

        logger.info("\nPer Task Type Accuracy")
        logger.info("=" * 60)
        for task_type, counts in sorted(type_results.items()):
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            type_results[task_type]["accuracy"] = accuracy
            logger.info(f"{task_type}: {accuracy:.4f} ({accuracy*100:.2f}%) - {counts['correct']}/{counts['total']}")
        logger.info("=" * 60 + "\n")

        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'type_results': type_results
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("\nEvaluating results...")
        all_results = []
        correct_count = 0
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            if result['is_correct']: correct_count += 1
            if self.debug or (len(all_results) <= 5):
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('idx', 'N/A')}")
                logger.info(f"Task Type: {result['metadata']['type']}")
                logger.info(f"Question: {str(result.get('question', ''))[:200]}...")
                logger.info(f"Raw Output: {result['raw_output'][:200]}...")
                logger.info(f"Extracted: {result['processed_answer']} | GT: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
                logger.info("=" * 60)
        logger.info(f"\n✓ Evaluation completed: {correct_count}/{len(all_results)} correct")
        return all_results

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{self.task_name}_{self.model_name}.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump({'results': results, 'statistics': statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
