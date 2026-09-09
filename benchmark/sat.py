import json
import logging
import os
import random
import re
import textwrap
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

class SATDataset(BaseDataset):
    """SAT dataset (local parquet or HuggingFace dataset)"""
    
    def __init__(
        self,
        dataset_name: str = "FlagEval/SAT",
        subset: str = "default",
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "SAT",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False
    ):
        super().__init__(instruct_following)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.task_name = task_name
        self.model_name = model_name
        self.backbone = backbone
        self.debug = debug
        self.thinking_model = thinking_model

    def get_default_instruct(self) -> str:
        """Return default instruction for SAT"""
        return ""

    @staticmethod
    def resize_image_if_needed(image: Any, max_long_edge: int = 3500) -> Any:
        """
        Resize image proportionally if its longest edge exceeds max_long_edge

        Args:
            image: PIL Image object or other image format
            max_long_edge: Maximum long edge limit (default 3500 pixels)

        Returns:
            Processed image object
        """
        width, height = image.size
        long_edge = max(width, height)

        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize image using high-quality LANCZOS resampling
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            return resized_image

        return image

    def load_dataset(self) -> Any:
        """Load SAT dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess SAT samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            prompt = sample['question'].strip()
            choices = sample['answers']
            correct_answer_text = sample['correct_answer']

            # Shuffle the options
            shuffled_choices = choices.copy()
            random.shuffle(shuffled_choices)

            # Format as "option1, option2, ... or last_option"
            if len(shuffled_choices) > 1:
                answer_str = ", ".join(shuffled_choices[:-1]) + " or " + shuffled_choices[-1]
            else:
                answer_str = shuffled_choices[0]

            sat_prompt = f"\nChoose between the following options: {answer_str}\nPlease directly answer the question by providing the exact text of your chosen answer.\n"
            problem = prompt + sat_prompt + self.instruct_following
            question = textwrap.dedent(problem).strip()

            images = sample['images']
            if images is not None:
                if isinstance(images, list):
                    images = [self.resize_image_if_needed(img) for img in images]
                else:
                    images = self.resize_image_if_needed(images)

            prepared_dataset.append({
                'question': question,
                'answer': correct_answer_text,
                'image': images,
                'metadata': {
                    'idx': idx,
                    'question_id': sample['question_id'],
                    'task': sample['question_type'],
                    'choices': choices,
                    'correct_answer_text': correct_answer_text
                }
            })

        return prepared_dataset
    
    @staticmethod
    def extract_answer_from_text(text: str, choices: List[str]) -> Optional[str]:
        """
        Extract answer text from generated text by matching against provided choices

        Args:
            text: The generated text from the model
            choices: List of possible answer choices

        Returns:
            The matched choice text or None if no match found
        """
        text = str(text).strip()

        # Normalize text for comparison (lowercase, remove extra whitespace)
        normalized_text = ' '.join(text.lower().split())

        # Try to find exact or partial matches with choices
        best_match = None
        best_match_score = 0

        for choice in choices:
            normalized_choice = ' '.join(choice.lower().split())

            # Exact match
            if normalized_choice == normalized_text:
                return choice

            # Check if choice is contained in the text
            if normalized_choice in normalized_text:
                # Prefer longer matches
                if len(normalized_choice) > best_match_score:
                    best_match = choice
                    best_match_score = len(normalized_choice)

        return best_match

    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for SAT evaluation"""
        original_raw_output = raw_output_text

        # Extract content from <answer></answer> tags for thinking models
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        question = processed_sample['question']
        answer = processed_sample['answer']
        metadata = processed_sample['metadata']
        idx = metadata['idx']
        choices = metadata['choices']

        # Convert to string
        raw_output_str = str(raw_output_text)

        # Try to extract answer text from response by matching against choices
        extracted_answer = self.extract_answer_from_text(raw_output_str, choices)

        # Check if answer is correct by comparing text
        if extracted_answer:
            # Normalize both strings for comparison
            is_correct = (' '.join(extracted_answer.lower().split()) == ' '.join(answer.lower().split()))
            processed_prediction = extracted_answer
        else:
            is_correct = False
            processed_prediction = None

        return {
            'idx': idx,
            'question': question,
            'ground_truth': answer,
            'metadata': metadata,
            'raw_output': original_raw_output,
            'processed_answer': processed_prediction,
            'is_correct': is_correct,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        """Evaluate all results"""
        logger.info("\nEvaluating results...")
        all_results = []
        
        correct_count = 0
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # Log first few samples in detail
            if self.debug or (len(all_results) <= 5):
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('idx', 'N/A')}")
                logger.info(f"Question: {str(result.get('question', ''))[:100]}...")
                logger.info(f"Raw Output: {result['raw_output']}")
                logger.info(f"Processed Prediction: {result.get('processed_answer', 'N/A')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
                logger.info("=" * 60)
        
        logger.info(f"\n✓ Evaluation completed: {correct_count}/{len(all_results)} correct")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for SAT"""
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
        
        # Per-task statistics
        sub_task_results = {}
        for result in results:
            sub_task = result['metadata']['task']
            if sub_task not in sub_task_results:
                sub_task_results[sub_task] = {"total": 0, "correct": 0}
            sub_task_results[sub_task]["total"] += 1
            if result['is_correct']:
                sub_task_results[sub_task]["correct"] += 1
        
        logger.info("\nPer-task Accuracy")
        logger.info("=" * 60)
        for sub_task, counts in sorted(sub_task_results.items()):
            sub_task_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            logger.info(f"{sub_task}: {sub_task_accuracy:.4f} ({sub_task_accuracy*100:.2f}%) - {counts['correct']}/{counts['total']}")
        logger.info("=" * 60 + "\n")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'sub_task_results': sub_task_results
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save evaluation results to file"""
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{self.task_name}_{self.model_name}.json'
        
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'statistics': statistics,
            }, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name