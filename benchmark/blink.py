import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class BLINKDataset(BaseDataset):
    """BLINK Benchmark Dataset"""

    SUBSETS = [
        "Counting",
        "Relative_Depth",
        "Spatial_Relation"
    ]

    def __init__(
        self,
        dataset_name: str = "BLINK-Benchmark/BLINK",
        subset: str = None,
        split: str = "val",
        instruct_following: str = None,
        task_name: str = "BLINK",
        model_name: str = None,
        backbone: str = None,
        debug: bool = False,
        thinking_model: bool = False,
        prompt_policy: str = "original",
    ):
        super().__init__(instruct_following)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.task_name = task_name
        self.model_name = model_name
        self.backbone = backbone
        self.debug = debug
        self.thinking_model=thinking_model
        self.prompt_policy = validate_prompt_policy(prompt_policy)

        # Validate subset
        if self.subset and self.subset not in self.SUBSETS:
            logger.warning(f"Subset '{self.subset}' not in predefined list: {self.SUBSETS}")

    def get_default_instruct(self) -> str:
        """Return default instruction for BLINK"""
        return ""

    @staticmethod
    def format_choice_letters(num_choices: int) -> str:
        """Format available option letters for the answer-format instruction."""
        if num_choices <= 0:
            return "A, B, C, or D"
        letters = [chr(ord("A") + i) for i in range(num_choices)]
        if len(letters) == 1:
            return letters[0]
        if len(letters) == 2:
            return f"{letters[0]} or {letters[1]}"
        return f"{', '.join(letters[:-1])}, or {letters[-1]}"

    def load_dataset(self) -> Any:
        """Load BLINK dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        if self.subset:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        else:
            # Load all subsets
            logger.info("Loading all BLINK subsets...")
            all_data = []
            for subset_name in self.SUBSETS:
                logger.info(f"Loading subset: {subset_name}")
                subset_data = load_dataset(self.dataset_name, name=subset_name, split=self.split)
                all_data.extend(list(subset_data))
            # Create a combined dataset-like object
            dataset = all_data

        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess BLINK samples

        Expected fields in BLINK dataset:
        - idx: unique identifier
        - sub_task: task category
        - image_1, image_2, ..., image_x: multiple images
        - choices: answer choices (already embedded in prompt)
        - answer: ground truth answer
        - prompt: the question text with choices already embedded
        """
        prepared_dataset = []

        for idx, sample in enumerate(dataset):
            # Extract basic fields
            question_id = sample.get("idx", "")
            sub_task = sample.get("sub_task", "")
            prompt = sample.get("prompt", "").strip()
            answer = sample.get("answer", "")
            answer = answer[1]
            choices = sample.get("choices", [])

            # Collect all images (image_1, image_2, ..., image_x)
            images = []
            image_idx = 1
            while f"image_{image_idx}" in sample:
                img = sample[f"image_{image_idx}"]
                if img is not None:
                    images.append(img)
                image_idx += 1

            # Construct the full question with an instruction matching the actual choice count.
            if self.prompt_policy == "original":
                full_prompt = prompt + "\nPlease answer with the letter of your choice (A, B, C, or D)."
            else:
                choice_letters = self.format_choice_letters(len(choices))
                full_prompt = prompt + f"\nPlease answer with the letter of your choice ({choice_letters})."
            if self.instruct_following:
                full_prompt = prompt + "\n" + self.instruct_following

            question = textwrap.dedent(full_prompt).strip()

            # Prepare the sample
            prepared_sample = {
                'question': question,
                'answer': answer,
                'image': images,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'sub_task': sub_task,
                    'answer': answer,
                    'choices': choices,
                    'num_images': len(images)
                }
            }

            prepared_dataset.append(prepared_sample)

        return prepared_dataset

    @staticmethod
    def extract_answer_from_text(text: str) -> str:
        """
        Extract answer letter (A-L) from generated text

        Supports multiple formats:
        - Answer: (A)
        - **Answer: (A)
        - localized format: Answer: (A)
        - Direct search for (A), (B), (C), etc.
        - Single letter A-L
        """
        text = str(text).strip()

        # Strategy 1: Try "Answer:" or localized "Answer:" patterns
        answer_patterns = [
            r'(?:Answer|Answer)[:\s]*\(([A-L])\)',  # Answer: (A) or Answer: (A)
            r'\*\*Answer[:\s]*\(([A-L])\)',        # **Answer: (A)
            r'Answer[:\s]*\(([A-L])\)',            # Answer: (A)
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        # Strategy 2: Search for all occurrences of (A), (B), (C), etc.
        all_matches = list(re.finditer(r'\(([A-L])\)', text, re.IGNORECASE))
        if all_matches:
            return all_matches[-1].group(1).upper()

        # Strategy 3: Check for single letter A-L
        if len(text) < 10:
            match = re.search(r'\b([A-L])\b', text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for BLINK evaluation"""
        original_raw_output = raw_output_text
        question = processed_sample['question']
        answer = processed_sample['answer']
        metadata = processed_sample['metadata']
        idx = metadata['idx']
        if self.thinking_model:
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()
        # Convert to string
        raw_output_str = str(raw_output_text)

        # Try to extract answer from the response
        extracted_answer = self.extract_answer_from_text(raw_output_str)

        # Determine if the answer is correct
        # Normalize both answers for comparison
        ground_truth = str(answer).strip().upper()

        if extracted_answer:
            is_correct = (extracted_answer == ground_truth)
            processed_prediction = extracted_answer
        else:
            is_correct = False
            processed_prediction = None

        return {
            'idx': idx,
            'question': question,
            'ground_truth': ground_truth,
            'metadata': metadata,
            'raw_output': original_raw_output,
            'processed_answer': processed_prediction,
            'is_correct': is_correct,
        }

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for BLINK"""
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

        # Per sub_task statistics
        subtask_results = {}
        for result in results:
            sub_task = result['metadata']['sub_task']
            if sub_task not in subtask_results:
                subtask_results[sub_task] = {"total": 0, "correct": 0, "accuracy": 0.0}
            
            subtask_results[sub_task]["total"] += 1
            if result['is_correct']:
                subtask_results[sub_task]["correct"] += 1

        logger.info("\nPer Sub-task Accuracy")
        logger.info("=" * 60)
        
        for sub_task, counts in sorted(subtask_results.items()):
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            subtask_results[sub_task]["accuracy"] = accuracy
            
            logger.info(f"{sub_task}: {accuracy:.4f} ({accuracy*100:.2f}%) - {counts['correct']}/{counts['total']}")
            
        logger.info("=" * 60 + "\n")

        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'subtask_results': subtask_results
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
                logger.info(f"Question: {str(result.get('question', ''))[:200]}...")
                logger.info(f"Raw Output: {result['raw_output']}")
                logger.info(f"Processed Prediction: {result.get('processed_answer', 'N/A')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
                logger.info("=" * 60)

        logger.info(f"\n✓ Evaluation completed: {correct_count}/{len(all_results)} correct")
        return all_results

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save evaluation results to file"""
        os.makedirs('logs/results', exist_ok=True)

        # Include subset in filename if specified
        if self.subset:
            result_file_name = f'logs/results/{self.task_name}_{self.subset}_{self.model_name}.json'
        else:
            result_file_name = f'logs/results/{self.task_name}_{self.model_name}.json'

        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'statistics': statistics,
            }, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
