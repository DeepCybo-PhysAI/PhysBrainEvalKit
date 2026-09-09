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

class CVBenchDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "nyu-visionx/CV-Bench",
        subset: str = "default",
        split: str = "test",
        instruct_following: str = None,
        task_name: str = "CV-Bench",
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
        self.thinking_model = thinking_model
        self.prompt_policy = validate_prompt_policy(prompt_policy)
    
    def get_default_instruct(self) -> str:
        """Return default instruction for CV-Bench.

        CV-Bench uses a per-sample answer-format instruction because Count
        can have 4, 5, or 6 choices. The concrete instruction is built in
        prepare_dataset() from the sample choices.
        """
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
        """Load CV-Bench dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset
    
    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess CV-Bench samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            prompt = sample['prompt'].strip()
            if self.prompt_policy == "original":
                problem = prompt + '\n' + self.instruct_following
            else:
                choices = sample['choices']
                choice_letters = self.format_choice_letters(len(choices))
                answer_instruction = f"Please answer with the letter of your choice ({choice_letters})."
                if self.instruct_following:
                    answer_instruction = self.instruct_following
                problem = prompt + "\n" + answer_instruction
            question = textwrap.dedent(problem).strip()
            answer = sample['answer']

            prepared_dataset.append({
                'question': question,
                'answer': answer,
                'image': sample['image'],
                'metadata': {
                    'idx': idx,
                    'question_id': sample['idx'],
                    'task': sample['task'],
                    'answer': sample['answer'],
                    'choices': sample['choices'],
                }
            })

        return prepared_dataset
    
    @staticmethod
    def extract_answer_from_text(text: str) -> str:
        """
        Extract answer letter (A-L) from generated text
        
        Supports multiple formats:
        - Answer: (A)
        - **Answer: (A)
        - Answer: (A)
        - Direct search for (A), (B), (C), etc.
        """
        text = str(text).strip()
        
        # Strategy 1: Try "Answer:" or "Answer:" patterns
        answer_patterns = [
            r'(?:Answer|Answer)[:\s]*\(([A-L])\)',  # Answer: (A)
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
        """Process raw output for CV-Bench evaluation"""
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

        # Convert to string
        raw_output_str = str(raw_output_text)
        
        # Extract letter from answer format (A) -> A
        answer_letter = answer.strip("()") if answer.startswith("(") and answer.endswith(")") else answer
        
        # Try to extract answer letter from response
        extracted_answer = self.extract_answer_from_text(raw_output_str)
        
        # Check if answer is correct
        if extracted_answer:
            is_correct = (extracted_answer.upper() == answer_letter.upper())
            processed_prediction = f"({extracted_answer})"
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
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for CV-Bench"""
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
