import json
import logging
import os
import re
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class ERQADataset(BaseDataset):
    """ERQA (Embodied Relational Question Answering) Dataset"""
    
    def __init__(
        self,
        dataset_name: str = "FlagEval/ERQA",
        subset: str = None,
        split: str = "test",
        instruct_following: str = None,
        task_name: str = "ERQA",
        model_name: str = None,
        backbone: str = None,
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
        """Return default instruction for ERQA"""
        return ""
    
    def load_dataset(self) -> Any:
        """Load ERQA dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        if self.subset:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset
    
    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess ERQA samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            original_question = sample['question'].strip()
            answer = sample['answer']  # Already in format "A", "B", "C", or "D"
            question_id = sample['question_id']
            question_type = sample['question_type']
            images = sample['images']  # List of images
            visual_indices = sample['visual_indices']
            
            # Add instruction if provided
            if self.instruct_following:
                question = original_question + "\n" + self.instruct_following
            else:
                question = original_question
            
            prepared_dataset.append({
                'question': question,
                'answer': answer,
                'image': images,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'question_type': question_type,
                    'answer': answer,
                    'num_images': len(images),
                    'visual_indices': visual_indices,
                    'original_question': original_question,
                }
            })
        
        return prepared_dataset
    
    @staticmethod
    def extract_answer_from_text(text: str) -> str:
        """
        Extract answer letter (A-D) from generated text
        
        Supports multiple formats:
        - Answer: A
        - Answer: (A)
        - **Answer: (A)
        - Answer: (A)
        - Single letter A-D at end of text
        """
        text = str(text).strip()
        
        # Strategy 1: Check for standalone letter at end
        standalone_end_match = re.search(r'\b([A-D])\b[.\s]*$', text, re.IGNORECASE)
        if standalone_end_match:
            return standalone_end_match.group(1).upper()
        
        # Strategy 2: Try "Answer:" or "Answer:" patterns
        answer_patterns = [
            r'(?:Answer|Answer)[:\s]*\(([A-D])\)',
            r'\*\*Answer[:\s]*\(([A-D])\)',
            r'Answer[:\s]*\(([A-D])\)',
            r'(?:Answer|Answer)[:\s]*([A-D])[.\s]*',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        # Strategy 3: Search all occurrences of (A), (B), (C), (D)
        all_matches = list(re.finditer(r'\(([A-D])\)', text, re.IGNORECASE))
        if all_matches:
            return all_matches[-1].group(1).upper()
        
        # Strategy 4: Find any standalone letter
        standalone_match = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
        if standalone_match:
            return standalone_match.group(1).upper()
        
        return None
    
    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for ERQA evaluation"""
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
        question_id = metadata['question_id']
        question_type = metadata['question_type']
        
        # Convert to string
        raw_output_str = str(raw_output_text)
        
        # Extract answer letter
        answer_letter = answer.strip().upper()
        extracted_answer = self.extract_answer_from_text(raw_output_str)
        
        # Check if answer is correct
        if extracted_answer:
            is_correct = (extracted_answer.upper() == answer_letter.upper())
            processed_prediction = extracted_answer
        else:
            is_correct = False
            processed_prediction = None
        
        return {
            'idx': metadata['idx'],
            'question_id': question_id,
            'question_type': question_type,
            'question': question,
            'metadata': metadata,
            'ground_truth': answer_letter,
            'raw_output': original_raw_output,
            'processed_answer': processed_prediction,
            'is_correct': is_correct,
        }
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for ERQA"""
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
        
        # Per question type statistics
        question_type_results = {}
        for result in results:
            question_type = result['metadata']['question_type']
            if question_type not in question_type_results:
                question_type_results[question_type] = {"total": 0, "correct": 0}
            question_type_results[question_type]["total"] += 1
            if result['is_correct']:
                question_type_results[question_type]["correct"] += 1
        
        logger.info("\nPer Question Type Accuracy")
        logger.info("=" * 60)
        for question_type, counts in sorted(question_type_results.items()):
            question_type_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            logger.info(f"{question_type}: {question_type_accuracy:.4f} ({question_type_accuracy*100:.2f}%) - {counts['correct']}/{counts['total']}")
        logger.info("=" * 60 + "\n")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'question_type_results': question_type_results
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
                logger.info(f"Question ID: {result.get('question_id', 'N/A')}")
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
