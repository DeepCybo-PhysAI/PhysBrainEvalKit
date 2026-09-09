import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class EmbSpatialDataset(BaseDataset):
    """EmbSpatial-Bench (Embodied Spatial Benchmark) Dataset"""
    
    def __init__(
        self,
        dataset_name: str = "FlagEval/EmbSpatial-Bench",
        subset: str = None,
        split: str = "test",
        instruct_following: str = None,
        task_name: str = "EmbSpatial-Bench",
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
        """Return default instruction for EmbSpatial-Bench"""
        return ""
    
    def load_dataset(self) -> Any:
        """Load EmbSpatial-Bench dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        if self.subset:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset
    
    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess EmbSpatial-Bench samples"""
        prepared_dataset = []
        map_answer_options = {
            0: "A", 1: "B", 2: "C", 3: "D",
        }

        for idx, sample in enumerate(dataset):
            question_raw = sample["question"].strip()
            answer_options = sample["answer_options"]
            answer_idx = sample["answer"]
            
            # Determine answer letter and text
            if answer_idx in map_answer_options:
                answer_letter = map_answer_options[answer_idx]
                text_answer = f"({answer_letter}) {answer_options[answer_idx]}"
            else:
                # Fallback if index out of range
                answer_letter = str(answer_idx)
                text_answer = str(answer_idx)

            question_id = sample["question_id"]
            relation = sample["relation"]
            
            # Construct Multiple Choice Question Text
            question_text = question_raw
            for i, option in enumerate(answer_options):
                if i in map_answer_options:
                    option_letter = map_answer_options[i]
                    question_text = f"{question_text}\n({option_letter}) {option}"
            
            # Add Instruction
            full_prompt = question_text + "\n" + self.instruct_following
            question = textwrap.dedent(full_prompt).strip()
            
            prepared_dataset.append({
                'question': question,
                'answer': answer_letter,  # Storing "A", "B", etc.
                'image': sample['image'],
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'answer': answer_letter,
                    'answer_options': answer_options,
                    'text_answer': text_answer,
                    'relation': relation
                }
            })
        
        return prepared_dataset
    
    @staticmethod
    def extract_answer_from_text(text: str) -> str:
        """
        Extract answer letter (A-D) from generated text
        
        Supports multiple formats:
        - Answer: (A)
        - **Answer: (A)
        - Answer: (A)
        - Direct search for (A), (B), (C), (D)
        """
        text = str(text).strip()
        
        # Strategy 1: Try "Answer:" or "Answer:" patterns
        answer_patterns = [
            r'(?:Answer|Answer)[:\s]*\(([A-D])\)',  # Answer: (A)
            r'\*\*Answer[:\s]*\(([A-D])\)',        # **Answer: (A)
            r'Answer[:\s]*\(([A-D])\)',            # Answer: (A)
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        # Strategy 2: Search for all occurrences of (A), (B), (C), (D)
        all_matches = list(re.finditer(r'\(([A-D])\)', text, re.IGNORECASE))
        if all_matches:
            return all_matches[-1].group(1).upper()
        
        # Strategy 3: Check for single letter A-D
        if len(text) < 10:
            match = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
        return None
    
    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for EmbSpatial-Bench evaluation"""
        original_raw_output = raw_output_text

        # Extract content from <answer></answer> tags for thinking models
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        question = processed_sample['question']
        answer = processed_sample['answer']  # This is "A", "B", etc.
        metadata = processed_sample['metadata']
        idx = metadata['idx']
        question_id = metadata['question_id']

        # Convert to string
        raw_output_str = str(raw_output_text)
        
        # Try to extract answer letter from the response
        extracted_answer = self.extract_answer_from_text(raw_output_str)
        
        # Determine if the answer is correct
        if extracted_answer:
            is_correct = (extracted_answer.upper() == answer.upper())
            processed_prediction = extracted_answer
        else:
            is_correct = False
            processed_prediction = None
        
        return {
            'idx': idx,
            'question_id': question_id,
            'question': question,
            'ground_truth': answer,
            'metadata': metadata,
            'raw_output': original_raw_output,
            'processed_answer': processed_prediction,
            'is_correct': is_correct,
        }
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for EmbSpatial-Bench"""
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
        
        # Per relation statistics
        relation_results = {}
        for result in results:
            relation = result['metadata']['relation']
            if relation not in relation_results:
                relation_results[relation] = {"total": 0, "correct": 0}
            relation_results[relation]["total"] += 1
            if result['is_correct']:
                relation_results[relation]["correct"] += 1
        
        logger.info("\nPer Relation Accuracy")
        logger.info("=" * 60)
        for relation, counts in sorted(relation_results.items()):
            relation_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            logger.info(f"{relation}: {relation_accuracy:.4f} ({relation_accuracy*100:.2f}%) - {counts['correct']}/{counts['total']}")
        logger.info("=" * 60 + "\n")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'relation_results': relation_results
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
                logger.info(f"Question: {str(result.get('question', ''))}...")
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
