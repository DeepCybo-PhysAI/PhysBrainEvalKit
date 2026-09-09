import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

class EgoPlan2Dataset(BaseDataset):
    """
    EgoPlan Dataset Adapter (HuggingFace version: IffYuan/ego-plan)
    """

    def __init__(
        self,
        dataset_name: str = "IffYuan/ego-plan",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "EgoPlan2",
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
        return ""

    def load_dataset(self) -> Any:
        """Load EgoPlan dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        
        for idx, sample in enumerate(dataset):
            video_frames = sample["video"]
            question = sample["question"]
            options = sample["options"] 
            
            choices_text = "\n".join(options)
            full_prompt = (
                f"{question}\n{choices_text}\n"
                "Please answer with only the letter of your choice (A, B, C, or D)."
            )
            answer = sample["answer"]
            
            prepared.append({
                "question": full_prompt,
                "answer": answer,
                "video": [video_frames],
                "image": sample["image"],
                "metadata": {
                    "idx": idx,
                    "question_id": sample["id"],
                    "domain": sample["domain"],
                    "scenario": sample["scenario"],
                    "video_source": sample["video_source"],
                    "options": options
                },
            })

        return prepared

    @staticmethod
    def extract_answer_from_text(text: str) -> Optional[str]:
        """
        Extract answer letter (A-D) from generated text

        Supports multiple formats:
        - Answer: (A)
        - **Answer: (A)
        - Answer: (A)
        - Direct search for (A), (B), (C), (D)
        - Single letter A-D
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

        # Strategy 3: Direct search for letter A-D (word boundary)
        match = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """
        Evaluate a single sample prediction using the logic from the EgoPlan script.
        """
        original_raw_output = raw_output_text

        # Extract content from <answer></answer> tags for thinking models
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        question = prepared_sample['question']
        answer = prepared_sample['answer']
        metadata = prepared_sample['metadata']
        idx = metadata['idx']
        question_id = metadata['question_id']

        # Extract answer using the improved method
        pred_choice = self.extract_answer_from_text(raw_output_text)

        # Calculate Score
        score = 1.0 if pred_choice == answer else 0.0

        return {
            'idx': idx,
            'question_id': question_id,
            'question': question,
            'raw_output': original_raw_output,
            "domain": metadata["domain"],
            "scenario": metadata["scenario"],
            "question_type": "multiple_choice", # Unified type
            'ground_truth': answer,
            'processed_answer': pred_choice,
            'is_correct': score,
        }

    def evaluate_results(
        self,
        prepared_dataset: List[Dict[str, Any]],
        raw_outputs: List[str]
    ) -> List[Dict[str, Any]]:
        logger.info("Evaluating EgoPlan Predictions...")
        results = []

        for sample, output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            try:
                res = self.process_raw_output(sample, output)
                results.append(res)
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.get('id')}: {e}")
                results.append({
                    "idx": sample["metadata"]["idx"],
                    "is_correct": 0.0,
                    "error": str(e),
                    "question_id": sample["metadata"]["question_id"],
                    "domain": sample["metadata"].get("domain", "unknown"),
                    "raw_output": output,
                })

        return results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute Overall Accuracy and Domain-wise Accuracy.
        """
        if not results:
            return {}

        total_score = 0.0
        count = 0

        domain_stats = defaultdict(list)

        for r in results:
            score = r.get("is_correct", 0.0)
            domain = r.get("domain", "unknown")

            total_score += score
            count += 1

            domain_stats[domain].append(score)

        # 1. Overall Accuracy
        overall_acc = total_score / count if count > 0 else 0.0

        final_stats = {
            "overall_accuracy": overall_acc
        }

        # 2. Domain-wise Accuracy
        for domain, scores in domain_stats.items():
            final_stats[f"domain_{domain}_acc"] = sum(scores) / len(scores)

        # Logging
        logger.info("\n" + "="*40)
        logger.info(f"EgoPlan Statistics (Total: {count})")
        logger.info("="*40)
        logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")

        logger.info("\n[Domain Results]")
        for d, scores in domain_stats.items():
            acc = sum(scores)/len(scores)
            logger.info(f"  {d}: {acc*100:.2f}% ({len(scores)})")
        logger.info("="*40)

        return {
            "overall_score": overall_acc,
            "category_results": final_stats
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        """Save evaluation results to file"""
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "statistics": statistics,
            }, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
