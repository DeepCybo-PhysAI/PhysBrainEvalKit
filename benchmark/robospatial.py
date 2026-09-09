import ast
import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from core.framework_integration import robospatial_statistics
from core.point_utils import clean_model_answer, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class RoboSpatialDataset(BaseDataset):
    """
    RoboSpatial-Home benchmark dataset
    
    Supports three spatial reasoning categories:
    - context: Spatial Context (identifying vacant areas)
    - compatibility: Spatial Compatibility (checking if object can fit)
    - configuration: Spatial Configuration (relative positioning)
    
    Reference: https://github.com/chanhee-luke/RoboSpatial-Eval
    Dataset: https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home

    instruct_following only for context category!!!
    """

    def __init__(
        self,
        dataset_name: str = "chanhee-luke/RoboSpatial-Home",
        category: Optional[str] = None,  # "context", "compatibility", "configuration", or None for all
        instruct_following: Optional[str] = None,
        task_name: str = "RoboSpatial",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False,
        prompt_policy: str = "internal",
    ):
        super().__init__(instruct_following)
        self.dataset_name = dataset_name
        self.category = category
        self.task_name = task_name
        self.model_name = model_name
        self.backbone = backbone
        self.debug = debug
        self.thinking_model = thinking_model
        self.instruct_following = instruct_following
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def get_default_instruct(self) -> str:
        """Return default instruction based on model type"""
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
        elif self.backbone == "gemini_robotics":
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl' or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    def load_dataset(self) -> Any:
        """Load RoboSpatial-Home dataset from HuggingFace"""
        if self.category:
            logger.info(f"Loading dataset: {self.dataset_name}, Category: {self.category}")
            dataset = load_dataset(self.dataset_name, split=self.category)
        else:
            logger.info(f"Loading dataset: {self.dataset_name} (all categories)")
            # Load all three splits and combine them
            context_data = load_dataset(self.dataset_name, split="context")
            compatibility_data = load_dataset(self.dataset_name, split="compatibility")
            configuration_data = load_dataset(self.dataset_name, split="configuration")
            
            # Combine all splits
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([context_data, compatibility_data, configuration_data])
        
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess RoboSpatial-Home samples"""
        prepared: List[Dict[str, Any]] = []
        
        for sample_idx, sample in enumerate(dataset):
            try:
                # Get category from sample
                category = sample["category"]
                prompt = str(sample["question"]).strip()
                
                # Construct instruction based on category
                if category == "context":
                    # Spatial Context: Identify empty space
                    target_text = "Your answer should be formatted as a list of tuples"
                    instruction_idx = prompt.find(target_text)
                    if instruction_idx != -1:
                        prompt = prompt[:instruction_idx].strip()
                    
                    if self.prompt_policy == "original":
                        problem = prompt + self.instruct_following
                    else:
                        problem = prompt + "\nMark exactly one point that meets the requirements.\n" + self.instruct_following
                    question = problem.strip()
                elif category in ["compatibility", "configuration"]:
                    # Spatial Compatibility/Configuration: Yes/No questions
                    instr = (
                        "Please answer the question based on the image. "
                        "Respond with only 'yes' or 'no'."
                    )
                    question = textwrap.dedent(prompt + "\n" + instr).strip()

                if category == "context":
                    mask = sample['mask']
                    image = sample['img']
                    width, height = image.size
                else:
                    mask = None
                    image = sample['img']
                    width, height = image.size

                prepared.append({
                    "question": question,
                    'image': sample['img'],
                    'answer': sample['answer'],
                    "metadata": {
                        "idx": sample_idx,
                        "category": category,
                        "original_question": prompt,
                        "mask": mask,
                        "width": width,
                        "height": height,
                    },
                })
            except Exception as e:
                logger.error(f"Failed to prepare sample {sample_idx}: {e}. Skipping.")
        
        if self.debug:
            prepared = prepared[:20]
            logger.info(f"Debug mode: Limited to {len(prepared)} samples")
        
        return prepared

    @staticmethod
    def extract_yes_no_from_text(text: str) -> Optional[str]:
        """
        Extract yes or no answer from generated text
        
        Supports multiple formats:
        - Answer: Yes
        - **Answer: No
        - Answer: Yes (Chinese)
        - Direct search for yes/no
        """
        text = str(text).strip()
        
        # Strategy 1: Try "Answer:" or "Answer:" patterns
        answer_patterns = [
            r'(?:Answer|Answer)[:\s]*(yes|no)',  # Answer: yes
            r'\*\*Answer[:\s]*(yes|no)',        # **Answer: yes
            r'Answer[:\s]*(yes|no)',            # Answer: yes
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].lower()
        
        # Strategy 2: Direct search for yes or no (word boundary)
        yes_match = re.search(r'\b(yes)\b', text, re.IGNORECASE)
        no_match = re.search(r'\b(no)\b', text, re.IGNORECASE)
        
        # If both found, return the last occurrence
        if yes_match and no_match:
            if yes_match.start() > no_match.start():
                return 'yes'
            else:
                return 'no'
        elif yes_match:
            return 'yes'
        elif no_match:
            return 'no'
        
        return None

    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for RoboSpatial-Home evaluation"""
        original_raw_output = raw_output_text

        # Extract common variables
        question = processed_sample['question']
        answer = processed_sample['answer']
        metadata = processed_sample['metadata']
        idx = metadata['idx']
        category = metadata['category']
        
        # Process compatibility and configuration categories (Yes/No questions)
        if category in ["compatibility", "configuration"]:
            # Extract yes/no answer from raw output
            answer_text, _ = clean_model_answer(raw_output_text)
            extracted_answer = self.extract_yes_no_from_text(answer_text)
            gt_lower = answer.lower().strip()
        
            is_correct = (extracted_answer == gt_lower)

            return {
                'idx': idx,
                'category': category,
                'question': question,
                'ground_truth': answer,
                'metadata': metadata,
                'raw_output': original_raw_output,
                'processed_output': extracted_answer,
                'is_correct': is_correct,
                'eval_type': 'binary',
            }
        
        # Process context category (Spatial point prediction).
        elif category == "context":
            scored = score_single_mask_sample(
                raw_output_text,
                mask=metadata["mask"],
                width=metadata["width"],
                height=metadata["height"],
                backbone=self.backbone,
            )
            return {
                "idx": idx,
                "category": category,
                "question": question,
                "raw_output": original_raw_output,
                "eval_type": "spatial",
                **scored,
            }
        else:
            raise ValueError(f"Unknown category: {category}")


    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        """Evaluate all results"""
        logger.info("\nEvaluating results...")
        all_results: List[Dict[str, Any]] = []
        
        correct_count = 0
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            
            if result.get("is_correct"):
                correct_count += 1
            
            # Log first few samples in detail for debugging
            if self.debug or (len(all_results) <= 5):
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('idx', 'N/A')}")
                logger.info(f"Category: {result.get('category', 'N/A')}")
                logger.info(f"Question: {str(result.get('question', ''))[:150]}...")
                logger.info(f"Raw Output: {result['raw_output'][:200]}...")
                
                if result.get('category') == 'context':
                    # Context category - show points accuracy
                    logger.info(f"Processed Points: {result.get('processed_points', [])}")
                    logger.info(f"Points in Mask: {result.get('points_in_mask', 0)}/{result.get('total_points', 0)}")
                    logger.info(f"Accuracy Score: {result.get('accuracy_score', 0.0):.4f}")
                else:
                    # Binary categories
                    logger.info(f"Processed Output: {result.get('processed_output', 'N/A')}")
                    logger.info(f"Ground Truth: {result.get('ground_truth', 'N/A')}")
                    logger.info(f"Is Correct: {result.get('is_correct', False)}")
                
                logger.info(f"Eval Type: {result.get('eval_type', 'N/A')}")
                logger.info("=" * 60)
        
        logger.info(f"\n✓ Evaluation completed")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Report context metrics and both strict/non-strict mixed overalls."""
        context_results = [r for r in results if r.get("category") == "context"]
        binary_results = [
            r for r in results
            if r.get("category") in {"compatibility", "configuration"}
        ]
        binary_correct = sum(1 for r in binary_results if r.get("is_correct"))
        statistics = robospatial_statistics(
            context_results,
            binary_correct=binary_correct,
            binary_total=len(binary_results),
        )

        binary_metrics = {}
        for category in ("compatibility", "configuration"):
            category_results = [r for r in binary_results if r.get("category") == category]
            correct = sum(1 for r in category_results if r.get("is_correct"))
            binary_metrics[category] = {
                "accuracy": correct / len(category_results) if category_results else 0.0,
                "correct": correct,
                "total": len(category_results),
            }

        statistics["binary_metrics"] = binary_metrics
        statistics["average_accuracy"] = statistics["point_metrics"]["strict_macro_f1"]
        logger.info(
            "RoboSpatial mixed strict/non-strict overall=%.4f/%.4f",
            statistics["strict_overall_score"],
            statistics["non_strict_overall_score"],
        )
        return statistics

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save evaluation results to file"""
        os.makedirs("logs/results", exist_ok=True)
        
        # Include category in filename if specified
        if self.category:
            result_file_name = f"logs/results/{self.task_name}_{self.category}_{self.model_name}.json"
        else:
            result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "statistics": statistics
            }, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
