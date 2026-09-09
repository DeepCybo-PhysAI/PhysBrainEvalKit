import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from core.point_utils import compute_point_metrics, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class PartAffordDataset(BaseDataset):
    """Part-Affordance-2K (IffYuan/Part-Affordance-2K)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/Part-Affordance-2K",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "Part-Affordance-2K",
        model_name: Optional[str] = None,
        debug: bool = False,
        backbone: Optional[str] = None,
        thinking_model: bool = False,
        prompt_policy: str = "internal",
    ):
        super().__init__(instruct_following)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.task_name = task_name
        self.model_name = model_name
        self.debug = debug
        self.thinking_model = thinking_model
        self.backbone = backbone
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def get_default_instruct(self) -> str:
        """Return default instruction based on model type"""
        if self.prompt_policy == "original" and self.backbone == "qwen3":
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. All values must be integers between 0 and 1000."""
        if self.backbone in ["gemma4", "qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            # return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. All values must be integers between 0 and 1000."""
            return """The answer must be a JSON list: [{\"point_2d\": [x, y]}]\nCoordinates must be integers normalized to [0, 1000]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
        elif self.backbone == "gemini_robotics":
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone=="internvl" or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess Part-Affordance-2K samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            question_id = sample['question_id']
            problem = sample['problem']
            image = sample['image']   
            width, height = image.size
            mask = sample['mask']

            if self.prompt_policy == "original":
                question = f"Provide one or more points coordinate of objects region this sentence describes: {problem.strip()}."
            else:
                question = f"Provide exactly one point in the object region described by this sentence: {problem.strip()}."
            full_problem = question + '\n' + self.instruct_following
            formatted_question = textwrap.dedent(full_problem).strip()

            prepared_dataset.append({
                'question': formatted_question,
                'answer': None,
                'image': image,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'task': 'part_affordance',
                    'category_type': sample.get('category_type'),
                    'mask': mask,
                    'width': width,
                    'height': height,
                }
            })

        return prepared_dataset

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        metadata = prepared_sample["metadata"]
        scored = score_single_mask_sample(
            raw_output_text,
            mask=metadata["mask"],
            width=metadata["width"],
            height=metadata["height"],
            backbone=self.backbone,
        )
        return {
            "idx": metadata["idx"],
            "question_id": metadata["question_id"],
            "question": prepared_sample["question"],
            "category_type": metadata.get("category_type"),
            "raw_output": raw_output_text,
            **scored,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("\nEvaluating results...")
        all_results: List[Dict[str, Any]] = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            all_results.append(self.process_raw_output(sample, raw_output))
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = compute_point_metrics(results)
        metrics.update({
            "overall_accuracy": metrics["strict_micro_f1"],
            "average_accuracy": metrics["strict_macro_f1"],
            "total_points_predicted": sum(r["total_points"] for r in results),
            "total_points_in_mask": sum(r["points_in_mask"] for r in results),
        })
        logger.info(
            "Point metrics: non-strict micro P/R/F1=%.4f/%.4f/%.4f, "
            "strict micro P/R/F1=%.4f/%.4f/%.4f",
            metrics["non_strict_micro_precision"],
            metrics["non_strict_micro_recall"],
            metrics["non_strict_micro_f1"],
            metrics["strict_micro_precision"],
            metrics["strict_micro_recall"],
            metrics["strict_micro_f1"],
        )
        return metrics

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        import os

        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
