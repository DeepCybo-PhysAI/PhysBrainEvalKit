import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.point_utils import compute_point_metrics, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class RefSpatialBenchDataset(BaseDataset):
    """RefSpatial-Bench (BAAI/RefSpatial-Bench)"""

    def __init__(
        self,
        dataset_name: str = "BAAI/RefSpatial-Bench",
        subset: Optional[str] = None,
        split: Optional[str] = None,
        instruct_following: Optional[str] = None,
        task_name: str = "RefSpatial-Bench",
        model_name: Optional[str] = None,
        debug: bool = False,
        backbone: Optional[str] = None,
        thinking_model: bool = False,
        prompt_policy: str = "original",
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
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
        elif self.backbone == "gemini_robotics":
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl'  or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Loading all splits: location, placement, unseen")
        from datasets import concatenate_datasets
        
        splits = ['location', 'placement', 'unseen']
        datasets = []
        for split_name in splits:
            split_data = load_dataset(self.dataset_name, split=split_name)
            split_data = split_data.map(lambda x, s=split_name: {**x, 'split_name': s})
            datasets.append(split_data)
            logger.info(f"Loaded split '{split_name}': {len(split_data)} samples")
        
        dataset = concatenate_datasets(datasets)
        logger.info(f"All splits combined. Total samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess RefSpatial-Bench samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            question_id = sample['id']
            prompt = sample['prompt']
            image = sample['image']   
            width, height = image.size
            mask = sample['mask']

            problem = prompt + '\n' + self.instruct_following
            formatted_question = problem.strip()

            prepared_dataset.append({
                'question': formatted_question,
                'answer': None,
                'image': image,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'mask': mask,
                    'width': width,
                    'height': height,
                    'split_name': sample.get('split_name'),
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
            "question_id": metadata.get("question_id"),
            "question": prepared_sample["question"],
            "raw_output": raw_output_text,
            "split_name": metadata.get("split_name", "unknown"),
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
        metrics.update(
            {
                "total_points_predicted": sum(r["total_points"] for r in results),
                "total_points_in_mask": sum(r["points_in_mask"] for r in results),
                "split": self.split if self.split else "all",
            }
        )
        split_metrics = {}
        for split_name in sorted({r.get("split_name", "unknown") for r in results}):
            split_results = [
                result for result in results
                if result.get("split_name", "unknown") == split_name
            ]
            split_metrics[split_name] = compute_point_metrics(split_results)
        metrics["split_metrics"] = split_metrics
        metrics["per_category"] = {
            name: values["strict_micro_f1"]
            for name, values in split_metrics.items()
        }
        logger.info(
            "RefSpatial final strict/non-strict micro F1=%.4f/%.4f",
            metrics["strict_micro_f1"],
            metrics["non_strict_micro_f1"],
        )
        return metrics

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        import os

        os.makedirs("logs/results", exist_ok=True)
        split_name = self.split if self.split else "all"
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}_{split_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
