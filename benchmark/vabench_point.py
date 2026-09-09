import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import re
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.point_utils import compute_point_metrics, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class VABenchPointDataset(BaseDataset):
    """VABench-P (IffYuan/VABench-P)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/VABench-P",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "VABench-P",
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
        if self.prompt_policy == "original" and self.backbone == "qwen3":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
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
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess VABench-P samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            question_id = sample['id']
            question = sample['problem']
            image = sample['image']
            mask = sample['mask']
                
            width, height = image.size

            if self.prompt_policy == "original":
                problem = question + '\n' + self.instruct_following
            else:
                problem = question + '\nMark exactly one point that meets the requirements.\n' + self.instruct_following
            formatted_question = textwrap.dedent(problem).strip()

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
                }
            })

        if self.debug:
            prepared_dataset = prepared_dataset[:20]
            logger.info(f"Debug mode: processing first 20 samples only")

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
