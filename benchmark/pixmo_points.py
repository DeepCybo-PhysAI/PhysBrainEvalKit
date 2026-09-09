import ast
import json
import logging
import math
import re
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.final_point_metrics import attach_score_fields, score_sample
from core.framework_integration import count_points_in_any_mask
from core.point_utils import (
    compute_point_metrics,
    convert_points_to_pixels,
    distance_first_mask_matching,
    parse_point_output,
    validate_mask_size,
)

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


def _parse_gt_point(value: Any) -> List[float]:
    """Parse PixMo's 0--100 representative point into [x, y]."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            value = [value["x"], value["y"]]
        elif "point" in value:
            value = value["point"]
        elif "point_2d" in value:
            value = value["point_2d"]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid PixMo GT point: {value!r}")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y) or not (
        0 <= x <= 100 and 0 <= y <= 100
    ):
        raise ValueError(f"PixMo GT point outside finite 0--100 range: {value!r}")
    return [x, y]


class PixmoPointsDataset(BaseDataset):
    """Pixmo Points (IffYuan/pixmo-points-eval)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/pixmo-points-eval",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "PixmoPoints",
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
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. If there is no object to annotate in the image, output "No object"."""
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000. If there is no object to annotate in the image, output \"No object\"."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            # return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. If there is no object to annotate in the image, output "No object"."""
            return """The answer must be a JSON list: [{\"point_2d\": [x, y]}]\nCoordinates must be integers normalized to [0, 1000].\nFor each object or region matching the label, provide one point_2d coordinate. If no matching object or region exists, return an empty list []."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. If there is no object to annotate in the image, output "No object"."""
        elif self.backbone == "gemini_robotics":
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>. If there is no object to annotate in the image, output "No object"."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl' or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points. If there is no object to annotate in the image, output \"No object\"."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess Pixmo Points samples"""
        prepared_dataset = []

        for idx, sample in enumerate(tqdm(dataset, desc="Preparing dataset")):
            # Extract data from sample
            image = sample['image']  # Image is already loaded in dataset
            label = sample['label']
            points = sample['points']
            masks = sample['masks']  # Sequence of boolean masks
            question = sample['question']
            hash_match = sample['hash_match']

            width, height = image.size
            gt_points_percent = [_parse_gt_point(point) for point in points]
            gt_points_pixels = [
                [point[0] / 100.0 * width, point[1] / 100.0 * height]
                for point in gt_points_percent
            ]
            # Create question based on label
            problem = question + '\n' + self.instruct_following
            formatted_question = textwrap.dedent(problem).strip()

            has_no_object = not gt_points_pixels
            instance_masks = [] if has_no_object else masks
            if not has_no_object and len(gt_points_pixels) != len(instance_masks):
                raise ValueError(
                    f"PixMo idx={idx}: {len(gt_points_pixels)} GT points != "
                    f"{len(instance_masks)} masks"
                )

            prepared_dataset.append({
                'question': formatted_question,
                'answer': None,
                'image': image,
                'metadata': {
                    'idx': idx,
                    'label': label,
                    'points': points,
                    'gt_points_percent': gt_points_percent,
                    'gt_points_pixels': gt_points_pixels,
                    'hash_match': hash_match,
                    'masks': instance_masks,
                    'width': width,
                    'height': height,
                    'has_no_object': has_no_object,
                }
            })

        return prepared_dataset

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        metadata = prepared_sample["metadata"]
        parsed = parse_point_output(str(raw_output_text))
        abs_points = convert_points_to_pixels(
            parsed.points,
            metadata["width"],
            metadata["height"],
            self.backbone,
        )
        has_no_object = bool(metadata.get("has_no_object", False))
        for mask in metadata["masks"]:
            validate_mask_size(mask, metadata["width"], metadata["height"])
        if has_no_object:
            gt_count = 1
            matched = 0
            raw_hits = 0
        else:
            gt_count = len(metadata["masks"])
            matched = distance_first_mask_matching(
                abs_points,
                metadata["gt_points_pixels"],
                metadata["masks"],
            )
            raw_hits = count_points_in_any_mask(abs_points, metadata["masks"])
        score = score_sample(
            predicted_count=len(parsed.points),
            matched_count=matched,
            raw_hit_count=raw_hits,
            gt_count=gt_count,
            parse_status=parsed.status,
            has_no_object=has_no_object,
        )
        return attach_score_fields(
            {
                "idx": metadata["idx"],
                "label": metadata["label"],
                "question": prepared_sample["question"],
                "raw_output": raw_output_text,
                "answer_text": parsed.answer_text,
                "parse_status": parsed.status,
                "parsed_points": parsed.points,
                "processed_points": abs_points,
                "gt_points_pixels": metadata["gt_points_pixels"],
                "matching_protocol": "distance_first_hungarian_then_assigned_mask",
                "masks_with_points": matched,
                "total_masks": gt_count,
                "points_in_any_gt_mask": raw_hits,
                "accuracy_score": matched / gt_count if gt_count else 0.0,
                "has_no_object": has_no_object,
            },
            score,
        )

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
            "total_masks": sum(r["total_masks"] for r in results),
            "masks_with_points": sum(r["masks_with_points"] for r in results),
        })
        logger.info(
            "PixMo point metrics: micro P/R/F1=%.4f/%.4f/%.4f",
            metrics["non_strict_micro_precision"],
            metrics["non_strict_micro_recall"],
            metrics["non_strict_micro_f1"],
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
