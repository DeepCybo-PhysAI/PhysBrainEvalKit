import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.point_utils import compute_point_metrics, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)

class PIOBenchDataset(BaseDataset):
    """PIO-Bench (IffYuan/PIO-Bench) Dataset Class"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/PIO-Bench",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "PIO-Bench",
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
        elif self.backbone in ["qwen3_5", "qwen3"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "gemini_robotics":
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone in ["gemini-2.5"]:
            # specific handling for gemini-2.5
            return """Output exactly one point that best localizes the description.
            Strictly provide only the tuple without any additional text, labels, or explanations. The answer should be presented in JSON format as follows: [{\"point\": [x, y]}]."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl' or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    def load_dataset(self) -> Any:
        logger.info(f"Loading dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared_dataset = []
        for idx, item in enumerate(dataset):
            image = item["image"]  # image bytes
            polygon = item["polygon"]
            width = item["width"]
            height = item["height"]
            lang = item.get("lang", "")
            dataset_name = item.get("dataset_name", "")
            s1_or_s2 = item.get("s1_or_s2", "")
            subclasses = item.get("subclasses", [])

            if self.prompt_policy == "original":
                problem = textwrap.dedent(
                    f"{lang}\n{self.instruct_following}"
                    if self.instruct_following else lang
                ).strip()
            else:
                problem = textwrap.dedent(
                    f"{lang}\nMark exactly one point that meets the requirements.\n{self.instruct_following}"
                    if self.instruct_following else lang
                ).strip()

            prepared_dataset.append({
                "question": problem,
                "answer": None,
                "image": image,
                "metadata": {
                    "idx": idx,
                    "polygon": polygon,
                    "width": width,
                    "height": height,
                    "dataset_name": dataset_name,
                    "s1_or_s2": s1_or_s2,
                    "subclasses": subclasses,
                }
            })

        if self.debug:
            prepared_dataset = prepared_dataset[:20]
            logger.info(f"Debug mode: processing first 20 samples only")

        return prepared_dataset
    
    @staticmethod
    def polygon_to_mask(polygon: List[List[int]], width: int, height: int) -> Image.Image:
        """
        Convert COCO-style polygon [[x1,y1,...]] to PIL grayscale mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        # polygon might be list of lists
        poly_list = [np.array(p).reshape(-1, 2).astype(np.int32) for p in polygon]
        cv2.fillPoly(mask, poly_list, 255)
        return Image.fromarray(mask)

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        metadata = prepared_sample["metadata"]
        width = metadata["width"]
        height = metadata["height"]
        image = prepared_sample["image"]
        if hasattr(image, "size") and image.size != (width, height):
            raise ValueError(
                f"PIO image size {image.size} does not match metadata {(width, height)}"
            )
        mask = self.polygon_to_mask(metadata["polygon"], width, height)
        scored = score_single_mask_sample(
            raw_output_text,
            mask=mask,
            width=width,
            height=height,
            backbone=self.backbone,
        )
        return {
            "idx": metadata["idx"],
            "question": prepared_sample["question"],
            "raw_output": raw_output_text,
            "s1_or_s2": metadata.get("s1_or_s2", ""),
            **scored,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating results...")
        results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            results.append(self.process_raw_output(sample, raw_output))
        return results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = compute_point_metrics(results)
        metrics.update({
            "overall_accuracy": metrics["strict_micro_f1"],
            "average_accuracy": metrics["strict_macro_f1"],
            "total_points_predicted": sum(r["total_points"] for r in results),
            "total_points_in_mask": sum(r["points_in_mask"] for r in results),
        })
        for subset in ("s1", "s2"):
            subset_results = [r for r in results if r.get("s1_or_s2") == subset]
            subset_metrics = compute_point_metrics(subset_results)
            metrics[f"{subset}_metrics"] = subset_metrics
            metrics[f"{subset}_accuracy"] = subset_metrics["strict_macro_f1"]
            metrics[f"{subset}_samples"] = len(subset_results)
        logger.info(
            "PIO point metrics: non-strict micro P/R/F1=%.4f/%.4f/%.4f",
            metrics["non_strict_micro_precision"],
            metrics["non_strict_micro_recall"],
            metrics["non_strict_micro_f1"],
        )
        return metrics

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        import os
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
