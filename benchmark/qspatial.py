import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class QSpatialBenchDataset(BaseDataset):
    """Q-Spatial-Bench quantitative spatial reasoning dataset."""

    def __init__(
        self,
        dataset_name: str = "andrewliao11/Q-Spatial-Bench",
        subset: Optional[str] = None,
        split: str = "QSpatial_plus",
        instruct_following: Optional[str] = None,
        task_name: str = "Q-Spatial-Bench",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False,
        success_delta: float = 2.0,
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
        self.success_delta = success_delta

    def get_default_instruct(self) -> str:
        return (
            'Answer the question by providing a numeric answer consisting of a scalar '
            'and a distance unit at the end of your response. Use exactly this format: '
            '\\scalar{scalar} \\distance_unit{distance unit}'
        )

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            image = sample["image"]
            if image is None:
                raise ValueError(
                    "QSpatial_scannet samples do not include images in the HF dataset. "
                    "Download ScanNet images first and extend this loader to read "
                    f"image_path={sample.get('image_path')}."
                )

            question = textwrap.dedent(
                sample["question"].strip() + "\n" + self.instruct_following
            ).strip()

            prepared_dataset.append({
                "question": question,
                "answer": {
                    "value": sample["answer_value"],
                    "unit": sample["answer_unit"],
                },
                "image": image.convert("RGB") if hasattr(image, "convert") else image,
                "metadata": {
                    "idx": idx,
                    "question_id": sample.get("image_path", str(idx)),
                    "question_type": sample.get("question_type"),
                    "image_path": sample.get("image_path"),
                    "answer_value": sample["answer_value"],
                    "answer_unit": sample["answer_unit"],
                },
            })

        return prepared_dataset

    @staticmethod
    def _get_multiplier(unit: str) -> float:
        unit = str(unit).lower().strip().rstrip(".")
        if unit in ["meters", "meter", "m", "metre", "metres"]:
            return 100.0
        if unit in ["centimeters", "centimeter", "cm"]:
            return 1.0
        if unit in ["feet", "foot", "ft"]:
            return 30.48
        if unit in ["inch", "inches", "in"]:
            return 2.54
        if unit in ["millimeters", "millimeter", "mm"]:
            return 0.1
        logger.warning(f"Unknown Q-Spatial unit: {unit}; using multiplier 1.0")
        return 1.0

    @classmethod
    def _to_centimeters(cls, value: float, unit: str) -> float:
        return float(value) * cls._get_multiplier(unit)

    @staticmethod
    def _parse_prediction(text: str) -> Dict[str, Any]:
        scalar_matches = re.findall(r"\\?scalar\{([^}]*)\}", str(text))
        unit_matches = re.findall(r"\\?distance_unit\{([^}]*)\}", str(text))

        parsed_scalar = None
        parsed_unit = None

        if scalar_matches:
            scalar_numbers = re.findall(r"-?\d+\.?\d*", scalar_matches[-1])
            if scalar_numbers:
                parsed_scalar = float(np.array(scalar_numbers, dtype=float).mean())

        if unit_matches:
            parsed_unit = unit_matches[-1].strip()

        if parsed_scalar is None:
            numbers = re.findall(r"-?\d+\.?\d*", str(text))
            if numbers:
                parsed_scalar = float(numbers[-1])

        if parsed_unit is None:
            unit_match = re.search(
                r"\b(centimeters?|cm|meters?|metres?|m|feet|foot|ft|inches?|in|millimeters?|mm)\b",
                str(text),
                re.IGNORECASE,
            )
            if unit_match:
                parsed_unit = unit_match.group(1)

        return {
            "parsed_scalar": parsed_scalar,
            "parsed_unit": parsed_unit,
        }

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        original_raw_output = raw_output_text

        if self.thinking_model:
            raw_output_text = re.sub(
                r"^(.*?</think>|<think>.*?</think>)",
                "",
                str(raw_output_text),
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_output_text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        metadata = prepared_sample["metadata"]
        gt_value_cm = self._to_centimeters(metadata["answer_value"], metadata["answer_unit"])
        parsed = self._parse_prediction(raw_output_text)

        pred_value_cm = None
        ratio_error = None
        success = False
        parse_error = None

        try:
            if parsed["parsed_scalar"] is None or parsed["parsed_unit"] is None:
                raise ValueError("Could not parse scalar and unit from model output")
            pred_value_cm = self._to_centimeters(parsed["parsed_scalar"], parsed["parsed_unit"])
            if gt_value_cm <= 0 or pred_value_cm <= 0:
                raise ValueError("Ground-truth and prediction must be positive distances")
            ratio_error = max(pred_value_cm / gt_value_cm, gt_value_cm / pred_value_cm)
            success = ratio_error < self.success_delta
        except Exception as exc:
            parse_error = str(exc)

        return {
            "idx": metadata["idx"],
            "question_id": metadata["question_id"],
            "question": prepared_sample["question"],
            "question_type": metadata["question_type"],
            "raw_output": original_raw_output,
            "ground_truth_value": metadata["answer_value"],
            "ground_truth_unit": metadata["answer_unit"],
            "ground_truth_value_cm": gt_value_cm,
            "parsed_scalar": parsed["parsed_scalar"],
            "parsed_unit": parsed["parsed_unit"],
            "pred_value_cm": pred_value_cm,
            "ratio_error": ratio_error,
            "success": success,
            "parse_error": parse_error,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating Q-Spatial-Bench predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            all_results.append(self.process_raw_output(sample, raw_output))
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        successful = sum(1 for r in results if r["success"])
        parsed = sum(1 for r in results if r["pred_value_cm"] is not None)
        success_rate = successful / total_samples if total_samples else 0.0
        parse_rate = parsed / total_samples if total_samples else 0.0

        type_stats = defaultdict(lambda: {"total": 0, "success": 0, "parsed": 0})
        for result in results:
            q_type = result.get("question_type") or "unknown"
            type_stats[q_type]["total"] += 1
            type_stats[q_type]["success"] += int(result["success"])
            type_stats[q_type]["parsed"] += int(result["pred_value_cm"] is not None)

        per_type = {}
        for q_type, stats in sorted(type_stats.items()):
            per_type[q_type] = {
                "total": stats["total"],
                "success": stats["success"],
                "parsed": stats["parsed"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] else 0.0,
                "parse_rate": stats["parsed"] / stats["total"] if stats["total"] else 0.0,
            }

        logger.info("\n" + "=" * 60)
        logger.info("Q-Spatial-Bench Statistics")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Parsed predictions: {parsed}")
        logger.info(f"Successful predictions: {successful}")
        logger.info(f"Parse rate: {parse_rate:.4f} ({parse_rate*100:.2f}%)")
        logger.info(f"Success rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
        logger.info("=" * 60)

        return {
            "success_rate": success_rate,
            "parse_rate": parse_rate,
            "total_samples": total_samples,
            "successful_predictions": successful,
            "parsed_predictions": parsed,
            "success_delta": self.success_delta,
            "per_question_type": per_type,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        filename = f"{self.task_name}_{self.model_name}_{self.split}.json"
        result_file_name = os.path.join("logs/results", filename)

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name

