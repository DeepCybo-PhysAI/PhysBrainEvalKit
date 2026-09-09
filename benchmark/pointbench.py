import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional
import re
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

from core.final_point_metrics import attach_score_fields, score_sample
from core.point_utils import (
    check_points_in_mask,
    compute_point_metrics,
    convert_points_to_pixels,
    parse_point_output,
    validate_mask_size,
)

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)
EXPECTED_MASK_SIZE_MISMATCH_IDS = ["54", "445", "775"]


class PointBenchDataset(BaseDataset):
    """PointBench (IffYuan/PointBench)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/PointBench",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "PointBench",
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
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            # return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
            return """The answer must be a JSON list: [{\"point_2d\": [x, y]}]\nCoordinates must be integers normalized to [0, 1000]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "gemini_robotics":
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should follow the json format: [{\"point\": <point>,\"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
        elif self.backbone == "molmo":
            return """Provide one or some points with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl'  or self.backbone =='magma':
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
        """Preprocess PointBench samples"""
        prepared_dataset = []
        resized_mask_ids = []
        for idx, sample in enumerate(dataset):
            question_id = sample['id']
            question = sample['question']
            category = sample['category']
            image = sample['image']
            mask = sample['mask']
            expected_count = sample.get('expected_count', 1)

            width, height = image.size
            original_mask_size = mask.size
            if original_mask_size != (width, height):
                resized_mask_ids.append(str(question_id))
                mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)

            problem = question + '\n' + self.instruct_following
            if category == "counting":
                problem += "\nMark only one point on each object that meets the requirements."
            else:
                problem += "\nMark only one point that meets the requirements."
            formatted_question = textwrap.dedent(problem).strip()

            prepared_dataset.append({
                'question': formatted_question,
                'answer': None,
                'image': image,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'category': category,
                    'mask': mask,
                    'original_mask_size': original_mask_size,
                    'width': width,
                    'height': height,
                    'expected_count': expected_count,
                    'original_points': sample.get('original_points'),
                }
            })

        if len(dataset) == 966 and resized_mask_ids != EXPECTED_MASK_SIZE_MISMATCH_IDS:
            raise ValueError(
                f"PointBench mask-size mismatch IDs changed: {resized_mask_ids}"
            )

        if self.debug:
            prepared_dataset = prepared_dataset[:20]
            logger.info(f"Debug mode: processing first 20 samples only")

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
        validate_mask_size(metadata["mask"], metadata["width"], metadata["height"])
        hits, _ = check_points_in_mask(abs_points, metadata["mask"])
        expected_count = int(metadata["expected_count"])
        score = score_sample(
            predicted_count=len(parsed.points),
            matched_count=min(hits, expected_count),
            raw_hit_count=hits,
            gt_count=expected_count,
            parse_status=parsed.status,
            exact_count_required=metadata["category"] == "counting",
        )
        return attach_score_fields(
            {
                "idx": metadata["idx"],
                "category": metadata["category"],
                "question": prepared_sample["question"],
                "raw_output": raw_output_text,
                "answer_text": parsed.answer_text,
                "parse_status": parsed.status,
                "parsed_points": parsed.points,
                "processed_points": abs_points,
                "points_in_mask": hits,
                "total_points": len(parsed.points),
                "expected_count": expected_count,
                "count_correct": len(parsed.points) == expected_count,
                "accuracy_score": hits / len(parsed.points) if parsed.points else 0.0,
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
            "total_points_predicted": sum(r["total_points"] for r in results),
            "total_points_in_mask": sum(r["points_in_mask"] for r in results),
        })
        category_stats = {}
        for category in sorted({r["category"] for r in results}):
            category_results = [r for r in results if r["category"] == category]
            category_stats[category] = compute_point_metrics(category_results)
        metrics["category_stats"] = category_stats
        logger.info(
            "PointBench metrics: non-strict micro P/R/F1=%.4f/%.4f/%.4f, "
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
