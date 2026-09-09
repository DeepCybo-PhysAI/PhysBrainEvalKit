import io
import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
import pyarrow.compute as pc
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class MMSIBenchDataset(BaseDataset):
    """MMSI-Bench multi-image spatial intelligence dataset."""

    def __init__(
        self,
        dataset_name: str = "datasets/MMSI-Bench",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "MMSI-Bench",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False,
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
        return "Please answer with only the option letter, such as A, B, C, or D."

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset_path = Path(self.dataset_name)
        if dataset_path.is_dir():
            parquet_path = dataset_path / "MMSI_Bench.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Could not find MMSI_Bench.parquet under {dataset_path}")
            dataset = load_dataset("parquet", data_files={self.split: str(parquet_path)}, split=self.split)
        elif dataset_path.is_file():
            dataset = load_dataset("parquet", data_files={self.split: str(dataset_path)}, split=self.split)
        elif self.subset:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    @staticmethod
    def _decode_image(image: Any) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            if image.get("path"):
                return Image.open(image["path"]).convert("RGB")
        try:
            return Image.fromarray(image).convert("RGB")
        except Exception:
            logger.warning(f"Could not decode MMSI image of type {type(image)}")
            return None

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        image_dataset = dataset.select_columns(["images"])
        metadata_dataset = dataset.remove_columns(["images"])
        image_counts = pc.list_value_length(dataset.data.column("images")).to_pylist()
        prepared_dataset = []
        for idx, sample in enumerate(metadata_dataset):
            question = textwrap.dedent(
                f"{sample['question'].strip()}\n{self.instruct_following}"
            ).strip()
            answer = self.extract_answer_from_text(sample.get("answer", ""))

            prepared_dataset.append({
                "question": question,
                "answer": answer,
                "image": {
                    "__lazy_hf_images__": True,
                    "dataset": image_dataset,
                    "row_index": idx,
                    "column": "images",
                },
                "metadata": {
                    "idx": idx,
                    "id": sample.get("id"),
                    "question_type": sample.get("question_type"),
                    "thought": sample.get("thought"),
                    "difficulty": sample.get("difficulty"),
                    "mean_normed_duration_seconds": sample.get("mean_normed_duration_seconds"),
                    "num_images": image_counts[idx],
                    "raw_answer": sample.get("answer"),
                },
            })
        return prepared_dataset

    @staticmethod
    def extract_answer_from_text(text: str) -> Optional[str]:
        text = str(text).strip()
        patterns = [
            r"(?:Answer|Answer)[:\s]*\(?([A-D])\)?",
            r"Correct\s+Answer[:\s]*\(?([A-D])\)?",
            r"^([A-D])\s*[.)]?$",
            r"\(([A-D])\)",
            r"\b([A-D])\b[.\s]*$",
            r"\b([A-D])\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        return None

    def process_raw_output(self, processed_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
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

        prediction = self.extract_answer_from_text(raw_output_text)
        answer = processed_sample["answer"]
        is_correct = prediction == answer if prediction and answer else False

        return {
            "idx": processed_sample["metadata"]["idx"],
            "sample_id": processed_sample["metadata"].get("id"),
            "question": processed_sample["question"],
            "ground_truth": answer,
            "metadata": processed_sample["metadata"],
            "raw_output": original_raw_output,
            "processed_answer": prediction,
            "is_correct": is_correct,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[Any]) -> List[Dict[str, Any]]:
        logger.info("Evaluating MMSI-Bench predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            draws = raw_output if isinstance(raw_output, list) else [raw_output]
            parsed = [self.process_raw_output(sample, draw) for draw in draws]
            result = dict(parsed[0])
            result["raw_outputs"] = [item["raw_output"] for item in parsed]
            result["pass_at_k"] = {str(k): any(item["is_correct"] for item in parsed[:k]) for k in (1, 8, 16, 32) if len(parsed) >= k}
            result["is_correct"] = result["pass_at_k"].get("1", False)
            all_results.append(result)
            if self.debug or len(all_results) <= 5:
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('sample_id')}")
                logger.info(f"Question Type: {result['metadata'].get('question_type')}")
                logger.info(f"Prediction: {result.get('processed_answer')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_predictions = sum(1 for result in results if result["is_correct"])
        overall_accuracy = correct_predictions / total_samples if total_samples else 0.0
        pass_at_k = {str(k): sum(bool(result.get("pass_at_k", {}).get(str(k), False)) for result in results) / total_samples for k in (1, 8, 16, 32)}

        per_type_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        per_difficulty_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for result in results:
            metadata = result.get("metadata", {})
            question_type = str(metadata.get("question_type") or "unknown")
            difficulty = str(metadata.get("difficulty") or "unknown")
            per_type_counts[question_type]["total"] += 1
            per_type_counts[question_type]["correct"] += int(result["is_correct"])
            per_difficulty_counts[difficulty]["total"] += 1
            per_difficulty_counts[difficulty]["correct"] += int(result["is_correct"])

        per_type = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_type_counts.items())
        }
        per_difficulty = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_difficulty_counts.items())
        }

        logger.info(f"MMSI-Bench overall accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")

        return {
            "overall_accuracy": overall_accuracy,
            "pass_at_k": pass_at_k,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "question_type_results": per_type,
            "difficulty_results": per_difficulty,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
