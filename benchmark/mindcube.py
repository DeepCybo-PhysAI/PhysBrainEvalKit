import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from .base import BaseDataset
from core.hf_data import resolve_snapshot

logger = logging.getLogger(__name__)


class MindCubeDataset(BaseDataset):
    """MindCube multi-view spatial reasoning dataset."""

    SPLIT_FILES = {
        "default": "MindCube.jsonl",
        "full": "MindCube.jsonl",
        "test": "MindCube_tinybench.jsonl",
        "tiny": "MindCube_tinybench.jsonl",
        "tinybench": "MindCube_tinybench.jsonl",
        "train": "MindCube_train.jsonl",
    }

    def __init__(
        self,
        dataset_name: str = "VLyb/MindCube-TinyBench",
        subset: Optional[str] = None,
        split: str = "tinybench",
        instruct_following: Optional[str] = None,
        task_name: str = "MindCube",
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
        self.data_root: Optional[Path] = None

    def get_default_instruct(self) -> str:
        return "Please answer with only the option letter, such as A, B, C, or D."

    def _resolve_jsonl_path(self) -> Path:
        dataset_path = resolve_snapshot(self.dataset_name)
        if dataset_path.is_file():
            self.data_root = dataset_path.parent.parent if dataset_path.parent.name == "raw" else dataset_path.parent
            return dataset_path

        if dataset_path.is_dir():
            split_key = str(self.split or "tinybench").lower()
            filename = self.SPLIT_FILES.get(split_key, self.split)
            candidates = [
                dataset_path / "data" / "raw" / filename,
                dataset_path / "raw" / filename,
                dataset_path / filename,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    if candidate.parent.name == "raw":
                        self.data_root = candidate.parent.parent
                    elif (dataset_path / "data").is_dir():
                        self.data_root = dataset_path / "data"
                    else:
                        self.data_root = dataset_path
                    return candidate

        raise FileNotFoundError(
            "Could not resolve MindCube jsonl. Expected a jsonl file or a directory "
            "containing data/raw/MindCube_tinybench.jsonl, MindCube.jsonl, or MindCube_train.jsonl; "
            f"got dataset_name={self.dataset_name!r}, split={self.split!r}."
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        jsonl_path = self._resolve_jsonl_path()
        logger.info(f"Dataset JSONL: {jsonl_path}")
        logger.info(f"Image root: {self.data_root}")

        dataset = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dataset.append(json.loads(line))

        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(self._resolve_image_path(image_path)).convert("RGB")

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.is_absolute():
            if self.data_root is None:
                raise ValueError("MindCube data_root is not initialized")
            path = self.data_root / path
        return path

    def prepare_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            image_paths = [str(path) for path in sample.get("images", [])]
            images = [self._resolve_image_path(path) for path in image_paths]
            answer = self.extract_answer_from_text(sample.get("gt_answer", ""))

            question = textwrap.dedent(
                f"{str(sample.get('question', '')).strip()}\n{self.instruct_following}"
            ).strip()

            prepared_dataset.append({
                "question": question,
                "answer": answer,
                "image": images,
                "metadata": {
                    "idx": idx,
                    "id": sample.get("id"),
                    "category": sample.get("category"),
                    "type": sample.get("type"),
                    "meta_info": sample.get("meta_info"),
                    "image_paths": image_paths,
                    "num_images": len(images),
                    "raw_answer": sample.get("gt_answer"),
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

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating MindCube predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            if self.debug or len(all_results) <= 5:
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('sample_id')}")
                logger.info(f"Prediction: {result.get('processed_answer')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_predictions = sum(1 for result in results if result["is_correct"])
        overall_accuracy = correct_predictions / total_samples if total_samples else 0.0

        per_type_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        per_category_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for result in results:
            metadata = result.get("metadata", {})
            type_name = str(metadata.get("type") or "unknown")
            per_type_counts[type_name]["total"] += 1
            per_type_counts[type_name]["correct"] += int(result["is_correct"])

            categories = metadata.get("category") or ["unknown"]
            if not isinstance(categories, list):
                categories = [categories]
            for category in categories:
                category = str(category)
                per_category_counts[category]["total"] += 1
                per_category_counts[category]["correct"] += int(result["is_correct"])

        per_type = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_type_counts.items())
        }
        per_category = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_category_counts.items())
        }

        logger.info(f"MindCube overall accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "per_type_results": per_type,
            "per_category_results": per_category,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
