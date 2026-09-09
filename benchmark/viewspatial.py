import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class ViewSpatialDataset(BaseDataset):
    """ViewSpatial-Bench dataset in lmms-eval HuggingFace parquet format."""

    def __init__(
        self,
        dataset_name: str = "datasets/ViewSpatial_lmmseval",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "ViewSpatial",
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
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        image_dataset = dataset.select_columns(["images"])
        metadata_dataset = dataset.remove_columns(["images"])
        prepared_dataset = []
        for idx, sample in enumerate(metadata_dataset):
            question = textwrap.dedent(
                f"{sample['question'].strip()}\n{sample['choices'].strip()}\n{self.instruct_following}"
            ).strip()
            answer = self.extract_answer_from_text(sample["answer"])
            image_paths = sample.get("image_path") or []
            num_images = len(image_paths) if isinstance(image_paths, list) else 1

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
                    "question_type": sample.get("question_type"),
                    "image_path": image_paths,
                    "choices": sample.get("choices"),
                    "raw_answer": sample.get("answer"),
                    "num_images": num_images,
                },
            })
        return prepared_dataset

    @staticmethod
    def extract_answer_from_text(text: str) -> Optional[str]:
        text = str(text).strip()
        patterns = [
            r"(?:Answer|Answer)[:\s]*\(?([A-D])\)?",
            r"^([A-D])\s*[.)]",
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
        is_correct = prediction == answer if prediction else False

        return {
            "idx": processed_sample["metadata"]["idx"],
            "question": processed_sample["question"],
            "ground_truth": answer,
            "metadata": processed_sample["metadata"],
            "raw_output": original_raw_output,
            "processed_answer": prediction,
            "is_correct": is_correct,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating ViewSpatial predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            if self.debug or len(all_results) <= 5:
                logger.info("=" * 60)
                logger.info(f"Question Type: {result['metadata'].get('question_type')}")
                logger.info(f"Prediction: {result.get('processed_answer')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_predictions = sum(1 for result in results if result["is_correct"])
        overall_accuracy = correct_predictions / total_samples if total_samples else 0.0

        question_type_results = {}
        for result in results:
            question_type = result["metadata"].get("question_type") or "unknown"
            question_type_results.setdefault(question_type, {"total": 0, "correct": 0})
            question_type_results[question_type]["total"] += 1
            question_type_results[question_type]["correct"] += int(result["is_correct"])

        for question_type, counts in question_type_results.items():
            counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else 0.0

        logger.info(f"ViewSpatial overall accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "question_type_results": dict(sorted(question_type_results.items())),
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
