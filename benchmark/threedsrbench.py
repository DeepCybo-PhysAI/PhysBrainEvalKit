import base64
import csv
import io
import sys
import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class ThreeDSRBenchDataset(BaseDataset):
    """3DSRBench circular TSV evaluation dataset."""

    TYPE_MAPPING = {
        "location": ["location_above", "location_closer_to_camera", "location_next_to"],
        "height": ["height_higher"],
        "orientation": ["orientation_in_front_of", "orientation_on_the_left", "orientation_viewpoint"],
        "multi_object": [
            "multi_object_closer_to",
            "multi_object_facing",
            "multi_object_viewpoint_towards_object",
            "multi_object_parallel",
            "multi_object_same_direction",
        ],
    }

    def __init__(
        self,
        dataset_name: str = "datasets/3DSRBench/3DSRBench/3dsrbench_v1_vlmevalkit_circular.tsv",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "3DSRBench",
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

    def load_dataset(self) -> List[Dict[str, str]]:
        logger.info(f"Dataset TSV: {self.dataset_name}")
        csv.field_size_limit(sys.maxsize)
        with open(self.dataset_name, "r", encoding="utf-8") as f:
            dataset = list(csv.DictReader(f, delimiter="\t"))
        logger.info(f"Dataset loaded. Number of rows: {len(dataset)}")
        return dataset

    @staticmethod
    def _decode_image(encoded: str) -> Image.Image:
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        return image

    @staticmethod
    def _base_qid(qid: str) -> str:
        qid = str(qid)
        if len(qid) >= 2 and qid[-2] == "-":
            return qid[:-2]
        return qid

    @staticmethod
    def _choices(sample: Dict[str, Any]) -> List[tuple[str, str]]:
        choices = []
        for letter in ["A", "B", "C", "D"]:
            value = sample.get(letter)
            if value is not None and str(value).strip():
                choices.append((letter, str(value).strip()))
        return choices

    def prepare_dataset(self, dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            choices = self._choices(sample)
            choice_text = "\n".join(f"{letter}. {text}" for letter, text in choices)
            question = textwrap.dedent(
                f"{sample['question'].strip()}\n{choice_text}\n{self.instruct_following}"
            ).strip()

            prepared_dataset.append({
                "question": question,
                "answer": sample["answer"].strip().upper(),
                "image": self._decode_image(sample["image"]),
                "metadata": {
                    "idx": idx,
                    "index": sample.get("index"),
                    "qid": sample.get("qid"),
                    "base_qid": self._base_qid(sample.get("qid", "")),
                    "category": sample.get("category"),
                    "image_source": sample.get("image_source"),
                    "image_url": sample.get("image_url"),
                    "choices": {letter: text for letter, text in choices},
                },
            })
        return prepared_dataset

    @staticmethod
    def extract_answer_from_text(text: str) -> Optional[str]:
        text = str(text).strip()
        patterns = [
            r"(?:Answer|Answer)[:\s]*\(?([A-D])\)?",
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
        logger.info("Evaluating 3DSRBench predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            if self.debug or len(all_results) <= 5:
                logger.info("=" * 60)
                logger.info(f"QID: {result['metadata'].get('qid')}")
                logger.info(f"Prediction: {result.get('processed_answer')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_rows = len(results)
        correct_rows = sum(1 for result in results if result["is_correct"])
        row_accuracy = correct_rows / total_rows if total_rows else 0.0

        grouped = {}
        for result in results:
            base_qid = result["metadata"].get("base_qid") or result["metadata"].get("qid")
            category = result["metadata"].get("category") or "unknown"
            if base_qid in grouped:
                grouped[base_qid]["correct"] = grouped[base_qid]["correct"] and result["is_correct"]
            else:
                grouped[base_qid] = {"correct": bool(result["is_correct"]), "category": category}

        total_questions = len(grouped)
        correct_questions = sum(1 for item in grouped.values() if item["correct"])
        overall_accuracy = correct_questions / total_questions if total_questions else 0.0

        per_category_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for item in grouped.values():
            category = item["category"]
            per_category_counts[category]["total"] += 1
            per_category_counts[category]["correct"] += int(item["correct"])

        per_category = {}
        for category, counts in sorted(per_category_counts.items()):
            per_category[category] = {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }

        per_type = {}
        for type_name, categories in self.TYPE_MAPPING.items():
            total = sum(per_category.get(category, {}).get("total", 0) for category in categories)
            correct = sum(per_category.get(category, {}).get("correct", 0) for category in categories)
            per_type[type_name] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }

        logger.info(f"3DSRBench grouped accuracy: {overall_accuracy:.4f} ({correct_questions}/{total_questions})")
        logger.info(f"3DSRBench row accuracy: {row_accuracy:.4f} ({correct_rows}/{total_rows})")

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_questions,
            "correct_predictions": correct_questions,
            "row_accuracy": row_accuracy,
            "total_rows": total_rows,
            "correct_rows": correct_rows,
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
