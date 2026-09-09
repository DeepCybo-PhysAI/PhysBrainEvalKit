import json
import logging
import os
import re
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class Ego3DBenchDataset(BaseDataset):
    """Ego3D-Bench multi-view outdoor spatial reasoning benchmark."""

    EXACT_NUMBER_CATEGORIES = {
        "Ego_Centric_Absolute_Distance",
        "Object_Centric_Absolute_Distance",
    }

    IMAGE_ORDER_BY_SOURCE = {
        "nuscenes": ["Front_Left", "Front", "Front_Right", "Back_Right", "Back", "Back_Left"],
        "waymo": ["Front", "Front_Left", "Side_Left", "Front_Right", "Side_Right"],
        "argoverse": ["Front_Left", "Front", "Front_Right", "Side_Right", "Back_Right", "Back_Left", "Side_Left"],
    }

    def __init__(
        self,
        dataset_name: str = "vbdai/Ego3D-Bench",
        subset: Optional[str] = None,
        split: str = "test",
        image_root: str = os.environ.get("EGO3DBENCH_IMAGE_ROOT", "datasets/Ego3D-Bench/images"),
        instruct_following: Optional[str] = None,
        task_name: str = "Ego3D-Bench",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False,
    ):
        super().__init__(instruct_following)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.image_root = Path(image_root)
        self.task_name = task_name
        self.model_name = model_name
        self.backbone = backbone
        self.debug = debug
        self.thinking_model = thinking_model

    def get_default_instruct(self) -> str:
        return ""

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    @classmethod
    def get_image_order(cls, source: str, image_map: Dict[str, str]) -> List[str]:
        source_key = str(source).lower()
        if source_key in cls.IMAGE_ORDER_BY_SOURCE:
            return cls.IMAGE_ORDER_BY_SOURCE[source_key]
        return [key for key in image_map.keys() if image_map.get(key)]

    @staticmethod
    def prettify_view_name(view: str) -> str:
        return view.replace("_", " ")

    def resolve_image_path(self, image_name: str) -> Path:
        path = Path(image_name)
        if not path.is_absolute():
            path = self.image_root / path
        return path

    def load_image(self, image_name: str) -> Image.Image:
        image_path = self.resolve_image_path(image_name)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Ego3D-Bench image not found: {image_path}. "
                "Download the official image zip from vbdai/Ego3D-Bench and pass --image_root."
            )
        with Image.open(image_path) as img:
            return img.convert("RGB").copy()

    def build_question(self, sample: Dict[str, Any], view_names: List[str]) -> str:
        question = str(sample["question"]).replace("<image>", "[image provided]").strip()
        options = sample.get("options")
        category = sample.get("category")

        view_text = ", ".join(f"{idx + 1}. {self.prettify_view_name(view)} view" for idx, view in enumerate(view_names))
        prompt_parts = [
            f"The images are provided in this order: {view_text}.",
            question,
        ]

        if options:
            prompt_parts.extend(str(option).strip() for option in options)
            letters = [self.extract_option_letter(option) for option in options]
            valid_letters = ", ".join(letter for letter in letters if letter)
            instruction = f"Answer with only the letter of your choice"
            if valid_letters:
                instruction += f" ({valid_letters})"
            instruction += ". Put the final answer inside <answer>...</answer> tags."
        elif category in self.EXACT_NUMBER_CATEGORIES:
            instruction = (
                "Answer with the numeric distance in meters. "
                "Put only the final number inside <answer>...</answer> tags."
            )
        else:
            instruction = "Put the final answer inside <answer>...</answer> tags."

        prompt_parts.append(self.instruct_following or instruction)
        return textwrap.dedent("\n".join(part for part in prompt_parts if part)).strip()

    def prepare_samples(self, samples: Any, start_index: int = 0) -> List[Dict[str, Any]]:
        def prepare_one(indexed_sample: Any) -> Dict[str, Any]:
            local_idx, sample = indexed_sample
            idx = start_index + local_idx
            image_map = sample["images"]
            image_order = self.get_image_order(sample.get("source"), image_map)
            available_views = [view for view in image_order if image_map.get(view)]
            images = [self.load_image(image_map[view]) for view in available_views]
            if not images:
                raise ValueError(f"Ego3D-Bench sample {idx} has no available images.")

            question = self.build_question(sample, available_views)
            options = sample.get("options")
            category = sample.get("category")
            question_type = "exact_number" if category in self.EXACT_NUMBER_CATEGORIES else "multiple_choice"

            return {
                "question": question,
                "answer": sample["answer"],
                "image": images,
                "metadata": {
                    "idx": idx,
                    "question_id": sample.get("idx", idx),
                    "source": sample.get("source"),
                    "category": category,
                    "question_type": question_type,
                    "options": options,
                    "image_order": available_views,
                    "image_files": [image_map[view] for view in available_views],
                },
            }

        indexed_samples = list(enumerate(samples))
        io_workers = max(1, int(os.environ.get("EGO3DBENCH_IO_WORKERS", "16")))
        if io_workers == 1 or len(indexed_samples) < 2:
            return [prepare_one(item) for item in indexed_samples]
        # HDFS image reads are I/O-bound. executor.map preserves sample order,
        # so evaluation IDs and deterministic sampling remain unchanged.
        with ThreadPoolExecutor(max_workers=min(io_workers, len(indexed_samples))) as pool:
            return list(pool.map(prepare_one, indexed_samples))

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        if self.debug and hasattr(dataset, "select"):
            dataset = dataset.select(range(min(20, len(dataset))))
            logger.info("Debug mode: processing first 20 samples only")

        prepared_dataset = self.prepare_samples(dataset)

        if self.debug and not hasattr(dataset, "select"):
            prepared_dataset = prepared_dataset[:20]
            logger.info("Debug mode: processing first 20 samples only")

        return prepared_dataset

    @staticmethod
    def strip_answer_tags(text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"^(.*?</think>|<think>.*?</think>)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        return text

    @staticmethod
    def extract_option_letter(option: Any) -> Optional[str]:
        match = re.match(r"\s*([A-Z])\s*[\).:]", str(option).strip(), re.IGNORECASE)
        return match.group(1).upper() if match else None

    @classmethod
    def option_text_by_letter(cls, options: Any) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for option in options or []:
            letter = cls.extract_option_letter(option)
            if letter:
                text = re.sub(r"^\s*[A-Z]\s*[\).:]\s*", "", str(option), flags=re.IGNORECASE).strip()
                mapping[letter] = text.lower()
        return mapping

    @classmethod
    def extract_choice(cls, text: str, options: Any = None) -> Optional[str]:
        answer_text = cls.strip_answer_tags(text).strip()

        patterns = [
            r"(?:answer|choice|option|Answer)[:\s]*\(?([A-Z])\)?",
            r"\(([A-Z])\)",
            r"\b([A-Z])\b",
        ]
        valid_letters = set(cls.option_text_by_letter(options).keys()) or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for pattern in patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            for match in reversed(matches):
                letter = str(match).upper()
                if letter in valid_letters:
                    return letter

        normalized = answer_text.lower().strip(" .。")
        option_map = cls.option_text_by_letter(options)
        for letter, option_text in option_map.items():
            if normalized == option_text or normalized in option_text or option_text in normalized:
                return letter

        yes_no_aliases = {
            "yes": "A",
            "no": "B",
        }
        if normalized in yes_no_aliases and yes_no_aliases[normalized] in valid_letters:
            return yes_no_aliases[normalized]

        return None

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        answer_text = Ego3DBenchDataset.strip_answer_tags(text)
        match = re.search(r"[-+]?\d*\.?\d+", answer_text)
        if not match:
            return None
        return float(match.group())

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        metadata = prepared_sample["metadata"]
        question_type = metadata["question_type"]
        ground_truth = prepared_sample["answer"]

        result = {
            "idx": metadata["idx"],
            "question_id": metadata["question_id"],
            "source": metadata["source"],
            "category": metadata["category"],
            "question_type": question_type,
            "question": prepared_sample["question"],
            "ground_truth": ground_truth,
            "raw_output": raw_output_text,
            "image_order": metadata["image_order"],
        }

        if question_type == "exact_number":
            pred_number = self.extract_number(raw_output_text)
            pred_number_for_metric = min(float(pred_number), 100.0) if pred_number is not None else None
            gt_number = float(ground_truth)
            abs_error = abs(pred_number_for_metric - gt_number) if pred_number_for_metric is not None else None
            squared_error = (pred_number_for_metric - gt_number) ** 2 if pred_number_for_metric is not None else None
            result.update({
                "processed_answer": pred_number,
                "metric_answer": pred_number_for_metric,
                "ground_truth_number": gt_number,
                "absolute_error": abs_error,
                "squared_error": squared_error,
                "is_valid": pred_number is not None,
            })
        else:
            pred_choice = self.extract_choice(raw_output_text, metadata.get("options"))
            gt_choice = str(ground_truth).strip().upper()
            result.update({
                "processed_answer": pred_choice,
                "is_correct": pred_choice == gt_choice,
                "options": metadata.get("options"),
            })

        return result

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating Ego3D-Bench predictions...")
        results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            try:
                results.append(self.process_raw_output(sample, raw_output))
            except Exception as exc:
                logger.error(f"Error evaluating sample {sample.get('metadata', {}).get('idx')}: {exc}")
                results.append({
                    "idx": sample.get("metadata", {}).get("idx"),
                    "question_id": sample.get("metadata", {}).get("question_id"),
                    "category": sample.get("metadata", {}).get("category"),
                    "raw_output": raw_output,
                    "error": str(exc),
                })
        return results

    def compute_statistics(self, results: List[Dict[str, Any]], log: bool = True) -> Dict[str, Any]:
        total_samples = len(results)
        choice_results = [r for r in results if r.get("question_type") == "multiple_choice"]
        exact_results = [r for r in results if r.get("question_type") == "exact_number"]
        valid_exact = [r for r in exact_results if r.get("is_valid")]

        choice_correct = sum(1 for r in choice_results if r.get("is_correct"))
        choice_accuracy = choice_correct / len(choice_results) if choice_results else 0.0

        rmse = None
        mae = None
        if valid_exact:
            rmse = float(np.sqrt(np.mean([r["squared_error"] for r in valid_exact])))
            mae = float(np.mean([r["absolute_error"] for r in valid_exact]))

        category_stats: Dict[str, Dict[str, Any]] = {}
        grouped = defaultdict(list)
        for result in results:
            grouped[result.get("category", "unknown")].append(result)

        for category, items in sorted(grouped.items()):
            if category in self.EXACT_NUMBER_CATEGORIES:
                valid = [r for r in items if r.get("is_valid")]
                category_stats[category] = {
                    "total_samples": len(items),
                    "valid_samples": len(valid),
                    "rmse": float(np.sqrt(np.mean([r["squared_error"] for r in valid]))) if valid else None,
                    "mae": float(np.mean([r["absolute_error"] for r in valid])) if valid else None,
                }
            else:
                correct = sum(1 for r in items if r.get("is_correct"))
                category_stats[category] = {
                    "total_samples": len(items),
                    "correct": correct,
                    "accuracy": correct / len(items) if items else 0.0,
                }

        if log:
            logger.info("\n" + "=" * 60)
            logger.info("Ego3D-Bench Statistics")
            logger.info("=" * 60)
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Multiple-choice samples: {len(choice_results)}")
            logger.info(f"Multiple-choice accuracy: {choice_accuracy:.4f} ({choice_accuracy*100:.2f}%)")
            logger.info(f"Exact-number samples: {len(exact_results)}")
            logger.info(f"Valid exact-number predictions: {len(valid_exact)}")
            if rmse is not None:
                logger.info(f"Exact-number RMSE: {rmse:.4f}")
                logger.info(f"Exact-number MAE: {mae:.4f}")
            logger.info("=" * 60)

        return {
            "overall_accuracy": choice_accuracy,
            "exact_number_rmse": rmse,
            "exact_number_mae": mae,
            "total_samples": total_samples,
            "multiple_choice_samples": len(choice_results),
            "multiple_choice_correct": choice_correct,
            "exact_number_samples": len(exact_results),
            "exact_number_valid_samples": len(valid_exact),
            "category_stats": category_stats,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name

    def save_partial_results(
        self,
        results: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        progress: Dict[str, Any],
    ) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.partial.json"
        tmp_file_name = f"{result_file_name}.tmp"

        payload = {
            "is_partial": True,
            "progress": progress,
            "results": results,
            "statistics": statistics,
        }
        with open(tmp_file_name, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        os.replace(tmp_file_name, result_file_name)

        logger.info(f"Partial results saved to: {result_file_name}")
        return result_file_name
