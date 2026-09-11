import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset
from core.hf_data import resolve_snapshot

logger = logging.getLogger(__name__)


class ViewSpatialDataset(BaseDataset):
    """ViewSpatial-Bench dataset in Hugging Face JSON or parquet format."""

    JSON_FILE = "ViewSpatial-Bench.json"
    ARCHIVE_FILES = {
        "scannetv2_val": "scannetv2_val.zip",
        "val2017": "val2017.zip",
    }

    def __init__(
        self,
        dataset_name: str = "lidingm/ViewSpatial-Bench",
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
        self.data_root: Optional[Path] = None
        self._archive_members: Dict[Path, set[str]] = {}

    def get_default_instruct(self) -> str:
        return "Please answer with only the option letter, such as A, B, C, or D."

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")

        dataset_path = Path(self.dataset_name).expanduser()
        if dataset_path.is_dir() or (
            not dataset_path.is_absolute()
            and not dataset_path.exists()
            and len(dataset_path.parts) == 2
        ):
            snapshot = resolve_snapshot(self.dataset_name)
            json_path = snapshot / self.JSON_FILE
            if json_path.is_file():
                self.data_root = snapshot
                with json_path.open("r", encoding="utf-8") as file:
                    dataset = json.load(file)
                if not isinstance(dataset, list):
                    raise ValueError(f"Expected a list of samples in {json_path}")
                logger.info(
                    "Loaded ViewSpatial JSON dataset: %d samples from %s",
                    len(dataset),
                    json_path,
                )
                return dataset

            parquet_paths = sorted((snapshot / "data").glob(f"{self.split}-*.parquet"))
            if parquet_paths:
                self.data_root = snapshot
                data_files = {self.split: [str(path) for path in parquet_paths]}
                dataset = load_dataset("parquet", data_files=data_files, split=self.split)
                logger.info(f"Dataset parquet loaded. Number of samples: {len(dataset)}")
                return dataset

        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        # The older lmms-eval export includes decoded `images`; the public
        # lidingm/ViewSpatial-Bench snapshot exposes only `image_path` and
        # ships the source images in two ZIP archives.
        column_names = set(getattr(dataset, "column_names", []) or [])
        has_embedded_images = "images" in column_names
        image_dataset = dataset.select_columns(["images"]) if has_embedded_images else None
        metadata_dataset = dataset.remove_columns(["images"]) if has_embedded_images else dataset
        prepared_dataset = []
        for idx, sample in enumerate(metadata_dataset):
            sample = dict(sample)
            question = textwrap.dedent(
                f"{sample['question'].strip()}\n{sample['choices'].strip()}\n{self.instruct_following}"
            ).strip()
            answer = self.extract_answer_from_text(sample["answer"])
            image_paths = sample.get("image_path") or []
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            if not isinstance(image_paths, list):
                image_paths = list(image_paths)
            num_images = len(image_paths)

            if has_embedded_images:
                image = {
                    "__lazy_hf_images__": True,
                    "dataset": image_dataset,
                    "row_index": idx,
                    "column": "images",
                }
            else:
                image = [self._resolve_image_reference(path) for path in image_paths]

            prepared_dataset.append({
                "question": question,
                "answer": answer,
                "image": image,
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

    def _resolve_image_reference(self, image_path: Any) -> Any:
        """Resolve a ViewSpatial image to a local path or ZIP member descriptor."""
        path_text = str(image_path).strip().replace("\\", "/")
        path = Path(path_text).expanduser()
        if path.is_file():
            return str(path.resolve())

        if self.data_root is None:
            raise FileNotFoundError(
                f"Cannot resolve ViewSpatial image {image_path!r}: dataset root is unknown"
            )

        relative_path = path_text.lstrip("./")
        direct_candidates = [
            self.data_root / relative_path,
            self.data_root / relative_path.removeprefix("ViewSpatial-Bench/"),
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return str(candidate.resolve())

        archive_key = next(
            (key for key in self.ARCHIVE_FILES if f"/{key}/" in f"/{relative_path}/"),
            None,
        )
        if archive_key is None:
            raise FileNotFoundError(
                f"ViewSpatial image path does not identify a known archive: {image_path!r}"
            )

        archive_path = self.data_root / self.ARCHIVE_FILES[archive_key]
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"ViewSpatial image archive is missing: {archive_path}. "
                "Download the complete lidingm/ViewSpatial-Bench dataset snapshot."
            )

        import zipfile

        members = self._archive_members.get(archive_path)
        if members is None:
            with zipfile.ZipFile(archive_path) as archive:
                members = set(archive.namelist())
            self._archive_members[archive_path] = members

        candidates = [
            relative_path,
            relative_path.removeprefix("ViewSpatial-Bench/"),
        ]
        member_name = next((candidate for candidate in candidates if candidate in members), None)
        if member_name is None:
            suffixes = tuple(f"/{candidate}" for candidate in candidates)
            matches = [member for member in members if member.endswith(suffixes)]
            if len(matches) == 1:
                member_name = matches[0]

        if member_name is None:
            raise FileNotFoundError(
                f"ViewSpatial image is not present in {archive_path}: {image_path!r}"
            )

        return {
            "type": "zip_image",
            "archive": str(archive_path.resolve()),
            "member": member_name,
        }

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
