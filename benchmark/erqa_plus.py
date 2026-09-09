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
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)


class ERQAPlusDataset(BaseDataset):
    """ERQA-plus embodied relational QA dataset."""

    JSON_FILE = "erqa_plus_gpt_1766_revised_gemini.json"

    def __init__(
        self,
        dataset_name: str = "datasets/erqa-plus",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "ERQA-PLUS",
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
        return ""

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset_path = Path(self.dataset_name)

        if dataset_path.is_dir():
            self.data_root = dataset_path
            json_path = dataset_path / self.JSON_FILE
            if json_path.is_file():
                dataset = self._load_json(json_path)
                logger.info(f"Dataset JSON loaded. Number of samples: {len(dataset)}")
                return dataset

            parquet_paths = sorted((dataset_path / "data").glob("train-*.parquet"))
            if parquet_paths:
                self._validate_parquet_files(dataset_path, parquet_paths)
                data_files = {self.split: [str(path) for path in parquet_paths]}
                dataset = load_dataset("parquet", data_files=data_files, split=self.split)
                logger.info(f"Dataset parquet loaded. Number of samples: {len(dataset)}")
                return dataset

            raise FileNotFoundError(
                "Could not find ERQA-plus annotations. Expected either "
                f"{json_path} or data/train-*.parquet under {dataset_path}. "
                "The current local directory appears to contain images/cache metadata only; "
                "rerun `hf download huggingdas/erqa-plus --repo-type dataset --local-dir "
                f"{dataset_path}` until the JSON or parquet files are present."
            )

        if dataset_path.is_file():
            self.data_root = dataset_path.parent
            if dataset_path.suffix.lower() == ".json":
                dataset = self._load_json(dataset_path)
                logger.info(f"Dataset JSON loaded. Number of samples: {len(dataset)}")
                return dataset
            if dataset_path.suffix.lower() == ".parquet":
                self._validate_parquet_files(dataset_path.parent, [dataset_path])
                dataset = load_dataset("parquet", data_files={self.split: str(dataset_path)}, split=self.split)
                logger.info(f"Dataset parquet loaded. Number of samples: {len(dataset)}")
                return dataset
            raise ValueError(f"Unsupported ERQA-plus dataset file: {dataset_path}")

        if self.subset:
            dataset = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def _load_json(self, json_path: Path) -> List[Dict[str, Any]]:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in [self.split, "train", "data", "rows", "examples", "samples"]:
                value = data.get(key)
                if isinstance(value, list):
                    return value
            if all(isinstance(value, dict) for value in data.values()):
                return list(data.values())

        raise ValueError(f"Unsupported ERQA-plus JSON structure in {json_path}")

    def _expected_parquet_names(self, dataset_path: Path) -> List[str]:
        trees_dir = dataset_path / ".cache" / "huggingface" / "trees"
        if not trees_dir.is_dir():
            return []

        expected = set()
        for tree_path in trees_dir.glob("*.json"):
            try:
                tree = json.loads(tree_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            files = tree.get("files", {}) if isinstance(tree, dict) else {}
            for rel_path in files:
                if re.fullmatch(r"data/train-\d{5}-of-\d{5}\.parquet", rel_path):
                    expected.add(Path(rel_path).name)
        return sorted(expected)

    def _validate_parquet_files(self, dataset_path: Path, parquet_paths: List[Path]) -> None:
        expected_names = self._expected_parquet_names(dataset_path)
        if expected_names:
            present_names = {path.name for path in parquet_paths}
            missing_names = [name for name in expected_names if name not in present_names]
            if missing_names:
                raise FileNotFoundError(
                    "ERQA-plus parquet files are incomplete. Missing: "
                    f"{', '.join(missing_names)}. Rerun the dataset download before evaluation."
                )

        invalid_paths = []
        for path in parquet_paths:
            if path.stat().st_size < 8:
                invalid_paths.append(path)
                continue
            with path.open("rb") as file:
                header = file.read(4)
                file.seek(-4, os.SEEK_END)
                footer = file.read(4)
            if header != b"PAR1" or footer != b"PAR1":
                invalid_paths.append(path)

        if invalid_paths:
            raise ValueError(
                "ERQA-plus parquet files are missing the parquet magic bytes, "
                "which usually means they are truncated or corrupted: "
                + ", ".join(str(path) for path in invalid_paths)
            )

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
            logger.warning(f"Could not decode ERQA-plus image of type {type(image)}")
            return None

    def _load_image_path(self, image_path: str) -> Image.Image:
        path = Path(str(image_path).strip())
        candidates = [path]
        if not path.is_absolute():
            rel = str(path)
            if rel.startswith("./"):
                rel = rel[2:]
            if self.data_root is not None:
                candidates = [self.data_root / rel, self.data_root / path, path]

        for candidate in candidates:
            if candidate.is_file():
                return Image.open(candidate).convert("RGB")

        raise FileNotFoundError(f"Could not find ERQA-plus image path: {image_path}")

    def _images_from_sample(self, sample: Dict[str, Any]) -> List[Image.Image]:
        images: List[Image.Image] = []

        all_images = sample.get("all_images")
        if all_images is not None:
            if not isinstance(all_images, list):
                all_images = [all_images]
            images = [image for image in (self._decode_image(item) for item in all_images) if image is not None]

        if not images and sample.get("image") is not None:
            image = self._decode_image(sample.get("image"))
            if image is not None:
                images = [image]

        if not images:
            image_paths = sample.get("image_paths") or sample.get("images") or []
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            images = [self._load_image_path(path) for path in image_paths]

        return images

    @staticmethod
    def _format_choices(choices: Any) -> str:
        if not choices:
            return ""
        if isinstance(choices, str):
            return choices.strip()
        if isinstance(choices, list):
            return "\n".join(str(choice).strip() for choice in choices if str(choice).strip())
        return str(choices).strip()

    def _format_question(self, sample: Dict[str, Any]) -> str:
        question = str(sample.get("question", "")).strip()
        choices = self._format_choices(sample.get("choices"))
        if choices and "choices:" not in question.lower():
            question = f"{question}\nChoices:\n{choices}"

        instruction = str(self.instruct_following or "").strip()
        if instruction and instruction not in question:
            question = f"{question}\n{instruction}"

        return textwrap.dedent(question).strip()

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            sample = dict(sample)
            images = self._images_from_sample(sample)
            answer = self.extract_answer_from_text(sample.get("answer", ""))
            question = self._format_question(sample)
            image_paths = sample.get("image_paths") or []
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            prepared_dataset.append({
                "question": question,
                "answer": answer,
                "image": images,
                "metadata": {
                    "idx": idx,
                    "qid": sample.get("qid") or sample.get("question_id") or sample.get("id"),
                    "question_type": sample.get("question_type"),
                    "explanation": sample.get("explanation"),
                    "choices": sample.get("choices"),
                    "visual_indices": sample.get("visual_indices"),
                    "image_paths": image_paths,
                    "num_images": len(images),
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
            "sample_id": processed_sample["metadata"].get("qid"),
            "question_type": processed_sample["metadata"].get("question_type"),
            "question": processed_sample["question"],
            "ground_truth": answer,
            "metadata": processed_sample["metadata"],
            "raw_output": original_raw_output,
            "processed_answer": prediction,
            "is_correct": is_correct,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("Evaluating ERQA-plus predictions...")
        all_results = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            result = self.process_raw_output(sample, raw_output)
            all_results.append(result)
            if self.debug or len(all_results) <= 5:
                logger.info("=" * 60)
                logger.info(f"Sample ID: {result.get('sample_id')}")
                logger.info(f"Question Type: {result.get('question_type')}")
                logger.info(f"Prediction: {result.get('processed_answer')}")
                logger.info(f"Ground Truth: {result['ground_truth']}")
                logger.info(f"Is Correct: {result['is_correct']}")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        correct_predictions = sum(1 for result in results if result["is_correct"])
        overall_accuracy = correct_predictions / total_samples if total_samples else 0.0

        per_type_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        per_image_count = defaultdict(lambda: {"total": 0, "correct": 0})
        for result in results:
            metadata = result.get("metadata", {})
            question_type = str(metadata.get("question_type") or "unknown")
            num_images = str(metadata.get("num_images") or 0)
            per_type_counts[question_type]["total"] += 1
            per_type_counts[question_type]["correct"] += int(result["is_correct"])
            per_image_count[num_images]["total"] += 1
            per_image_count[num_images]["correct"] += int(result["is_correct"])

        question_type_results = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_type_counts.items())
        }
        image_count_results = {
            name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for name, counts in sorted(per_image_count.items(), key=lambda item: int(item[0]))
        }

        logger.info(f"ERQA-plus overall accuracy: {overall_accuracy:.4f} ({correct_predictions}/{total_samples})")

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "question_type_results": question_type_results,
            "image_count_results": image_count_results,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
