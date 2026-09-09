import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.point_utils import (
    compute_point_metrics,
    score_single_mask_sample,
    validate_mask_size,
)

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class RoboRefitDataset(BaseDataset):
    """RoboRefIt point grounding evaluated against a single target mask."""

    def __init__(
        self,
        dataset_name: str = "IffYuan/Roborefit",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "RoboRefit",
        model_name: Optional[str] = None,
        debug: bool = False,
        backbone: Optional[str] = None,
        thinking_model: bool = False,
        data_root: Optional[str] = None,
        qa_jsonl: Optional[str] = None,
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
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self.qa_jsonl = Path(qa_jsonl).expanduser().resolve() if qa_jsonl else None
        self.prompt_policy = validate_prompt_policy(prompt_policy)

    def get_default_instruct(self) -> str:
        """Return the point output protocol used by the selected backbone."""
        if self.prompt_policy == "original" and self.backbone == "qwen3":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        if self.backbone in {"gemma4", "qwen3_5", "qwen3", "gemini-2.5"}:
            return (
                'The answer must be a JSON list: [{"point_2d": [x, y]}]\n'
                "Coordinates must be integers normalized to [0, 1000]."
            )
        if self.backbone in {"qwen2.5", "qwen2_5", "mimo"}:
            return (
                'The answer must be a JSON list: [{"point_2d": [x, y]}]. '
                "Use absolute image pixel coordinates."
            )
        if self.backbone == "gemini_robotics":
            return (
                'The answer should follow the JSON format: [{"point": <point>, '
                '"label": <label1>}]. The point must be in [y, x] order normalized '
                "to [0, 1000]."
            )
        if self.backbone == "molmo":
            return (
                'Provide one point in XML format, for example: <point x="63.5" '
                'y="44.5" alt="target">target</point>.'
            )
        if self.backbone in {"gpt", "pelican", "internvl", "magma"}:
            return (
                "Your answer must be a list containing exactly one (x, y) tuple, "
                "where both coordinates are normalized to [0, 1]."
            )
        raise ValueError(f"Unsupported backbone: {self.backbone}")

    @staticmethod
    def _resolve_local_path(
        value: Any,
        *,
        data_root: Path,
        field: str,
        qa_path: Path,
        line_number: int,
    ) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {field} at {qa_path}:{line_number}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = data_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"RoboRefIt {field} not found at {qa_path}:{line_number}: {path}"
            )
        return path

    def _load_local_dataset(self) -> List[Dict[str, Any]]:
        qa_path = self.qa_jsonl or self.data_root / "qa.jsonl"
        data_root = qa_path.parent if self.qa_jsonl else self.data_root
        if not qa_path.is_file():
            raise FileNotFoundError(f"RoboRefIt qa.jsonl not found: {qa_path}")

        rows: List[Dict[str, Any]] = []
        with qa_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question = row.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Missing question at {qa_path}:{line_number}")

                image_path = self._resolve_local_path(
                    row.get("image_path") or row.get("image"),
                    data_root=data_root,
                    field="image_path",
                    qa_path=qa_path,
                    line_number=line_number,
                )
                mask_path = self._resolve_local_path(
                    row.get("mask_path") or row.get("mask"),
                    data_root=data_root,
                    field="mask_path",
                    qa_path=qa_path,
                    line_number=line_number,
                )
                row["question_id"] = row.get(
                    "question_id", row.get("id", len(rows))
                )
                row["image_path"] = str(image_path)
                row["mask_path"] = str(mask_path)
                rows.append(row)

        logger.info("Loaded %d corrected RoboRefIt rows from %s", len(rows), qa_path)
        return rows

    def load_dataset(self) -> Any:
        if self.data_root or self.qa_jsonl:
            return self._load_local_dataset()

        logger.info("Dataset: %s, Split: %s", self.dataset_name, self.split)
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)
        logger.info("Dataset loaded. Number of samples: %d", len(dataset))
        return dataset

    @staticmethod
    def _load_image(value: Any, *, mode: str, field: str) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert(mode)
        if isinstance(value, (str, os.PathLike)):
            with Image.open(value) as image:
                return image.convert(mode).copy()
        raise ValueError(
            f"RoboRefIt sample requires a PIL image or path in {field}"
        )

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Load local images/masks and build an explicitly single-point prompt."""
        prepared_dataset: List[Dict[str, Any]] = []
        for idx, sample in enumerate(dataset):
            question_id = sample.get("question_id", sample.get("id", idx))
            question = str(sample["question"]).strip()
            image_value = sample.get("image_path") or sample.get("image")
            mask_value = sample.get("mask_path") or sample.get("mask")
            image = self._load_image(
                image_value, mode="RGB", field="image_path/image"
            )
            mask = self._load_image(mask_value, mode="L", field="mask_path/mask")
            width, height = image.size

            declared_size = (sample.get("width"), sample.get("height"))
            if all(value is not None for value in declared_size):
                declared_size = tuple(int(value) for value in declared_size)
                if declared_size != (width, height):
                    raise ValueError(
                        f"RoboRefIt declared image size {declared_size} does not "
                        f"match decoded image size {(width, height)} for question "
                        f"{question_id}"
                    )
            validate_mask_size(mask, width, height)
            if mask.getbbox() is None:
                raise ValueError(f"RoboRefIt mask is empty for question {question_id}")

            if self.prompt_policy == "original":
                prompt = (
                    "Provide a point annotation for the interaction target in: "
                    f"{question}.\n{self.instruct_following}"
                )
            else:
                prompt = (
                    "Provide exactly one point on the interaction target described by: "
                    f"{question}.\n{self.instruct_following}"
                )
            prepared_dataset.append(
                {
                    "question": textwrap.dedent(prompt).strip(),
                    "answer": None,
                    "image": image,
                    "metadata": {
                        "idx": idx,
                        "question_id": question_id,
                        "task": "roborefit",
                        "mask": mask,
                        "mask_path": sample.get("mask_path"),
                        "mask_bbox": sample.get("mask_bbox"),
                        "mask_area": sample.get("mask_area"),
                        "bbox": sample.get("bbox"),
                        "width": width,
                        "height": height,
                    },
                }
            )

        return prepared_dataset

    def process_raw_output(
        self,
        prepared_sample: Dict[str, Any],
        raw_output_text: str,
    ) -> Dict[str, Any]:
        metadata = prepared_sample["metadata"]
        scored = score_single_mask_sample(
            raw_output_text,
            mask=metadata["mask"],
            width=metadata["width"],
            height=metadata["height"],
            backbone=self.backbone,
        )
        return {
            "idx": metadata["idx"],
            "question_id": metadata.get("question_id"),
            "question": prepared_sample["question"],
            "raw_output": raw_output_text,
            "mask_path": metadata.get("mask_path"),
            "mask_bbox": metadata.get("mask_bbox"),
            "mask_area": metadata.get("mask_area"),
            "bbox": metadata.get("bbox"),
            **scored,
        }

    def evaluate_results(
        self,
        prepared_dataset: List[Dict[str, Any]],
        raw_outputs: List[str],
    ) -> List[Dict[str, Any]]:
        logger.info("\nEvaluating RoboRefIt results against target masks...")
        return [
            self.process_raw_output(sample, raw_output)
            for sample, raw_output in tqdm(
                zip(prepared_dataset, raw_outputs),
                total=len(prepared_dataset),
                desc="Evaluation",
            )
        ]

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = compute_point_metrics(results)
        metrics.update(
            {
                "overall_accuracy": metrics["strict_micro_f1"],
                "average_accuracy": metrics["strict_macro_f1"],
                "total_points_predicted": sum(
                    result["total_points"] for result in results
                ),
                "total_points_in_mask": sum(
                    result["points_in_mask"] for result in results
                ),
            }
        )
        logger.info(
            "RoboRefIt mask metrics: non-strict micro P/R/F1=%.4f/%.4f/%.4f, "
            "strict micro P/R/F1=%.4f/%.4f/%.4f",
            metrics["non_strict_micro_precision"],
            metrics["non_strict_micro_recall"],
            metrics["non_strict_micro_f1"],
            metrics["strict_micro_precision"],
            metrics["strict_micro_recall"],
            metrics["strict_micro_f1"],
        )
        return metrics

    def save_results(
        self,
        results: List[Dict[str, Any]],
        statistics: Dict[str, Any],
    ) -> str:
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"
        with open(result_file_name, "w", encoding="utf-8") as handle:
            json.dump(
                {"results": results, "statistics": statistics},
                handle,
                ensure_ascii=False,
                indent=4,
            )
        logger.info("Results saved to: %s", result_file_name)
        return result_file_name
