import json
import logging
import os
import re
import tarfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

EVAL_BINARY_INSTRUCTION = " Please answer yes or no without any other extra text."


def transform_training_qa(question: str, answer: str, category: str) -> tuple[str, str]:
    """Apply the train_explicit_style transform used to build the SFT data."""
    question = question.strip()
    if "Q:" in question:
        prefix, suffix = question.rsplit("Q:", 1)
        prefix, suffix = prefix.rstrip(), suffix.strip()
    else:
        normalized = question
        if normalized.endswith(EVAL_BINARY_INSTRUCTION):
            normalized = normalized[: -len(EVAL_BINARY_INSTRUCTION)].rstrip()
        suffixes = (
            "what is likely to happen next?",
            "what action is possible right now?",
            "what just happened?",
            "possible right now?",
            "satisfied?",
            "immediate next step?",
            "next 5 steps?",
        )
        prefix = suffix = ""
        for candidate in suffixes:
            if normalized == candidate:
                suffix = candidate
                break
            if normalized.endswith(candidate):
                prefix = normalized[: -len(candidate)].rstrip()
                suffix = candidate
                break
        if not suffix:
            raise ValueError(f"Unsupported RoboVQA question template for {category}: {question}")

    if category == "past_description" and suffix == "what just happened?":
        transformed = "what just happened?"
        instruction = (
            "Answer with only one short action phrase describing the completed action. "
            "Do not explain or add any other text."
        )
    elif category == "future_prediction" and suffix == "what is likely to happen next?":
        transformed = "what is likely to happen next?"
        instruction = (
            "Answer with only one short action phrase describing the most likely next action. "
            "Do not explain or add any other text."
        )
    elif category == "affordance" and suffix == "what action is possible right now?":
        transformed = "what action is possible right now?"
        instruction = (
            "Answer with only one short action phrase describing an action that is possible now. "
            "Do not explain or add any other text."
        )
    elif category == "affordance" and suffix == "possible right now?":
        if not prefix:
            raise ValueError("Discriminative affordance question has an empty action")
        transformed = f"{prefix}. is it possible right now?"
        instruction = "Answer only yes or no, without explanation or any other text."
    elif category == "success" and suffix == "satisfied?":
        if not prefix:
            raise ValueError("Success question has an empty action")
        transformed = f"{prefix}. is it satisfied?"
        instruction = "Answer only yes or no, without explanation or any other text."
    elif category in {"planning", "immediate_planning_with_context20"} and suffix == "immediate next step?":
        if not prefix:
            raise ValueError(f"{category} question has an empty context")
        transformed = f"{prefix}. what is the immediate next step?"
        instruction = (
            "Answer with only one short action phrase describing the immediate next action. "
            "Do not explain or add any other text."
        )
    elif category == "remaining5_planning_with_context20" and suffix == "next 5 steps?":
        if not prefix:
            raise ValueError("Remaining-5 question has an empty context")
        transformed = f"{prefix}. what are the next 5 steps?"
        instruction = (
            'Answer with up to five consecutively numbered entries in the format "1- ... 2- ...". '
            'If the task will be completed in fewer than five actions, list the remaining actions '
            'and then write "done" as the final numbered entry. If it is already complete, answer '
            '"1- done". Do not add any text after "done" or outside the numbered list.'
        )
    else:
        raise ValueError(f"Unsupported RoboVQA question template for {category}: {question}")

    transformed_answer = answer.strip().rstrip(".").rstrip()
    if not transformed_answer:
        raise ValueError("Answer became empty after formatting")
    if category == "success" or (category == "affordance" and suffix == "possible right now?"):
        transformed_answer = transformed_answer.lower()
        if transformed_answer not in {"yes", "no"}:
            raise ValueError(f"Expected a yes/no answer for {category}, found: {answer}")
    if category == "remaining5_planning_with_context20" and not transformed_answer.startswith("1-"):
        raise ValueError(f"Remaining-5 answer is not numbered: {answer}")
    return f"{transformed} {instruction}", transformed_answer


class RoboVQADataset(BaseDataset):
    """RoboVQA adapter for the legacy HF data and local multi-frame data."""

    def __init__(
        self,
        dataset_name: str = "IffYuan/RoboVQA",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "RoboVQA",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        thinking_model: bool = False,
        data_root: Optional[str] = None,
        qa_jsonl: Optional[str] = None,
        prompt_policy: str = "raw",
        system_prompt: Optional[str] = None,
        expected_num_frames: Optional[int] = None,
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
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self.qa_jsonl = Path(qa_jsonl).expanduser().resolve() if qa_jsonl else None
        self.prompt_policy = prompt_policy
        self.system_prompt = system_prompt
        self.expected_num_frames = expected_num_frames

    def get_default_instruct(self) -> str:
        return ""

    def load_dataset(self) -> Any:
        """Load RoboVQA dataset from HuggingFace"""
        if self.data_root or self.qa_jsonl:
            qa_path = self.qa_jsonl or self.data_root / "qa.jsonl"
            data_root = self.data_root or qa_path.parent
            frames_archive = data_root / "frames.tar"
            archive_members: Optional[set[str]] = None
            if not qa_path.is_file():
                raise FileNotFoundError(f"RoboVQA qa.jsonl not found: {qa_path}")

            rows = []
            with qa_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    frame_paths = row.get("frame_paths")
                    if not isinstance(frame_paths, list) or not frame_paths:
                        raise ValueError(f"Missing frame_paths at {qa_path}:{line_number}")
                    if self.expected_num_frames and len(frame_paths) != self.expected_num_frames:
                        raise ValueError(
                            f"Expected {self.expected_num_frames} frames at {qa_path}:{line_number}, "
                            f"found {len(frame_paths)}"
                        )
                    image_paths = [(data_root / frame_path).resolve() for frame_path in frame_paths]
                    if all(path.is_file() for path in image_paths):
                        row["images"] = [str(path) for path in image_paths]
                    elif frames_archive.is_file():
                        if archive_members is None:
                            with tarfile.open(frames_archive, mode="r:") as archive:
                                archive_members = {member.name for member in archive if member.isfile()}
                        missing = next(
                            (frame_path for frame_path in frame_paths if frame_path not in archive_members),
                            None,
                        )
                        if missing:
                            raise FileNotFoundError(
                                f"RoboVQA frame not found in {frames_archive}: {missing}"
                            )
                        row["images"] = [
                            {
                                "type": "tar_image",
                                "archive": str(frames_archive.resolve()),
                                "member": frame_path,
                            }
                            for frame_path in frame_paths
                        ]
                    else:
                        missing = next(path for path in image_paths if not path.is_file())
                        raise FileNotFoundError(
                            f"RoboVQA frame not found: {missing}; archive not found: {frames_archive}"
                        )
                    rows.append(row)
            storage = frames_archive if archive_members is not None else data_root / "frames"
            logger.info(
                "Loaded local RoboVQA dataset: %d rows from %s (frames: %s)",
                len(rows),
                qa_path,
                storage,
            )
            return rows

        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []

        for sample in dataset:
            idx = sample["id"]
            question_idx = sample["question_idx"]
            question = sample["question"]
            images = sample["images"]
            answer = str(sample["answer"])

            match = re.search(r"<task:([^:]+)", question_idx)
            category = str(sample.get("task_type") or (match.group(1) if match else "Unknown"))

            if self.prompt_policy == "train_explicit_style":
                question, answer = transform_training_qa(str(question), answer, category)
                formatted_answer = answer.strip()
            elif self.prompt_policy != "raw":
                raise ValueError(f"Unsupported RoboVQA prompt policy: {self.prompt_policy}")
            else:
                formatted_answer = answer.strip().lower()

            problem = question + '\n' + self.instruct_following
            formatted_question = textwrap.dedent(problem).strip()

            prepared.append(
                {
                    "question": formatted_question,
                    "answer": formatted_answer,
                    "image": images,
                    "system_prompt": self.system_prompt,
                    "metadata": {
                        "idx": idx,
                        "question_idx": question_idx,
                        "category": category,
                        "clip_id": sample.get("clip_id"),
                        "task_type": sample.get("task_type"),
                        "task_spec": sample.get("task_spec"),
                        "num_images": len(images),
                    },
                }
            )

        if self.debug:
            prepared = prepared[:20]
        return prepared

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        """Process raw output for RoboVQA evaluation"""
        original_raw_output = raw_output_text

        # Extract content from <answer></answer> tags for thinking models
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        metadata = prepared_sample["metadata"]
        idx = metadata["idx"]
        question_idx = metadata["question_idx"]
        category = metadata["category"]
        question = prepared_sample["question"]
        gt_answer = prepared_sample["answer"]

        predicted_text = str(raw_output_text).strip().lower()
        if predicted_text.endswith("."):
            predicted_text = predicted_text[:-1]
        predicted_text = predicted_text.replace("<", "").replace(">", "")

        try:
            bleu_score = sacrebleu.sentence_bleu(predicted_text, [gt_answer], tokenize="intl").score
        except Exception:
            bleu_score = 0.0

        return {
            "idx": idx,
            "question_idx": question_idx,
            "category": category,
            "question": question,
            "raw_output": original_raw_output,
            "processed_output": predicted_text,
            "ground_truth": gt_answer,
            "bleu_score": bleu_score,
            "category": category,
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        """Evaluate all results"""
        logger.info("\nEvaluating results...")
        all_results: List[Dict[str, Any]] = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            try:
                result = self.process_raw_output(sample, raw_output)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process sample {sample.get('metadata', {}).get('idx', 'N/A')}: {e}. Skipping.")
        return all_results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)
        if total_samples == 0:
            return {}

        overall_bleu = sum(r["bleu_score"] for r in results) / total_samples
        category_scores: dict[str, list[float]] = defaultdict(list)
        for r in results:
            category_scores[r["category"]].append(r["bleu_score"])

        category_stats = {
            cat: {"bleu_score": sum(scores) / len(scores), "count": len(scores)}
            for cat, scores in category_scores.items()
        }

        logger.info("\nOverall Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Samples Processed: {total_samples}")
        logger.info(f"Overall BLEU Score: {overall_bleu:.4f}")
        logger.info("\nCategory-wise BLEU Scores:")
        for cat, stats in category_stats.items():
            logger.info(f"  {cat}: {stats['bleu_score']:.4f} ({stats['count']} samples)")
        logger.info("=" * 60)

        return {"overall_bleu": overall_bleu, "total_samples": total_samples, "category_results": category_stats}

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        """Save evaluation results to file"""
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "statistics": statistics,
            }, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
