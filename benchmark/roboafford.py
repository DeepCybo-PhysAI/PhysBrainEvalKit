import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from core.point_utils import compute_point_metrics, score_single_mask_sample

from .base import BaseDataset
from .prompt_policy import validate_prompt_policy

logger = logging.getLogger(__name__)


class RoboAffordDataset(BaseDataset):
    """RoboAfford (Zray26/roboafford-eval)"""

    def __init__(
        self,
        dataset_name: str = "Zray26/roboafford-eval",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "RoboAfford",
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
        """Return the point output protocol used by the selected backbone."""
        if self.prompt_policy == "original" and self.backbone == "qwen3":
            return """The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]."""
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

    def load_dataset(self) -> Any:
        logger.info("Dataset: %s, Split: %s", self.dataset_name, self.split)
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)
        logger.info("Dataset loaded. Number of samples: %d", len(dataset))
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Build single-point prompts for samples with one target mask."""
        prepared_dataset: List[Dict[str, Any]] = []
        for sample_idx, sample in enumerate(dataset):
            question = str(sample["question"])
            if self.prompt_policy != "original":
                question = question.strip()
            target_text = "Your answer should be formatted as a list of tuples"
            text_idx = question.find(target_text)
            if text_idx != -1:
                question = question[:text_idx]
                if self.prompt_policy != "original":
                    question = question.strip()

            category = sample["category"]
            image = sample["image"]
            width, height = image.size
            if self.prompt_policy == "original":
                point_instruction = (
                    "Provide your answer as 2D point coordinates: each tuple "
                    "contains the x and y coordinates of a point meeting the "
                    "above conditions."
                )
                prompt = f"{question}{point_instruction}".strip()
                prompt = f"{prompt}\n{self.instruct_following}"
            else:
                prompt = (
                    f"{question}\n"
                    "Provide exactly one point on the target region.\n"
                    f"{self.instruct_following}"
                )
            prepared_dataset.append(
                {
                    "question": textwrap.dedent(prompt).strip(),
                    "answer": None,
                    "image": image,
                    "metadata": {
                        "idx": sample_idx,
                        "task": category,
                        "mask": sample["mask"],
                        "width": width,
                        "height": height,
                    },
                }
            )

        return prepared_dataset

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
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
            "question": prepared_sample["question"],
            "task": metadata.get("task"),
            "raw_output": raw_output_text,
            **scored,
        }

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
        logger.info(
            "RoboAfford point metrics: non-strict micro P/R/F1=%.4f/%.4f/%.4f, "
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
    
    def visualize(self, result_file_path: str, output_dir: str = "logs/visualizations", vis_mask: bool = True):
        def formatted_question_to_vis_text(question: str) -> str:
            question_mark_idx = question.find('?')
            if question_mark_idx != -1:
                return question[:question_mark_idx].strip()
            return question.strip()
        
        import cv2

        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, self.task_name, self.model_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        with open(result_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = data['results']
        dataset = self.load_dataset()
        
        for res in results:
            idx = res["idx"]
            sample = dataset[idx]
            question = sample['question']
            image_pil = sample['image'].convert('RGB')
            img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            h, w, c = img.shape
            mask = np.array(sample['mask'])
            if vis_mask:
                mask_overlay = img.copy()
                mask_overlay[mask > 0] = [0, 255, 0] 
                cv2.addWeighted(mask_overlay, 0.3, img, 0.7, 0, img)

            processed_points = res["processed_points"]
            for pt in processed_points:
                x, y = int(pt[0]), int(pt[1])
                
                is_in_mask = False
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        is_in_mask = True

                inner_color = (255, 0, 0) if is_in_mask else (128, 128, 128)
                outer_color = (255, 255, 255) 
                
                cv2.circle(img, (x, y), 6, outer_color, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 4, inner_color, -1, cv2.LINE_AA)

            bottom_text = formatted_question_to_vis_text(question)
            text_area_h = 60
            bottom_bar = np.zeros((text_area_h, w, c), dtype=img.dtype)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(bottom_text, font, 1.0, 1)
            
            max_display_w = int(w * 0.95)
            font_scale = min(0.7, max_display_w / text_w) if text_w > 0 else 0.5
            
            (final_w, final_h), _ = cv2.getTextSize(bottom_text, font, font_scale, 1)
            text_x = (w - final_w) // 2
            text_y = (text_area_h + final_h) // 2
            
            cv2.putText(bottom_bar, question, (text_x, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            final_img = cv2.vconcat([img, bottom_bar])
            save_path = os.path.join(vis_dir, f"sample_{idx}_acc_{res['accuracy_score']:.2f}.png")
            cv2.imwrite(save_path, final_img)
            print(f"Saving visualization to {save_path}")
