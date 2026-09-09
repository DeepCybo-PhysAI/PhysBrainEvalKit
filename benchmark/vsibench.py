import io
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
    "object_rel_distance", "route_planning", "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation",
]
METRICS_FOR_MCA = {"accuracy": "exact_match"} 
METRICS_FOR_NA = {"MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)"}
WORST_CASE_FOR_METRICS = {"accuracy": 0., "MRA:.5:.95:.05": 0.}

class VSIBenchDataset(BaseDataset):
    """VSI-Bench (HF dataset, video frames stored as bytes)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/vsi-bench",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "VSI-Bench",
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
        return ""

    def load_dataset(self) -> Any:
        """Load dataset from HuggingFace"""
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        if self.debug:
            dataset = dataset.select(range(min(20, len(dataset))))

        self.raw_dataset = dataset
        video_dataset = dataset.select_columns(["videos"])
        metadata_dataset = dataset.remove_columns(["videos"])
        for i, sample in enumerate(metadata_dataset):
            prepared.append({
                "question": sample["question"]+self.instruct_following, 
                "answer": str(sample["ground_truth"]).strip().lower(),
                "video": [{
                    "__lazy_hf_video__": True,
                    "dataset": video_dataset,
                    "row_index": i,
                    "column": "videos",
                }],
                "metadata": {
                    "id": sample["id"],
                    "dataset": sample.get("dataset"),
                    "scene_name": sample.get("scene_name"),
                    "question_type": sample.get("question_type"),
                    "options": sample.get("options"),
                    "video_path": sample.get("video_path"),
                },
            })

        return prepared

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:

        def exact_match(pred, target):
            return 1. if pred.lower() == target.lower() else 0.
        
        def to_float(pred):
            try:
                match = re.search(r'-?\d+\.?\d*', str(pred))
                if match: 
                    return float(match.group(0))
            except (ValueError, TypeError):
                pass
            word_to_num = {
                'zero': 0, 'one': 1,
                'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12,
                'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
                'twenty': 20
            }
            for w in word_to_num:
                if w in str(pred).lower():
                    return float(word_to_num[w])

            return None

        def abs_dist_norm(pred, target):
            return abs(pred - target) / target 
        
        def mean_relative_accuracy(pred, target, start, end, interval):
            num_pts = (end - start) / interval + 2
            conf_intervs = np.linspace(start, end, int(num_pts))
            accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
            return accuracy.mean()

        original_raw_output = raw_output_text

        # Extract content from <answer></answer> tags for thinking models
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        meta = prepared_sample["metadata"]
        q_type = meta['question_type']
        gt_text = prepared_sample['answer']
        question_text = prepared_sample['question']
        results = {
            "id": meta["id"],
            "question": question_text, 
            "question_type": q_type,
            "raw_output": original_raw_output, 
            "ground_truth": gt_text,   
            "score": 0.0,
            "metric_type": "unknown"
        }
        cleaned_pred = raw_output_text.strip().split(' ')[0].rstrip('.').strip()
        if q_type in MCA_QUESTION_TYPES:
            score = exact_match(cleaned_pred, gt_text)
            results['score'] = score
            results['metric_type'] = 'accuracy'
            
        elif q_type in NA_QUESTION_TYPES:
            pred_float = to_float(cleaned_pred)
            gt_float = to_float(gt_text)
            if pred_float is not None and gt_float is not None:
                mra = mean_relative_accuracy(pred_float, gt_float, start=0.5, end=0.95, interval=0.05)
                results['score'] = mra
            else:
                results['score'] = WORST_CASE_FOR_METRICS['MRA:.5:.95:.05']
            results['metric_type'] = 'mra'
            
        else:
            logger.warning(f"Unknown question type: {q_type}")
            results['score'] = 0.0

        return results

    def evaluate_results(
        self,
        prepared_dataset: List[Dict[str, Any]],
        raw_outputs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Iterate over dataset and outputs to generate evaluation results.
        """
        logger.info("Evaluating VSI-Bench Predictions...")
        results = []

        for sample, output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            try:
                res = self.process_raw_output(sample, output)
                results.append(res)
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.get('id')}: {e}")

                results.append({
                    "id": sample["metadata"]["id"],
                    "question_type": sample["metadata"]["question_type"],
                    "score": 0.0,
                    "error": str(e)
                })

        return results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results to compute final metrics per category and overall.
        """
        if not results:
            return {}

        type_stats = defaultdict(list)
        
        for r in results:
            q_type = r["question_type"]
            type_stats[q_type].append(r["score"])

        final_stats = {}
        for q_type, scores in type_stats.items():
            metric_name = "accuracy" if q_type in MCA_QUESTION_TYPES else "mra"
            final_stats[f"{q_type}_{metric_name}"] = sum(scores) / len(scores)

        direction_keys = [
            'object_rel_direction_easy_accuracy',
            'object_rel_direction_medium_accuracy',
            'object_rel_direction_hard_accuracy'
        ]
        dir_scores = [final_stats.pop(k) for k in direction_keys if k in final_stats]
        if dir_scores:
            final_stats['object_rel_direction_accuracy'] = sum(dir_scores) / len(dir_scores)

        if final_stats:
            overall = sum(final_stats.values()) / len(final_stats)
        else:
            overall = 0.0
        
        final_stats['overall'] = overall

        logger.info("\n" + "="*40)
        logger.info(f"VSI-Bench Statistics (Total: {len(results)})")
        logger.info("="*40)
        for k, v in final_stats.items():
            logger.info(f"{k}: {v*100:.2f}%")
        logger.info("="*40)

        return {
            "overall_score": overall,
            "category_results": final_stats,
             "predictions": results 
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        """Save detailed results and stats to JSON."""
        os.makedirs("logs/results", exist_ok=True)
        filename = f"{self.task_name}_{self.model_name}_results.json"
        path = os.path.join("logs/results", filename)

        output_data = {
            "model": self.model_name,
            "results": results,
            "statistics": statistics
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to {path}")
        return path
