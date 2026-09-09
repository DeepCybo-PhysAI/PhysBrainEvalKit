import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from tqdm import tqdm

from core.point_utils import omni_decode_points

from .base import BaseDataset

logger = logging.getLogger(__name__)


def interpolate_trajectory(trajectory, new_length):
    """Interpolate trajectory to a fixed number of points"""
    if len(trajectory) <= 1 or new_length <= 1:
        return trajectory

    old_indices = np.arange(len(trajectory))
    new_indices = np.linspace(0, len(trajectory) - 1, new_length)

    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    x_interpolator = interp1d(old_indices, x_coords, kind='linear')
    y_interpolator = interp1d(old_indices, y_coords, kind='linear')

    new_x_coords = x_interpolator(new_indices)
    new_y_coords = y_interpolator(new_indices)

    return [[float(x), float(y)] for x, y in zip(new_x_coords, new_y_coords)]


def calculate_rmse_mae(pred_trajectory, ans_trajectory):
    """Calculate RMSE and MAE between predicted and ground truth trajectories"""
    if len(pred_trajectory) != len(ans_trajectory):
        logger.warning(f"Trajectory length mismatch: pred={len(pred_trajectory)}, ans={len(ans_trajectory)}")
        return None, None

    squared_diffs = []
    abs_diffs = []

    for pred_point, ans_point in zip(pred_trajectory, ans_trajectory):
        dx = pred_point[0] - ans_point[0]
        dy = pred_point[1] - ans_point[1]

        squared_diff = dx**2 + dy**2
        squared_diffs.append(squared_diff)

        abs_diff = (abs(dx) + abs(dy)) / 2
        abs_diffs.append(abs_diff)

    rmse = np.sqrt(np.mean(squared_diffs))
    mae = np.mean(abs_diffs)

    return rmse, mae


def calculate_discrete_frechet(P, Q):
    """
    Calculate the Discrete Fréchet Distance between two trajectories P and Q.
    P and Q are lists of points [[x1, y1], [x2, y2], ...].
    """
    P = np.array(P)
    Q = np.array(Q)
    
    n = len(P)
    m = len(Q)
    
    if n == 0 or m == 0:
        return None

    dist_matrix = cdist(P, Q, 'euclidean')
    ca = np.ones((n, m)) * -1.0
    ca[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], dist_matrix[0, j])

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(dist_matrix[i, j], min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]))

    return ca[n-1, m-1]


class VABenchVisualTraceDataset(BaseDataset):
    """VABench Visual Trace Dataset (Embodied1/vabench-v)"""

    def __init__(
        self,
        dataset_name: str = "IffYuan/vabench-v",
        subset: Optional[str] = None,
        split: str = "test",
        instruct_following: Optional[str] = None,
        task_name: str = "VABench-VisualTrace",
        model_name: Optional[str] = None,
        debug: bool = False,
        backbone: Optional[str] = None,
        thinking_model: bool = False
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

    def get_default_instruct(self) -> str:
        """Return default instruction based on model type"""
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            # 0-1000 normalized pixel coordinates (Qwen3 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            # absolute pixel coordinates (Qwen2.5 Standard)
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl' or self.backbone =='magma':
            # gpt format
            # 0-1 normalized pixel coordinates
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
            
    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        """Preprocess VABench Visual Trace samples"""
        prepared_dataset = []
        for idx, sample in enumerate(dataset):
            question_id = sample.get('id', str(hash(sample['problem']))[:8])
            question = sample['problem'].strip()
            image = sample['image'].convert("RGB")
            ans_traj = sample['answer']
            if isinstance(ans_traj, str):
                ans_traj = json.loads(ans_traj)

            width, height = image.size

            problem = question + '\n' + self.instruct_following
            formatted_question = textwrap.dedent(problem).strip()

            prepared_dataset.append({
                'question': formatted_question,
                'answer': ans_traj,
                'image': image,
                'metadata': {
                    'idx': idx,
                    'question_id': question_id,
                    'ans_traj': ans_traj,
                    'width': width,
                    'height': height,
                }
            })

        if self.debug:
            prepared_dataset = prepared_dataset[:20]
            logger.info(f"Debug mode: processing first 20 samples only")

        return prepared_dataset

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        original_predicted = raw_output_text

        # Extract answer from reasoning model output
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        # Parse predicted trajectory
        pred_traj = omni_decode_points(str(raw_output_text))

        metadata = prepared_sample['metadata']
        width = metadata['width']
        height = metadata['height']
        ans_traj = metadata['ans_traj']
        idx = metadata['idx']
        question_id = metadata['question_id']
        question = prepared_sample['question']

        # Scale trajectories to normalized coordinates (0-1000)
        ans_traj_scaled = []
        for point in ans_traj:
            x, y = point
            x = (x / width) * (1000 - 1)
            y = (y / height) * (1000 - 1)
            ans_traj_scaled.append([x, y])
        ans_traj_normalized = ans_traj_scaled

        pred_traj_scaled = []
        pred_traj_normalized=[]
        if len(pred_traj) > 0:
            if self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
                for point in pred_traj:
                    x, y = point
                    x = (x / width) * (1000 - 1)
                    y = (y / height) * (1000 - 1)
                    pred_traj_scaled.append([x, y])
                pred_traj_normalized = pred_traj_scaled
            elif self.backbone in ["gemma4", "qwen3_5", "qwen3", "gemini-2.5"]:
                # 0-1000 to abs
                pred_traj_normalized = pred_traj
            elif self.backbone in ['molmo']:
                for point in pred_traj:
                    x_abs = int(round(point[0]*10))
                    y_abs = int(round(point[1]*10))
                    pred_traj_scaled.append([x_abs, y_abs])    
                pred_traj_normalized = pred_traj_scaled                
            elif self.backbone in ['gpt','pelican','internvl','magma']:
                for point in pred_traj:
                    x_abs = int(round(point[0]*1000))
                    y_abs = int(round(point[1]*1000))
                    pred_traj_scaled.append([x_abs, y_abs])
                pred_traj_normalized = pred_traj_scaled      
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        rmse = None
        mae = None
        dfd = None  # Discrete Fréchet Distance

        if len(pred_traj_normalized) > 1 and len(ans_traj_normalized) > 1:
            # Interpolate to align trajectory lengths for RMSE/MAE
            new_length = max(len(pred_traj_normalized), len(ans_traj_normalized))
            pred_traj_interp = interpolate_trajectory(pred_traj_normalized, new_length)
            ans_traj_interp = interpolate_trajectory(ans_traj_normalized, new_length)
            rmse, mae = calculate_rmse_mae(pred_traj_interp, ans_traj_interp)

            # DFD measures curve shape similarity without interpolation
            dfd = calculate_discrete_frechet(pred_traj_normalized, ans_traj_normalized)

        return {
            "idx": idx,
            "question_id": question_id,
            "question": question,
            "raw_output": original_predicted,
            "pred_traj": pred_traj,
            "ans_traj": ans_traj,
            "pred_traj_normalized": pred_traj_normalized,
            "ans_traj_normalized": ans_traj_normalized,
            "rmse": min(float(rmse),1000) if rmse is not None else None,
            "mae": min(float(mae),1000) if mae is not None else None,
            "dfd": min(float(dfd),1000) if dfd is not None else None
        }

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info("\nEvaluating results...")
        all_results: List[Dict[str, Any]] = []
        for sample, raw_output in tqdm(zip(prepared_dataset, raw_outputs), total=len(prepared_dataset), desc="Evaluation"):
            all_results.append(self.process_raw_output(sample, raw_output))
        return all_results

    def normalize_score(self, distance: float, max_distance: float = 1000.0) -> float:
        """
        Normalize distance metric to a 0-100 score.
        Score = 100 * (distance / max_distance), clamped to [0, 100].
        Lower distance -> Lower score.

        Args:
            distance: The distance metric (RMSE, MAE, or DFD)
            max_distance: Maximum expected distance for normalization (default: 1000 for 0-1000 coordinate space)

        Returns:
            Normalized score in range [0, 100]
        """
        if distance is None:
            return None
        normalized = 100.0 * (distance / max_distance)
        return max(0.0, min(100.0, normalized))

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = len(results)

        rmses = []
        maes = []
        dfds = []
        valid_samples = 0

        for result in results:
            if result['rmse'] is not None and result['mae'] is not None:
                rmses.append(result['rmse'])
                maes.append(result['mae'])
                if result.get('dfd') is not None:
                    dfds.append(result['dfd'])
                valid_samples += 1

        avg_rmse = np.mean(rmses) if rmses else None
        avg_mae = np.mean(maes) if maes else None
        avg_dfd = np.mean(dfds) if dfds else None

        normalized_rmse_score = self.normalize_score(avg_rmse, max_distance=1000.0) if avg_rmse is not None else None
        normalized_mae_score = self.normalize_score(avg_mae, max_distance=1000.0) if avg_mae is not None else None
        normalized_dfd_score = self.normalize_score(avg_dfd, max_distance=1000.0) if avg_dfd is not None else None  

        logger.info("\nOverall Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Samples Processed: {total_samples}")
        logger.info(f"Valid Samples: {valid_samples}")
        if avg_rmse is not None:
            logger.info(f"Average RMSE: {avg_rmse:.4f} | Normalized Score: {normalized_rmse_score:.2f}/100")
        if avg_mae is not None:
            logger.info(f"Average MAE: {avg_mae:.4f} | Normalized Score: {normalized_mae_score:.2f}/100")
        if avg_dfd is not None:
            logger.info(f"Average Discrete Fréchet Distance: {avg_dfd:.4f} | Normalized Score: {normalized_dfd_score:.2f}/100")
        logger.info("=" * 60)

        return {
            "avg_rmse": float(avg_rmse) if avg_rmse is not None else None,
            "avg_mae": float(avg_mae) if avg_mae is not None else None,
            "avg_dfd": float(avg_dfd) if avg_dfd is not None else None,
            "normalized_rmse_score": float(normalized_rmse_score) if normalized_rmse_score is not None else None,
            "normalized_mae_score": float(normalized_mae_score) if normalized_mae_score is not None else None,
            "normalized_dfd_score": float(normalized_dfd_score) if normalized_dfd_score is not None else None,
            "valid_samples": valid_samples,
            "total_samples": total_samples,
        }

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]):
        os.makedirs("logs/results", exist_ok=True)
        result_file_name = f"logs/results/{self.task_name}_{self.model_name}.json"

        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {result_file_name}")
        return result_file_name
