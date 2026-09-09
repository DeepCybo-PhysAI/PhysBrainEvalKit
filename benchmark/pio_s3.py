import io
import json
import logging
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

from core.point_utils import omni_decode_points

from .base import BaseDataset

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.5"
DEFAULT_JUDGE_MAX_WORKERS = 8
DEFAULT_JUDGE_MAX_CONCURRENT_REQUESTS = 8

# ==============================================================================
#  Trajectory Evaluation Prompt (Modified for 1-10 Scale & Gripper Focus)
# ==============================================================================
TRAJECTORY_EVAL_PROMPT = """### Role
You are an expert evaluator in robotic manipulation and visual reasoning. Your job is to assess the quality of predicted trajectories based on task instructions and visual inputs.

You are given:
- A task instruction describing an object manipulation task.
- An image showing a predicted trajectory.

**Note:**
- In the image, the **red circle** indicates the start point, and the **blue diamond** indicates the end point.
- The trajectory shifts from red to blue, indicating the path over time.
- You should **evaluate the predicted trajectory as the motion path of the robot's gripper (end-effector)** needed to perform the task.

**Evaluation Criteria (listed in order of importance):**

1. **Task Alignment and Success (most important)** - Does the gripper trajectory clearly and accurately fulfill the task instruction?  
   - **The trajectory must start at the correct location (e.g., current gripper position) and move towards the target object/location effectively.** - Large deviations in the starting or ending point (e.g., moving to the wrong object, wrong destination, or stopping short of the goal) should result in a low score.  
   - If the task is not accomplished (due to incorrect goal interpretation or spatial execution), the score should be low regardless of other qualities.

2. **Feasibility** - Is the movement physically plausible, smooth, and continuous?  
   - Are there any unrealistic discontinuities, sharp turns, or impossible transitions?  
   - Even if the movement is feasible, it should not receive a high score if the task is not completed.

3. **Obstacle Avoidance / Safety** - Does the trajectory reasonably avoid collisions with surrounding objects (other than the target object)?  
   - Minor risks may be tolerated if the task is completed successfully, but major or clear collisions should reduce the score.

**Scoring Guideline:**
- If the task is **not accomplished**, or if the start or end point is significantly incorrect, the score should typically be **5 or below**.
- If the task is completed but the trajectory has issues (e.g., roughness, minor risk of collision), a score in the **6–8** range is appropriate.
- A **score of 9–10** should be given only when the trajectory clearly completes the task, with good start/end accuracy, smooth motion, and reasonable safety.

Based on these criteria, provide a single overall score from 1 (very poor) to 10 (excellent), reflecting how well the task is accomplished.

Respond strictly in the following format:
Score: <1-10>  
Explanation: <brief justification>

The task instruction is:  
{task_instruction}

Please give your response."""


# ==============================================================================
#  Trajectory Visualization
# ==============================================================================
def visualize_trajectory(image_path: str, save_path: str, points: List[List[float]]) -> None:
    """
    Draws the trajectory on the image with task description at the top and saves it.
    Uses absolute pixel coordinates.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    h, w = image.shape[:2]

    # Draw trajectory
    if len(points) < 2:
        logger.warning(f"Trajectory has less than 2 points ({len(points)}), skipping drawing")
        cv2.imwrite(save_path, image)
        return

    num_points = len(points)

    # Draw lines connecting points with color gradient from red to blue
    for i in range(num_points - 1):
        # Color gradient: red (start) -> blue (end)
        ratio = i / (num_points - 1)
        b = int(255 * ratio)
        r = int(255 * (1 - ratio))
        color = (b, 0, r)  # BGR format

        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))

        cv2.line(image, pt1, pt2, color, thickness=3)

    # 1. Draw start point (Red Circle)
    start_pt = (int(points[0][0]), int(points[0][1]))
    cv2.circle(image, start_pt, 8, (0, 0, 255), -1)

    # 2. Draw end point (Blue Diamond)
    end_pt = (int(points[-1][0]), int(points[-1][1]))
    # Drawing a diamond using OpenCV marker
    cv2.drawMarker(image, end_pt, (255, 0, 0), markerType=cv2.MARKER_DIAMOND, 
                   markerSize=15, thickness=3)

    # Save output
    cv2.imwrite(save_path, image)

# ==============================================================================
#  Judge Evaluation
# ==============================================================================
class TrajectoryScore(BaseModel):
    """Expected response format from the judge model."""
    explanation: str
    score: int


def parse_trajectory_score(output: str) -> TrajectoryScore:
    """
    Parse the judge output to extract score (1-10) and explanation.
    """
    logging.info(f"Raw GPT output: {output}")

    # 1. Clean up markdown code blocks if present
    clean_output = re.sub(r'```python|```', '', output, flags=re.IGNORECASE).strip()

    # 2. Try to parse Python class-style definition
    class_pattern = re.compile(
        r'explanation\s*:\s*str\s*=\s*["\'](.*?)["\']\s*\n.*?'
        r'score\s*:\s*int\s*=\s*(\d+)',
        re.DOTALL | re.IGNORECASE
    )
    
    match = class_pattern.search(clean_output)
    if match:
        explanation = match.group(1).strip()
        score = int(match.group(2))
        return TrajectoryScore(explanation=explanation, score=score)

    # 3. Fallback: Simple Regex for "Score: <number>" pattern from the prompt
    score_match = re.search(r'Score\s*:\s*(\d+)', clean_output, re.IGNORECASE)
    explanation_match = re.search(r'Explanation\s*:\s*(.*)', clean_output, re.IGNORECASE | re.DOTALL)
    
    if score_match:
        score = int(score_match.group(1))
        explanation = explanation_match.group(1).strip() if explanation_match else clean_output
        return TrajectoryScore(explanation=explanation, score=score)

    # 4. Fallback: Brute-force extraction
    score_match = re.search(r'score.*=\s*(\d+)', clean_output, re.IGNORECASE)
    if not score_match:
         score_match = re.search(r'(\d+)/10', clean_output, re.IGNORECASE) # Catch "8/10"
    
    score = int(score_match.group(1)) if score_match else 1
    
    return TrajectoryScore(explanation=clean_output[:200], score=score)


def evaluate_trajectory_with_gpt4o(
    image_path: str,
    client: OpenAI,
    task_description: str,
    openai_model: str = DEFAULT_JUDGE_MODEL,
    openai_seed: int = 1234,
    openai_max_tokens: int = 512,
    openai_temperature: float = 0.2,
    verbose: bool = False,
) -> TrajectoryScore:
    """
    Evaluate a trajectory visualization using a vision-capable judge on a 1-10 scale.
    """
    if client is None:
        logger.error("OpenAI client is None.")
        return TrajectoryScore(explanation="OpenAI client not available", score=1)

    try:
        # Load and encode image
        with open(image_path, "rb") as img_file:
            import base64
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Format prompt with the specific task
        formatted_prompt = TRAJECTORY_EVAL_PROMPT.format(task_instruction=task_description)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": formatted_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=openai_temperature,
            max_tokens=openai_max_tokens,
            seed=openai_seed
        )

        raw_response = extract_openai_response_text(response)
        if looks_like_html(raw_response):
            raise ValueError(
                "Judge endpoint returned HTML instead of a chat completion. "
                "Check that --openai_base_url points to an OpenAI-compatible API path, usually ending with /v1."
            )
        if verbose:
            print(f"Judge Output: {raw_response}")

        score_result = parse_trajectory_score(raw_response)

        # Ensure score is in valid range [1, 10]
        score_result.score = max(1, min(10, score_result.score))

        return score_result

    except Exception as e:
        logger.error(f"Error in trajectory evaluation: {str(e)}")
        return TrajectoryScore(explanation=f"Error: {str(e)}", score=1)


def extract_openai_response_text(response: Any) -> str:
    """Extract chat completion content from SDK objects, dicts, or proxy strings."""
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return response
        return extract_openai_response_text(parsed)

    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is not None:
                return str(content)
        if "content" in response:
            return str(response["content"])
        return json.dumps(response, ensure_ascii=False)

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return extract_openai_response_text(model_dump())

    return str(response)


def looks_like_html(text: str) -> bool:
    prefix = text.lstrip()[:200].lower()
    return prefix.startswith("<!doctype html") or prefix.startswith("<html")


# ==============================================================================
#  PIO-S3 Dataset Class
# ==============================================================================
class PIOS3Dataset(BaseDataset):
    """
    PIO-S3 Dataset Class for trajectory evaluation
    Evaluation: judge-model based trajectory scoring (1-10 scale)
    """

    def __init__(
        self,
        dataset_name: str = "IffYuan/PIO-S3",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "PIO-S3-Verified",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        openai_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_seed: int = 1234,
        judge_max_tokens: int = 512,
        judge_temperature: float = 0.2,
        judge_max_workers: int = DEFAULT_JUDGE_MAX_WORKERS,
        judge_max_concurrent_requests: int = DEFAULT_JUDGE_MAX_CONCURRENT_REQUESTS,
        thinking_model: bool = False,
        save_visualizations: bool = True,
        visualization_dir: str = "logs/pio_s3_visualizations"
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
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.judge_model = judge_model
        self.judge_seed = judge_seed
        self.judge_max_tokens = judge_max_tokens
        self.judge_temperature = judge_temperature
        self.judge_max_workers = judge_max_workers
        self.judge_semaphore = threading.Semaphore(judge_max_concurrent_requests)

        self.openai_key = openai_key or os.environ.get("PIO_S3_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        self.openai_base_url = openai_base_url or os.environ.get("PIO_S3_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        self.client: Optional[OpenAI] = None

        if not self.openai_key:
            raise ValueError(
                "No OpenAI-compatible judge API key provided. Set PIO_S3_OPENAI_KEY, "
                "OPENAI_API_KEY, API_KEY, or pass --openai_key."
            )

        try:
            self.client = OpenAI(api_key=self.openai_key, base_url=self.openai_base_url)
            logger.info("OpenAI client initialized for PIO-S3 evaluation.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

        os.makedirs(self.visualization_dir, exist_ok=True)

    def get_default_instruct(self) -> str:
        # ... (Same as original)
        if self.backbone == "gemma4":
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]. All coordinates must be integers normalized to the range 0 to 1000."""
        elif self.backbone in ["qwen3_5", "qwen3", "gemini-2.5"]:
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "qwen2.5" or self.backbone == "qwen2_5" or self.backbone=='mimo':
            return """The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}]."""
        elif self.backbone == "molmo":
            return """Provide one point with the format in xml format. For example: <point x="63.5" y="44.5" alt="Mt Rainier">Mt Rainier</point>."""
        elif self.backbone == "gpt" or self.backbone == "pelican" or self.backbone =='internvl' or self.backbone =='magma':
            return "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    def load_dataset(self) -> Any:
        logger.info(f"Loading dataset: {self.dataset_name}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        prompt_template = (
            "You are currently a robot performing robotic manipulation tasks. The task instruction is: {task}. Use 2D points to trace the movement trajectory of the robot arm end-effector (not the manipulated object) as it moves to complete the task. You must provide the points in the order of the trajectory, and the number of points must be 8.\n"
            "You will also be offered the starting points of the current end-effector (annotated in red marker in the given image)\n"
        )       
        for idx, sample in enumerate(dataset):
            image = sample['image'].convert("RGB")
            if image is None:
                continue
            width, height = image.size
            raw_task = sample.get("question", "")
            formatted_question = prompt_template.format(task=raw_task)
            
            prepared.append({
                "question": formatted_question + self.instruct_following,
                "image": image,
                "metadata": {
                    "idx": idx,
                    "task": raw_task,
                    "sample_id": sample.get("id", f"sample_{idx}"),
                    'width': width,
                    'height': height,
                },
            })
        return prepared

    def process_raw_output(
        self,
        prepared_sample: Dict[str, Any],
        raw_output_text: str
    ) -> Dict[str, Any]:
        meta = prepared_sample["metadata"]
        question = prepared_sample["question"]
        sample_id = meta["sample_id"]
        width = meta['width']
        task = meta["task"]
        height = meta['height']

        # Remove thinking tags logic (same as original)
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()

        results = {
            "idx": meta["idx"],
            "sample_id": sample_id,
            "question": question,
            "raw_output": raw_output_text,
            "score": 0.0,  # 0-100 percentage
            "raw_score": 1,  # 1-10 scale
            "explanation": "",
            "num_points": 0
        }

        try:
            points_ = omni_decode_points(raw_output_text)
            points = []
            results["num_points"] = len(points_)
            
            if len(points_) > 0:
                if self.backbone in ["qwen2.5", "qwen2_5", 'mimo']:
                    points = points_
                elif self.backbone in ["gemma4", "qwen3_5", "qwen3", "gemini-2.5"]:
                    for point in points_:
                        x_abs = int(round(point[0] / 1000.0 * width))
                        y_abs = int(round(point[1] / 1000.0 * height))
                        points.append([x_abs, y_abs])
                elif self.backbone in ['molmo']:
                    for point in points_:
                        x_abs = int(round(point[0] / 100.0 * width))
                        y_abs = int(round(point[1] / 100.0 * height))
                        points.append([x_abs, y_abs])    
                elif self.backbone in ['gpt','pelican','internvl','magma']:
                    for point in points_:
                        x_abs = int(round(point[0] * width))
                        y_abs = int(round(point[1] * height))
                        points.append([x_abs, y_abs])
                else:
                    raise ValueError(f"Unsupported backbone: {self.backbone}")
            
            if len(points) < 2:
                logger.warning(f"Sample {sample_id}: Insufficient points ({len(points)})")
                results["explanation"] = "Insufficient trajectory points (need at least 2)"
                return results

            # Save image temporarily
            image = prepared_sample["image"]
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))

            temp_image_path = os.path.join(self.visualization_dir, f"temp_{sample_id}.png")
            vis_image_path = os.path.join(self.visualization_dir, f"{sample_id}_trajectory.png")

            image.save(temp_image_path)

            # Create visualization
            visualize_trajectory(
                temp_image_path,
                vis_image_path,
                points
            )

            # Evaluate with the configured judge model.
            with self.judge_semaphore:
                eval_result = evaluate_trajectory_with_gpt4o(
                    vis_image_path,
                    client=self.client,
                    task_description=task,
                    openai_model=self.judge_model,
                    openai_seed=self.judge_seed,
                    openai_max_tokens=self.judge_max_tokens,
                    openai_temperature=self.judge_temperature,
                    verbose=self.debug,
                )

            # Convert score from 1-10 to percentage (0-100)
            # Formula: ((score - 1) / 9) * 100
            # 1 -> 0%, 5.5 -> 50%, 10 -> 100%
            percentage_score = ((eval_result.score - 1) / 9) * 100

            results["score"] = percentage_score
            results["raw_score"] = eval_result.score
            results["explanation"] = eval_result.explanation

            # Clean up
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if not self.save_visualizations and os.path.exists(vis_image_path):
                os.remove(vis_image_path)

        except Exception as e:
            logger.error(f"Processing failed for sample {sample_id}: {e}")
            results["error"] = str(e)

        return results

    def evaluate_results(self, prepared_dataset, raw_outputs):
        logger.info(f"Evaluating PIO-S3-Verified trajectories with {self.judge_model}...")
        results = []
        with ThreadPoolExecutor(max_workers=self.judge_max_workers) as executor:
            futures = [
                executor.submit(self.process_raw_output, sample, output)
                for sample, output in zip(prepared_dataset, raw_outputs)
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Scoring trajectories"):
                results.append(f.result())
        return results

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics for 1-10 scoring.
        """
        if not results:
            return {}

        scores = [r.get("score", 0.0) for r in results]
        raw_scores = [r.get("raw_score", 1) for r in results]
        num_points_list = [r.get("num_points", 0) for r in results]

        overall_score = sum(scores) / len(scores) if scores else 0.0
        avg_raw_score = sum(raw_scores) / len(raw_scores) if raw_scores else 1.0
        avg_num_points = sum(num_points_list) / len(num_points_list) if num_points_list else 0.0

        # Count score distribution (1-10)
        score_dist = {i: raw_scores.count(i) for i in range(1, 11)}

        stats = {
            "overall_score": overall_score,  # 0-100 percentage
            "avg_raw_score": avg_raw_score,  # 1-10 scale
            "avg_num_points": avg_num_points,
            "score_distribution": score_dist,
            "total_samples": len(results)
        }

        logger.info(f"Overall Score: {overall_score:.2f}%")
        logger.info(f"Average Raw Score: {avg_raw_score:.2f}/10")
        logger.info(f"Average Trajectory Points: {avg_num_points:.1f}")

        return {"overall_score": overall_score, "detailed_stats": stats}

    # save_results ... (Same as original)
    def save_results(self, results, statistics):
        os.makedirs("logs/results", exist_ok=True)
        filename = f"{self.task_name}_{self.model_name}_results.json"
        path = os.path.join("logs/results", filename)
        output_data = {
            "model": self.model_name,
            "task": self.task_name,
            "statistics": statistics,
            "results": results
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {path}")
        return path
