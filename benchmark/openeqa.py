import json
import logging
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from .base import BaseDataset

logger = logging.getLogger(__name__)

# ==============================================================================
# Parallel Configuration
# ==============================================================================
DEFAULT_JUDGE_MODEL = "gpt-5.5"
DEFAULT_JUDGE_MAX_WORKERS = 8
DEFAULT_JUDGE_MAX_CONCURRENT_REQUESTS = 8
DEFAULT_JUDGE_MAX_RETRIES = 5

# ============================================================================== 
#  Prompt Templates
# ==============================================================================
PROMPT_MMBENCH = """You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}
"""

PROMPT_MMBENCH_EXTRA = """You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.

Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}
"""

# ============================================================================== 
# LLM Match Scoring Logic with Semaphore
# ==============================================================================
def parse_score(output: str, tag: str = "Your mark:") -> int:
    if not output:
        raise ValueError("Empty output string")

    if tag in output:
        output = output.split(tag, 1)[1]
    match = re.search(r'\d+', output)
    if match:
        return int(match.group(0))
    raise ValueError(f"Cannot parse score from output: {output}")

def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    client: Optional[OpenAI] = None,
    openai_model: str = DEFAULT_JUDGE_MODEL,
    openai_seed: int = 1234,
    openai_max_tokens: int = 2048,
    openai_temperature: float = 0.2,
    max_retries: int = DEFAULT_JUDGE_MAX_RETRIES,
    verbose: bool = False,
) -> int:
    """Calculates the score using OpenAI API with retry logic."""
    if prediction is None:
        return 0

    if client is None:
        logger.error("OpenAI client is None.")
        return 0

    if extra_answers and len(extra_answers) > 0:
        formatted_prompt = PROMPT_MMBENCH_EXTRA.format(
            question=question,
            answer=answer,
            extra_answers=str(extra_answers),
            prediction=prediction
        )
    else:
        formatted_prompt = PROMPT_MMBENCH.format(
            question=question,
            answer=answer,
            prediction=prediction
        )

    messages = [{"role": "user", "content": formatted_prompt}]

    for attempt in range(max_retries):
        try:
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
            if not raw_response:
                logger.warning(f"Empty response, retry {attempt + 1}/{max_retries}")
                continue

            if verbose:
                print(f"Judge Input: {formatted_prompt}")
                print(f"Judge Output: {raw_response}")

            score = parse_score(raw_response)
            return min(max(score, 1), 5)

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All retries failed: {e}")
                return 0

    return 0


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
# OpenEQADataset Class
# ==============================================================================
class OpenEQADataset(BaseDataset):
    """
    Open-EQA Dataset Class

    Data Source: 'IffYuan/open-eqa' (Parquet/Arrow format with bytes)
    Evaluation: Custom LLM-Match Score using embedded hardcoded prompts.
    """

    def __init__(
        self,
        dataset_name: str = "IffYuan/open-eqa",
        subset: Optional[str] = None,
        split: str = "train",
        instruct_following: Optional[str] = None,
        task_name: str = "OpenEQA",
        model_name: Optional[str] = None,
        backbone: Optional[str] = None,
        debug: bool = False,
        openai_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_seed: int = 1234,
        judge_max_tokens: int = 2048,
        judge_temperature: float = 0.2,
        judge_max_workers: int = DEFAULT_JUDGE_MAX_WORKERS,
        judge_max_concurrent_requests: int = DEFAULT_JUDGE_MAX_CONCURRENT_REQUESTS,
        judge_max_retries: int = DEFAULT_JUDGE_MAX_RETRIES,
        thinking_model: bool = False
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
        self.judge_model = judge_model
        self.judge_seed = judge_seed
        self.judge_max_tokens = judge_max_tokens
        self.judge_temperature = judge_temperature
        self.judge_max_workers = judge_max_workers
        self.judge_max_retries = judge_max_retries
        self.judge_semaphore = threading.Semaphore(judge_max_concurrent_requests)
        
        self.openai_key = openai_key or os.environ.get("OPENEQA_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        self.openai_base_url = openai_base_url or os.environ.get("OPENEQA_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        self.client: Optional[OpenAI] = None
        
        if not self.openai_key:
            raise ValueError(
                "No OpenAI-compatible judge API key provided. Set OPENEQA_OPENAI_KEY, "
                "OPENAI_API_KEY, API_KEY, or pass --openai_key."
            )

        try:
            self.client = OpenAI(api_key=self.openai_key, base_url=self.openai_base_url)
            logger.info("OpenAI client initialized once in OpenEQADataset.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def get_default_instruct(self) -> str:
        return ""

    def load_dataset(self) -> Any:
        logger.info(f"Dataset: {self.dataset_name}, Subset: {self.subset}, Split: {self.split}")
        dataset = load_dataset(self.dataset_name, split=self.split)
        logger.info(f"Dataset loaded. Number of samples: {len(dataset)}")
        return dataset

    def prepare_dataset(self, dataset: Any, start_index: int = 0) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for idx, sample in enumerate(dataset):
            frames_bytes = sample["videos"]
            extra_answers = sample["extra_answers"]
            prepared.append({
                "question": sample["question"],
                "answer": str(sample["answer"]), 
                "video": [frames_bytes],     
                "metadata": {
                    "idx": start_index + idx,
                    "question_id": sample.get("question_id"),
                    "category": sample.get("category"),
                    "episode_history": sample.get("episode_history"),
                    "extra_answers": extra_answers
                },
            })
        return prepared

    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:
        meta = prepared_sample["metadata"]
        question = prepared_sample["question"]
        gt_answer = prepared_sample["answer"]
        extra_answers = meta["extra_answers"]
        
        if self.thinking_model:
            raw_output_text = re.sub(r'^(.*?</think>|<think>.*?</think>)', '', raw_output_text, flags=re.DOTALL | re.IGNORECASE).strip()
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text), re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()
                
        if raw_output_text.endswith('.'):
            raw_output_text = raw_output_text[:-1]
        raw_output_text = raw_output_text.replace('<', '').replace('>', '').strip()

        results = {
            "idx": meta["idx"],
            "question_id": meta["question_id"],
            "category": meta["category"],
            "question": question,
            "prediction": raw_output_text,
            "ground_truth": gt_answer,
            "score": 0.0,
            "raw_llm_score": 0
        }
        if not raw_output_text:
            results["error"] = "empty_model_prediction"
            logger.error(
                f"Skipping judge for ID {meta['question_id']}: empty model prediction"
            )
            return results
        try:
            with self.judge_semaphore:
                score_raw = get_llm_match_score(
                    question=question,
                    answer=gt_answer,
                    prediction=raw_output_text,
                    extra_answers=extra_answers,
                    client=self.client,
                    openai_model=self.judge_model,
                    openai_seed=self.judge_seed,
                    openai_max_tokens=self.judge_max_tokens,
                    openai_temperature=self.judge_temperature,
                    max_retries=self.judge_max_retries,
                    verbose=self.debug
                )
            # Normalize score from 1-5 scale to 0-1 scale: (score - 1) / 4
            llm_score = max((score_raw - 1) / 4, 0)
            results["score"] = llm_score
            results["raw_llm_score"] = score_raw
        except Exception as e:
            logger.error(f"Scoring failed for ID {meta['question_id']}: {e}")
            results["score"] = 0.0
            results["error"] = str(e)
        return results

    def evaluate_results(self, prepared_dataset: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Evaluating OpenEQA predictions with {self.judge_model}...")
        results = []

        with ThreadPoolExecutor(max_workers=self.judge_max_workers) as executor:
            futures = [executor.submit(self.process_raw_output, sample, output)
                       for sample, output in zip(prepared_dataset, raw_outputs)]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
                results.append(f.result())
        return results

    def compute_statistics(self, results: List[Dict[str, Any]], log: bool = True) -> Dict[str, Any]:
        if not results:
            return {}
        category_scores = defaultdict(list)
        for r in results:
            category_scores[r.get("category", "unknown")].append(r.get("score", 0.0))
        final_stats = {}
        total_scores = []
        for cat, scores in category_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            final_stats[f"{cat}_score"] = avg_score
            total_scores.extend(scores)
        overall = sum(total_scores) / len(total_scores) if total_scores else 0.0
        final_stats["overall_score"] = overall
        if log:
            logger.info(f"Overall Score: {overall:.4f}")
        return {"overall_score": overall, "category_results": final_stats}

    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:
        os.makedirs("logs/results", exist_ok=True)
        filename = f"{self.task_name}_{self.model_name}_results.json"
        path = os.path.join("logs/results", filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "statistics": statistics}, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {path}")
        return path

    def save_partial_results(
        self,
        results: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        progress: Dict[str, Any],
    ) -> str:
        os.makedirs("logs/results", exist_ok=True)
        filename = f"{self.task_name}_{self.model_name}_results.partial.json"
        path = os.path.join("logs/results", filename)
        tmp_path = f"{path}.tmp"
        payload = {
            "is_partial": True,
            "progress": progress,
            "results": results,
            "statistics": statistics,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)
        logger.info(f"Partial results saved to {path}")
        return path
