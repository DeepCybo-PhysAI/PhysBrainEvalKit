import base64
import os
import json
import logging
import random
import re
import warnings
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional
import numpy as np
import networkx as nx
from tqdm import tqdm

from .base import BaseDataset

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DEFAULT_VLABENCH_DATASET_PATH = os.environ.get("VLABENCH_DATASET_PATH", "datasets/VLABench/vlm_evaluation_v1.0")

def convert_to_str(value):
    return str(value) if value is not None else None

# predefined subtask patterns
SUBTASK_PATTERN = [
    ["pick", "place"],
    ["pick", "insert"],
    ["pick", "pour", "place"],
    ["pick", "pour"],
    ["pick", "pull"],
    ["pick", "lift"],
    ["pick", "push", "pull"],
    ["pick", "push", "place"],
    ["pick", "push"],
    ["pick", "open_door"],
    ["press"]
]

def build_input_with_embedded_prompt(input_instruction, with_CoT=False):  
    embedded_prompt = """
In Skill Lab, we provide a series of robot skills to support efficient automated operations for various tasks. Each skill has a specific call format, which includes the skill name and corresponding parameters.

The available skills are as follows:

1. **pick**: Used to grasp and pick up a target object.
   - Call format:
     ```json
     {
       "name": "pick",
       "params": {
         "target_entity_name": Target Number
       }
     }
     ```

2. **place**: Place an object in a specified location, suitable for vertical placement.
   - Call format:
     ```json
     {
       "name": "place",
       "params": {
         "target_container_name": Target Number
       }
     }
     ```

3. **press**: Press a specified location or button.
   - Call format:
     ```json
     {
       "name": "press",
       "params": {
         "target_entity_name": Target Number
       }
     }
     ```

4. **open_door**: Open a door.
   - Call format:
     ```json
     {
       "name": "open_door",
       "params": {
       }
     }
     ```

5. **insert**: Insert an item into a target location.
   - Call format:
     ```json
     {
       "name": "insert",
       "params": {
         "target_container_name": Target Number
       }
     }
     ```

6. **pull**: Pull the robotic arm horizontally.
   - Call format:
     ```json
     {
       "name": "pull",
       "params": {
       }
     }
     ```

7. **pour**: Pour a liquid or granular substance.
   - Call format:
     ```json
     {
       "name": "pour",
       "params": {
         "target_container_name": Target Number
       }
     }
     ```

8. **push**: Push a target object horizontally.
   - Call format:
     ```json
     {
       "name": "push",
       "params": {
         "target_container_name": Target Number
       }
     }
     ```

9. **lift**: Lift the robotic arm vertically.
   - Call format:
     ```json
     {
       "name": "lift",
       "params": {
       }
     }
     ```

These call formats ensure that each skill operation has clearly defined parameters, allowing the system to accurately execute the specified automated tasks.

---

You will receive the following input:

1. **Image input**: Two images
   - The first image shows four different perspectives of objects (without labels).
   - The second image contains the same four perspectives of objects, but each view is labeled with a number (representing each object's identifier).

2. **Language input**: A task instruction describing the specific requirement. Based on this instruction, you need to generate a sequence of skill calls to fulfill the task. Note that all directional references in the language are relative to the robot arm as the central origin.

---

### Task Requirements:

Based on the image and language inputs, generate a sequence of skill calls. Each skill call sequence should contain the skill name (extracted from the task instruction), the skill operation parameters (if the skill requires parameters), and the target entity or container number (obtained from the labeled image).

---

### Generation Steps:

1. **Extract Task Instruction**: Identify operation requirements (e.g., adjust, inspect, move) and the target object number or view from the language input.

2. **Combine with Image Information**: Using the labeled image, match the objects described in the task instruction with their corresponding identifiers to determine the target number.

3. **Generate Skill Call Sequence**: Based on the task instruction, use the extracted operation requirements as skill names and assign them the appropriate number information.

4. **Output Format**: Generate a skill call sequence in the following structure:
```json
[
    {
        "name": "Skill Name 1",
        "params": {
            "parameter": "value"
        }
    },
    {
        "name": "Skill Name 2",
        "params": {
            "parameter": "value"
        }
    }
]
```

Since the evaluation process will extract patterns from the skill sequence to build an operation graph, the output skill call sequence should ideally match one or more of the following patterns to ensure correctness:

Sub-skill sequence patterns:
- ["pick", "place"]
- ["pick", "insert"]
- ["pick", "pour", "place"]
- ["pick", "pour"]
- ["pick", "pull"]
- ["pick", "lift"]
- ["pick", "push", "pull"]
- ["pick", "push", "place"],
- ["pick", "push"],
- ["pick", "open_door"]
- ["press"]

For example, if the output only includes a single "pick" skill, it will be considered incorrect during the evaluation.
Please pay close attention to spatial information and ensure accurate pattern selection. For example, if the object needs to be picked up and displayed, use the pattern `["pick", "lift"]`. If the object needs to be pulled out and displayed, use the pattern `["pick", "pull"]`. If `["pick", "place"]` is used incorrectly in place of these patterns, it will be considered an error.
"""
    
    prepared_input = embedded_prompt + "\n\nLanguage instruction:\n" + input_instruction + "\n\nPlease generate the skill call sequence in JSON format as specified above."  
    if with_CoT:
        prepared_input = prepared_input + "\n\nPlease analyze the problem step by step and give the answer"
    return prepared_input

def find_subtasks(skill_sequence, patterns):
    """
    find all matching subtasks in skill_sequence based on patterns, return a list of subtasks in order
    """
    subtasks = []
    sequence = [skill["name"] for skill in skill_sequence]
    i = 0

    while i < len(sequence):
        matched = False
        for pattern in patterns:
            if sequence[i:i + len(pattern)] == pattern:
                # print(i, "to", i + len(pattern), "match", pattern)
                subtasks.append(skill_sequence[i:i + len(pattern)])
                i += len(pattern)
                matched = True
                break
        if not matched:
            i += 1
    return subtasks
   
def build_graph(skill_sequence, patterns, dependency="Sequential"):
    """
    Build graph structure based on dependency relationship, where each subtask is a node of the graph
    """
    G = nx.DiGraph()

    G.add_node("START", subtask="START", target_entity=None, target_container=None)
    
    subtasks = find_subtasks(skill_sequence, patterns)
    node_count = 1
    node_ids = []
    
    for subtask in subtasks:
        subtask_name = "-".join([skill["name"] for skill in subtask])
        
        target_entities = [skill["params"].get("target_entity_name") for skill in subtask]
        target_containers = [skill["params"].get("target_container_name") for skill in subtask]

        target_entities = [entity for entity in target_entities if entity is not None]
        target_containers = [container for container in target_containers if container is not None]
        target_entity = convert_to_str(target_entities[-1] if target_entities else None)
        target_container = convert_to_str(target_containers[-1] if target_containers else None)
        
        # add subtask node to the graph
        node_id = f"{subtask_name}_{node_count}"
        G.add_node(node_id, 
                   subtask=subtask_name,
                   target_entity=target_entity,
                   target_container=target_container)
        node_ids.append(node_id)
        node_count += 1
    
    if dependency == "Sequential":
        # linear dependency between subtasks
        previous_node = "START"
        for node_id in node_ids:
            G.add_edge(previous_node, node_id)
            previous_node = node_id
    elif dependency == "Seq-independent":
        # independent sequential dependency between subtasks
        for node_id in node_ids:
            G.add_edge("START", node_id)
    elif isinstance(dependency, dict):
        # custom dependency relationship
        for src, dests in dependency.items():
            for dest in dests:
                if src <= len(node_ids) and dest <= len(node_ids):
                    G.add_edge(node_ids[src - 1], node_ids[dest - 1])
                else:
                    raise ValueError("The dependency index is out of range.")
        
        for node_id in node_ids:
            if G.in_degree(node_id) == 0:
                G.add_edge("START", node_id)

    
    return G

def exact_match_percentage(graph1, graph2):
    """
    Compute the exact match percentage of graph2 relative to graph1, where the match requires the pattern, target_entity, target_container

    Params:
        graph1: expert subtask graph
        graph2: the graph to match
    """
    exact_match_count = 0
    total_nodes = len([node for node in graph1.nodes if node != "START"])

    # Obtain the topological order of graph1 and graph2.
    topo_order1 = list(nx.topological_sort(graph1))
    topo_order2 = list(nx.topological_sort(graph2))

    # Create the hierarchical structure of the nodes in graph1 and graph2
    layers1 = {}
    layers2 = {}
    
    for node in topo_order1:
        if node == "START":
            layers1[node] = 0
        else:
            layers1[node] = max(layers1[predecessor] for predecessor in graph1.predecessors(node)) + 1

    for node in topo_order2:
        if node == "START":
            layers2[node] = 0
        else:
            layers2[node] = max(layers2[predecessor] for predecessor in graph2.predecessors(node)) + 1

    # compare the nodes in each layer
    max_layer = max(layers1.values())
    for layer in range(1, max_layer + 1):
        nodes_layer1 = [node for node, lvl in layers1.items() if lvl == layer]
        nodes_layer2 = [node for node, lvl in layers2.items() if lvl == layer]
        
        for node1 in nodes_layer1:
            pattern1 = graph1.nodes[node1].get("subtask")
            target_entity1 = graph1.nodes[node1].get("target_entity")
            target_container1 = graph1.nodes[node1].get("target_container")

            match_found = False
            for node2 in nodes_layer2:
                pattern2 = graph2.nodes[node2].get("subtask")
                target_entity2 = graph2.nodes[node2].get("target_entity")
                target_container2 = graph2.nodes[node2].get("target_container")

                if pattern1 == pattern2 and target_entity1 == target_entity2 and target_container1 == target_container2:

                    predecessors1 = set(graph1.predecessors(node1))
                    predecessors2 = set(graph2.predecessors(node2))

                    if len(predecessors1) == len(predecessors2):
                        matched_predecessors = all(
                            any(
                                graph2.nodes[p2].get("subtask") == graph1.nodes[p1].get("subtask") and
                                graph2.nodes[p2].get("target_entity") == graph1.nodes[p1].get("target_entity") and
                                graph2.nodes[p2].get("target_container") == graph1.nodes[p1].get("target_container")
                                for p2 in predecessors2
                            )
                            for p1 in predecessors1
                        )

                        if matched_predecessors:
                            exact_match_count += 1
                            nodes_layer2.remove(node2)
                            match_found = True
                            break
                        
            if not match_found:
                continue

    exact_match_score = (exact_match_count / total_nodes) * 100

    return exact_match_score

def get_exact_match(skill_sequence1, skill_sequence2, dependency):
    graph1 = build_graph(skill_sequence1, SUBTASK_PATTERN, dependency=dependency)
    graph2 = build_graph(skill_sequence2, SUBTASK_PATTERN, dependency=dependency)
    exact_match_score = exact_match_percentage(graph1, graph2)
    return exact_match_score

def calculate_skill_and_entity_scores(sequence1, sequence2):
    """
    Calculate the skill match score and entity/container recognition match score of sequence2
    relative to sequence1, without considering the order and the correspondence between skills and entities.
    
    Parameters:
        sequence1: list - The standard skill sequence (includes skills and target objects).
        sequence2: list - The skill sequence to be compared.

    Returns:
        dict - A dictionary containing the skill match score and the percentage of correct entity/container recognition.
    """

    skills1 = [skill["name"] for skill in sequence1]
    skills2 = [skill["name"] for skill in sequence2]

    entities1 = []
    entities2 = []

    for skill in sequence1:
        if skill["params"].get("target_entity_name") is not None:
            entities1.append(("target_entity", convert_to_str(skill["params"].get("target_entity_name"))))
        if skill["params"].get("target_container_name") is not None:
            entities1.append(("target_container", convert_to_str(skill["params"].get("target_container_name"))))
    
    for skill in sequence2:
        if skill["params"].get("target_entity_name") is not None:
            entities2.append(("target_entity", convert_to_str(skill["params"].get("target_entity_name"))))
        if skill["params"].get("target_container_name") is not None:
            entities2.append(("target_container", convert_to_str(skill["params"].get("target_container_name"))))

    skills1_counter = Counter(skills1)
    skills2_counter = Counter(skills2)
    skill_match_count = sum((skills1_counter & skills2_counter).values())  
    
    entities1_counter = Counter(entities1)
    entities2_counter = Counter(entities2)
    entity_match_count = sum((entities1_counter & entities2_counter).values())

    total_skills = len(skills1)
    total_entities = len(entities1)
    skill_match_score = (skill_match_count / total_skills) * 100 if total_skills > 0 else 0
    entity_match_score = (entity_match_count / total_entities) * 100 if total_entities > 0 else 0

    return {
        "skill_match_score": skill_match_score,
        "entity_match_score": entity_match_score
    }

def calculate_skill_with_entity_scores(sequence1, sequence2):
    """
     Calculate the simultaneous skill and entity/container recognition match score of sequence2
     relative to sequence1, without considering the order.
    
     Parameters:
       sequence1: list - The standard skill sequence (includes skills and target objects).
       sequence2: list - The skill sequence to be compared.
    
     Returns:
       dict - A dictionary containing the percentage of correct skill and entity/container recognition.
    """
    skill_with_entity1 = [(skill["name"], convert_to_str(skill["params"].get("target_entity_name")), convert_to_str(skill["params"].get("target_container_name"))) for skill in sequence1]
    skill_with_entity2 = [(skill["name"], convert_to_str(skill["params"].get("target_entity_name")), convert_to_str(skill["params"].get("target_container_name"))) for skill in sequence2]

    skill_with_entity1_counter = Counter(skill_with_entity1)
    skill_with_entity2_counter = Counter(skill_with_entity2)
    skill_with_entity_match_count = sum((skill_with_entity1_counter & skill_with_entity2_counter).values())  # minimum intersection match count

    total_skills = len(skill_with_entity1)
    skill_with_entity_match_score = (skill_with_entity_match_count / total_skills) * 100 if total_skills > 0 else 0

    return {
        "skill_with_entity_match_score": skill_with_entity_match_score
    }

def get_final_score(standard_skill_sequence, model_skill_sequence, dependency):
    """
     Calculate the matching score of the model's skill sequence relative to the standard skill sequence.
    
     Parameters:
       standard_skill_sequence: list - The standard skill sequence.
       model_skill_sequence: list - The skill sequence output by the model.
       dependency: str - The type of dependency relation.
    
     Returns:
       dict - A dictionary containing the skill match score and the percentage of correct entity/container recognition.
    """

    skill_entity_scores = calculate_skill_and_entity_scores(standard_skill_sequence, model_skill_sequence)

    skill_with_entity_scores = calculate_skill_with_entity_scores(standard_skill_sequence, model_skill_sequence)

    exact_match_score = get_exact_match(standard_skill_sequence, model_skill_sequence, dependency)

    score_weight = {
        "skill_match_score": 0.4,
        "entity_match_score": 0.4,
        "skill_with_entity_match_score": 0.1,
        "exact_match_score": 0.1
    }
    total_score = (
        skill_entity_scores["skill_match_score"] * score_weight["skill_match_score"] +
        skill_entity_scores["entity_match_score"] * score_weight["entity_match_score"] +
        skill_with_entity_scores["skill_with_entity_match_score"] * score_weight["skill_with_entity_match_score"] +
        exact_match_score * score_weight["exact_match_score"]
    )
    return {
        "skill_match_score": skill_entity_scores["skill_match_score"],
        "entity_match_score": skill_entity_scores["entity_match_score"],
        "skill_with_entity_match_score": skill_with_entity_scores["skill_with_entity_match_score"],
        "exact_match_score": exact_match_score,
        "total_score": total_score
    }



class VLABenchDataset(BaseDataset):
    """Base dataset class defining the interface for all datasets"""
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        instruct_following: Optional[str] = None,
        task_name: str = "VLABench",
        model_name: Optional[str] = None,
        debug: bool = False,
        backbone: Optional[str] = None,
        thinking_model: bool = False
    ):
        super().__init__(instruct_following)
        self.dataset_path = dataset_path or os.environ.get("VLABENCH_DATASET_PATH", DEFAULT_VLABENCH_DATASET_PATH)
        self.subset = subset
        self.task_name = task_name
        self.model_name = model_name
        self.debug = debug
        self.thinking_model = thinking_model
        self.backbone = backbone
        self.seq_independent_task = ["cook_dishes", "texas_holdem"]
    
    def get_default_instruct(self) -> str:
        return ""
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        - dataset_path/
          - {dimension}/  (for example, M&T/, CommonSence/)
            - {task_name}/
              - example{num}/
                - input/
                  - input.png
                  - input_mask.png
                  - instruction.txt
                - output/
                  - operation_sequence.json
        """
        logger.info(f"Loading VLABench dataset from: {self.dataset_path}")
        if not self.dataset_path or not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"VLABench dataset directory not found: {self.dataset_path}. "
                "Download VLABench/vlm_evaluation_v1.0 and set VLABENCH_DATASET_PATH "
                "or pass --dataset_path."
            )
        
        samples = []
        
        if self.subset:
            if isinstance(self.subset, (list, tuple)):
                dimensions = list(self.subset)
            else:
                dimensions = [self.subset]
        else:
            dimensions = ["M&T", "CommenSence", "Semantic", "Spatial", "PhysicsLaw", "Complex"]
        
        for dimension in dimensions:
            dim_path = os.path.join(self.dataset_path, dimension)
            if not os.path.isdir(dim_path) and dimension in {"CommenSence", "CommonSence"}:
                for alias in ["CommenSence", "CommonSence"]:
                    alias_dim_path = os.path.join(self.dataset_path, alias)
                    if os.path.isdir(alias_dim_path):
                        dim_path = alias_dim_path
                        dimension = alias
                        break
            if not os.path.isdir(dim_path):
                logger.warning(f"Skipping missing VLABench dimension directory: {dim_path}")
                continue
            tasks = [t for t in os.listdir(dim_path) 
                    if os.path.isdir(os.path.join(dim_path, t))]
            
            for task_name in tasks:
                task_path = os.path.join(dim_path, task_name)
                examples = [e for e in os.listdir(task_path) 
                           if e.startswith("example") and os.path.isdir(os.path.join(task_path, e))]
                
                for example_dir in sorted(examples):
                    example_path = os.path.join(task_path, example_dir)
                    example_num = int(example_dir.replace("example", ""))

                    input_mask_path = os.path.join(example_path, "input", "input_mask.png")
                    input_pic_path = os.path.join(example_path, "input", "input.png")
                    input_instruction_path = os.path.join(example_path, "input", "instruction.txt")
                    output_path = os.path.join(example_path, "output", "operation_sequence.json")
                    
                    with open(input_instruction_path, 'r', encoding='utf-8') as f:
                        instruction = f.read().strip()
                    with open(output_path, 'r', encoding='utf-8') as f:
                        operation_sequence = json.load(f)
                    sample = {
                        "dimension": dimension,
                        "task_name": task_name,
                        "example_num": example_num,
                        "instruction": instruction,
                        "input_image_path": input_pic_path,
                        "input_mask_path": input_mask_path,
                        "operation_sequence": operation_sequence,
                        "is_seq_independent": task_name in self.seq_independent_task
                    }
                    samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from VLABench dataset")
        
        if self.debug:
            samples = samples[:20]
            logger.info(f"Debug mode: Using first {len(samples)} samples")
            
        return samples
    
    def prepare_dataset(self, dataset: List[Dict[str, Any]], start_index: int = 0) -> List[Dict[str, Any]]:
        prepared = []
        
        for idx, sample in enumerate(dataset):
            global_idx = start_index + idx
            instruction = sample["instruction"]
            image_path = sample["input_image_path"]
            mask_path = sample["input_mask_path"]
            question = build_input_with_embedded_prompt(
                input_instruction=instruction,
                with_CoT=self.thinking_model
            )
            # Keep paths lazy. HFInferenceEngine opens only the images assigned
            # to the current persistent worker after sample sharding.
            images = [image_path, mask_path]
            ground_truth = json.dumps(sample["operation_sequence"], ensure_ascii=False)
            
            prepared.append({
                "question": question,
                "answer": ground_truth,
                "image": images,
                "metadata": {
                    "id": global_idx,
                    "dimension": sample["dimension"],
                    "task_name": sample["task_name"],
                    "example_num": sample["example_num"],
                    "is_seq_independent": sample["is_seq_independent"],
                    "ground_truth": sample["operation_sequence"],
                }
            })
        
        return prepared
    
    def process_raw_output(self, prepared_sample: Dict[str, Any], raw_output_text: str) -> Dict[str, Any]:

        original_raw_output = raw_output_text
        
        if self.thinking_model:
            answer_match = re.search(r'<answer>(.*?)</answer>', str(raw_output_text),
                                   re.DOTALL | re.IGNORECASE)
            if answer_match:
                raw_output_text = answer_match.group(1).strip()
        
        metadata = prepared_sample["metadata"]
        task_name = metadata["task_name"]
        ground_truth = metadata["ground_truth"]
        is_seq_independent = metadata["is_seq_independent"]
        raw_text = "" if raw_output_text is None else str(raw_output_text).strip()

        model_output = None
        try:
            json_data = raw_text.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(json_data)
            model_output = {"skill_sequence": parsed, "origin_output": raw_text}
        except Exception:
            try:
                parsed = json.loads(raw_text)
                model_output = {"skill_sequence": parsed, "origin_output": raw_text}
            except Exception:
                model_output = {"format_error": "format_error", "origin_output": raw_text}
        
        if "format_error" in model_output:
            scores = {
                "skill_match_score": 0.0,
                "entity_match_score": 0.0,
                "skill_with_entity_match_score": 0.0,
                "exact_match_score": 0.0,
                "total_score": 0.0
            }
        else:
            
            model_skill_sequence = model_output["skill_sequence"]
            dependency = "Sequential" if not is_seq_independent else "Seq-independent"
            try:
                scores = get_final_score(
                    ground_truth["skill_sequence"], 
                    model_skill_sequence, 
                    dependency=dependency
                )
            except:
                scores = {
                    "skill_match_score": 0.0,
                    "entity_match_score": 0.0,
                    "skill_with_entity_match_score": 0.0,
                    "exact_match_score": 0.0,
                    "total_score": 0.0
                }
        
        result = {
            "id": metadata["id"],
            "dimension": metadata["dimension"],
            "task_name": task_name,
            "example_num": metadata["example_num"],
            "question": prepared_sample["question"],
            "raw_output": original_raw_output,
            "processed_output": model_output,
            "ground_truth": ground_truth,
            "scores": scores,
            "total_score": scores["total_score"],
            "is_seq_independent": is_seq_independent,
            "error": "format_error" if "format_error" in (model_output or {}) else None
        }
        
        return result
    
    def evaluate_results(
        self,
        prepared_dataset: List[Dict[str, Any]],
        raw_outputs: List[str]
    ) -> List[Dict[str, Any]]:
        logger.info("Evaluating VLABench predictions...")
        results = []
        if len(prepared_dataset) != len(raw_outputs):
            raise ValueError(f"prepared_dataset length ({len(prepared_dataset)}) != raw_outputs length ({len(raw_outputs)})")
        for sample, output in tqdm(zip(prepared_dataset, raw_outputs), 
                                  total=len(prepared_dataset), 
                                  desc="Evaluating VLABench"):

            result = self.process_raw_output(sample, output)
            results.append(result)
        
        return results
    
    def compute_statistics(self, results: List[Dict[str, Any]], log: bool = True) -> Dict[str, Any]:
        if not results:
            return {}
        
        dim_stats = defaultdict(lambda: {
            "total": 0,
            "skill_match_score_sum": 0.0,
            "entity_match_score_sum": 0.0,
            "skill_with_entity_match_score_sum": 0.0,
            "exact_match_score_sum": 0.0,
            "total_score_sum": 0.0,
            "tasks": defaultdict(lambda: {
                "total": 0,
                "total_score_sum": 0.0
            })
        })

        for result in results:
            dim = result["dimension"]
            task_name = result["task_name"]
            
            dim_stats[dim]["total"] += 1
            dim_stats[dim]["skill_match_score_sum"] += result["scores"]["skill_match_score"]
            dim_stats[dim]["entity_match_score_sum"] += result["scores"]["entity_match_score"]
            dim_stats[dim]["skill_with_entity_match_score_sum"] += result["scores"]["skill_with_entity_match_score"]
            dim_stats[dim]["exact_match_score_sum"] += result["scores"]["exact_match_score"]
            dim_stats[dim]["total_score_sum"] += result["total_score"]
            
            dim_stats[dim]["tasks"][task_name]["total"] += 1
            dim_stats[dim]["tasks"][task_name]["total_score_sum"] += result["total_score"]
        

        final_stats = {}
        overall_total = 0
        overall_total_score_sum = 0.0
        
        for dim, stats in dim_stats.items():
            total = stats["total"]
            if total > 0:
                final_stats[dim] = {
                    "num_samples": total,
                    "skill_match_score_avg": stats["skill_match_score_sum"] / total,
                    "entity_match_score_avg": stats["entity_match_score_sum"] / total,
                    "skill_with_entity_match_score_avg": stats["skill_with_entity_match_score_sum"] / total,
                    "exact_match_score_avg": stats["exact_match_score_sum"] / total,
                    "total_score_avg": stats["total_score_sum"] / total,
                    "tasks": {}
                }
                
                for task_name, task_stats in stats["tasks"].items():
                    if task_stats["total"] > 0:
                        final_stats[dim]["tasks"][task_name] = {
                            "num_samples": task_stats["total"],
                            "total_score_avg": task_stats["total_score_sum"] / task_stats["total"]
                        }
                
                overall_total += total
                overall_total_score_sum += stats["total_score_sum"]
        
        if overall_total > 0:
            final_stats["overall"] = {
                "num_samples": overall_total,
                "total_score_avg": overall_total_score_sum / overall_total
            }
        
        if log:
            logger.info("\n" + "="*60)
            logger.info("VLABench Evaluation Statistics")
            logger.info("="*60)
            logger.info(f"Total samples evaluated: {overall_total}")

            for dim, stats in final_stats.items():
                if dim == "overall":
                    continue
                logger.info(f"\nDimension: {dim} ({stats['num_samples']} samples)")
                logger.info(f"  Total Score: {stats['total_score_avg']:.2f}%")
                logger.info(f"  Skill Match: {stats['skill_match_score_avg']:.2f}%")
                logger.info(f"  Entity Match: {stats['entity_match_score_avg']:.2f}%")
                logger.info(f"  Skill+Entity Match: {stats['skill_with_entity_match_score_avg']:.2f}%")
                logger.info(f"  Exact Match: {stats['exact_match_score_avg']:.2f}%")

            if "overall" in final_stats:
                logger.info(f"\nOverall Score: {final_stats['overall']['total_score_avg']:.2f}%")

            logger.info("="*60)
        
        return {
            "overall_score": final_stats.get("overall", {}).get("total_score_avg", 0.0),
            "dimension_results": final_stats,
        }
    
    def save_results(self, results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> str:

        import os
        import json
        
        os.makedirs("logs/results", exist_ok=True)
        
        model_name = self.model_name or "unknown_model"
        filename = f"{self.task_name}_{model_name}_results.json"
        path = os.path.join("logs/results", filename)
        
        output_data = {
            "model": model_name,
            "backbone": self.backbone,
            "task_name": self.task_name,
            "subset": self.subset,
            "thinking_model": self.thinking_model,
            "statistics": statistics,
            "results": results
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Results saved to {path}")
        return path

    def save_partial_results(
        self,
        results: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        progress: Dict[str, Any],
    ) -> str:
        os.makedirs("logs/results", exist_ok=True)

        model_name = self.model_name or "unknown_model"
        filename = f"{self.task_name}_{model_name}_results.partial.json"
        path = os.path.join("logs/results", filename)
        tmp_path = f"{path}.tmp"

        output_data = {
            "is_partial": True,
            "model": model_name,
            "backbone": self.backbone,
            "task_name": self.task_name,
            "subset": self.subset,
            "thinking_model": self.thinking_model,
            "progress": progress,
            "statistics": statistics,
            "results": results,
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)

        logger.info(f"Partial results saved to {path}")
        return path
