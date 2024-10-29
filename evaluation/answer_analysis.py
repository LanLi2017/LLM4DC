import json
from typing import Union, List, Dict, Any
from difflib import SequenceMatcher
from math import isclose

# Define a utility to convert JSON strings to Python objects
def parse_input(answer: Union[str, float, List, Dict]) -> Any:
    if isinstance(answer, str):
        try:
            # Try to parse JSON strings into Python objects
            return json.loads(answer)
        except json.JSONDecodeError:
            return answer.lower().strip()  # Normalize strings for comparison
    elif isinstance(answer, float):
        return round(answer, 2)  # Round floats to two decimal places if needed
    return answer  # If already in desired format

# Calculate exact match accuracy
def accuracy_metric(gt: Any, pred: Any) -> float:
    return 1.0 if gt == pred else 0.0

# Calculate precision, recall, and F1 for lists (assuming items are unique)
def precision_recall_f1(gt: List, pred: List) -> Dict[str, float]:
    gt_set, pred_set = set(gt), set(pred)
    true_positives = len(gt_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gt_set) if gt_set else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1_score}

# Calculate semantic distance for string answers using sequence matching
def semantic_similarity(gt: str, pred: str) -> float:
    return SequenceMatcher(None, gt, pred).ratio()  # Returns a ratio between 0 and 1

# Evaluate an answer based on the ground truth
def evaluate_answer(gt: Any, pred: Any) -> Dict[str, float]:
    # Parse inputs
    gt, pred = parse_input(gt), parse_input(pred)
    
    # Initialize results
    results = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "semantic_similarity": 0}
    
    # Check type and apply appropriate metrics
    if isinstance(gt, float) and isinstance(pred, float):
        results["accuracy"] = 1.0 if isclose(gt, pred, rel_tol=1e-2) else 0.0  # Accuracy for floats with tolerance
    
    elif isinstance(gt, str) and isinstance(pred, str):
        results["accuracy"] = accuracy_metric(gt, pred)
        results["semantic_similarity"] = semantic_similarity(gt, pred)
    
    elif isinstance(gt, list) and isinstance(pred, list):
        metrics = precision_recall_f1(gt, pred)
        results.update(metrics)
        results["accuracy"] = accuracy_metric(gt, pred)
    
    elif isinstance(gt, dict) and isinstance(pred, dict):
        gt_keys, pred_keys = list(gt.keys()), list(pred.keys())
        results.update(precision_recall_f1(gt_keys, pred_keys))  # Precision/Recall on keys
        results["accuracy"] = accuracy_metric(gt, pred)
        # Check semantic similarity for each key-value pair
        similarity = [semantic_similarity(str(gt[k]), str(pred.get(k, ""))) for k in gt_keys]
        results["semantic_similarity"] = sum(similarity) / len(similarity) if similarity else 0

    return results