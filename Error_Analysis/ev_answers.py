import importlib.util
import inspect
from typing import List
import requests
import json
import re
import difflib
from collections import Counter
# from spellchecker import SpellChecker
from datetime import datetime
import pandas as pd
import ast
import random
import logging 
from typing import Union, List, Dict, Any
from math import isclose
from difflib import SequenceMatcher
from bert_score import score
# from history_update_problem.call_or import export_rows
# from call_or import *

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
def calculate_answer_metrics(gt: Any, pred: Any) -> Dict[str, float]:
    # Parse inputs
    gt, pred = parse_input(gt), parse_input(pred)
    
    # Initialize results
    results = {"accuracy": 0, "semantic_similarity": 0, "precision": 0, "recall":0, "f1":0}
    
    # Check type and apply appropriate metrics
    if isinstance(gt, float) and isinstance(pred, float):
        results["accuracy"] = 1.0 if isclose(gt, pred, rel_tol=1e-2) else 0.0  # Accuracy for floats with tolerance
        precision, recall = results["accuracy"], results["accuracy"]
        results.update({"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0})
        # results['bertscore'] = score([f"{pred}"], [f"{gt}"], lang='en', verbose=True)
        results["semantic_similarity"] = semantic_similarity(str(gt), str(pred))
    elif isinstance(gt, int) and isinstance(pred, int):
        results["accuracy"] = 1.0 if isclose(gt, pred, rel_tol=1e-2) else 0.0  # Accuracy for floats with tolerance
        precision, recall = results["accuracy"], results["accuracy"]
        results.update({"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0})
        # results['bertscore'] = score([f"{pred}"], [f"{gt}"], lang='en', verbose=True)
        results["semantic_similarity"] = semantic_similarity(str(gt), str(pred))


    elif isinstance(gt, str) and isinstance(pred, str):
        results["accuracy"] = accuracy_metric(gt.lower(), pred.lower())
        precision, recall = results["accuracy"], results["accuracy"]
        results.update({"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0})
        # results['bertscore'] = score([pred], [gt], lang='en', verbose=True)
        results["semantic_similarity"] = semantic_similarity(gt, pred)

    
    elif isinstance(gt, list) and isinstance(pred, list):
        if type(gt[0]) == str:
            gt = [x.lower() for x in gt]
            if len(pred) > 0:
                if type(pred[0]) == str:
                    pred = [x.lower() for x in pred]
        metrics = precision_recall_f1(gt, pred)
        results.update(metrics)
        results["accuracy"] = accuracy_metric(gt, pred)
        # results['bertscore'] = score([f"{pred}"], [f"{gt}"], lang='en', verbose=True)
        similarity = semantic_similarity(f"{gt}", f"{pred}")
        results["semantic_similarity"] = similarity

    
    elif isinstance(gt, dict) and isinstance(pred, dict):
        gt = {key.lower(): value for key, value in gt.items()}
        pred = {key.lower(): value for key, value in pred.items()}
        gt_keys, pred_keys = list(gt.keys()), list(pred.keys())
        if type(gt[gt_keys[0]]) == dict:
            precision_recall_f1_results = []
            for k in gt_keys:
                if k in pred.keys():
                    pred_input = pred[k].values()
                else:
                    pred_input = []
                precision_recall_f1_results.append(precision_recall_f1(gt[k].values(), pred_input))
            precision = sum([x['precision'] for x in precision_recall_f1_results])/len([x['precision'] for x in precision_recall_f1_results])
            recall = sum([x['recall'] for x in precision_recall_f1_results])/len([x['recall'] for x in precision_recall_f1_results])
            f1 =sum([x['f1'] for x in precision_recall_f1_results])/len([x['f1'] for x in precision_recall_f1_results])
            results.update({'precision': precision, 'recall':recall, 'f1': f1})  # Precision/Recall on keys

        elif type(gt[gt_keys[0]]) == list:
            precision_recall_f1_results = []
            for k in gt_keys:
                if k in pred:
                    pred_input = pred[k]
                else: 
                    pred_input = []
                if type(gt[k][0]) ==str:
                    gt_input = [x.lower() for x in gt[k]]
                    if len(pred_input) > 0:
                        if type(pred_input[0]) == str:
                            pred_input = [x.lower() for x in pred_input]    
                else:
                    gt_input = gt[k]
                precision_recall_f1_results.append(precision_recall_f1(gt[k], pred_input)) 
            precision = sum([x['precision'] for x in precision_recall_f1_results])/len([x['precision'] for x in precision_recall_f1_results])
            recall = sum([x['recall'] for x in precision_recall_f1_results])/len([x['recall'] for x in precision_recall_f1_results])
            f1 =sum([x['f1'] for x in precision_recall_f1_results])/len([x['f1'] for x in precision_recall_f1_results])
            results.update({'precision': precision, 'recall':recall, 'f1': f1})  # Precision/Recall on keys
        

        results["accuracy"] = accuracy_metric(gt, pred)
        # Check semantic similarity for each key-value pair
        similarity = [semantic_similarity(str(gt[k]), str(pred.get(k, ""))) for k in gt_keys]
        results["semantic_similarity"] = sum(similarity) / len(similarity) if similarity else 0
        
        # results['bertscore'] = score([f"{pred}"], [f"{gt}"], lang='en', verbose=True)
    p, r, f1 = score([f"{pred}"], [f"{gt}"], lang='en', verbose=True)
    results.update({'bertscore_p': p.detach().cpu().tolist(),
    'bertscore_r': r.detach().cpu().tolist(),
    'bertscore_f1': f1.detach().cpu().tolist()})
    # print(results)
    return results

def load_answer_dataset(datafile_path):
    """
    load json file, each line is a json dictionary

    datafile_path: str
    return: data:  list_of_dictionary
    """
    data = []
    with open(datafile_path, 'r') as f:
        for l in f:
            data.append(json.loads(l))
    return data

def eval_answers(answer_gt_path, answer_preds_llama):
    answer_gt = load_answer_dataset(answer_gt_path)
    answer_gt = pd.DataFrame(answer_gt)
    answer_preds_llama = load_answer_dataset(answer_preds_llama)
    answer_preds_llama = pd.DataFrame(answer_preds_llama)
    # answer_compare = answer_gt.merge(answer_preds_llama[['pp_id', 'answer']], on='pp_id', how='left', suffixes=('_gt', '_preds'))
    answer_compare = answer_gt.merge(
        answer_preds_llama[['pp_id', 'answer']], 
        on='pp_id', 
        how='inner',  # <-- keeps only matching pp_id rows
        suffixes=('_gt', '_preds')
    )
    results = []
    for i, row in answer_compare.iterrows():
        
        gt = row['answer_gt']
        preds = row['answer_preds']
        
        single_result = calculate_answer_metrics(gt, preds)
        single_result['pp_id'] = row['pp_id']
        results.append(single_result)
        # break
    return pd.DataFrame(results)

 

# 'dirty', 
models = ['llama3.1',  'mistral', 'gemma2','gemma2base']
for model in models[:2]: #['mistral', 'gemma2', 'llama3.1']:
    print(model)
    answer_gt_path = '/projects/bces/lanl2/LLM4DC/evaluation/answer_1-154_gt.json'
    answer_preds = f'/projects/bces/lanl2/LLM4DC/Error_Analysis/log_answer_{model}.json'

    eval_answer_results = eval_answers(answer_gt_path, answer_preds)
    eval_answer_results['bertscore_p'] = eval_answer_results['bertscore_p'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else None)
    eval_answer_results['bertscore_r'] = eval_answer_results['bertscore_r'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else None)
    eval_answer_results['bertscore_f1'] = eval_answer_results['bertscore_f1'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else None)
    eval_answer_results.to_csv(f'/projects/bces/lanl2/LLM4DC/Error_Analysis/log_{model}_answer_result.csv')
