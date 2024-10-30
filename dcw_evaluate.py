# LLM-based history update solution
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
# from history_update_problem.call_or import export_rows
from call_or import *

from evaluation import *


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

def eval_answers():
    answer_gt = load_answer_dataset(answer_gt_path)
    answer_gt = pd.DataFrame(answer_gt)
    answer_preds_llama = load_answer_dataset(answer_preds_llama)
    answer_preds_llama = pd.DataFrame(answer_preds_llama)
    answer_compare = answer_gt.merge(answer_preds_llama[['pp_id', 'answer']], on='pp_id', how='left', suffixes=('_gt', '_preds'))

    results = []
    for i, row in answer_compare.iterrows()::
        gt = row['answer_gt']
        preds = row['answer_preds']
        single_result = calculate_answer_metrics(gt, preds)
        results.append(single_results)


    pass 

def eval_workflows():
    pass

def eval_dataset():
    pass

if __name__ == '__main__':
    models = ['llama3.1']
    answer_gt_path = '/projects/bces/lanl2/LLM4DC/evaluation/answer_1-110_small_table.json'
    answer_preds_llama = '/projects/bces/lanl2/LLM4DC/evaluation/answer_1-110_llama3.1.json'

    
    # eval_answer_results = eval_answers(answer_gt_path, answer_preds_llama)
    model = models[0]
    wf_gt_folder = '/projects/bces/lanl2/LLM4DC/datasets'
    wf_pred_folder = f'/projects/bces/lanl2/LLM4DC/CoT.response/{model}/recipes_llm'

    if query_id >= 62 and query_id <= 91:
        target_path = f'{wf_gt_folder}/ppp_datasets/cleaned_tables/ppp_sample_p{query_id}.csv'
        wf_gt_fp = f"{wf_gt_folder}/ppp_datasets/workflows/ppp_sample_p{query_id}.json"
        wf_pred_fp = f"{wf_pred_folder}/ppp_test_{query_id}.json"
    elif query_id >= 92:
        target_path = f'{wf_gt_folder}/dish_datasets/cleaned_tables/dish_sample_p{query_id}.csv'
        wf_gt_fp = f"{wf_gt_folder}/dish_datasets/workflows/dish_sample_p{query_id}.json"
        wf_pred_fp = f"{wf_pred_folder}/dish_test_{query_id}.json"
    elif query_id >= 31 and query_id <= 61:
        target_path = f'{wf_gt_folder}/chi_food_inspection_datasets/cleaned_tables/chi_sample_p{query_id}.csv'
        wf_gt_fp = f"{wf_gt_folder}/chi_food_inspection_datasets/workflows/chi_sample_p{query_id}.json"
        wf_pred_fp = f"{wf_pred_folder}/chi_test_{query_id}.json"
    elif query_id <31:
        target_path = f'{wf_gt_folder}/purpose-prepared-datasets/menu/menu_p{query_id}'
        wf_gt_fp = f"{wf_gt_folder}/purpose-prepared-datasets/menu/workflows/menu_p{query_id}.json"
        wf_pred_fp = f"{wf_pred_folder}/menu_test_{query_id}.json"
