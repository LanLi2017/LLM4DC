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

def eval_answers(answer_gt_path, answer_preds_llama):
    answer_gt = load_answer_dataset(answer_gt_path)
    answer_gt = pd.DataFrame(answer_gt)
    answer_preds_llama = load_answer_dataset(answer_preds_llama)
    answer_preds_llama = pd.DataFrame(answer_preds_llama)
    answer_compare = answer_gt.merge(answer_preds_llama[['pp_id', 'answer']], on='pp_id', how='left', suffixes=('_gt', '_preds'))

    results = []
    for i, row in answer_compare.iterrows():
        gt = row['answer_gt']
        preds = row['answer_preds']
        single_result = calculate_answer_metrics(gt, preds)
        single_result['pp_id'] = row['pp_id']
        results.append(single_result)
    return pd.DataFrameresults


def eval_workflows(pp_id, gt_wf_fp, pred_wf_fp):
    print(gt_wf_fp)
    gt_ops_list = parse_recipe(pp_id, recipe=gt_wf_fp)
    pred_ops_list = parse_recipe(pp_id, recipe=pred_wf_fp)
    print(gt_ops_list)
    print(pred_ops_list)

def eval_dataset(pp_id, gd_ds_fp, pred_ds_fp):
    result_dict = retrieve_tg_cols()
    tg_cols = result_dict[pp_id]
    gd_df = pd.read_csv(gd_ds_fp)
    pred_df = pd.read_csv(pred_ds_fp)
    res = average_match_ratio(gd_df, pred_df, tg_cols)
    print(pp_id, res)


if __name__ == '__main__':
    models = ['llama3.1', 'gemma2', 'mistral']
    answer_gt_path = '/projects/bces/lanl2/LLM4DC/evaluation/answer_1-110_small_table.json'
    answer_preds_llama = '/projects/bces/lanl2/LLM4DC/evaluation/answer_1-110_llama3.1.json'

    
    # eval_answer_results = eval_answers(answer_gt_path, answer_preds_llama)
    model = models[2]
    wf_gt_folder = '/projects/bces/lanl2/LLM4DC/datasets'
    wf_pred_folder = f'/projects/bces/lanl2/LLM4DC/CoT.response/{model}/recipes_llm'
    ds_pred_folder = f'/projects/bces/lanl2/LLM4DC/CoT.response/{model}/datasets_llm'
    
    query_contents = pd.read_csv('/projects/bces/lanl2/LLM4DC/purposes/queries.csv')
    for query_id in range(111):
        row = query_contents[query_contents['ID'] == query_id]
        if len(row) == 0:
            continue
        if query_id >= 62 and query_id <= 91:
            target_path = f'{wf_gt_folder}/ppp_datasets/cleaned_tables/ppp_sample_p{query_id}.csv'
            pred_path = f"{ds_pred_folder}/{model}_ppp_test_{query_id}.csv"
            wf_gt_fp = f"{wf_gt_folder}/ppp_datasets/workflows/ppp_sample_p{query_id}.json"
            wf_pred_fp = f"{wf_pred_folder}/ppp_test_{query_id}.json"
        elif query_id >= 92:
            target_path = f'{wf_gt_folder}/dish_datasets/cleaned_tables/dish_sample_p{query_id}.csv'
            pred_path = f"{ds_pred_folder}/{model}_dish_test_{query_id}.csv"
            wf_gt_fp = f"{wf_gt_folder}/dish_datasets/workflows/dish_sample_p{query_id}.json"
            wf_pred_fp = f"{wf_pred_folder}/dish_test_{query_id}.json"
        elif query_id >= 31 and query_id <= 61:
            target_path = f'{wf_gt_folder}/chi_food_inspection_datasets/cleaned_tables/chi_sample_p{query_id}.csv'
            pred_path = f"{ds_pred_folder}/{model}_chi_test_{query_id}.csv"
            wf_gt_fp = f"{wf_gt_folder}/chi_food_inspection_datasets/workflows/chi_sample_p{query_id}.json"
            wf_pred_fp = f"{wf_pred_folder}/chi_test_{query_id}.json"
        elif query_id <31:
            target_path = f'{wf_gt_folder}/purpose-prepared-datasets/menu/menu_p{query_id}.csv'
            pred_path = f"{ds_pred_folder}/{model}_menu_test_{query_id}.csv"
            wf_gt_fp = f"{wf_gt_folder}/purpose-prepared-datasets/menu/workflows/menu_p{query_id}.json"
            wf_pred_fp = f"{wf_pred_folder}/menu_test_{query_id}.json"
    
        # eval_workflows(query_id, wf_gt_fp, wf_pred_fp)
        eval_dataset(query_id, target_path, pred_path)