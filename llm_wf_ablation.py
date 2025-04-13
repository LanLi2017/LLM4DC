# LLM-based history update solution
import importlib.util
import inspect
from typing import List
import requests
import json
import re
from glob import glob
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

import ollama
from ollama import Client
from ollama import generate


def export_intermediate_tb(project_id):
    # Call API to retrieve intermediate table
    rows = []
    csv_reader = export_rows(project_id)
    rows = list(csv_reader)
    columns = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=columns)
    return df


def gen_table_str(df, num_rows=3, tg_col=None, flag=[]):
    # Sample the first 'num_rows' rows
    num_rows = min(num_rows, len(df))
    df = df.sample(n=num_rows)
    dropna=True
    max_length = 20

    # If no target column is specified, generate the full table
    if not tg_col:
        if flag:
            df = df[flag]
            
        # Find the maximum length for each column for proper alignment
        col_widths = [max(len(str(col)), df[col].astype(str).map(len).max()) + 2 for col in df.columns]

        # Prepare the formatted column schema line
        column_schema = 'col: | ' + ' | '.join([f'{col:<{col_widths[i]}}' for i, col in enumerate(df.columns)]) + ' |'
        # Prepare the formatted rows with row numbers
        rows_lines = []
        for i, row in df.iterrows():
            row_str = ' | '.join([f'{str(value):<{col_widths[j]}}' for j, value in enumerate(row)])
            rows_lines.append(f'row {i+1}: | {row_str} |')
        
        # Combine the column schema with the formatted rows
        table_str = column_schema + '\n' + '\n'.join(rows_lines)
        return table_str

    # If a target column is specified, return just that column's values
    else:
        column_values = df[tg_col]
        if dropna:
            column_values = column_values.replace("", float("NaN"))
            column_values = column_values.dropna()
        if len(column_values) >= max_length:
            column_values = column_values.sample(n=max_length, random_state=42)
        formatted_output = [f"col: {tg_col}"]
        for i, value in enumerate(column_values, start=1):
            formatted_output.append(f"row {i}: {value}")
        return '\n'.join(formatted_output)


def get_function_arguments(script_path: str, function_name: str) -> List[str]:
    """
    Get the arguments of a function from a given Python script.

    Parameters:
        script_path (str): Path to the Python script.
        function_name (str): Name of the function to inspect.

    Returns:
        List[str]: List of argument names.
    """
    # Load the script as a module
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the function object
    func = getattr(module, function_name)
    
    # Get the function signature
    sig = inspect.signature(func)
    
    # Extract argument names
    args = [param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty]
    
    return args


def extract_exp(content, refs=None):
    # Count occurrences of each *reference* in the generated content by LLM
    if refs:
        # 1. select columns; 2. select operations
        ref_counts = Counter()
        for ref in refs:
            # Adjust the pattern to allow for optional leading formatting characters
            pattern = r'(?:(?:\*\*|\`|\`\`)?\s*)' + re.escape(ref) + r'(?:(?=\s)|(?=\*\*|\`|\`\`)|$)'
            ref_counts[ref] = len(re.findall(pattern, content))
    
        # Find the maximum occurrence count
        max_count = max(ref_counts.values(), default=0)
        
        # Retrieve operation names with the maximum occurrence
        most_freq_ref = [res for res, count in ref_counts.items() if count == max_count and count > 0]
        print(f'most_freq_ref: {most_freq_ref}')
        return most_freq_ref[0] if most_freq_ref else False
    else:
        # this is to extract arguments 
        matches = re.findall(r'`{1,3}(.*?)`{1,3}', content, re.DOTALL)
        
        if matches:
            code_blocks = [match.strip().replace('; ', '\n') for match in matches]
            return code_blocks
        else:
            print(f'Current content cannot be parsed: {content}')
            print("No code blocks found.")
            return False


def gen(prompt, context, model, options={'temperature':0.0}):
    """
    options ref: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values 
    {'temperature':
    'stop':
    'num_predict':
    'top_p'
    'mirostat': 0(default), 1(mirostat1),2(mirostat2)
    }
    """
    r = generate(model=model, 
                 prompt=prompt, 
                 context=context,
                 options=options,
                stream=True
                )
    res=[]
    
    for part in r:
        response_part = part['response']
        res.append(response_part)
        if part['done'] is True:
            return part['context'], ''.join(res)
    raise ValueError


# parse edits by LLMs into a list
def parse_edits(raw_string):
    # Remove newlines and spaces
    raw_string = raw_string.replace('\n', '').strip()
    
    # pattern =  r'\[(\{.*?\})*,*\]```'
    result = re.findall(r'(\[(:?\n?.*\n?)*\])', raw_string, re.DOTALL)
    if result:
        for r in result:
            raw_string = r[0]
    
    # matches = re.findall(pattern, raw_string)
    # Parse the string using ast.literal_eval
    raw_string.strip('python').strip('sql')
    parsed_edits = eval(raw_string)
    
    return parsed_edits


def wf_gen(model,purpose,df):
    #  input purpose, table
    #  return project_id (call API to process data), and the recipe
    av_cols = df.columns.to_list() # current column schema 

    # prompt_clean = """
    # You are an expert in data cleaning theory and practices, you are able to recognize whether current data(i.p., column and cell values) is clean (high quality) enough for the provided objectives. 
    # The pipeline of evalauting whether a column is of good quality: 
    # (1). Profiling the column, check it from column schema level and instance level;
    # "Whether the column name is meaningful or not?" 
    # "what are the distributions of data instances?" "are they clearly represented in this column?"
    # (2). Assess the profiling results from four dimensions as following: 
    # - **Accuracy**: Whether the target column is free from obvious errors, inconsistencies, or biases
    # - **Relevance**: Whether the target column exists in the dataset to address the objectives.
    # - **Completeness**: Whether the target column has a reasonable sample size and contains enough data instances (not too many missing values)
    # - **Conciseness**: Whether the spellings in the target column are standardized, no same semantics but different representations exist
    # (3) if any of the quality dimensions are False, then you SHOULD return a chain of proper data cleaning operations as workflow to improve data quality.
    # In the following, you will learn operations and args.
    # """
    
    # I: prompt of operation learning 
    # operation-learn (learn_ops_.txt): when to select a proper operation 
    #  for operations with arguments, add more examples from arguements learn 
    with open('prompts/learn_ops_ab.txt', 'r')as f_learn_ops:
        learn_ops = f_learn_ops.read()
    
    # context = [learn_ops]

    # II: full chain example prompt
    with open('prompts/full_chain.txt', 'r')as f_learn_wf:
        learn_wf = f_learn_wf.read()
    
    # Step 1: Prime the model with learn_ops
    context, _ = gen(learn_ops, [], model)

    # III: instruction: the task is to generate a chain of operations given a purpose
    # input: table, purpose, output: full_chain
    tb_str = gen_table_str(df, num_rows=5)
    ops_pool = ["upper", "trim", "mass_edit", "regexr_transform", "numeric", "date"]

    learn_wf += f"""\
\n\n Based on table contents and Purpose provided as following, ONLY generate a workflow in ``` ``` with NO Explanations.\n"""\
+f"""\
/*
{tb_str}
*/

Purpose: {purpose}
Workflow:
    """
    # options_sel_op = {
    #             'temperature': 0.3,
    #             'stop': ["\n\n\n\n"],
    #             'num_predict': -1,
    #             'top_k': 60,
    #             'top_p': 0.95,
    #             'mirostat': 1  # 0 (default), 1 (mirostat1), 2 (mirostat2)
    #         }

    _, gen_wf_desc = gen(learn_wf, context, model)
    gen_wf = extract_exp(gen_wf_desc)
    print(f'Generated workflow: {gen_wf}')
    return gen_wf


# def load_clean_json(filepath):
#     with open(filepath, 'r') as f:
#         content = f.read()

#     # Trim everything before the actual JSON
#     match = re.search(r'(\[|\{)', content)
#     if match:
#         content = content[match.start():]
#     else:
#         raise ValueError(f"No JSON array/object found in {filepath}")
    
#     return json.loads(content)

import json
import re

def load_even_broken_json(filepath):
    """
    Load a possibly malformed JSON file:
    - Skip junk before JSON starts
    - Join lines into a proper JSON string
    - Parse and return the data (or None if unparseable)
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Step 1: Skip lines before actual JSON (e.g. 'sql', empty, etc.)
    json_start_index = next((i for i, line in enumerate(lines) 
                             if line.strip().startswith('[') or line.strip().startswith('{')), None)

    if json_start_index is None:
        print(f" No JSON object or array found in file: {filepath}")
        return None

    json_str = ''.join(lines[json_start_index:]).strip()

    # Step 2: Fix common issues (unescaped control characters in strings)
    # Don't overdo it — just ensure we don't have raw \n, \t, etc. inside strings
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in {filepath}: {e}")
        return None


def extract_purpose_id(filename):
    """Extracts purpose ID from filename like 'base_gemma2_chi_test_p31.json'"""
    match = re.search(r'_p(\d+)\.json$', filename)
    return int(match.group(1)) if match else None

def create_projects(project_name, ds_fp):
    _, proj_id = create_project(data_fp=ds_fp, project_name=project_name)
    return proj_id


def gen_full_chain():
    model = "gemma2:27b"
    # model = "mistral:7b-instruct"
    # model = "gemma2base:9b"
    # model = "llama3.1"
    model_name = model.split(':')[0]

    log_dir = f"ablation/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    pp_f = 'purposes/all_purposes.csv'
    pp_df = pd.read_csv(pp_f)
    
    logging_dir = f"{log_dir}/logging"
    os.makedirs(logging_dir, exist_ok=True)

    chains_dir = f"{log_dir}/workflow_unparsed"
    os.makedirs(chains_dir, exist_ok=True)

    # ds_file = "datasets/menu_data.csv"
    # ds_name = "menu_test"
    # 25-27 53,54 56-57 58-60 rerun date...
    for index, row in pp_df.iloc[:].iterrows():
        timestamp = datetime.now()
        timestamp_str = f'{timestamp.month}{timestamp.day}{timestamp.hour}{timestamp.minute}'
        print(timestamp_str)
        pp_id = row['ID']
        pp_v = row['Purposes']
        print(f"Row {index}: id = {pp_id}, purposes = {pp_v}")
        if 1 <= pp_id <= 30:
            ds_name = "menu_test"
            ds_file = f"datasets/menu_datasets/menu_p{pp_id}.csv"
        elif 31<= pp_id <=60:
            ds_name = "chi_test"
            ds_file = f"datasets/CFI_datasets/chi_food_data_p{pp_id}.csv"
        elif 62<=pp_id<=91:
            ds_name = "ppp_test"
            ds_file = f"datasets/ppp_datasets/ppp_data_p{pp_id}.csv"
        elif 92<= pp_id <=110:
            ds_name = "dish_test"
            ds_file = f"datasets/dish_datasets/dish_data_p{pp_id}.csv" 
        elif 111<=pp_id<=126:
            ds_name = "flights_test"
            ds_file = f"datasets/flights/flights_data_p{pp_id}.csv"
        elif 127<=pp_id<=154:
            ds_name="hos_test"
            ds_file = f"datasets/hospital/hos_data_p{pp_id}.csv"
        #TODO: logging file name 
        logging_name = f"{logging_dir}/{model_name}_{ds_name}_{pp_id}.log"
        logging.basicConfig(filename=logging_name, level=logging.INFO) # TODO: change filename 

        df = pd.read_csv(ds_file)
        gen_workflow = wf_gen(model,purpose=pp_v, df=df)
        # Ensure the filename ends with .json
        filename = f'{project_name}.json'
        wf_filepath = os.path.join(chains_dir, filename)

        # Write the generated workflow
        with open(wf_filepath, 'w', encoding='utf-8') as fp:
            if gen_workflow:
                fp.write(gen_workflow[0])
    

def test_main():
    # model = "gemma2:27b"
    # model = "mistral:7b-instruct"
    # model = "gemma2base:9b"
    model = "llama3.1"
    model_name = model.split(':')[0]

    # ollama.pull(model)
    log_dir = f"ablation/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    pp_f = 'purposes/all_purposes.csv'
    pp_df = pd.read_csv(pp_f)

    ds_dir = f"{log_dir}/datasets_llm"
    os.makedirs(ds_dir, exist_ok=True)

    recipe_dir = f"{log_dir}/recipes_llm"
    os.makedirs(recipe_dir, exist_ok=True)

    chains_dir = f"{log_dir}/workflow_unparsed"
    os.makedirs(chains_dir, exist_ok=True)

    logging_dir = f"{log_dir}/logging"
    os.makedirs(logging_dir, exist_ok=True)
    
    json_files = sorted(glob(os.path.join(chains_dir, "*.json")))

    # Example: control start/stop
    start_idx = 0
    end_idx = len(json_files)  # or set to something like 20

    for i in range(start_idx, end_idx):
        filepath = json_files[i]
    # for filepath in glob(os.path.join(chains_dir, "*.json")):
        filename = os.path.basename(filepath)
        project_name = filename.replace(".json", "")
        print(project_name)
        pp_id = extract_purpose_id(filename)
        print(f"Current purpose: {pp_id}, index of the file: {i}")

        # data = load_clean_json(filepath)
        data = load_even_broken_json(filepath)
        if not data:
            logging.info(f"Skipped empty file: {filename}")
            continue

        if 1 <= pp_id <= 30:
            ds_name = "menu_test"
            ds_file = f"datasets/menu_datasets/menu_p{pp_id}.csv"
        elif 31<= pp_id <=60:
            ds_name = "chi_test"
            ds_file = f"datasets/CFI_datasets/chi_food_data_p{pp_id}.csv"
        elif 62<=pp_id<=91:
            ds_name = "ppp_test"
            ds_file = f"datasets/ppp_datasets/ppp_data_p{pp_id}.csv"
        elif 92<= pp_id <=110:
            ds_name = "dish_test"
            ds_file = f"datasets/dish_datasets/dish_data_p{pp_id}.csv" 
        elif 111<=pp_id<=126:
            ds_name = "flights_test"
            ds_file = f"datasets/flights/flights_data_p{pp_id}.csv"
        elif 127<=pp_id<=154:
            ds_name="hos_test"
            ds_file = f"datasets/hospital/hos_data_p{pp_id}.csv"
        
        proj_names_list = extract_proj_names()
        project_id = None
        if project_name not in proj_names_list:
            project_id = create_projects(project_name, ds_file)
            print(f"Project {project_name} creation finished.")
            logging.info(f"Project {project_name} creation finished.")
        
        for ops in data:
            try:
                col = ops['columnName']
                op_name = ops['op']
                if op_name=='numeric':
                    text_transform(project_id, column=col, expression="value.toNumber()")
                elif op_name=='date':
                    text_transform(project_id, column=col, expression="value.toDate()")
                elif op_name=='upper':
                    text_transform(project_id, column=col, expression="value.toUpper()")
                elif op_name =='trim':
                    text_transform(project_id, column=col, expression="value.trim()")
                elif op_name=='mass_edit':
                    edits_v = ops['args']
                    mass_edit(project_id, column=col, edits=edits_v)
                else:
                    print("no corresponding operation")
                    pass
            except Exception as e:
                logging.info(e)
                pass
        print(f"Done with project {pp_id}")
            
        # download dataset 
        df = export_intermediate_tb(project_id)
        ds_path = f"{ds_dir}/{project_name}.csv"
        df.to_csv(ds_path)
        
        # download recipes 
        data = get_operations(project_id)
        recipe_path = f"{recipe_dir}/{project_name}.json"
        with open(recipe_path, "w") as workflow:
            json.dump(data, workflow, indent=4)  # `indent=4` adds pretty formatting


if __name__ == '__main__':
    test_main()