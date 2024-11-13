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
# from history_update_problem.call_or import export_rows
from call_or import *

import ollama
from ollama import Client
from ollama import generate
import logging

map_ops_func = {
"core/column-split": split_column,
"core/column-addition": add_column,
"core/text-transform": text_transform,
"core/mass-edit": mass_edit,
"core/column-rename": rename_column,
"core/column-removal": remove_column
}


def export_ops_list(project_id, st=0):
    ops = get_operations(project_id)
    op_list = [dict['op'] for dict in ops]
    functions_list = [map_ops_func[operation].__name__ for operation in op_list]
    return ops, functions_list


def parse_text_transform(ops_list, functions_list):
    """This is to decompose text_transform to common_transform and regex_based transform"""
    for idx, op in enumerate(ops_list):
        op_name = op['op']
        if op_name=="core/text-transform":
            exp = op['expression']
            if exp=="value.trim()":
                functions_list[idx] = "trim"
            elif exp=="value.toUppercase()":
                functions_list[idx] = "upper"
            elif exp=="value.toNumber()":
                functions_list[idx] = "numeric"
            elif exp=="value.toDate()":
                functions_list[idx] = "date"
            elif exp.startswith("jython"):
                functions_list[idx] = "regexr_transform"
            elif exp=="value.toString()":
                functions_list[idx] = "date"
            else:
                raise NotImplementedError
    return functions_list

def export_intermediate_tb(project_id):
    # Call API to retrieve intermediate table
    rows = []
    csv_reader = export_rows(project_id)
    rows = list(csv_reader)
    columns = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=columns)
    return df


def format_sel_col(df):
    table_caption = "A mix of simple bibliographic description of the menus"
    # df = pd.read_csv(fp)
    columns = df.columns.tolist() # column schema information 
    col_priority = []
    for col in df.columns:
        # Get the column name
        column_name = col
        # Get three row values from the column
        row_values = df[col].head(3).tolist()
        # Append column name and three row values as a sublist
        col_priority.append([column_name] + row_values)
    res = {
        "table_caption": table_caption,
        "columns": columns,
        "table_column_priority": col_priority
    }
    format_res = json.dumps(res, indent=2)
    print(format_res)
    return format_res


def gen_table_str(df, num_rows=3, tg_col=None, flag=[]):
    # Sample the first 'num_rows' rows
    df = df.head(num_rows)
    
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


def extract_tg_cols(content):
    # This is to extract a list of target columns from the LLM generation 
    print(content)
    extracted_content = re.search(r"```(.*?)```", content, re.DOTALL)

    # If match is found, parse the content
    if extracted_content:
        parsed_list = ast.literal_eval(extracted_content.group(1))
    else:
        parsed_list = []
    return parsed_list


def extract_exp(content, refs=None):
    # Count occurrences of each *reference* in the generated content by LLM
    print(f'content: {content}')
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
        if most_freq_ref:
            return most_freq_ref[0]
        else:
            return False
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
    # seperated_line = raw_string.split('\n')
    # new_string = ''
    # for line in seperated_line:
    #     new_string += line.replace(r'^\#.*?', '') + '\n'
    raw_string.strip('python').strip('sql')
    parsed_edits = eval(raw_string)
    # parsed_edits = ast.literal_eval(raw_string)
    
    return parsed_edits


def wf_gen(project_id, log_data, model, purpose):
    df = export_intermediate_tb(project_id) # Return current intermediate table
    tb_str = gen_table_str(df, num_rows=10)
    av_cols = df.columns.to_list() # current column schema 
    ops_gen = {}
    ops_data = []
    errors = []
    context =[]
    # TASK I: select target column(s)
    with open("prompts/f_select_column.txt", 'r')as f:
        sel_col_learn = f.read()
    print(f'current purpose: {purpose}')
    prompt_sel_col = sel_col_learn + f"""\
\n\nBased on table contents and Purpose provided as following, output the Selected columns in ``` ```ONLY.
/*
{format_sel_col(df)}
*/
Purpose: {purpose}
Selected columns:
                                    """
    print(prompt_sel_col)

    context, sel_col_desc = gen(prompt_sel_col, context, model)
    
    # print(f'description of selected column: {sel_col_desc}')
    ext_res = extract_exp(sel_col_desc)[0]
    print(ext_res)
    ext_res = ext_res.strip('sql').strip('python')
    tg_cols = ast.literal_eval(ext_res)
    print(f'Target columns: {tg_cols}')

    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    with open("prompts/eod.txt", 'r')as f0:
        eod_learn = f0.read()

    num_votes = 3 # run gen() multiple times to generate end_of_dc decisions
    
    while tg_cols:
        eod_flag = "False"
        sel_col = tg_cols[0]
        print(f'Current selected column: {sel_col} from the target column list: {tg_cols}')
        sum_eod = f"Generate proper operations to improve accuracy, completeness, conciseness of the column: {sel_col}"
        # st: start step id, [st:] is to chunk the functions_list on current sel_col ONLY
        st = 0
        # Clean the column one by one
        df = export_intermediate_tb(project_id) # Return current intermediate table
        sel_cols_df = df[tg_cols]
        while eod_flag == "False":
            # ```upper```,  ```trim```, ```add_column```, ```split_column```, ```mass_edit```,  ```regexr_transform```, , ```numeric```, and ```date```
            # ops_pool = ["upper", "trim", "add_column", "split_column", "mass_edit", "regexr_transform", "numeric", "date"]
            ops_pool = ["upper", "trim", "mass_edit", "regexr_transform", "numeric", "date"]
            context = []
            eod_desc = False
            col_values = df[sel_col]
            num_rows = len(df)
            av_cols = df.columns.to_list() # current column names list 

            # TASK II: select operations
            sel_op = None
            ops_history, functions_list = export_ops_list(project_id, st)
            if functions_list:
                functions_list = parse_text_transform(ops_history, functions_list)
                print(f'Applied operation history: {functions_list}')
            col_str = gen_table_str(df, num_rows=15, tg_col=sel_col) # only keep first 15 rows for operation selection
            sel_cols_str = gen_table_str(sel_cols_df, num_rows=15)
            print(f'Selected first {num_rows} rows for current table: {col_str}')

            # context-learn (full_chain_demo): how the previous operation are related to the current one
            # operation-learn (learn_ops.txt): when to select a proper operation 
            with open('prompts/learn_ops.txt', 'r')as f_learn_ops:
                learn_ops = f_learn_ops.read()
            # with open('prompts/learn_ops_lf.txt', 'r')as f_learn_ops:
            #     learn_ops = f_learn_ops.read()
            # with open('prompts/full_chain_demo.txt', 'r')as f_chain:
            #     dynamic_plan = f_chain.read()
            
            prompt_sel_ops = learn_ops +\
                             f"""\
\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool} and output the operation name in ``` ```.\n"""\
                             +f"""\
/*
{col_str}
*/
Purpose: {purpose}
Target column: {sel_col}
Explanation: {sum_eod}
Selected Operation: 
                              """
            print("----start-------")
            print(prompt_sel_ops)

            options_sel_op = {
                    'temperature': 0.1,
                    'stop': ["Explanation:", "'','','',", ],
                    'num_predict': -1,
                    'top_k': 60,
                    'top_p': 0.95,
                    'mirostat': 1 #0(default), 1(mirostat1),2(mirostat2)
                }
            # TODO: Quality control
            context, sel_op_desc = gen(prompt_sel_ops, context, model, options_sel_op)
            print(sel_op_desc)
            sel_op = extract_exp(sel_op_desc, ops_pool)
            print(f'selected operation: {sel_op}')

            # TASK III: Learn function arguments (share the same context with sel_op)
            # return first 15 rows for generating arguments [different ops might require different number of rows]
            if sel_op not in ['numeric', 'trim', 'upper', 'date', 'regexr_transform']:
                args = get_function_arguments('call_or.py', sel_op)
                args.remove('project_id')  # No need to predict project_id
                args.remove('column')
                print(f'Current args need to be generated: {args}')
            elif sel_op == "regexr_transform":
                args = get_function_arguments('call_or.py', 'text_transform')
                args.remove('project_id')
                args.remove('column')
            else:
                print(f'No arguments need to generate for {sel_op}')
            
            with open(f'prompts/{sel_op}.txt', 'r') as f1:
                prompt_sel_args = f1.read()

            # Prepare the operation purpose
            prompt_eod = eod_learn + f"""
                                        \n\nBased on table contents, Objective, and Flag provided as following, output Explanations.
                                        /*
                                        {col_str}
                                        */

                                        Objective: {purpose}
                                        Flag: ```False```
                                        Explanations: 
                                        """
            _, eod_desc = gen(prompt_eod, [], model, {'temperature': 0.2}) #clear out context
            prompt_eod_desc_summarization = f"""please generate a one-sentence summarization and a one-sentence data cleaning objective for next operation according to the detailed data quality issue mentioned by **3.Assessing profiling results from four dimensions:** from the: \n{eod_desc}"""
            _, one_sent_eod_desc = gen(prompt_eod_desc_summarization, [], model, {'temperature': 0.2, 'top_p': 0.95})
            # Regular expression to extract the desired sentence
            # eod_pattern= r"Next operation:\s*(.*?)\."
            print(one_sent_eod_desc)
            eod_pattern = r"\*\*Data Cleaning Objective:\*\*\s*(.*?)\."
            # Search for the pattern in the text
            eod_match = re.search(eod_pattern, one_sent_eod_desc, re.DOTALL)
            # Extract the matched sentence if found
            sum_eod = eod_match.group(1).strip() + '.' if eod_match else one_sent_eod_desc
            print(sum_eod)
            
            # >>>>Start Arguments Generation>>>>
            context = []
            if sel_op == 'split_column':
                prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output Separator in ``` ``` . """\
                                + f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    Separator: 
                                    """
                context, sep_desc = gen(prompt_sel_args, context, model)
                sep = extract_exp(sep_desc)
                print(f'Predicted separator for operation split column: {sep}')
                sel_args= {'column':sel_col, 'separator':sep}
                split_column(project_id, **sel_args)
            elif sel_op == 'add_column':
                prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output expression and new_column separately in ``` ```."""\
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    Expression:, New_column:
                                    """
                context, res_dict_desc = gen(prompt_sel_args, context, model)
                [exp, new_col] = extract_exp(res_dict_desc)
                print(f"Expression of add_column: {exp}, New column created: {new_col}")
                sel_args = {'column': sel_col, 'expression': exp, 'new_column': new_col} 
                add_column(project_id, column=sel_col, expression=exp, new_column=new_col)
            elif sel_op == 'regexr_transform':
                # tb_str = gen_table_str(df, num_rows=50, tg_col=sel_col)
                col_str = gen_table_str(df, num_rows=30, tg_col=sel_col)
                prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output expression in ``` ``` (Ensure the expression format statisifies ALL requirements in the **Check**). """ \
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    Expression: 
                                    """
                # print(f'updated prompt for selecting arguments: {prompt_sel_args}')
                context, exp_desc = gen(prompt_sel_args, context, model)
                exp = extract_exp(exp_desc)[0].replace('jython\n', 'jython:')+ '\nreturn value'
                print(f'********predicted expression: {exp}')
                text_transform(project_id, column=sel_col, expression=exp)
            elif sel_op == 'numeric':
                text_transform(project_id, column=sel_col, expression="value.toNumber()")
            elif sel_op == 'date':
                text_transform(project_id, column=sel_col, expression="value.toDate()")
                text_transform(project_id, column=sel_col, expression="value.toString()")
            elif sel_op == 'trim':
                text_transform(project_id, column=sel_col, expression="value.trim()")
            elif sel_op == 'upper':
                text_transform(project_id, column=sel_col, expression="value.toUppercase()")
            elif sel_op == 'mass_edit':
                # sel_cols_str = gen_table_str(sel_cols_df, num_rows=num_rows)
                # sum_edo = sum_eod.replace('\n', ' ')
                col_str = gen_table_str(df, num_rows=num_rows, flag=['City', 'Zip', 'State'])
                print(col_str)
                prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output edits (a list of dictionaries) in ``` ```."""\
                                + f"""\n
/*
{col_str}
*/
Purpose: {purpose}
Current Operation Purpose: {sum_eod}
edits: 
"""
#                 prompt_sel_args += """\n\nBased on the following table contents, Purpose, and Current Operation Objective, generate the JSON format **edits** ONLY in ```  ````.""" \
#                     + f"""\n
# /*
# {col_str}
# */
# Purpose: {purpose}
# Current Operation Objective: {sel_op_desc}
# edits: """
                print("prompts for generating edits:")
                print(prompt_sel_args)
                options = {
                    'temperature': 0.2,
                }
                try:
                    context, edits_desc = gen(prompt_sel_args, context, model, options)
                    edits_v = extract_exp(edits_desc)
                    print(f'descriptions for edits: \n\n {edits_desc}')
                
                    if edits_v:
                        edits_v = edits_v[0].replace("edits: ", "")
                        print('edits_v_before parsed, ', edits_v)
                        edits_v = parse_edits(edits_v)
                        print('parsed edits', edits_v)
                        mass_edit(project_id, column=sel_col, edits=edits_v)
                    else: 
                        print('No edits are parsed')
                except:
                    pass
                # raise NotImplementedError
            # Re-execute intermediate table, retrieve current data cleaning workflow
            cur_df = export_intermediate_tb(project_id)
            cur_av_cols = cur_df.columns.to_list() # check if column schema gets changed, current - former
            diff = list(set(cur_av_cols) - set(av_cols))
            if diff:
                print(f'column schema get changed: {diff}')
                # add extra columns generated during the data cleaning process
                tg_cols.extend(diff)
                diff = False
            else:
                diff = True
            cur_col = cur_df[sel_col]
            cur_col_str = gen_table_str(cur_df, num_rows=30, tg_col=sel_col)
            ops_history, functions_list = export_ops_list(project_id, st)
            functions_list = parse_text_transform(ops_history, functions_list)
            log_data['Operations'] = functions_list
            print(f"start id: {st}; column: {sel_col}; column schema gets modified: {diff}; \nfunctions list: {functions_list}")

            # TASK VI:
            # Keep passing intermediate table and data cleaning objective, until eod_flag is True. End the iteration.
            iter_prompt = eod_learn + f"""
                                    \n\nBased on table contents and Objective provided as following, output Flag in ``` ```.
                                    /*
                                    {cur_col_str}
                                    */
                                    Objective: {purpose}
                                    Flag:
                                    """
            eod_flag_list = []
            eod_desc_list = []
            for _ in range(num_votes):
                context, eod_desc = gen(iter_prompt, [], model, {'temperature':0.7})
                eod_flag = extract_exp(eod_desc, ['False', 'True'])
                eod_flag_list.append(eod_flag)
                eod_desc_list.append(eod_desc)
            if any([x == "True" for x in eod_flag_list]):
                eod_flag = "True"
                ops_data += functions_list # appending the operations if done...
                ops_gen[sel_col] = functions_list #TODO... the functions_list are the whole...    
                ops_gen['Error_Running'] = errors
            else:
                eod_flag  = "False"
                mask = [int(x == "False") for x in eod_flag_list]
                eod_desc = random.choice([value for value, m in zip(eod_desc_list, mask) if m == 1])
                
            print(f'Decision of end of data cleaning on column {sel_col}: {eod_flag}')
        log_data['Columns'].append(sel_col)
        tg_cols.pop(0)
        st += len(functions_list)
        print(f"remaining columns: {tg_cols}")
    # log_data["Operations"] = list(set(ops_data))
    print(f'The full operation chain: {ops_gen}')
    print(f'The whole process: {log_data}')
    return log_data


def create_projects(project_name, ds_fp):
    _, proj_id = create_project(data_fp=ds_fp, project_name=project_name)
    return proj_id

def main():
    # model = "llama3.2"
    model = "llama3.1:8b-instruct-fp16"
    log_dir = "CoT.response"
    os.makedirs(log_dir, exist_ok=True)

    pp_par_folder = "purposes"
    # purpose_file = ["menu_about", "ppp_about", "dish_about", "chi_food_inspect_about"]
    purpose_file = ["menu_about"]
    pp_paths = [f"{pp_par_folder}/{file}.csv" for file in purpose_file]

    ds_par_folder = "datasets"
    # ds_file = ["menu_data", "ppp_data", "dish_data", "chi_food_data"]
    ds_file = ["menu_data"]
    ds_paths = [f"{ds_par_folder}/{file}.csv" for file in ds_file]
    
    # start from menu
    rounds = list(range(len(pp_paths))) #[0,1,2,3]

    for round in rounds:
        # Four datasets: Four rounds
        pp_f = pp_paths[round]
        ds = ds_paths[round]
        ds_name = ds_file[round]
        pp_df = pd.read_csv(pp_f)
        logs = []
        for index, row in pp_df.iterrows():
            timestamp = datetime.now()
            timestamp_str = f'{timestamp.month}{timestamp.day}{timestamp.hour}{timestamp.minute}'
            print(timestamp_str)
            pp_id = row['ID']
            pp_v = row['Purposes']
            print(f"Row {index}: id = {pp_id}, purposes = {pp_v}")
            project_name = f"{ds_name}_{pp_id}_{timestamp_str}"
            log_data = {
                "ID": pp_id,
                "Purposes": pp_v,
                "Columns": [],
                "Operations": [],
                "Error_Running":[]
            }
            proj_names_list = extract_proj_names()
            if project_name in proj_names_list:
                print(f"Project {project_name} already exists!")
                print(project_name)
                project_id = get_project_id(project_name)
                ops_history, funcs = export_ops_list(project_id)
                # if ops_history:
                #     print(f"Data cleaning task has been finished in {project_id}: {project_name}")
                #     pass
                # else:
                wf_res, log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
                logs.append(log_data)
            else:
                project_id = create_projects(project_name, ds)
                print(f"Project {project_name} creation finished.")
                wf_res, log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
                logs.append(log_data)
            # log_file = open(, "w")
            # Initialize empty log data
            with open(f"{log_dir}/{ds_name}_{pp_id}_log_{timestamp_str}.txt", "w") as log_f:
                json.dump(log_data, log_f, indent=4)


def pull_datasets():
    parent_folder = "CoT.response/datasets_llm"
    projects = list_projects()
    for proj_id, v in projects.items():
        dataset_name = v['name']
        project_id = int(proj_id)
        print(dataset_name)
        df = export_intermediate_tb(project_id)
        filepath = f"{parent_folder}/{dataset_name}"
        # if not os.path.exists(filepath):
        df.to_csv(f"{parent_folder}/{dataset_name}.csv")


def pull_recipes():
    parent_folder = "CoT.response/recipes_llm"
    projects = list_projects()
    for proj_id, v in projects.items():
        dataset_name = v['name']
        project_id = int(proj_id)
        print(dataset_name)
        data = get_operations(project_id)
        filepath = f"{parent_folder}/{dataset_name}.json"
        if not os.path.exists(filepath):
           with open(filepath, "w") as workflow:
                json.dump(data, workflow, indent=4)  # `indent=4` adds pretty formatting
        else:
            print(f"{filepath} Has Been Existed!")

def test_main():
    # model = "gemma2:9b" #"llama3.1:8b-instruct-fp16"
    # ollama.pull(model)
    logging.basicConfig(filename='my_log.log', level=logging.DEBUG) # TODO: change filename 
    logging.info('Hello, world!')


    models = [
    "llama3.1:8b-instruct-fp16" ,
    "llama3.2",
    "phi3",
    "gemma2",
    "mistral"
    ]
    model = "llama3.1:8b-instruct-fp16"
    log_dir = "CoT.response"
    os.makedirs(log_dir, exist_ok=True)

    pp_f = 'purposes/pp_instance_level.csv'
    pp_df = pd.read_csv(pp_f)
    
    # ds_file = "datasets/menu_data.csv"
    # ds_name = "menu_test"
    for index, row in pp_df.iloc[39:40].iterrows():
        timestamp = datetime.now()
        timestamp_str = f'{timestamp.month}{timestamp.day}{timestamp.hour}{timestamp.minute}'
        print(timestamp_str)
        pp_id = row['ID']
        pp_v = row['Purposes']
        print(f"Row {index}: id = {pp_id}, purposes = {pp_v}")
        if 1<= pp_id <=30:
            ds_name = "menu_test"
            ds_file = "datasets/menu_data.csv"
        elif 31<= pp_id <=61:
            ds_name = "chi_test"
            ds_file = "datasets/chi_food_data.csv"
        elif 62<=pp_id<=91:
            ds_name = "ppp_test"
            ds_file = "datasets/ppp_data_20.csv"
        elif pp_id > 91:
            ds_name = "dish_test"
            ds_file = "datasets/dish_data.csv"
        # project_name = f"{ds_name}_{pp_id}_{timestamp_str}"
        project_name = f"{ds_file.split('/')[-1].split('.')[0]}_{pp_id}"
        log_data = {
            "ID": pp_id,
            "Purposes": pp_v,
            "Columns": [],
            "Operations": [],
            "Error_Running":[]
        }
        proj_names_list = extract_proj_names()
        project_id = None
        if project_name in proj_names_list:
            print(f"Project {project_name} already exists!")
            print(project_name)
            project_id = get_project_id(project_name)
            ops_history, funcs = export_ops_list(project_id)
            log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
        else:
            project_id = create_projects(project_name, ds_file)
            print(f"Project {project_name} creation finished.")
            log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
        # with open(f"{log_dir}/{ds_name}_{pp_id}_log_{timestamp_str}.txt", "w") as log_f:
        #     json.dump(log_data, log_f, indent=4)
        with open(f"{log_dir}/{ds_name}_{pp_id}_log.txt", "w") as log_f:
            json.dump(log_data, log_f, indent=4)
    
    # Download all the prepared datasets
    pull_datasets()
    pull_recipes()

if __name__ == '__main__':
    # pull_recipes()
    # pull_datasets()
    test_main()
    # main()
    