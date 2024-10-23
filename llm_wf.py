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
    # return json.dumps(res, indent=2)
    return res


def gen_table_str(df, num_rows=3, tg_col=None):
    # Sample the first 'num_rows' rows
    df = df.head(num_rows)
    
    # If no target column is specified, generate the full table
    if not tg_col:
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
        matches = re.findall(r'```(.*?)```', content, re.DOTALL)
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
    parsed_edits = ast.literal_eval(raw_string)
    
    return parsed_edits


# Quality control on selecting operations
def is_valid_operation(data, operation):
    """If data-type operation: numeric or upper, or date is selected, check whether they fit for current data types or not"""
    data = data.dropna()
    if operation == "numeric":
        # Check if all values are numeric (convertible to a number)
        return all(is_numeric(value) for value in data if value)
    
    elif operation == "upper":
        # Check if all values are strings
        return all(isinstance(value, str) for value in data if value)
    
    elif operation == "date":
        # Check if all values are date strings that can be parsed as dates
        return all(is_valid_date(value) for value in data)
    
    else:
        return True

def is_numeric(value):
    """Helper function to check if a value can be converted to a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def is_valid_date(value):
    """Helper function to check if a value is a valid date string."""
    from dateutil.parser import parse
    try:
        parse(value)
        return True
    except (ValueError, TypeError):
        return False

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
    prompt_sel_col = sel_col_learn + f"""
                                    \n\nBased on table contents and purpose provided as following, output column name in ``` ```.
                                    /*
                                    {format_sel_col(df)}
                                    */
                                    Purpose: {purpose}
                                    Selected column:
                                    """

    context, sel_col_desc = gen(prompt_sel_col, context, model)
    
    # print(f'description of selected column: {sel_col_desc}')
    sel_cols = extract_exp(sel_col_desc, refs=av_cols)
    tg_cols = sel_cols.split(',')
    log_cols = tg_cols
    log_data['Columns'] = log_cols
    
    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    with open("prompts/eod.txt", 'r')as f0:
        eod_learn = f0.read()

    num_votes = 3 # run gen() multiple times to generate end_of_dc decisions
    eod_flag = "False" # initialize the end_of_dc_flag to start data_cleaning pipeline
    
    while tg_cols:
        sel_col = tg_cols[0]
        sum_eod = f"Generate proper operations to improve accuracy, completeness, conciseness of the column: {sel_col}"
        # st: start step id, [st:] is to chunk the functions_list on current sel_col ONLY
        st = 0
        # Clean the column one by one
        df = export_intermediate_tb(project_id) # Return current intermediate table
        sel_cols_df = df[tg_col]
        while eod_flag == "False":
            ops_pool = ["mass_edit", "split_column", "add_column", "regexr_transform", "upper", "numeric", "date", "trim"]
            context = []
            eod_desc = False
            col_values = df[sel_col]
            num_rows = len(df)
            av_cols = df.columns.to_list() # current column names list 

            # TASK II: select operations
            sel_op = ''
            ops_history, functions_list = export_ops_list(project_id, st)
            if functions_list:
                functions_list = parse_text_transform(ops_history, functions_list)
                print(f'Applied operation history: {functions_list}')
            col_str = gen_table_str(df, num_rows=15, tg_col=sel_col) # only keep first 15 rows for operation selection
            print(f'Selected first {num_rows} rows for current table: {col_str}')

            # context-learn (full_chain_demo): how the previous operation are related to the current one
            # operation-learn (learn_ops.txt): when to select a proper operation 
            with open('prompts/learn_ops.txt', 'r')as f_learn_ops:
                dynamic_plan = f_learn_ops.read()
            with open('prompts/full_chain_demo.txt', 'r')as f_chain:
                f_chain_learn = f_chain.read()
            if functions_list:
                # if applied operations already, use this context info to help generate Next Operation
                prompt_ops_chain = dynamic_plan + f"""\n\n Based on table contents, Purpose and Operation Chain provided as following, generate the Explanations.\n"""\
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Operation Chain: ```{functions_list}``` 
                                    Explanations:
                                    """
                context, chain_exp = gen(prompt_ops_chain, context, model)
            else:
                context = context
                chain_exp = ""
            print(f'Try to explain the current chain: {chain_exp}')
            
            # TBD: Basic operations are allowed to be applied only once
            prompt_sel_ops = dynamic_plan +\
                f"""\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool} and output the operation name in ``` ```.\n"""\
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {sum_eod}
                                    Selected Operation: 
                                    Target column: {sel_col}
                                    """
            print(prompt_sel_ops)
            options_sel_op = {
                'temperature': 0.2
            }
            
            # TODO: Quality control
            while not is_valid_operation(col_values, sel_op) or not (sel_op in ops_pool):
                print(f'current selected operation: {sel_op}')
                print(f'The result of checking valid: {is_valid_operation(col_values, sel_op)}')
                if sel_op == '':
                    context, sel_op_desc = gen(prompt_sel_ops, context, model, options_sel_op)
                    sel_op = extract_exp(sel_op_desc, ops_pool)
                else:
                    context = []
                    prompt_sel_ops = """You are an expert in data cleaning and able to choose appropriate Operations to prepare the table in good format BEFORE addressing the Purpose. 
Note that the operation chosen should aim at making data be in BETTER SHAPE that can be used for the purpose instead of addressing the purpose directly.""" +  \
                         f"""\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool_update} and output the operation name in ``` ```.\n"""\
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {sum_eod}
                                    Selected Operation: 
                                    Target column: {sel_col}
                                    """
                    context, sel_op_desc = gen(prompt_sel_ops, context, model, options_sel_op)
                    sel_op = extract_exp(sel_op_desc, ops_pool)
            print(f'selected operation: {sel_op}')
            raise NotImplementedError

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
            _, eod_desc = gen(prompt_eod, [], model) #clear out context
            prompt_eod_desc_summarization = f"""please generate a one-sentence summarization and a one-sentence data cleaning objective for next operation according to the detailed data quality issue mentioned by **3.Assessing profiling results from four dimensions:** from the: \n{eod_desc}"""
            _, one_sent_eod_desc = gen(prompt_eod_desc_summarization, [], model)
            # Regular expression to extract the desired sentence
            # eod_pattern= r"Next operation:\s*(.*?)\."
            eod_pattern = r"\*\*Data Cleaning Objective:\*\*\s*(.*?)\."
            # Search for the pattern in the text
            eod_match = re.search(eod_pattern, one_sent_eod_desc, re.DOTALL)
            # Extract the matched sentence if found
            sum_eod = eod_match.group(1).strip() + '.' if eod_match else one_sent_eod_desc
            
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
                #   We choose to return all the related columns
                #  [city, zip] should work together to repair the data
                sel_cols_str = gen_table_str(sel_cols_df, num_rows=num_rows)
                prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output edits (a list of dictionaries) in ``` ``` ONLY.""" \
                                    +f"""
                                    /*
                                    {sel_cols_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    edits: 
                                    """
                print("prompts for generating edits:")
                print(prompt_sel_args)
                options = {
                    'temperature': 0.0,
                    'stop': ["Explanation:", "'','','',", "\n\n", ],
                    'num_predict': -1,
                    'top_p': 0.95,
                    'mirostat': 1 #0(default), 1(mirostat1),2(mirostat2)
                }
                context, edits_desc = gen(prompt_sel_args, context, model, options)
                edits_v = extract_exp(edits_desc)
            
                if edits_v:
                    edits_v = edits_v[0].replace("edits: ", "")
                    edits_v = parse_edits(edits_v)
                    mass_edit(project_id, column=sel_col, edits=edits_v)
                else: 
                    print('no selected arguments')
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
        tg_cols.pop(0)
        st += len(functions_list)
        print(f"remaining columns: {tg_cols}")
    # log_data["Operations"] = list(set(ops_data))
    print(f'The full operation chain: {ops_gen}')
    print(f'The whole process: {log_data}')
    return functions_list, log_data


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

def test_main():
    model = "llama3.1:8b-instruct-fp16"
    log_dir = "CoT.response"
    os.makedirs(log_dir, exist_ok=True)

    pp_f = 'purposes/test_data.csv'
    pp_df = pd.read_csv(pp_f)
    
    ds_file = "datasets/menu_data.csv"
    ds_name = "menu_test"
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
            wf_res, log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
        else:
            project_id = create_projects(project_name, ds_file)
            print(f"Project {project_name} creation finished.")
            wf_res, log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
        with open(f"{log_dir}/{ds_name}_{pp_id}_log_{timestamp_str}.txt", "w") as log_f:
            json.dump(log_data, log_f, indent=4)

if __name__ == '__main__':
    test_main()
    # main()
    