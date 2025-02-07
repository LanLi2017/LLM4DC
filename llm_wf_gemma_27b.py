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
            elif exp.startswith("value.toDate") or exp.startswith("value.toString"):
                functions_list[idx] = "date"
            elif exp.startswith("jython"):
                functions_list[idx] = "regexr_transform"
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


def parse_column_list(column_str):
    """
    Safely parse a string representation of a list into an actual list.
    """
    try:
        parsed_list = ast.literal_eval(column_str)
        if isinstance(parsed_list, list):
            return parsed_list
    except:
        pass
    return []

def generate_target_columns(df, purpose, av_cols, model):
    """
    Generates a valid list of target columns using an LLM.
    
    - Ensures tg_cols is non-empty.
    - Removes duplicate columns.
    - Filters columns that are not in av_cols.
    - Keeps generating until a valid list is obtained (within MAX_RETRIES).
    Returns:
        list: A list of selected target columns.
    """
    
    # Read prompt template
    with open("prompts/f_select_column.txt", 'r') as f:
        sel_col_learn = f.read()

    prompt_sel_col = sel_col_learn + f"""
    
    \n\nBased on table contents and Purpose provided below, output Selected columns as a list inside ``` ```. 
/*
{format_sel_col(df)}
*/
Purpose: {purpose}
Selected columns:
    """
    logging.info(f"#TASK I: Select target columns:\n\n{prompt_sel_col}")
    retries = 0

    tg_cols = []

    while not tg_cols or not set(tg_cols).issubset(av_cols):  # Ensure valid, non-empty tg_cols

        context = []
        context, sel_col_desc = gen(prompt_sel_col, context, model)
        logging.info(sel_col_desc)
        print(sel_col_desc)

        try:
            tg_cols = ast.literal_eval(sel_col_desc)  # Try direct parsing
        except:
            # Extract content inside triple backticks
            matches = re.findall(r'```(?:\w+)?\n?(.*?)\n?```', sel_col_desc, re.DOTALL)
            matches = [m.strip() for m in matches if m.strip()]

            if matches:
                freq_counter = Counter(matches)
                ext_res = freq_counter.most_common(1)[0][0]  # Get most frequent match
            else:
                extracted_list = extract_exp(sel_col_desc)
                ext_res = extracted_list[0] if extracted_list else ""
            
            if ext_res:
                clean_ext_res = ext_res.replace("python\n", "", 1).strip()

                # Ensure we properly extract a valid list
                if clean_ext_res.startswith("[") and clean_ext_res.endswith("]"):
                    tg_cols = parse_column_list(clean_ext_res)
                elif clean_ext_res.startswith('"[') and clean_ext_res.endswith(']"'):
                    fixed_ext_res = clean_ext_res.strip('"')
                    tg_cols = parse_column_list(fixed_ext_res)
                else:
                    logging.warning(f"Invalid extracted result: {clean_ext_res}")
                    tg_cols = []
            else:
                tg_cols = []  # Ensure tg_cols is always defined

        # Ensure tg_cols is a valid list
        if not isinstance(tg_cols, list):
            tg_cols = []

        # Remove duplicates and filter out invalid columns
        tg_cols = [col for col in set(tg_cols) if col in av_cols]

        retries += 1
        if not tg_cols or not set(tg_cols).issubset(av_cols):
            logging.warning(f"Retry {retries}: Could not extract valid target columns. Regenerating...")
    return tg_cols

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


def convert_python_to_jython(python_code):
    """
    Converts a given Python regex-based transformation snippet into Jython-compatible code.

    Args:
        python_code (str): The Python code snippet.

    Returns:
        str: Jython-compatible code snippet.
    """
    # Ensure input code is stripped of extra spaces
    python_code = python_code.strip()
    # Remove Python-specific imports
    python_code = re.sub(r"^\s*import re\s*", "", python_code, flags=re.MULTILINE)
    # Remove comments (full-line comments and inline comments)
    python_code = re.sub(r"^\s*#.*", "", python_code, flags=re.MULTILINE)  # Full-line comments
    python_code = re.sub(r"\s+#.*", "", python_code)  # Inline comments
    # Ensure "value" is used correctly
    python_code = re.sub(r"\bvalue\s*=", "value =", python_code)

    # Convert f-strings to Jython-compatible string formatting
    python_code = re.sub(r'f\"(.*?)\"', r'"%s" % (\1)', python_code)

    # Ensure regex function names are correctly formatted for Jython
    python_code = re.sub(r"re\.(search|match|sub|compile)\(", r"re.\1(", python_code)

    # Remove any unintended code blocks (` ```python ... ``` `)
    python_code = re.sub(r"```python|```", "", python_code).strip()

    # Ensure return statement is present at the end
    if not re.search(r"\breturn\b", python_code):
        python_code += "\nreturn value"

    # Format final Jython code
    jython_code = "jython:import re\n" + python_code.strip()

    return jython_code


def wf_gen(project_id, log_data, model, logging, purpose):
    df = export_intermediate_tb(project_id) # Return current intermediate table
    av_cols = df.columns.to_list() # current column schema 
    ops_gen = {}
    ops_data = []
    errors = []
    context =[]
    # TASK I: select target column(s)
    tg_cols = generate_target_columns(df, purpose, av_cols, model)
    print(f"Final target columns: {tg_cols}")

    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    with open("prompts/eod.txt", 'r')as f0:
        eod_learn = f0.read()

    num_votes = 3 # run gen() multiple times to generate end_of_dc decisions
    
    while tg_cols:
        ops_pool = ["upper", "trim", "mass_edit", "regexr_transform", "numeric", "date"]
        eod_flag = "False"
        sel_col = tg_cols[0]
        print(f'Current selected column: {sel_col} from the target column list: {tg_cols}')
        sum_eod = f"Generate proper operations to improve accuracy, completeness, conciseness of the column: {sel_col}"
        # st: start step id, [st:] is to chunk the functions_list on current sel_col ONLY
        st = 0
        count_empty = 0
        options_sel_op = {
                    'temperature': 0.1,
                    # 'stop': ["\n\n\n\n"],
                    'num_predict': -1,
                    'top_k': 60,
                    'top_p': 0.95,
                    'mirostat': 1 #0(default), 1(mirostat1),2(mirostat2)
                }
        while eod_flag == "False":

            # ```upper```,  ```trim```, ```add_column```, ```split_column```, ```mass_edit```,  ```regexr_transform```, , ```numeric```, and ```date```
            # ops_pool = ["upper", "trim", "add_column", "split_column", "mass_edit", "regexr_transform", "numeric", "date"]
            context = []
            eod_desc = False
            # Clean the column one by one
            df = export_intermediate_tb(project_id) # Return current intermediate table
            sel_cols_df = df[tg_cols]
            col_values = df[sel_col]
            num_rows = len(df)
            av_cols = df.columns.to_list() # current column names list 

            # TASK II: select operations
            sel_op = None
            ops_history, functions_list = export_ops_list(project_id, st)
            if functions_list:
                functions_list = parse_text_transform(ops_history, functions_list)
                print(f'Applied operation history: {functions_list}')
                # if 'trim' in functions_list:
                #     ops_pool = [op_name for op_name in ops_pool if op_name!='trim']
            col_str = gen_table_str(df, num_rows=15, tg_col=sel_col) # only keep first 15 rows for operation selection
            # TBD: how many rows we will show here?
            tb_str = gen_table_str(sel_cols_df, num_rows=30)

            # operation-learn (learn_ops_.txt): when to select a proper operation 
            with open('prompts/learn_ops_.txt', 'r')as f_learn_ops:
                learn_ops = f_learn_ops.read()
            
            prompt_sel_ops = learn_ops +\
                             f"""\
\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool} and output the operation name in ``` ```.\n"""\
                             +f"""\
/*
{tb_str}
*/
Purpose: {purpose}
Target column: {sel_col}
Explanation: {sum_eod}
Selected Operation: 
                              """
            print("----start-------")
            print(prompt_sel_ops)
            logging.info(f"#TASK II: select operations: \n\n {prompt_sel_ops}")  
            options_sel_op = {
                'temperature': 0.3,
                'stop': ["\n\n\n\n"],
                'num_predict': -1,
                'top_k': 60,
                'top_p': 0.95,
                'mirostat': 1  # 0 (default), 1 (mirostat1), 2 (mirostat2)
            }

            count_empty = 0
            sel_op = None

            while not sel_op or sel_op not in ops_pool:
                context, sel_op_desc = gen(prompt_sel_ops, context, model, options_sel_op)
                logging.info(f"\nGenerated operation description:\n{sel_op_desc}")
                
                sel_op = extract_exp(sel_op_desc, ops_pool)
                
                if sel_op in ops_pool:
                    break
                
                count_empty += 1
                logging.warning(f"Invalid operation selected. Retrying... (Attempt {count_empty})")

            print(f"Selected operation: {sel_op}")

            with open(f'prompts/{sel_op}.txt', 'r') as f1:
                prompt_sel_args = f1.read()

            # Prepare the operation purpose
            prompt_eod = eod_learn + f"""\
\n\nBased on table contents, Objective, and Flag provided as following, output Explanations.
/*
{col_str}
*/

Purpose: {purpose}
Flag: ```False```
Explanations: 
                                            """
            _, eod_desc = gen(prompt_eod, [], model, {'temperature': 0.2}) #clear out context
            prompt_eod_desc_summarization = f"""please generate a one-sentence summarization and a one-sentence data cleaning objective for next operation according to the detailed data quality issue mentioned by **3.Assessing profiling results from four dimensions:** from the: \n{eod_desc}"""
            _, one_sent_eod_desc = gen(prompt_eod_desc_summarization, [], model, {'temperature': 0.2, 'top_p': 0.95})
            # Regular expression to extract the desired sentence
            # eod_pattern= r"Next operation:\s*(.*?)\."
            print(one_sent_eod_desc)
            logging.info(f'data cleaning objectives: {one_sent_eod_desc}')
            eod_pattern = r"\*\*Data Cleaning Objective:\*\*\s*(.*?)\."
            # Search for the pattern in the text
            eod_match = re.search(eod_pattern, one_sent_eod_desc, re.DOTALL)
            # Extract the matched sentence if found
            sum_eod = eod_match.group(1).strip() + '.' if eod_match else one_sent_eod_desc
            print(sum_eod)
                
            # >>>>Start Arguments Generation>>>>
            context = []
            if sel_op == 'regexr_transform':
                # tb_str = gen_table_str(df, num_rows=50, tg_col=sel_col)
                col_str = gen_table_str(df, num_rows=30, tg_col=sel_col)
                prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output expression in ``` ```. """ \
                                    +f"""\
/*
{col_str}
*/
Purpose: {purpose}
Current Operation Purpose: {sum_eod}
Python Expression:
                                    """
                # print(f'updated prompt for selecting arguments: {prompt_sel_args}')
                context, py_exp = gen(prompt_sel_args, context, model)
                # exp = extract_exp(exp_desc)[0].replace('jython\n', 'jython:')+ '\nreturn value'
                exp = convert_python_to_jython(py_exp)
                logging.info(f"#TASK III: generate regexr arguments: \n\n {exp}")
                print(f'********predicted expression: {exp}')
                text_transform(project_id, column=sel_col, expression=exp)
            elif sel_op == 'numeric':
                text_transform(project_id, column=sel_col, expression="value.toNumber()")
            elif sel_op == 'date':
                text_transform(project_id, column=sel_col, expression="value.toDate()")
                # col_str = gen_table_str(df, num_rows=num_rows, tg_col=sel_col)
                # print(col_str)
#                 prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output Expression in ``` ```."""\
#                                 + f"""\n
# /*
# {col_str}
# */
# Purpose: {purpose}
# # Current Operation Purpose: {sum_eod}
# # Expression: 
# # """
#                 context, date_desc = gen(prompt_sel_args, context, model)
#                 print(f'Generated description for date: {date_desc}')
#                 match = re.search(r'```(?:sql|.*?)\s*(value\.toDate\(.*?\))\s*```', date_desc, re.DOTALL)
#                 if match:
#                     date_exp = match.group(1)
#                     print(f'Generated arguments for date: {date_exp}')
#                     text_transform(project_id, column=sel_col, expression=date_exp)
#                 else:
#                     # Try to capture expressions that start with `value.toString(...)`
#                     alt_match = re.search(r'```(?:sql|.*?)\s*(value\.toString\(.*?\))\s*```', date_desc, re.DOTALL)
                    
#                     if alt_match:
#                         # Modify expression to use `value.toDate().toString("yyyy-MM-dd")`
#                         date_exp = 'value.toDate().toString("yyyy-MM-dd")'
#                         print(f'Converted value.toString(...) to: {date_exp}')
#                         text_transform(project_id, column=sel_col, expression=date_exp)
#                     else:
#                         pass
            elif sel_op == 'trim':
                text_transform(project_id, column=sel_col, expression="value.trim()")
            elif sel_op == 'upper':
                text_transform(project_id, column=sel_col, expression="value.toUppercase()")
            elif sel_op == 'mass_edit':
                # sel_cols_str = gen_table_str(sel_cols_df, num_rows=num_rows)
                # sum_edo = sum_eod.replace('\n', ' ')
                col_str = gen_table_str(df, num_rows=num_rows, tg_col=sel_col)
                # print(col_str)
                prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output edits (a list of dictionaries) in ``` ```. DO NOT add any comments in the list!"""\
                                + f"""\n
/*
{col_str}
*/
Purpose: {purpose}
Current Operation Purpose: {sum_eod}
edits: 
"""
                print("prompts for generating edits:")
                print(prompt_sel_args)
                options = {
                    'temperature': 0.2,
                    'stop':['\n\n\n']
                }
                try:
                    context, edits_desc = gen(prompt_sel_args, context, model, options)
                    edits_v = extract_exp(edits_desc)
                    print(f'descriptions for edits: \n\n {edits_desc}')
                    logging.info(f"#TASK III: generate mass_edit arguments: \n\n {edits_desc}")
                
                    if edits_v:
                        edits_v = edits_v[0].replace("edits: ", "")
                        edits_v = parse_edits(edits_v)
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
            thread_length = 5 # the longest number of steps on a single column 12/8
            if any([x == "True" for x in eod_flag_list]) or len(functions_list)>thread_length:
                eod_flag = "True"
                ops_data += functions_list # appending the operations if done...
                ops_gen[sel_col] = functions_list #TODO... the functions_list are the whole...    
                ops_gen['Error_Running'] = errors
            else:
                eod_flag  = "False"
                mask = [int(x == "False") for x in eod_flag_list]
                eod_desc = random.choice([value for value, m in zip(eod_desc_list, mask) if m == 1])
                logging.info(f"#TASK IV: data quality inspection: \n\n {eod_desc}")
            if count_empty >= 5:
                eod_flag = "True"
                
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


def test_main():
    # ollama.pull(model)
    # models = [
    # "llama3.1:8b-instruct-fp16",
    # "LLama3.3(70b)"
    # ]
    # ollama run deepseek-r1:8b
    model = "gemma2:27b"
    # model = "llama3.1:8b-instruct-fp16"
    # ollama.pull(model)
    model_name = model.split(':')[0]

    # ollama.pull(model)
    log_dir = f"CoT.response/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    # pp_f = 'purposes/queries.csv'
    # pp_f = 'purposes/all_purposes.csv'
    pp_f = 'purposes/pp_rerun.csv'
    pp_df = pd.read_csv(pp_f)

    ds_dir = f"{log_dir}/datasets_llm"
    os.makedirs(ds_dir, exist_ok=True)

    recipe_dir = f"{log_dir}/recipes_llm"
    os.makedirs(recipe_dir, exist_ok=True)

    ops_dir = f"{log_dir}/operation"
    os.makedirs(ops_dir, exist_ok=True)

    logging_dir = f"{log_dir}/logging"
    os.makedirs(logging_dir, exist_ok=True)
    
    # ds_file = "datasets/menu_data.csv"
    # ds_name = "menu_test"
    # 47, 105,106,112,144
    # 46-47, 92-94, 99-100, 131-132
    for index, row in pp_df.iloc[23:24].iterrows():
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
        # project_name = f"{ds_name}_{pp_id}_{timestamp_str}"
        #TODO: logging file name 
        logging_name = f"{logging_dir}/{model_name}_{ds_name}_{pp_id}.log"
        logging.basicConfig(filename=logging_name, level=logging.INFO) # TODO: change filename 
        
        project_name = f"{model_name}_{ds_name}_p{pp_id}"
        print(project_name)
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
            logging.info(f'Project {project_name} already exists!')
            print(project_name)
            project_id = get_project_id(project_name)
            ops_history, funcs = export_ops_list(project_id)
            log_data = wf_gen(project_id, log_data, model,logging, purpose=pp_v)
        else:
            project_id = create_projects(project_name, ds_file)
            print(f"Project {project_name} creation finished.")
            logging.info(f"Project {project_name} creation finished.")
            log_data = wf_gen(project_id, log_data, model, logging, purpose=pp_v)
        # with open(f"{log_dir}/{ds_name}_{pp_id}_log_{timestamp_str}.txt", "w") as log_f:
        #     json.dump(log_data, log_f, indent=4)

        #TODO: annotated operation file
        with open(f"{log_dir}/operation/{ds_name}_{pp_id}_log.txt", "w") as log_f:
            json.dump(log_data, log_f, indent=4)
        
        # download dataset 
        df = export_intermediate_tb(project_id)
        ds_path = f"{ds_dir}/{project_name}.csv"
        df.to_csv(ds_path)
        
        # download recipes 
        data = get_operations(project_id)
        recipe_path = f"{recipe_dir}/{project_name}.json"
        with open(recipe_path, "w") as workflow:
            json.dump(data, workflow, indent=4)  # `indent=4` adds pretty formatting
    
    # Download all the prepared datasets
    #TODO: change the dataset and workflow folder name
    # pull_datasets(model_name)
    # pull_recipes(model_name)

if __name__ == '__main__':
    # pull_recipes()
    # pull_datasets()
    test_main()