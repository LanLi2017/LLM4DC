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
"core/column-removal": remove_column,
"core/row-reorder": reorder_rows,
}


def export_ops_list(project_id, st=0):
    ops = get_operations(project_id)
    op_list = [dict['op'] for dict in ops]
    functions_list = [map_ops_func[operation].__name__ for operation in op_list]
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


def gen_col_str(df, col_name:str):
    column_values = df[col_name].astype(str).tolist()
    col_string = ' '.join(column_values)
    
    return col_string


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
    print(content)
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
        print(most_freq_ref)
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
            print("No code blocks found.")
            return False


def gen(prompt, context, model, temp=0):
    r = generate(model=model, 
                 prompt=prompt, 
                 context=context,
                 options={'temperature': temp},
                stream=True
                )
    res=[]
    for part in r:
        response_part = part['response']
        res.append(response_part)

        if part['done'] is True:
            return part['context'], ''.join(res)


# parse edits by LLMs into a list
def parse_edits(raw_string):
    # Remove newlines and spaces
    raw_string = raw_string.replace('\n', '').strip()
    
    # Parse the string using ast.literal_eval
    parsed_edits = ast.literal_eval(raw_string)
    
    return parsed_edits


def wf_gen(project_id, log_data, model, purpose):
    df = export_intermediate_tb(project_id) # Return current intermediate table
    tb_str = gen_table_str(df, num_rows=10)
    av_cols = df.columns.to_list() # current column schema 
    ops_gen = {}
    ops_data = []
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
    ops_pool = ["mass_edit", "split_column", "add_column", "text_transform"]
    eod_flag = "False" # initialize the end_of_dc_flag to start data_cleaning pipeline
    
    while tg_cols:
        sel_col = tg_cols[0]
        # st: start step id, [st:] is to chunk the functions_list on current sel_col ONLY
        st = 0
        # Clean the column one by one
        while eod_flag == "False":
            context = []
            eod_desc = False
            df = export_intermediate_tb(project_id) # Return current intermediate table
            num_rows = len(df)
            total_col_str = gen_table_str(df, num_rows, tg_col=sel_col) # no chunk
            av_cols = df.columns.to_list() # current column names list 

            # TASK II: select operations
            sel_op = ''
            functions_list = export_ops_list(project_id, st)
            print(f'Applied operation history: {functions_list}')
            num_rows = 15 # only keep top 15 rows
            col_str = gen_table_str(df, num_rows=num_rows, tg_col=sel_col)
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

            prompt_sel_ops = dynamic_plan +\
                f"""\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool} and output the operation name in ``` ```.\n"""\
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Selected Operation: 
                                    """

            while not (sel_op in ops_pool):
                context, sel_op_desc = gen(prompt_sel_ops, context, model)
                print(f'+++++++++selected operation description++++++\n {sel_op_desc}')
                sel_op = extract_exp(sel_op_desc, ops_pool)
            print(f'selected operation: {sel_op}')
            raise NotImplementedError

            # TASK III: Learn function arguments (share the same context with sel_op)
            # return first 15 rows for generating arguments [different ops might require different number of rows]
            args = get_function_arguments('call_or.py', sel_op)
            args.remove('project_id')  # No need to predict project_id
            args.remove('column')
            print(f'Current args need to be generated: {args}')
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
            eod_pattern= r"Next operation:\s*(.*?)\."
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
            elif sel_op == 'text_transform':
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
                print(col_str)
                # print(f'updated prompt for selecting arguments: {prompt_sel_args}')
                context, exp_desc = gen(prompt_sel_args, context, model)
                print(f'Predicted expression description: {exp_desc}')
                exp = extract_exp(exp_desc)[0].replace('jython\n', 'jython:')+ '\nreturn value'
                print(f'********predicted expression: {exp}')
                text_transform(project_id, column=sel_col, expression=exp)
            elif sel_op == 'mass_edit':
                # We choose to return the whole column to give LLMs an overview of all cases
                col_str = gen_table_str(df, num_rows=num_rows, tg_col=sel_col)
                print(col_str)
                prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output edits (a list of dictionaries) in ``` ``` ONLY, DO NOT add any extra comments or keywords.""" \
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    edits: 
                                    """
                print("prompts for generating edits:")
                print(prompt_sel_args)
                context, edits_desc = gen(prompt_sel_args, context, model)
        
                # Find all matches of the pattern in the provided text
                try:
                    edits_v = extract_exp(edits_desc)[0].replace("edits: ", "")
                    edits = parse_edits(edits_v)
                except:
                    prompt_sel_edits = """Incorrect format of edits, please regenerate the edits ONLY in ``` ```. """
                    context, edits_desc = gen(prompt_sel_edits, context, model)

                mass_edit(project_id, column=sel_col, edits=edits)
            elif sel_op == "reorder_rows":
                prompt_sel_args += """\n\nBased on table contents, Purpose and Current Operation Purpose provided as following, output the value of Sort_by ONLY in ``` ```. """ \
                                    +f"""
                                    /*
                                    {col_str}
                                    */
                                    Purpose: {purpose}
                                    Current Operation Purpose: {sum_eod}
                                    Sort_by: {sel_col}
                                    """
                context, sort_col_desc = gen(prompt_sel_args, context, model)
                sort_col = extract_exp(sort_col_desc)
                reorder_rows(project_id, sort_by=sort_col)
            
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
            functions_list = export_ops_list(project_id, st)
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
                context, eod_desc = gen(iter_prompt, [], model, temp=0.7)
                eod_flag = extract_exp(eod_desc, ['False', 'True'])
                eod_flag_list.append(eod_flag)
                eod_desc_list.append(eod_desc)
            if any([x == "False" for x in eod_flag_list]):
                eod_flag  = "False"
                mask = [int(x == "False") for x in eod_flag_list]
                eod_desc = random.choice([value for value, m in zip(eod_desc_list, mask) if m == 1])
            else:
                eod_flag = "True"
                ops_data += functions_list # appending the operations if done...
                ops_gen[sel_col] = functions_list #TODO... the functions_list are the whole...
            print(f'Decision of end of data cleaning on column {sel_col}: {eod_flag}')
        tg_cols.pop(0)
        st += len(functions_list)
        print(f"remaining columns: {tg_cols}")
    print(f'The full operation chain: {ops_gen}')
    log_data["Operations"] = list(set(ops_data))
    return functions_list, log_data


def create_projects(project_name, ds_fp):
    _, proj_id = create_project(data_fp=ds_fp, project_name=project_name)
    return proj_id

def main():
    model = "llama3.2"
    pp_par_folder = "purposes"
    purpose_file = ["menu_about", "ppp_about", "dish_about", "chi_food_inspect_about"]
    pp_paths = [f"{pp_par_folder}/{file}.csv" for file in purpose_file]

    ds_par_folder = "datasets"
    ds_file = ["menu_data", "ppp_data", "dish_data", "chi_food_data"]
    ds_paths = [f"{ds_par_folder}/{file}.csv" for file in ds_file]

    pp_f = pp_paths[0]
    ds = ds_paths[0]
    ds_name = ds_file[0]
    pp_df = pd.read_csv(pp_f)
    pp_v = pp_df.iloc[-1] ["Purposes"]
    pp_id = int(pp_df.iloc[-1]["ID"])
    project_name = f"{ds_name}_{pp_id}"
    print(project_name)
    ds_name = ds_file[0]
    # Test the last purpose and use the purpose id as the project name
    project_id = int(create_projects(project_name, ds))
    # project_id = 2109337273503
    log_dir = "CoT.response"
    os.makedirs(log_dir, exist_ok=True)
    # log_file = open(, "w")
    # Initialize empty log data
    log_data = {
        "ID": pp_id,
        "Purposes": pp_v,
        "Columns": [],
        "Operations": []
    }
    wf_res, log_data = wf_gen(project_id, log_data, model, purpose=pp_v)
    print(log_data)

    with open(f"{log_dir}/{project_name}.txt", "w") as log_f:
        json.dump([log_data], log_f, indent=4)


if __name__ == '__main__':
    main()
    