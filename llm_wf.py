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


def export_ops_list(project_id):
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
    # for row in csv_reader:
    #     rows.append(row)
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
        return most_freq_ref[0]
    else:
        # this is to extract arguments 
        matches = re.findall(r'```(.*?)```', content, re.DOTALL)
        if matches:
            code_blocks = [match.strip().replace('; ', '\n') for match in matches]
            return code_blocks
        else:
            print("No code blocks found.")
            return False


def gen(prompt, context, log_f, temp=0):
    r = generate(model=model, 
                 prompt=prompt, 
                 context=context,
                 options={'temperature': temp},
                stream=True
                )
    res=[]
    for part in r:
        response_part = part['response']
        log_f.write(response_part)
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


def wf_gen(project_id, log_f):
    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    with open("prompts/eod.txt", 'r')as f0:
            eod_learn = f0.read()

    num_votes = 3 # run gen() multiple times to generate end_of_dc decisions
    dc_obj = """ How many different types of event are recorded in the dataset?"""
    # dc_obj = """ How do the physical size of collected menus evolve during the 1900-2000 years?"""
    # ops_pool = ["mass_edit", "split_column", "add_column", "text_transform", "rename_column", "remove_column"]
    ops_pool = ["mass_edit", "text_transform"]
    print(ops_pool)

    functions_list = export_ops_list(project_id)
    print(f'Applied operation history: {functions_list}')
    eod_flag = "False" # initialize the end_of_dc_flag to start data_cleaning pipeline

    while eod_flag == "False":
        context = []
        eod_desc = False
        df = export_intermediate_tb(project_id) # Return current intermediate table
        tb_str = gen_table_str(df, num_rows=10)
        av_cols = df.columns.to_list()

        # TASK I: select target column(s)
        with open("prompts/f_select_column.txt", 'r')as f:
            sel_col_learn = f.read()

        prompt_sel_col = sel_col_learn + f"""
                                        \n\nBased on table contents and purpose provided as following, output column name in ``` ```.
                                        /*
                                        {format_sel_col(df)}
                                        */
                                        Purpose: {dc_obj}
                                        Selected column:
                                        """

        context, sel_col_desc = gen(prompt_sel_col, context, log_f)
        
        print(f'description of selected column: {sel_col_desc}')
        sel_col = extract_exp(sel_col_desc, refs=av_cols)
        print(f'selected column: {sel_col}')

        # TASK II: select operations
        sel_op = ''
        ops = get_operations(project_id)
        op_list = [dict['op'] for dict in ops]
        functions_list = [map_ops_func[operation].__name__ for operation in op_list]
        print(f'Applied operation history: {functions_list}')
        tb_str = gen_table_str(df, num_rows=15, tg_col=sel_col)
        print(f'Selected first {num_rows} rows for current table: {tb_str}')

        # context-learn (full_chain_demo): how the previous operation are related to the current one
        # operation-learn (learn_ops.txt): when to select a proper operation 
        with open('prompts/learn_ops.txt', 'r'))as f_learn_ops:
            dynamic_plan = f_learn_ops.read()
        with open('prompts/full_chain_demo.txt', 'r')as f_chain:
            f_chain_learn = f_chain.read()
        if functions_list:
            # if applied operations already, use this context info to help generate Next Operation
            prompt_ops_chain = dynamic_plan + f"""\n\n Based on table contents, Purpose and Operation Chain provided as following, generate the Explanations.\n"""\
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Operation Chain: ```{functions_list}``` 
                                Explanations:
                                """
            context, chain_exp = gen(prompt_ops_chain, context, log_f)
        else:
            context = context
            chain_exp = ""
        print(f'Try to explain the current chain: {chain_exp}')

        prompt_sel_ops = dynamic_plan +\
             f"""\n\n Based on table contents and Purpose provided as following, select a proper Operation from the {ops_pool} and output the operation name in ``` ```.\n"""\
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Selected Operation: 
                                """

        while not (sel_op in ops_pool):
            context, sel_op_desc = gen(prompt_sel_ops, context, log_f)
            print(f'+++++++++selected operation description++++++\n {sel_op_desc}')
            sel_op = extract_exp(sel_op_desc, ops_pool)
        print(f'selected operation: {sel_op}')

        # TASK III: Learn function arguments (share the same context with sel_op)
        # return first 15 rows for generating arguments [different ops might require different number of rows]
        tb_str = gen_table_str(df, num_rows=15, tg_col=sel_col)
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
                                    {tb_str}
                                    */

                                    Objective: {dc_obj}
                                    Flag: ```False```
                                    Explanations: 
                                    """
        _, eod_desc = gen(prompt_eod, [], log_f) #clear out context
        prompt_eod_desc_summarization = f"""please generate a one-sentence summarization and a one-sentence data cleaning objective for next operation according to the detailed data quality issue mentioned by **3.Assessing profiling results from four dimensions:** from the: \n{eod_desc}"""
        _, one_sent_eod_desc = gen(prompt_eod_desc_summarization, [], log_f)
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
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Current Operation Purpose: {sum_eod}
                                Separator: 
                                """
            context, sep_desc = gen(prompt_sel_args, context, log_f)
            sep = extract_exp(sep_desc)
            print(f'Predicted separator for operation split column: {sep}')
            sel_args= {'column':sel_col, 'separator':sep}
            split_column(project_id, **sel_args)
        elif sel_op == 'add_column':
            # prompt_sel_args += prompt_exp_lr
            prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output expression and new_column separately in ``` ```."""\
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Current Operation Purpose: {sum_eod}
                                Expression:, New_column:
                                """
            context, res_dict_desc = gen(prompt_sel_args, context, log_f)
            [exp, new_col] = extract_exp(res_dict_desc)
            print(f"Expression of add_column: {exp}, New column created: {new_col}")
            sel_args = {'column': sel_col, 'expression': exp, 'new_column': new_col} 
            add_column(project_id, column=sel_col, expression=exp, new_column=new_col)
        # elif sel_op == 'rename_column':
        #     prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output new_column in ``` ```.""" \
        #                         +f"""
        #                         /*
        #                         {tb_str}
        #                         */
        #                         Purpose: {dc_obj}
        #                         Current Operation Purpose: {sum_eod}
        #                         Arguments: column: {sel_col}, new_column: 
        #                         """
        #     context, new_col_desc = gen(prompt_sel_args, context, log_f)
        #     new_col = extract_exp(new_col_desc)
        #     sel_args = {'column': sel_col, 'new_column': new_col}
        #     rename_column(project_id, **sel_args)
        elif sel_op == 'text_transform':
            # tb_str = gen_table_str(df, num_rows=50, tg_col=sel_col)
            tb_str = gen_table_str(df, num_rows=30, tg_col=sel_col)
            prompt_sel_args += """\n\nBased on table contents, Purpose, and Current Operation Purpose provided as following, output expression in ``` ``` (Ensure the expression format statisifies ALL requirements in the **Check**). """ \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Current Operation Purpose: {sum_eod}
                                Expression: 
                                """
            print(tb_str)
            # print(f'updated prompt for selecting arguments: {prompt_sel_args}')
            context, exp_desc = gen(prompt_sel_args, context, log_f)
            print(f'Predicted expression description: {exp_desc}')
            exp = extract_exp(exp_desc)[0].replace('jython\n', 'jython:')+ '\nreturn value'
            print(f'********predicted expression: {exp}')
            text_transform(project_id, column=sel_col, expression=exp)
        elif sel_op == 'mass_edit':
            # We choose to return the whole column to give LLMs an overview of all cases
            tb_str = gen_table_str(df, num_rows=100, tg_col=sel_col)
            print(tb_str)
            prompt_sel_args += """\n\nBased on the table contents, Purpose, and Current Operation Purpose provided as following, output edits (a list of dictionaries) in ``` ``` ONLY, DO NOT add any extra comments or keywords.""" \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Current Operation Purpose: {sum_eod}
                                edits: 
                                """
            print("prompts for generating edits:")
            print(prompt_sel_args)
            context, edits_desc = gen(prompt_sel_args, context, log_f)
    
            # Find all matches of the pattern in the provided text
            try:
                print(f'Start extracting \n\n++++{edits_desc}++++')
                edits_v = extract_exp(edits_desc)[0].replace("edits: ", "")
                print(f'The extracted edits value: {edits_v}')
                edits = parse_edits(edits_v)
                print(f"++++parsed edits: {edits}")
            except:
                print(f'Something wrong with the extracting edits...')
                print(f'-------Predicted edits description: ++++{edits_desc}++++')
                prompt_sel_edits = """Incorrect format of edits, please regenerate the edits ONLY in ``` ```. """
                context, edits_desc = gen(prompt_sel_edits, context, log_f)

            mass_edit(project_id, column=sel_col, edits=edits)
        # elif sel_op == 'remove_column':
        #     prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output arguments column in ``` ```. """ \
        #                         +f"""
        #                         /*
        #                         {tb_str}
        #                         */
        #                         Purpose: {dc_obj}
        #                         Current Operation Purpose: {sum_eod}
        #                         Arguments: column: 
        #                         """
        #     context, rm_desc = gen(prompt_sel_args, context, log_f)
        #     rm_col = extract_exp(rm_desc, av_cols)
        #     sel_args = {'column': rm_col}
        #     remove_column(project_id, **sel_args)
        elif sel_op == "reorder_rows":
            prompt_sel_args += """\n\nBased on table contents, Purpose and Current Operation Purpose provided as following, output the value of Sort_by ONLY in ``` ```. """ \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Current Operation Purpose: {sum_eod}
                                Sort_by: {sel_col}
                                """
            context, sort_col_desc = gen(prompt_sel_args, context, log_f)
            sort_col = extract_exp(sort_col_desc)
            reorder_rows(project_id, sort_by=sort_col)
        
        # raise NotImplementedError
        # Re-execute intermediate table, retrieve current data cleaning workflow
        cur_df = export_intermediate_tb(project_id)
        functions_list = export_ops_list(project_id)

        # TASK VI:
        # Keep passing intermediate table and data cleaning objective, until eod_flag is True. End the iteration.
        iter_prompt = eod_learn + f"""
                                \n\nBased on table contents and Objective provided as following, output Flag in ``` ```.
                                /*
                                {gen_table_str(cur_df)}
                                */
                                Objective: {dc_obj}
                                Flag:
                                """
        eod_flag_list = []
        eod_desc_list = []
        for _ in range(num_votes):
            context, eod_desc = gen(iter_prompt, [], log_f)
            eod_flag = extract_exp(eod_desc, ['False', 'True'])
            print(f'What is the flag?: {eod_flag}')
            eod_flag_list.append(eod_flag)
            eod_desc_list.append(eod_desc)
        if any([x == "False" for x in eod_flag_list]):
            eod_flag  = "False"
            mask = [int(x == "False") for x in eod_flag_list]
            eod_desc = random.choice([value for value, m in zip(eod_desc_list, mask) if m == 1])
        else:
            eod_flag = "True"
            functions_list.append(eod_flag)
        print(f'Decision of end of data cleaning: {eod_flag}')
    
    print(f'The full operation chain: {functions_list}')
    log_f.close()
    return functions_list


def main():
    log_f = open("CoT.response/llm_dcw.txt", "w")
    project_id = 2098024566597
    wf_res = wf_gen(project_id, log_f)
    print(wf_res)


if __name__ == '__main__':
    main()
    