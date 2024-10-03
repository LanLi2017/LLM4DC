# LLM-based history update solution
import importlib.util
import inspect
from typing import List
import requests
import json
import re
import difflib
from collections import Counter
from spellchecker import SpellChecker

import pandas as pd
import ast

# from history_update_problem.call_or import export_rows
from call_or import *

import ollama
from ollama import Client
from ollama import generate

model = "llama3.1:8b-instruct-fp16" 
# ollama.pull(model)
# model = "llama3.1"

dynamic_plan = """

You are an expert in data cleaning and able to choose appropriate functions and arguments to prepare the data in good format and correct semantics. Available example demos to learn the data cleaning operations can be retrieved here:

split_column(): 
'''
If the table have the needed column but does not have the exact cell values to answer the question. In other words, the cell values from the column 
comprise the values to answer the question, we use **split_column()** to decompose the column for it. For example,
/*
col : Week | When | Kickoff | Opponent | Results; Final score | Results; Team record | Game site | Attendance
row 1 : 1 | Saturday, April 13 | 7:00 p.m. | at Rhein Fire | W 27–21 | 1–0 | Rheinstadion | 32,092
row 2 : 2 | Saturday, April 20 | 7:00 p.m. | London Monarchs | W 37–3 | 2–0 | Waldstadion | 34,186
row 3 : 3 | Sunday, April 28 | 6:00 p.m. | at Barcelona Dragons | W 33–29 | 3–0 | Estadi Olímpic de Montjuïc | 17,503
*/
Purpose: what is the date of the competition with highest attendance?
Operation: ```split_column```
Arguments: column: "When", separator: ","
Explanation: The question asks about the date of the competition with highest score. Each row is about one competition. We split the value from column "When" with separator ",", and create two new columns.
Output: April 13 | April 20 | April 28

'''

add_column():
'''
If the table does not have the needed column to answer the question, we use **add_column()** to add a new column for it. For example,
/*
col : week | when | kickoff | opponent | results; final score | results; team record | game site | attendance
row 1 : 1 | saturday, april 13 | 7:00 p.m. | at rhein fire | w 27–21 | 1–0 | rheinstadion | 32,092
row 2 : 2 | saturday, april 20 | 7:00 p.m. | london monarchs | w 37–3 | 2–0 | waldstadion | 34,186
row 3 : 3 | sunday, april 28 | 6:00 p.m. | at barcelona dragons | w 33–29 | 3–0 | estadi olímpic de montjuïc | 17,503
*/
Purpose: Return top 5 competitions that have the most attendance.
Operation: ```add_column```
Arguments: column: "attendance", expression: "value", new_column: "attendance number")
Explanation: We copy the value from column "attendance" and create a new column "attendance number" for each row.
Output: 32,092 | 34,186 | 17,503
'''

rename_column(): 
'''
If the table does not have the related column name to answer the question, we use **rename_column()** to find the most related column and rename the column with new, and more meaningful name. For example,
/*
col : Code | County | Former Province | Area (km2) | Population; Census 2009 | Capital
row 1 : 1 | Mombasa | Coast | 212.5 | 939,370 | Mombasa (City)
row 2 : 2 | Kwale | Coast | 8,270.3 | 649,931 | Kwale
row 3 : 3 | Kilifi | Coast | 12,245.9 | 1,109,735 | Kilifi
*/
Purpose: what is the total number of counties with a population in 2009 higher than 500,000?
Operation: ```rename_column```
Arguments: column: "Population; Census 2009", new_column: "Population"
Explanation: the question asks about the number of counties with a population in 2009 higher than 500,000. Each row is about one county. We rename the column "Population; Census 2009" as "Population".
Output: 939370 | 649311 | 1109735
'''

text_transform(): 
'''
If the question asks about the characteristics/patterns of cell values in a column, we use **text_transform()** to format and transform the items. For example,
/*
col : code | county | former province | area (km2) | population; census 2009 | capital
row 1 : 1 | mombasa | coast | 212.5 | 939,370 | mombasa (city)
row 2 : 2 | kwale | coast | 8,270.3 | 649,931 | kwale
row 3 : 3 | kilifi | coast | 12,245.9 | 1,109,735 | kilifi
*/
Purpose: Figure out the place that has a population in 2009 higher than 500,000.
Operation: ```text_transform```
Arguments: column: "Population; Census 2009", expression: "jython: return int(value)"
Explanation: For expression: "jython: return int(value)": value is cell values in the target column "population; census 2009", int() can transform value type into integers 
Output: 939370 | 649311 | 1109735

'''

mass_edit():
'''
If the question asks about items with the same value and the number of these items, we use **mass_edit()** to standardize the items. For example,
/*
col : LoanAmount | City     | State  | Zip 
row 1 : 30333    | Hon      | HI     |96814
row 2 : 149900   | HONOLULU | HI     | 96814 
row 3 : 148100   | Honolulu | HI     | 96814
row 4 : 334444   | CHI      | IL     | 60611
row 5 : 120      | urbana   | IL     | 61802
row 6 : 100000   | Chicagoo | IL     | 
*/
Purpose: Return how many cities are in the table.
Operation: ```mass_edit```
Arguments: column: "City", edits:[{'from': ['Hon', 'HONOLULU'], 'to': 'Honolulu'}, {'from': ['CHI', 'Chicagoo'], 'to': 'Chicago'}, {'from': ['urbana'], 'to': 'Urbana'}])
Explanation: Mispellings and different formats of data need to be revised. 
Output: Honolulu | Honolulu | Honolulu | Chicago | Urbana | Chicago

'''

remove_column():

'''
If the column contains too many missing values, to improve the data quality, we use **remove_column()** to drop the column. For example,
/*
col : rank | lane | player name| country | time  | player_name(preferred)
row 1 :  | 5 | olga tereshkova |  kaz    | 51.86 |
row 2 :  | 6 | manjeet kaur    |  ind    | 52.17 | NA
row 3 :  | 3 | asami tanno     |  jpn    | 53.04 |
*/
Purpose: return the player information, including both name and country 
Operation: ```remove_column```
Arguments: column: "player_name(preferred)"
Explanation: cell values in column player_name(preferred) are empty, therefore, we will remove it.

'''

reorder_rows():
'''
If the question asks about the order of items in a column, we use **reorder_rows()** to sort the items. For example,
/*
col : Position | Club | Played | Points | Wins | Draws | Losses | Goals for | Goals against | Goal Difference
row 1 : 1 | Malaga CF | 42 | 79 | 22 | 13 | 7 | 72 | 47 | +25
row 10 : 10 | CP Merida | 42 | 59 | 15 | 14 | 13 | 48 | 41 | +7
row 3 : 3 | CD Numancia | 42 | 73 | 21 | 10 | 11 | 68 | 40 | +28
*/
Purpose: what club placed in the last position?
Operation: ```reorder_rows```
Arguments: sort_by: "Position"
Explanation: the question asks about the club in the last position. Each row is about a club. We need to know the order of position from last the top. There is a column for position and the column name is Position. The datatype is Numerical.

'''

"""
exp_in_out = """
Data input before data cleaning:
/*
col : physical_description 
row 1 : CARD; 4.75X7.5;
row 2 : BROADSIDE; ILLUS; COL; 5.5X8.50; 
row 3 : BROADSIDE; ILLUS; COL; 3,5X7;
row 4 : CARD;ILLUS;5.25X8/25;
row 5 : 30x21cm folded; 30x42cm open
row 6 : CARD; ILLUS; 6 x 9.75 in.
row 7 : Booklet; 8.25 x 11.5 inches
*/

Expected data output after data cleaning:
/*
col : physical_description               | size              | unit
row 1 : CARD; 4.75X7.5;                  | 4.75X7.5          | 
row 2 : BROADSIDE; ILLUS; COL; 5.5X8.50; | 5.5X8.50          |
row 3 : BROADSIDE; ILLUS; COL; 3,5X7;    | 3.5X7             |      
row 4 : CARD;ILLUS;5.25X8/25;            | 5.25X8.25         |
row 5 : 30x21cm folded; 30x42cm open     | 30x21; 30x42      | cm
row 6 : CARD; ILLUS; 6 x 9.75 in.        | 6 x 9.75          | inches
row 7 : Booklet; 8.25 x 11.5 inches      | 8.25 x 11.5       | inches
*/
"""

map_ops_func = {
"core/column-split": split_column,
"core/column-addition": add_column,
"core/text-transform": text_transform,
"core/mass-edit": mass_edit,
"core/column-rename": rename_column,
"core/column-removal": remove_column,
"core/row-reorder": reorder_rows,
}


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


def gen_table_str(df, num_rows=3):
    # Sample the first 30 rows
    df = df.head(num_rows)
    # Prepend "row n:" to each row
    df.insert(0, 'Row', [f'row {i+1}:' for i in range(len(df))])
    # Convert the DataFrame to a Markdown string without the header
    rows_lines = [f"row {i+1}: | " + " | ".join(map(str, row)) + " |" for i, row in df.iterrows()]
    # rows_lines = [f"| " + " | ".join(map(str, row)) + " |" for _, row in df.iterrows()]
    # Add the column schema line
    column_names = " | ".join(df.columns)
    column_schema = f'col: | {column_names} |\n'
    # Combine the column schema with the DataFrame content
    table_str = column_schema + "\n".join(rows_lines)
    return table_str


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


def extract_exp(content):
    """This function is to extract python code from generated results"""
    match = re.search(r'```(.*?)```', content, re.DOTALL)
    if match:
        code_block = match.group(1).strip()
        code_block = code_block.replace('; ', '\n')
        return code_block
    else:
        print("No code block found.")
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


def exe_edits(strings):
    '''
    input a column of cell value, return clusters
    '''
    strings = [s for s in strings if s not in (None, "")]
    # Initialize a spell checker
    spell = SpellChecker()
    
    # Step 1: Correct spelling mistakes
    corrected_strings = [spell.correction(s) for s in strings]
    corrected_strings = [s for s in corrected_strings if s is not None]
    # Step 2: Group similar strings using difflib's get_close_matches
    clusters = []
    used_strings = set()

    for string in corrected_strings:
        # Skip strings that are already part of a cluster
        if string in used_strings:
            continue
        
        # Find close matches based on string similarity
        if string is not None:
            matches = difflib.get_close_matches(string, corrected_strings, n=len(strings), cutoff=0.7)
        
        # Mark these strings as used
        for match in matches:
            used_strings.add(match)
        
        # Step 3: Choose the most frequent correct spelling
        most_common = Counter(matches).most_common(1)[0][0]
        
        # Create a cluster dictionary
        clusters.append({
            'from': matches,
            'to': most_common.capitalize()  # Capitalize for proper formatting (e.g., "Chicago")
        })
    
    return clusters


if __name__ == "__main__":
    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    with open("prompts/eod.txt", 'r')as f0:
            eod_learn = f0.read()

    dc_obj = """ How many different events are recorded in the dataset?"""
    # dc_obj = """ How do the physical size of collected menus evolve during the 1900-2000 years?"""
    ops_pool = ["split_column", "add_column", "text_transform", "mass_edit", "rename_column", "remove_column"]
    log_f = open("CoT.response/llm_dcw.txt", "w")
    ops = [] # operation history 
    project_id = 2334363766799  
    df_init = export_intermediate_tb(project_id)

    ops = get_operations(project_id)
    op_list = [dict['op'] for dict in ops]
    functions_list = [map_ops_func[operation].__name__ for operation in op_list]
    if not functions_list:
        eod_flag = "False"
    else:
        prompt_eod = eod_learn + f"""
                                    \n\nBased on table contents and Objective provided as following, output Flag in ```` ```` without Explanations.
                                    /*
                                    {gen_table_str(df_init)}
                                    */
                                    Objective: {dc_obj}
                                    Flag:
                                    """
    
        context, eod_desc = gen(prompt_eod, [], log_f)
        print(eod_desc)
        eod_flag = extract_exp(eod_desc)
        print(eod_flag)
    
    print(eod_flag)
    while eod_flag == "False":
        context = []
        # The eod_flag is True.
        # Return current intermediate table
        df = export_intermediate_tb(project_id)
        tb_str = gen_table_str(df)
        av_cols = df.columns.to_list()

        # 1. LLM predict next operation: five op-demo and chain-of ops 
        # TASK I: select target column(s)
        with open("prompts/f_select_column.txt", 'r')as f:
            sel_col_learn = f.read()

        prompt_sel_col = sel_col_learn + f"""
                                        \n\nBased on table contents and purpose provided as following, output column names in a **Python List** without Explanations.
                                        /*
                                        {format_sel_col(df)}
                                        */
                                        Purpose: {dc_obj}
                                        Selected columns:
                                        """

        context, sel_col = gen(prompt_sel_col, context, log_f)

        # TASK II: select operations
        ops = get_operations(project_id)
        op_list = [dict['op'] for dict in ops]
        functions_list = [map_ops_func[operation].__name__ for operation in op_list]
        print(functions_list)
        if 'mass_edit' in functions_list:
            ops_pool.remove('mass_edit')
        print(f'current available operations: {ops_pool}')
        prompt_sel_ops = dynamic_plan + f""" Based on table contents and purpose provided as following, output Operation name in ``` ``` without Explanations. The available operations list: {ops_pool}"""\
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Operation: 
                                """

        context, sel_op = gen(prompt_sel_ops, context, log_f)
        print(sel_op)

        sel_op = sel_op.strip('`')
        
        while sel_op not in ops_pool:
            prompt_regen = f"""The selected operation is not found in {functions_list}. Please regenerate operation name for TASK II."""
            context, sel_op = gen(prompt_regen, context, log_f)
            sel_op = sel_op.strip('`')
        # TASK III: Learn function arguments (share the same context with sel_op)
        args = get_function_arguments('call_or.py', sel_op)
        args.remove('project_id')  # No need to predict project_id
        args.remove('column')
        print(f'Current args need to be generated: {args}')
        # prompt_sel_args = f"""<|begin_of_text|> Next predicted operation is {sel_op}"""
        with open(f'prompts/{sel_op}.txt', 'r') as f1:
            prompt_sel_args = f1.read()
        
        # update tb_str to use the full rows:
        tb_str = gen_table_str(df, num_rows=100)
        context = []
        if sel_op == 'split_column':
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output separator in " " without Explanations."""\
                               + f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: column: {sel_col}, separator: 
                                """
            context, sep = gen(prompt_sel_args, context, log_f)
            sel_args= {'column':sel_col, 'separator':sep}
            split_column(project_id, **sel_args)
        elif sel_op == 'add_column':
            # prompt_sel_args += prompt_exp_lr
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output expression and new_column in " " without Explanations."""\
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: column: {sel_col}, expression:, new_column:
                                """
            context, res_dict = gen(prompt_sel_args, context, log_f)
            sel_args = {'column': sel_col, 'expression': res_dict} 
            add_column(project_id, **sel_args)
        elif sel_op == 'rename_column':
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output new_column in " " without Explanations.""" \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: column: {sel_col}, new_column: 
                                """
            context, new_col = gen(prompt_sel_args, context, log_f)
            sel_args = {'column': sel_col, 'new_column': new_col}
            rename_column(project_id, **sel_args)
        elif sel_op == 'text_transform':
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output expression in " " without Explanations.""" \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: column: {sel_col}, expression: 
                                """
            context, exp = gen(prompt_sel_args, context, log_f)
            print(f'predicted expression: {exp}')
            sel_args = {'column': sel_col, 'expression': exp}
            text_transform(project_id, **sel_args)
        elif sel_op == 'mass_edit':
            # semi-automate.. python (edits) + LLM
            # print(f'selected column name: {sel_col}')
            sel_col = sel_col.strip("[]' ")
            # print(f'processed column name: {sel_col}')
            df_me = df[sel_col].dropna().tolist() # only input target column
            edits = exe_edits(df_me)
            sel_args = {'column': sel_col, 'edits': edits}
            mass_edit(project_id, **sel_args)
        elif sel_op == 'remove_column':
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output arguments column in " " without Explanations.""" \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: column: {sel_col}
                                """
            sel_args = {'column': sel_col}
            remove_column(project_id, **sel_args)
        elif sel_op == "reorder_rows":
            prompt_sel_args += """\n\nBased on table contents and purpose provided as following, output arguments sort_by in " " without Explanations.""" \
                                +f"""
                                /*
                                {tb_str}
                                */
                                Purpose: {dc_obj}
                                Arguments: sort_by: {sel_col}
                                """
            context, sort_col = gen(prompt_sel_args, context, log_f)
            sel_args = {'sort_by': sort_col}
            reorder_rows(project_id, **sel_args)
       
        # @TODO: Question: is Full_Chain_learn equal to eod_learn prompts?
        # with open("prompts/full_chain_demo.txt", 'r')as f2:
        #     full_chain_learn = f2.read()
        # prompt_full_chain = "Learn when to generate {{True}} for eod_flag and end the data cleaning functions generation:\n" + full_chain_learn
        
        # Re-execute intermediate table
        cur_df = export_intermediate_tb(project_id)

        # TASK VI:
        # Keep passing intermediate table and data cleaning objective, until eod_flag is True. End the iteration.
        iter_prompt = eod_learn + f"""
                                \n\nBased on table contents and Objective provided as following, output Flag in ```` ```` without Explanations.
                                /*
                                {gen_table_str(cur_df)}
                                */
                                Objective: {dc_obj}
                                Flag:
                                """
   
        context, eod_desc = gen(iter_prompt, [], log_f)
        eod_flag = extract_exp(eod_desc)
        print(eod_flag)
        print(f'LLMs believe current table is good enough to address objectives: {eod_flag}')

    # prompt += "Learn how to generate arguments for function add column: \n" + 
    log_f.close()
