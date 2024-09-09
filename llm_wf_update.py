# LLM-based history update solution
import importlib.util
import inspect
from typing import List
import requests
import json
import re

import pandas as pd

# from history_update_problem.call_or import export_rows
from call_or import *

import ollama
from ollama import Client
from ollama import generate

model = "llama3.1:8b-instruct-fp16" 
# ollama.pull(model)
# model = "llama3.1"
# model = 'stablelm-zephyr' 

# parser = argparse.ArgumentParser()
# parser.add_argument("--start", required=True, type=int)
# parser.add_argument("--end", required=True, type=int)
# parser.add_argument("--dry_run", default=False, action="store_true",
#     help="whether it's a dry run or real run.")
# parser.add_argument(
#     "--temperature", type=float, default=0.7,
#     help="temperature of 0 implies greedy sampling.")
# simple case: How many types of event in the menu table?
prep_learning = """
<|python_tag|>
class RefineProject:
    def text_transform(project_id, column, expression):
        '''
        OpenRefine's transformations can change {{column}} contents by applying {{expression}}:
        The default is GREL (General Refine Expression Language), Altertively, it accept python code.
        Example: text_transform(project_id, column="id", expression="return int(value)") 
        Using python code to transform values in column id as integer. 
        '''
        response = self.do_json('text-transform',
                                {
                                    'columnName': column,
                                    'expression': expression,
                                    'onError': on_error,
                                    'repeat': repeat,
                                    'repeatCount': repeat_count,
                                })
        return response

    def mass_edit(project_id, column, edits):
        '''
        Replacing data values with new values in {{column}} with new values, old and new values are defined in {{edits}}
        edits: [{'from': ['foo'], 'to': 'bar'}, {...}]
        Example: 
        mass_edit(project_id, "city", edits=[{'from':["Cicago"], 'to':"Chicago"}])
        Replace cell values in column city from spelling "Cicago" to "Chicago"
        '''
        edits = json.dumps(edits)
        response = self.do_json('mass-edit',
                                {
                                    'columnName': column, 'expression': expression, 'edits': edits})
        return response

    def add_column(project_id, column, new_column, expression):
        '''Add a new column named as {{new_column}} based on {{expression}}. if the expression is default, then copy-paste cell values from {{column}} to {{new_column}}
        using jython: followed the code if the expression is written in python.
        Example: add_column(project_id, "name", "full name", expression="jython:res=cells['first name'].value+','+cells['last name'].value\nreturn res" )'''
        if column_insert_index is None:
            column_insert_index = self.column_order[column] + 1
        response = self.do_json('add-column',
                                {
                                    'baseColumnName': column,
                                    'newColumnName': new_column,
                                    'expression': expression,
                                    'columnInsertIndex': column_insert_index,
                                    'onError': on_error})
        return response

    def split_column(project_id,column,separator):
        '''split {{column}} by {{separator}} to generate more new columns
        Example: split_column(project_id, "full_name", "," )
        Explain: cell values in column full_name are composite values{{first name, last name}} concatenated with ',', splitting column and we can extract
        first name and last name as the new columns.'''
        response = self.do_json('split-column',
                                {
                                    'columnName': column,
                                    'separator': separator,
                                    'mode': mode,
                                    'regex': regex,
                                    'guessCellType': guess_cell_type,
                                    'removeOriginalColumn': remove_original_column,
                                })
        return response

    def rename_column(project_id, column, new_column):
        '''Give old {{column}} name a more meaningful {{new_column}} name.
        Example: rename_column(project_id, 'column 1', 'city')
        Explain: the old column name {{column 1}} is updated as {{city}}, which becomes more meaningful.'''
        response = self.do_json('rename-column',
                                {
                                    'oldColumnName': column,
                                    'newColumnName': new_column,
                                })
        return response
        
    def remove_column(project_id, column):
        '''remove {{column}} if it is useless or of low quality, e.g., most cells are empty and cannot be inferred.
        Example: remove_column(project_id, 'currency')
        Explain: most of the cells in column {{currency}} are NA, so we remove this column.'''
        response = self.do_json('remove-column', {'columnName': column})
        return response
    
    def reorder_rows(project_id, sort_by=None):
        ''' reorder_rows if need to the results need to be retrieved through some order. e.g., what's the most/least/top N occurred (frequency)
        species' names.
        Example: reorder_rows(project_id, 'frequency'): sort the rows in project_id according to frequency
        if sort_by is not None:
            self.sorting = Sorting(sort_by)
        response = self.do_json('reorder-rows', {'sorting': self.sorting.as_json()})
        # clear sorting
        self.sorting = Sorting()
        return response
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

# map_func_fname = {
#     'split_column': 'prompts/split_column.txt' ,
#     'add_column': 'prompts/add_column.txt',
#     'text_transform': 'prompts/text_transform.txt',
#     'mass_edit': 'prompts/mass_edit.txt',
#     'rename_column':'prompts/rename_column.txt',
#     'remove_column': 'prompts/remove_column.txt',
# }

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
    return {
        "table_caption": table_caption,
        "columns": columns,
        "table_column_priority": col_priority
    }


def gen_table_str(df):
    # Sample the first 30 rows
    df = df.head(30)
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


def diff_check(func_name, old_df, new_df, target_col):
    """This function is using the diff of applied ops as a context to inspire
        LLMs to generate next function"""
    """Qs: which kind of diff refer to good cleaning function?"""
    # return:
    # column-level diff: {column-schema: }
    # cell-level diff: 
    if func_name=='text_transform':
         differences = {}
         len_df = len(new_df)
         assert len(old_df) == len(new_df)

         for i in range(len_df):
            old_value = old_df.iloc[i][target_col]
            new_value = new_df.iloc[i][target_col]
            if old_value != new_value:
                differences[i] = {old_value: new_value}
        
         prompt_changes = f"""The changes resulted by text_transform: a dictionary of pairs of old value and new value: {differences}"""
         return prompt_changes


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

def chunk_return(text):
    # Only capture the final conclusions. 
    results = re.findall(r'\b(True|False)\b', text)
    return results[-1]


if __name__ == "__main__":
    # Define EOD: End of Data Cleaning
    # Input:intermediate table; Example output format
    # Output: False/True
    __eod = """ 
            You are an expert in data cleaning theory and practices, you are able to recognize whether the data is 
            clean (high quality) enough for the provided objectives:
            The pipeline of evalauting whether a dataset is of good quality: 
            (1). Understand the data cleaning objectives that this dataset is instended to address. Ensure that the dataset is relevant and
            provides sufficient information to answer it.
            (2). Profiling the dataset, check dataset from column schema level and instance level;
            "what are the columns? Whether the column names are meaningful or not?" 
            "what are the distributions of data instances?" "are they clearly represented?"
            (3). Assess the profiling results from four dimensions as following: 
            - **accuracy**: Whether the dataset is free from obvious errors, inconsistencies, or biases
            - **relavance**: Whether it includes essential variables (target columns) to address the objectives.
            - **completeness**: Whether it has a reasonable sample size and contains enough data instances (not too many missing values)
            - **duplicates**: Whether the spellings are standardized, no same semantics but different representations exist
            (4) Return True or False according to results from (3), IF ONLY you are sure or return "True" for all dimensions: accuracy, relavance, completeness, and 
            duplicates, you would return True. Otherwise, you SHOULD summarize the conclusion as "False".
             """
    # TODO: ...give specific function failed example for evaluation check 
    __failed_op = """
        Data input:
        /*
        col : code | county | former province | area (km2) | population; census 2009 | capital
        row 1 : 1 | mombasA | coast | 212.5 | 939,370 | mombasa (city)
        row 2 : 2 | Kwale | coast | 8,270.3 | 649,931 | kwale
        row 3 : 3 | KILIFI| coast | 12,245.9 | 1,109,735 | kilifi
        */
        Purpose: Figure out how many counties are recorded in total.
        Rationale: cell values in target column county are in different formats required to be normalized.
       
        Failed Example I: Parsing Error
        ```
        def text_transform(column, expression):
            if column == "county":
                return eval(expression.replace("value", f"row['{column}']")).upper()
        ```
        Failed Reason: Operation structure does not follow required format. 
        (1).If only other auxilary functions are required, there is no need to start with def text_transform() again, write the function directly.
        (2). use arguments "value" to represent cell values in target column county. 
        (3). expression cannot be recognized.
        
        Failed Example II: Inappropriate Conversion
        ``` "jython: return int(value)" ```
        Failed Reason: For expression:"jython: return int(value)", it tries to convert the string type value in column county 
        to integer, which does not make sense. 
        
        Failed Example III: Wrong regular expression
        ```
        import re  
        pattern = re.compile(r'^\d+')
        match = pattern.match(value)
        if match:
            value = match.group(0)
            return int(value)   
        ```
        Failed Reason: For pattern generated in expression:"pattern = re.compile(r'^\d+')", it failed to capture the correct pattern in column county.
        Therefore, this operation will not be executable to transform the data. 

        """
    __ev_op = f"""
            Evaluation Instruction: This instruction is to teach you how to evaluate the performance of applied function: whether 
            the function correctly transforms the data.
            Checking the changes (dictionary type, every key-value pair represent:old value: new value) by different functions
            in different ways: 
            For text transform, the performance is good if new values are more consistent: same format, same semantics, less missing values,
            more correct spellings. Conversely, this function will decrease the data quality and should be reverted. Failed applied function examples
            can be found: {__failed_op}.
            """
    __op = __ev_op +\
            """
            Return True or False ONLY. NO EXPLANATIONS.
            Since you have selected one data cleaning function and generated arguments to transform the data at this step.
            Provided actual changes caused by this function,please refer to Failed examples to check whether it is correctly applied on the dataset. 
            The answer is important and you should be **strict** with the changes by function. 
            You CAN ONLY return True if the changes strongly show that new values are better than the old values.Otherwise, Return False.
            <|end_of_text|>
            """
    # Compare with using example in and out
    # __eod_exp = """ 
            # Return True or False ONLY. NO EXPLANATIONS.
            # Return True: If NO data preparation is needed anymore on the intermediate table: 
            # data values are in high quality to address the {{Data Cleaning Objective}}.
            # (similar to {{Expected data output after data cleaning}}). 
            # Otherwise, Return False.
            #  """
    # dc_obj = """ 
    #          The task is to figure out how the size of menus evolves over the decades. 
    #          For this task, you will use {{menu}} dataset from NYPL(New York Public Library). 
    #          Dataset Description: {{A mix of simple bibliographic description of the menus}},
    #          The relevant columns input: {{physical_description}}
    #          """
    dc_obj = """ 
             {{Data Cleaning Objective}}:
             The task is to figure out how many different events are recorded in the collected menus. 
             For this task, you will use {{menu}} dataset from NYPL(New York Public Library). 
             Dataset Description: {{A mix of simple bibliographic description of the menus}}.
             """
    log_f = open("CoT.response/llm_dcw.txt", "w")
    # fpath = "data.in/menu_llm.csv"
    ops = [] # operation history 
    project_id = 2334363766799  
    df_init = export_intermediate_tb(project_id)
    print("this is the dataset:")
    print(df_init)
    # df_init = pd.read_csv('data.in/menu_llm.csv')
    prompt_init = """<|begin_of_text|>""" + dc_obj + f""" intermediate table:{df_init} """\
                      + __eod + """<|end_of_text|>"""
                            # + exp_in_out \
   
    context, eod_desc = gen(prompt_init, [], log_f)
    eod_flag = chunk_return(eod_desc)
    print(eod_flag)

    while eod_flag == "False":
        context = []
        # The eod_flag is True.
        # Return current intermediate table
        df = export_intermediate_tb(project_id)
        av_cols = df.columns.to_list()

        # 1. LLM predict next operation: five op-demo and chain-of ops 
        # TASK I: select target column(s)
        with open("prompts/f_select_column.txt", 'r')as f:
            sel_col_learn = f.read()
        prompt_sel_col = "<|begin_of_text|> Learn how to select column based on given question:\n" + sel_col_learn
        sel_col_tb = format_sel_col(df)
        prompt_sel_col += "Table input:\n" + json.dumps(sel_col_tb)
        prompt_sel_col += f"Data cleaning objective: {dc_obj}"
        prompt_sel_col += f"""Available columns for chosen: {av_cols}.
                             TASK I: Step by step, Return one relevant column name(string) based on {{Data cleaning objective}} ONLY. NO EXPLANATIONS.
                             Example Return: column 1 <|end_of_text|>"""
        # prompt_sel_col += f"""Available columns for chosen: {av_cols}.
        #                      TASK I: Step by step, Return one relevant column name(string) based on {{Data cleaning objective}} ONLY.
        #                      Example Return: column 1 """
        context, sel_col = gen(prompt_sel_col, context, log_f)
        print(sel_col)
        while sel_col not in av_cols:
            prompt_regen = f"""The selected columns are not in {av_cols}. Please regenerate column name for TASK I."""
            context, sel_col = gen(prompt_regen, context, log_f)

        # TASK II: select operations
         
        prompt_sel_ops = """<|begin_of_text|> You are an expert in data cleaning and able to choose appropriate functions and arguments to prepare the data in good format
                and correct semantics. Available data cleaning functions include split_column (add more columns by splitting original composite values), 
                add_column (add new column), text_transform (apply expression to transform data), mass_edit (standardize data by replacing old values with new values), 
                rename_column (give more meaningful column names), remove_column."""+\
                "TASK II: Step by step, learn available python functions to process data in class RefineProject:" + prep_learning
        ops = get_operations(project_id)
        op_list = [dict['op'] for dict in ops]
        functions_list = [map_ops_func[operation].__name__ for operation in op_list]
        print(functions_list)
        prompt_sel_ops += f"Chain of operation history has been applied: {functions_list} ->\n"
        # prompt_sel_ops += f"Sample first 30 rows from the Intermediate Table: {gen_table_str(df)} \n"
        prompt_sel_ops += f"Data values in target column: {gen_col_str(df, sel_col)}"
        prompt_sel_ops += f"Data cleaning purpose: {dc_obj}"
        # prompt_sel_ops += """
        #                    Return one selected function name from Functions Pool of RefineProject ONLY. NO EXPLANATIONS.
        #                    Functions pool: split_column, add_column, text_transform, mass_edit, rename_column, remove_column.
        #                    This task is to make the data in a good quality that fit for {{Data cleaning purpose}}."""
        prompt_sel_ops += """
                           Functions pool: split_column, add_column, text_transform, mass_edit, rename_column, remove_column.
                           Step by step, return ONLY one selected function name from Functions Pool of RefineProject. <|end_of_text|>"""
        
        func_pool = ["split_column", "add_column", "text_transform", "mass_edit", "rename_column", "remove_column"]
        context, sel_op = gen(prompt_sel_ops, context, log_f)
        print(sel_op)
        print(f"selected function is {sel_op}")
        sel_op = sel_op.strip('`')
        
        while sel_op not in func_pool:
            prompt_regen = f"""The selected function is not found in {functions_list}. Please regenerate function name for TASK II."""
            context, sel_op = gen(prompt_regen, context, log_f)
            sel_op = sel_op.strip('`')
        # TASK III: Learn function arguments (share the same context with sel_op)
        args = get_function_arguments('call_or.py', sel_op)
        args.remove('project_id')  # No need to predict project_id
        args.remove('column')
        prompt_sel_args = f"""<|begin_of_text|> Next predicted function is {sel_op}"""
        # prompt_sel_args += f"Sample first 30 rows from the Intermediate Table: {gen_table_str(df)} \n"
        with open(f'prompts/{sel_op}.txt', 'r') as f1:
            sel_args_learn = f1.read()
        prompt_sel_args += f"""TASK III: Step by step, learn proper arguments based on intermediate table and data cleaning purpose:
                                {sel_args_learn}"""
        prompt_exp_lr = f"""
                        You are a professional python developer and can write a function to transform the data in proper
                        format. With the selected function and examples, please write a proper python function. 
                        """
        if sel_op == 'split_column':
            prompt_sel_args += f"""
                                Therefore, the answer is: split_column(column=, separator=?).
                                column SHOULD BE {sel_col}. What's the separator?
                                ONLY generate value for {{separator}}. <|end_of_text|>
                                """
            context, sep = gen(prompt_sel_args, context, log_f)
            sel_args= {'column':sel_col, 'separator':sep}
            split_column(project_id, **sel_args)
        elif sel_op == 'add_column':
            # prompt_sel_args += prompt_exp_lr
            prompt_sel_args += f"""
                                Therefore, the answer is: add_column(column=, expression=?, new_column=? ).
                                column SHOULD BE {sel_col}. 
                                Return format: A dictionary of expression and new_column.
                                No explanations. <|end_of_text|>
                                """
            context, res_dict = gen(prompt_sel_args, context, log_f)
            while not isinstance(res_dict, dict):
                prompt_sel_args += f"""Return format is incorrect, it should be a dictionary, keys: expression and new_column,
                                       Please regenerate the correct values for the provided keys. <|end_of_text|>"""
                context, res_dict = gen(prompt_sel_args, context, log_f)
            sel_args = {'column': sel_col}.update(res_dict) 
            add_column(project_id, **sel_args)
        elif sel_op == 'rename_column':
            prompt_sel_args += f"""
                                Therefore, the answer is: rename_column(column=, new_column=?).
                                column SHOULD BE {sel_col}. What's the new_column?
                                ONLY generate value for new_column. <|end_of_text|>
                                """
            context, new_col = gen(prompt_sel_args, context, log_f)
            sel_args = {'column': sel_col, 'new_column': new_col}
            rename_column(project_id, **sel_args)
        elif sel_op == 'text_transform':
            prompt_sel_args += prompt_exp_lr
            prompt_sel_args += f"""
                                Therefore, the answer is: text_transform(column=,expression=?). 
                                column SHOULD BE {sel_col}. What's the expression?
                                ONLY generate python code for expression. <|end_of_text|>
                                """
            context, exp = gen(prompt_sel_args, context, log_f)
            format_exp = extract_exp(exp)
            print(format_exp)
            while format_exp is False:
                print('regenerate....')
                context, exp = gen(prompt_sel_args, context, log_f)
                format_exp = extract_exp(exp)
                print('end')
            sel_args = {'column': sel_col, 'expression': f"{format_exp}"}
            text_transform(project_id, **sel_args)
        elif sel_op == 'mass_edit':
            prompt_sel_args += f"""
                                Therefore, the answer is: mass_edit(column=, edits=[?]).
                                column SHOULD BE {sel_col}. What's the edits?
                                ONLY generate value for the edits. <|end_of_text|>
                                """
            context, edits = gen(prompt_sel_args, context, log_f)
            sel_args = {'column': sel_col, 'edits': edits}
            mass_edit(project_id, **sel_args)
        elif sel_op == 'remove_column':
            prompt_sel_args += f"""
                                Therefore, the answer is: remove_column(column=).
                                column SHOULD BE {sel_col}.  <|end_of_text|>
                                """
            sel_args = {'column': sel_col}
            remove_column(project_id, **sel_args)
        elif sel_op == "reorder_rows":
            prompt_sel_args += f"""
                                Therefore, the answer is: reorder_rows(sort_by=?).
                                Sort by which column?
                                ONLY generate value for sort_by. <|end_of_text|>
                                """
            context, sort_col = gen(prompt_sel_args, context, log_f)
            sel_args = {'sort_by': sort_col}
            reorder_rows(project_id, **sel_args)
       
        with open("prompts/full_chain_demo.txt", 'r')as f2:
            full_chain_learn = f2.read()
        prompt_full_chain = "Learn when to generate {{True}} for eod_flag and end the data cleaning functions generation:\n" + full_chain_learn
        
        # Re-execute intermediate table
        cur_df = export_intermediate_tb(project_id)
        prompt_init_prov = f""" <|begin_of_text|> Understanding how the selected function and arguments perform on the dataset is important
                            for we will understand how the changes applied by the function and whether this function improve the
                            data quality to meet cleaning objectives. 
                            """
        changes = diff_check(sel_op, df, cur_df, sel_col)
        # Modify I: if number of changes==0: return False
        print(f"*********{changes}********")

        prompt_changes = prompt_init_prov + changes + __op
        # delete sel_args_learn
        # EOD Flag I: Does the selected function perform correctly on the dataset?
        context, eod_flag1 = gen(prompt_changes, [], log_f)
        print(f"LLMs believe the function is correctly applied: {eod_flag1}")
        
        # TODO: if False, revert function re-generate 

        # TASK V:
        # Keep passing intermediate table and data cleaning objective, until eod_flag is True. End the iteration.
        iter_prompt = + """<|begin_of_text|>""" + prompt_full_chain + dc_obj + f""" intermediate table:{cur_df} """\
                      + __eod + + """<|end_of_text|>"""
                            # + exp_in_out \
   
        context, eod_flag2 = gen(iter_prompt, [], log_f)
        print(f'LLMs believe current table is good enough to address objectives: {eod_flag2}')
        eod_flag = eod_flag1 and eod_flag2
        print(eod_flag)

    # prompt += "Learn how to generate arguments for function add column: \n" + 
    log_f.close()
