
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