import pandas as pd
import numpy as np
from dateutil.parser import parse

def retrieve_tg_cols(tg_cols_fp="/projects/bces/lanl2/LLM4DC/evaluation/target_columns_list.csv"):
    id_tg_cols = {}
    tg_df = pd.read_csv(tg_cols_fp)
    result_dict = tg_df.set_index('ID')['tg_columns'].to_dict()
    return result_dict


def is_iso_format_date(value: str) -> bool:
    """Check if a string follows an ISO 8601 date format."""
    try:
        parse(value)  # Attempt to parse the string as a date
        return True
    except (ValueError, TypeError):
        return False
    

def average_match_ratio(gd_df: pd.DataFrame, pred_df: pd.DataFrame, tg_columns: str) -> float:
    match_ratios = []
    tg_columns = tg_columns.split(',')
    target_columns = [item.strip() for item in tg_columns]

    for column in target_columns:
        match_count = 0
        non_null_matches = np.where(gd_df[column].notnull() & pred_df[column].notnull())[0]

        for idx in non_null_matches:
            gd_values = gd_df[column][idx]
            pred_values = pred_df[column][idx]

            # Ensure both values are strings before checking for dates
            if isinstance(gd_values, str) and isinstance(pred_values, str):
                if is_iso_format_date(gd_values) and is_iso_format_date(pred_values):
                    # Case-sensitive match for dates
                    if gd_values == pred_values:
                        match_count += 1
                    else:
                        print(f"Date Mismatch: {gd_values} <--> {pred_values}")
                else:
                    # Case-insensitive comparison for non-date strings
                    if gd_values.upper() == pred_values.upper():
                        match_count += 1
                    else:
                        print(f"String Mismatch: {gd_values} <--> {pred_values}")
            elif gd_values == pred_values:
                match_count += 1

        match_ratio = match_count / len(gd_df)
        match_ratios.append(match_ratio)

    average_ratio = sum(match_ratios) / len(target_columns)
    return average_ratio


# def average_match_ratio(gd_df: pd.DataFrame, pred_df: pd.DataFrame, tg_columns: str) -> float:
#     match_ratios = []
#     tg_columns = tg_columns.split(',')
#     if type(tg_columns) == list:
#         target_columns = [item.strip() for item in tg_columns]
#     else:
#         target_columns = [tg_columns]
#     # Iterate over each target column to calculate match ratio
#     for column in target_columns:
#         match_count = 0

#         # Filter for rows where both dataframes have non-null values in the column
#         non_null_matches = np.where(gd_df[column].notnull() & pred_df[column].notnull())[0]

#         # non_null_matches = gd_df[column].notnull() & pred_df[column].notnull()
#         # Count matches where values in both dataframes are equal
#         for idx in non_null_matches:
#             gd_values = gd_df[column][idx]
#             pred_values = pred_df[column][idx]
#             # print(f'ground truth: {gd_values}; predict results: {pred_values}')
#             if isinstance(gd_values, str) and isinstance(pred_values, str):
#                 if gd_values.upper() == pred_values.upper():
#                     match_count += 1
#                 else:
#                     print(f"{gd_values} <-->  {pred_values}")
#             elif gd_values == pred_values:
#                 print(f"{gd_values} <-!=->  {pred_values}")
#                 match_count += 1

#         # match_count = (gd_df[column][non_null_matches] == pred_df[column][non_null_matches]).sum()

#         # Calculate the match ratio for the column
#         # print(match_count)
#         # print(len(gd_df))
#         match_ratio = match_count / len(gd_df)  # Divide by total row count

#         # Append the match ratio for this column
#         match_ratios.append(match_ratio)

#     # Calculate and return the average of match ratios across all target columns
#     average_ratio = sum(match_ratios) / len(target_columns)
#     return average_ratio