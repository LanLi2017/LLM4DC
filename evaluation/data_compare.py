import pandas as pd


def retrieve_tg_cols(tg_cols_fp="target_columns_list.csv"):
    id_tg_cols = {}
    tg_df = pd.read_csv(tg_cols_fp)
    result_dict = tg_df.set_index('ID')['tg_columns'].to_dict()
    return result_dict


def average_match_ratio(gd_df: pd.DataFrame, pred_df: pd.DataFrame, tg_columns: str) -> float:
    match_ratios = []
    
    target_columns = [item.strip() for item in tg_columns.split(',')]

    # Iterate over each target column to calculate match ratio
    for column in target_columns:
        # Filter for rows where both dataframes have non-null values in the column
        non_null_matches = gd_df[column].notnull() & pred_df[column].notnull()

        # Count matches where values in both dataframes are equal
        match_count = (gd_df[column][non_null_matches] == pred_df[column][non_null_matches]).sum()

        # Calculate the match ratio for the column
        match_ratio = match_count / len(gd_df)  # Divide by total row count

        # Append the match ratio for this column
        match_ratios.append(match_ratio)

    # Calculate and return the average of match ratios across all target columns
    average_ratio = sum(match_ratios) / len(target_columns)
    return average_ratio

result_dict = retrieve_tg_cols()
query_id=89
tg_cols = result_dict[query_id]
print(f'target columns for purpose id {query_id}: {tg_cols}')
gd_fp = f'/projects/bces/lanl2/LLM4DC/datasets/ppp_datasets/cleaned_tables/ppp_sample_p{query_id}.csv'
print(f'ground truth file: {gd_fp}')
gd_df = pd.read_csv(gd_fp)

model = "mistral"
llm_folder = f"CoT.response/{model}/datasets_llm"
pred_fp = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/{model}_ppp_test_{query_id}.csv'
print(f'Model: {model} predicted file: \n\n{pred_fp}')
pred_df = pd.read_csv(pred_fp)
res = average_match_ratio(gd_df, pred_df, tg_cols)
print(res)