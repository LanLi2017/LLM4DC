import pandas as pd

def average_match_ratio(df1: pd.DataFrame, df2: pd.DataFrame, target_columns: list) -> float:
    # Ensure all target columns exist in both DataFrames
    missing_columns = [col for col in target_columns if col not in df1.columns or col not in df2.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} must be in both DataFrames")

    # Initialize a list to store match ratios for each target column
    match_ratios = []

    # Iterate over each target column to calculate match ratio
    for column in target_columns:
        # Filter for rows where both entries are numeric
        numeric_matches = df1[column].apply(lambda x: isinstance(x, (int, float))) & \
                          df2[column].apply(lambda x: isinstance(x, (int, float)))

        # Calculate the match ratio for numeric rows in the current column
        match_count = (df1[column][numeric_matches] == df2[column][numeric_matches]).sum()
        match_ratio = match_count / len(df1)  # Divide by total row length

        # Append the match ratio for this column
        match_ratios.append(match_ratio)

    # Calculate and return the average of match ratios
    average_ratio = sum(match_ratios) / len(target_columns)
    return average_ratio
