import os
import pandas as pd
from glob import glob
import numpy as np

def count_cell_errors(clean_df, dirty_df):
    """Count number of differing cells between clean and dirty DataFrames."""
    errors = 0
    total_cells = clean_df.shape[0] * clean_df.shape[1]

    # Ensure alignment
    clean_df = clean_df.reset_index(drop=True)
    dirty_df = dirty_df.reset_index(drop=True)

    # Compare cell-by-cell
    for col in clean_df.columns:
        if col in dirty_df.columns:
            clean_col = clean_df[col].astype(str)
            dirty_col = dirty_df[col].astype(str)
            errors += (clean_col != dirty_col).sum()
    
    return errors, total_cells

def compute_error_ratios(clean_dir, dirty_dir):
    clean_files = sorted(glob(os.path.join(clean_dir, '*.csv')))
    dirty_files = sorted(glob(os.path.join(dirty_dir, '*.csv')))

    table_error_ratios = []

    for clean_path, dirty_path in zip(clean_files, dirty_files):
        clean_df = pd.read_csv(clean_path)
        dirty_df = pd.read_csv(dirty_path)

        errors, total_cells = count_cell_errors(clean_df, dirty_df)
        error_ratio = errors / total_cells if total_cells > 0 else 0

        table_name = os.path.basename(clean_path)
        table_error_ratios.append((table_name, error_ratio))

        print(f"{table_name}: {errors} errors, {total_cells} total cells, error ratio = {error_ratio:.4f}")
    
    ratios = [ratio for _, ratio in table_error_ratios]
    std = np.std(ratios)
    # avg = np.mean(ratios)
    # Average ratio across all tables
    if table_error_ratios:
        avg_ratio = sum(ratio for _, ratio in table_error_ratios) / len(table_error_ratios)
        print(f"\nAverage error ratio across {len(table_error_ratios)} tables: {avg_ratio:.4f}")
        print(f"STD is {std}")
    else:
        print("No tables found for comparison.")

    return avg_ratio, std

if __name__ == '__main__':
    # Set your input/output table directories here
    log_file = 'error_ratio_table.csv'
    # tb = 'hospital'
    # tb = 'flights'
    # tb = 'ppp'
    tb = 'dish'
    # clean_dir = 'hospital/clean_tables'
    # dirty_dir = 'hospital'
    # clean_dir = 'flights/cleaned_tables'
    # dirty_dir = 'flights'
    # clean_dir = 'CFI_datasets/cleaned_tables'
    # dirty_dir = 'CFI_datasets'
    # clean_dir = 'menu_datasets/clean_tables'
    # dirty_dir = 'menu_datasets'
    # clean_dir = 'ppp_datasets/ori_dirty'
    # dirty_dir = 'ppp_datasets'
    clean_dir = 'dish_datasets/cleaned_tables'
    dirty_dir = 'dish_datasets'


    with open(log_file, 'a') as log:
        avg_ratio, std = compute_error_ratios(clean_dir, dirty_dir)
        log.write(f'Error ratio for dataset {tb}: {avg_ratio} \n')
        log.write(f'Standard deviation for the ratio: {std} \n')
    
