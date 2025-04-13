import os
import re
import json
from glob import glob
from collections import Counter

def load_clean_json(filepath):
    """Loads a JSON file, skipping headers like 'sql', and categorizes its status."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Find where the JSON starts
    json_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('[') or line.strip().startswith('{'):
            json_start_index = i
            break

    if json_start_index is None:
        return "no_json_start", None

    json_str = ''.join(lines[json_start_index:]).strip()
    if not json_str:
        return "empty", None

    try:
        data = json.loads(json_str)
        if not data:
            return "empty_json", data
        return "ok", data
    except json.JSONDecodeError:
        return "parse_error", None

def extract_info(file_path):
    """Extracts purpose ID, columns, and operations from a JSON file."""
    filename = os.path.basename(file_path)
    match = re.search(r'_p(\d+)\.json$', filename)
    purpose_id = int(match.group(1)) if match else None

    status, data = load_clean_json(file_path)
    if status != "ok":
        return status, None

    column_set = set()
    operations = []

    for op in data:
        col = op.get("columnName")
        if col:
            if isinstance(col, list):
                column_set.update(col)
            else:
                column_set.add(col)
        operations.append(op.get("op", ""))

    return status, {
        "ID": purpose_id,
        "Columns": sorted(column_set),
        "Operations": operations
    }

def process_all_files(folder_path):
    status_counter = Counter()
    results = []

    for filepath in glob(os.path.join(folder_path, "*.json")):
        status, info = extract_info(filepath)
        status_counter[status] += 1

        if info:
            results.append(info)
        else:
            print(f"Skipped: {os.path.basename(filepath)} due to {status}")

    print("\n=== Summary ===")
    for k, v in status_counter.items():
        print(f"{k}: {v}")

    return status_counter["empty"] + status_counter["empty_json"]

# Example usage:
model_name = "llama3.1"
empty_count = process_all_files(f"{model_name}/workflow_unparsed")
print(f"\n Number of empty files: {empty_count}")
