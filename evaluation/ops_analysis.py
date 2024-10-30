import numpy as np
import json
# TODO: parse json to operation list from ground_truth and prediction workflow json files
def parse_recipe(pp_id, recipe):
    res = []
    with open(recipe, 'r')as recipe_f:
        data = json.load(recipe_f)
    
    for op in data:
        op_name = op['op']
        if op_name=="core/text-transform":
            exp = op['expression']
            if exp=="value.trim()":
                res.append("trim")
            elif exp=="value.toUppercase()":
                res.append("upper")
            elif exp=="value.toNumber()":
                res.append("numeric")
            elif exp=="value.toDate()":
                res.append("date")
            elif exp.startswith("jython"):
                res.append("regexr_transform")
            elif exp=="value.toString()":
                res.append("date")
            else:
                res.append("text_transform")
        else:
            op_name = op_name.split('/')[-1].replace('-', '_')
            res.append(op_name)
    return {pp_id: res}
    


def calculate_operation_metrics(ground_truth, predictions):
    total_samples = len(ground_truth)
    exact_matches = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    results = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt)
        pred_set = set(pred)

        # Exact match for accuracy
        if gt_set == pred_set:
            exact_matches += 1

        # Precision and Recall for each sample
        true_positives = len(gt_set & pred_set)
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gt_set) if gt_set else 0

        # F1 Score for each sample
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        results.append({'accuracy': exact_matches, 'precision': precision, 'recall': recall, 'f1': 1})
        # Accumulate macro metrics
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate macro averages
    accuracy = exact_matches / total_samples
    macro_precision = total_precision / total_samples
    macro_recall = total_recall / total_samples
    macro_f1 = total_f1 / total_samples

    return pd.DataFrame(results), {'accuracy': accuracy, 'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f1': macro_f1}    