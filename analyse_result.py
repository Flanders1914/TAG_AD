import json
from pygod.metric import eval_roc_auc, eval_average_precision, eval_f1, eval_precision_at_k, eval_recall_at_k
import os
import torch

if __name__ == "__main__":

    result_file_list = []
    # find all result files
    for file in os.listdir("."):
        if file.endswith(".json") and not file.endswith("dataset.json"):
            result_file_list.append(file)

    result_dict = {}
    # sort result_file_list by name
    result_file_list.sort()
    for result_file in result_file_list:
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
            y_true = [result["ground_truth"] for result in results]
            y_pred = []
            count = 0
            for result in results:
                if result["prediction"] is not None:
                    y_pred.append(result["prediction"])
                else:
                    y_pred.append(0.5)
                    count += 1
            
            y_true = torch.tensor(y_true)
            y_pred = torch.tensor(y_pred)
        except Exception as e:
            continue
        print(f"Result file: {result_file}")
        print(f"AUC: {eval_roc_auc(y_true, y_pred)}")
        print(f"Average Precision: {eval_average_precision(y_true, y_pred)}")
        print(f"Precision at K: {eval_precision_at_k(y_true, y_pred, k=20)}")
        print(f"Recall at K: {eval_recall_at_k(y_true, y_pred, k=20)}")
        print(f"Count: {count}")
        print("="*100)