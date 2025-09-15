# python analyse_result.py --result_file pubmed_fixed_sbert_2_986_gpt-4o-mini.json
import json
import argparse
import sklearn.metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="cora_fixed_sbert_2_135_gpt-4o-mini.json")
    args = parser.parse_args()
    result_file = args.result_file
    with open(result_file, "r") as f:
        results = json.load(f)
    
    TP = len([result for result in results if result["ground_truth"] == 1 and result["prediction"] == 1])
    TN = len([result for result in results if result["ground_truth"] == 0 and result["prediction"] == 0])
    FP = len([result for result in results if result["ground_truth"] == 0 and result["prediction"] == 1])
    FN = len([result for result in results if result["ground_truth"] == 1 and result["prediction"] == 0])

    print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
    print(f"Precision: {TP / (TP + FP)}")
    print(f"Recall: {TP / (TP + FN)}")