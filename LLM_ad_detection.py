from data.raw_data_loader import LLMGNNDataLoader
from torch_geometric.data import Data
from anomaly_generator.utils import calculate_similarity
from openai_query import send_query_to_openai
import argparse
import json
from typing import List, Dict
import random

from torch_geometric.utils import k_hop_subgraph
from detector_prompts import (SYSTEM_PROMPT, NEIGHBORS_DESCRIPTION, USER_PROMPT_DETECTOR)
from deepinfra_query import send_query_to_deepinfra
from local_model_inference import use_local_model

DATA_DIR = "data/generated"
MAX_NEIGHBORS = 50
DEEPINFRA_MODEL = ["deepseek-ai/DeepSeek-V3-0324"]

def validation_check(data: Data):
    # check attributes, must have processed_text: List[str], anomaly_labels: torch.Tensor, anomaly_types: List[int] and updated_x: torch.Tensor, category_names: List[str]
    if not hasattr(data, "processed_text"):
        raise ValueError("processed_text is not found")
    if not hasattr(data, "anomaly_labels"):
        raise ValueError("anomaly_labels is not found")
    if not hasattr(data, "anomaly_types"):
        raise ValueError("anomaly_types is not found")
    if not hasattr(data, "updated_x"):
        raise ValueError("updated_x is not found")
    if not hasattr(data, "category_names"):
        raise ValueError("category_names is not found")
    # calculate similarity
    normal_similarity, anomaly_similarity = calculate_similarity(data)
    print(f"Normal similarity: {normal_similarity}, Anomaly similarity: {anomaly_similarity}")
    assert normal_similarity == 1 and anomaly_similarity < 1, "The similarity is not valid"
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="wikics_fixed_sbert_2_585.pt")
    parser.add_argument("--output_file", type=str, default="wikics_fixed_sbert_2_585_gpt-4o-mini.json")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_nodes", type=int, default=3000) # -1 means all nodes
    parser.add_argument("--existing_data_path", type=str, default=None)
    args = parser.parse_args()
    dataset_file = args.dataset_file
    data = LLMGNNDataLoader(data_dir=DATA_DIR).load_dataset(dataset_file, is_map_label=False)
    validation_check(data)

    anomaly_labels: List[int] = data.anomaly_labels.tolist()
    model_name = args.model_name
    max_nodes = args.max_nodes
    node_num = len(anomaly_labels)
    existing_data_path = args.existing_data_path

    selected_nodes = []
    if existing_data_path == None:
        if max_nodes != -1:
            # randomly select max_nodes nodes
            rnd = random.Random(42)
            selected_idxs = rnd.sample(range(node_num), max_nodes)
        else:
            # all nodes
            selected_idxs = list(range(node_num))
        # create the selected_nodes
        for idx in selected_idxs:
            selected_nodes.append(
                {
                    "index": idx,
                    "anomaly_label": anomaly_labels[idx],
                }
            )
    else:
        with open(existing_data_path, "r") as f:
            # exist data is the output of the previous run of the same dataset
            existing_data = json.load(f)
            for data_item in existing_data:
                selected_nodes.append(
                    {
                        "index": data_item["index"],
                        "anomaly_label": data_item["ground_truth"],
                    }
                )

    print(f"Total nodes: {len(selected_nodes)}")
    print(f"First 5 nodes: {selected_nodes[:5]}")
    user_prompt = USER_PROMPT_DETECTOR

    # formulate the testing datasets
    testing_data = []
    print("Formulating the testing datasets...")
    for i in range(len(selected_nodes)):
        if i % 100 == 0:
            print(f"Have processed {i} nodes")
        index = selected_nodes[i]["index"]
        label = selected_nodes[i]["anomaly_label"]
        prompt = make_prompt(data, index, user_prompt)
        # count the number of tokens in the prompt
        num_tokens = len(prompt.split())
        if num_tokens > 100000:
            print(f"The number of tokens in the prompt is too large: {num_tokens}")
            exit(1)
        testing_data.append({"index": index, "prompt": prompt, "ground_truth": label})

    # check if use the api
    if "gpt" in model_name.lower() or model_name in DEEPINFRA_MODEL:
        print(f"Using the api to detect the anomaly: {model_name}")
        print(f"Dataset: {dataset_file}")
        results = use_api(testing_data, model_name)
    else:
        results = use_local_model(testing_data, model_name)

    # calculate the accuracy
    accuracy = sum([result["ground_truth"] == result["prediction"] for result in results]) / len(results)
    print(f"Accuracy: {accuracy}")
    # save the results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
        

def make_prompt(data: Data, idx: int, user_prompt: str) -> str:
    first_hop_neighbors = get_k_hop_neighbors(data, idx, 1)
    if len(first_hop_neighbors) > MAX_NEIGHBORS:
        print(f"The number of first hop neighbors is greater than {MAX_NEIGHBORS}, sampling {MAX_NEIGHBORS} neighbors")
        rnd = random.Random(42)
        first_hop_neighbors = rnd.sample(first_hop_neighbors, MAX_NEIGHBORS)
    # get the text attributes and their category names
    neighbors_texts = []
    for neighbor_idx in first_hop_neighbors:
        neighbors_texts.append(data.processed_text[neighbor_idx])
    # get the text attribute of the current node
    current_node_text = data.processed_text[idx]
    # construct the prompt
    neighbors_description = ""
    for i in range(len(first_hop_neighbors)):
        neighbors_description += NEIGHBORS_DESCRIPTION.format(neighbor_index=i, neighbor_text=neighbors_texts[i])
    # construct the prompt
    prompt = user_prompt.format(raw_text=current_node_text, num_neighbors=len(first_hop_neighbors), neighbors_description=neighbors_description)
    return prompt

def get_k_hop_neighbors(data: Data, idx: int, k: int)-> List[int]:
    """
    Get the k-hop neighbors of the node
    """
    if k == 0:
        return []
    # get the k-hop subgraph
    subset_k, _, _, _ = k_hop_subgraph(idx, k, data.edge_index)
    if k == 1:
        exact_k = subset_k[subset_k != idx].tolist()
        return exact_k
    # get k-1 hop subgraph
    subset_k_1, _, _, _ = k_hop_subgraph(idx, k-1, data.edge_index)
    # get the neighbors of the node
    set_k   = set(subset_k.tolist())
    set_k_1 = set(subset_k_1.tolist())
    exact_k = list(set_k - set_k_1 - {idx})
    return exact_k

def use_api(testing_data: List[Dict], model_name: str) -> List[Dict]:
    """
    Use the openai/deepinfra api to detect the anomaly
    """
    results = []
    count = 0
    for data in testing_data:
        prediction = None
        print("Index:\n", data["index"])
        print("Input:\n", data["prompt"])
        print("ground_truth:\n", data["ground_truth"])
        while True:
            if model_name in DEEPINFRA_MODEL:
                api_response = send_query_to_deepinfra(data["prompt"], SYSTEM_PROMPT, model_name)
            else:
                api_response = send_query_to_openai(data["prompt"], SYSTEM_PROMPT, model_name)
            if "RESULT:TRUE" in api_response:
                prediction = 1
                break
            elif "RESULT:FALSE" in api_response:
                prediction = 0
                break
            else:
                print("Invalid response! Try again...")
        count += 1
        print(f"Have processed {count} nodes")
        print("ground_truth:\n", data["ground_truth"])
        print("prediction:\n", prediction)
        print("="*100)
        results.append({"index": data["index"], "ground_truth": data["ground_truth"], "prediction": prediction})
    return results


if __name__ == "__main__":
    main()