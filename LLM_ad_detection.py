# python LLM_ad_detection.py
from data.raw_data_loader import LLMGNNDataLoader
from torch_geometric.data import Data
from openai_query import send_query_to_openai
import argparse
import json
from typing import List, Dict
import random
import os
from torch_geometric.utils import k_hop_subgraph
from detector_prompts import (SYSTEM_PROMPT, NEIGHBORS_DESCRIPTION, USER_PROMPT_DETECTOR)
from deepinfra_query import send_query_to_deepinfra
from deepseek_query import send_query_to_deepseek
import yaml
import sys

DATA_DIR = "data/generated"
CONFIG_PATH = "./config.yaml"
MAX_NEIGHBORS = 10
MAX_NEIGHBORS_WORD_NUM = 1000
DEEPINFRA_MODEL = ["deepseek-ai/DeepSeek-V3.1"]
DEEPSEEK_MODEL = ["deepseek-chat"]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="citeseer_fixed_sbert_4_159.pt")
    parser.add_argument("--output_file", type=str, default="citeseer_fixed_sbert_4_159_DeepSeek-V3.1.json")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--max_nodes", type=int, default=1000) # -1 means all nodes
    args = parser.parse_args()
    dataset_file = args.dataset_file
    data = LLMGNNDataLoader(data_dir=DATA_DIR).load_dataset(dataset_file, is_map_label=False)
    dataset_name = dataset_file.split(".")[0]
    if "5" == dataset_name.split("_")[3]:
        print("Structural anomaly")
        data.processed_text = data.raw_texts

    anomaly_labels: List[int] = data.anomaly_labels.tolist()
    model_name = args.model_name
    max_nodes = args.max_nodes
    node_num = len(anomaly_labels)
    user_prompt = USER_PROMPT_DETECTOR

    # formulate the testing datasets
    testing_data = []
    testing_data_name = dataset_name+"_testing_dataset.json"
    if os.path.exists(testing_data_name):
        print("Loading the testing datasets from the existing file...")
        with open(testing_data_name, "r") as f:
            testing_data = json.load(f)
            print("Length of the testing data: ", len(testing_data))
    else:
        # select the nodes
        selected_nodes = []
        if max_nodes != -1:
            # randomly select max_nodes nodes, ensure the selected nodes has at least 4% of anomaly nodes
            while True:
                selected_idxs = random.sample(range(node_num), max_nodes)
                if sum([anomaly_labels[idx] for idx in selected_idxs]) >= 0.04 * max_nodes and sum([anomaly_labels[idx] for idx in selected_idxs]) <= 0.06 * max_nodes:
                    break
                print("The number of anomaly nodes is not in the range of 4% to 6%")
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
        print(f"Total nodes: {len(selected_nodes)}")
        print(f"First 5 nodes: {selected_nodes[:5]}")
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
        with open(testing_data_name, "w") as f:
            json.dump(testing_data, f, indent=2)
        print(f"Saved the testing datasets to {testing_data_name}")

    # check if use the api
    if "gpt" in model_name.lower() or model_name in DEEPINFRA_MODEL or model_name in DEEPSEEK_MODEL:
        print(f"Using the api to detect the anomaly: {model_name}")
        print(f"Dataset: {dataset_file}")
        results = use_api(testing_data, model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    # save the results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved the results to {args.output_file}")


def make_prompt(data: Data, idx: int, user_prompt: str) -> str:
    first_hop_neighbors = get_k_hop_neighbors(data, idx, 1)
    # reduce the redundancy of the first hop neighbors
    first_hop_neighbors = list(set(first_hop_neighbors))
    if len(first_hop_neighbors) > MAX_NEIGHBORS:
        print(f"The number of first hop neighbors is greater than {MAX_NEIGHBORS}, sampling {MAX_NEIGHBORS} neighbors")
        rnd = random.Random(42)
        first_hop_neighbors = rnd.sample(first_hop_neighbors, MAX_NEIGHBORS)
    # get the text attributes and their category names
    neighbors_texts = []
    for neighbor_idx in first_hop_neighbors:
        neighbor_text = data.processed_text[neighbor_idx]
        # Discard the neighbor text after MAX_NEIGHBORS_WORD_NUM words
        neighbor_text_split = neighbor_text.split()
        if len(neighbor_text_split) > MAX_NEIGHBORS_WORD_NUM:
            neighbor_text_split = neighbor_text_split[:MAX_NEIGHBORS_WORD_NUM]
        neighbor_text = " ".join(neighbor_text_split)
        neighbors_texts.append(neighbor_text)
    # get the text attribute of the current node
    current_node_text = data.processed_text[idx]
    # construct the prompt
    neighbors_description = ""
    for i in range(len(first_hop_neighbors)):
        neighbors_description += NEIGHBORS_DESCRIPTION.format(neighbor_index=i, neighbor_text=neighbors_texts[i])
    # construct the prompt
    prompt = user_prompt.format(raw_text=current_node_text, num_neighbors=len(first_hop_neighbors), max_neighbors_word_num=MAX_NEIGHBORS_WORD_NUM, neighbors_description=neighbors_description)
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
    try:
        # get the api key, model from config.yaml
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        deepinfra_api_key = config.get("DEEPINFRA_API_KEY")
        openai_api_key = config.get("OPENAI_KEY")
        deepseek_api_key = config.get("DEEPSEEK_API_KEY")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    temperature = 0

    # send the query to different api
    results = []
    count = 0
    for data in testing_data:
        if model_name in DEEPINFRA_MODEL:
            api_response = send_query_to_deepinfra(data["prompt"], SYSTEM_PROMPT, model_name, deepinfra_api_key, temperature)
        elif model_name in DEEPSEEK_MODEL:
            api_response = send_query_to_deepseek(data["prompt"], SYSTEM_PROMPT, model_name, deepseek_api_key, temperature)
        else:
            api_response = send_query_to_openai(data["prompt"], SYSTEM_PROMPT, model_name, openai_api_key, temperature)

        # parse the response to get the prediction
        prediction = None # default value=None
        result_idx = api_response.find("RESULT:")
        if result_idx != -1:
            res_str = api_response[result_idx+7]
            if res_str.isdigit() and int(res_str) >= 0 and int(res_str) < 10:
                score = int(res_str)
                if score == 1:
                    if result_idx+8 < len(api_response) and api_response[result_idx+8] == "0":
                        prediction = 1.0
                    else:
                        prediction = 0.1
                else:
                    prediction = int(res_str)/10
        # save the response
        count += 1
        print("Response:\n", api_response)
        print(f"Have processed {count} nodes")
        print("ground_truth:\n", data["ground_truth"])
        print("prediction:\n", prediction)
        print("="*100)
        results.append({"index": data["index"], "ground_truth": data["ground_truth"], "prediction": prediction})
    return results


if __name__ == "__main__":
    main()