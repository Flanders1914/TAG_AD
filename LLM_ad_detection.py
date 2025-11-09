# python LLM_ad_detection.py
from data.raw_data_loader import LLMGNNDataLoader
from torch_geometric.data import Data
from openai_query import inference_on_openai
import argparse
import json
from typing import List, Dict
import random
import os
from torch_geometric.utils import k_hop_subgraph
from detector_prompts import (ANALYSIS_FRAMEWORK_CONTEXTUAL_HUMAN_DESIGNED, ANALYSIS_FRAMEWORK_STRUCTURAL_HUMAN_DESIGNED, SYSTEM_PROMPT, NEIGHBORS_DESCRIPTION, USER_PROMPT_DETECTOR_CONTEXTUAL, USER_PROMPT_DETECTOR_STRUCTURAL, USER_PROMPT_DETECTOR_MIXED, ANALYSIS_FRAMEWORK_DUMMY)
from graph_structure_encoder import graph_structure_encoder
from deepinfra_query import inference_on_deepinfra
from deepseek_query import inference_on_deepseek
import sys

DATA_DIR = "data/generated"
# hyper-parameters for contextual anomaly prompt
MAX_NEIGHBORS = 20
MAX_MIXED_NEIGHBORS = 10
MAX_WORD_NUM = 1000
MAX_SAMPLE_TRIALS = 1000
MAX_PROMPT_TOKEN_NUM = 100000
# hyper-parameters for structural anomaly prompt
MAX_NODES = 100
MAX_EDGES = 100
MAX_MIXED_NODES = 50
MAX_MIXED_EDGES = 50
RANDOM_SEED = 42
# the supported models for the api
DEEPINFRA_MODEL = ["deepseek-ai/DeepSeek-V3-0324", "Qwen/Qwen3-14B", "google/gemma-3-27b-it"]
DEEPSEEK_MODEL = ["deepseek-chat"]
OPENAI_MODEL = ["gpt-4o-mini"]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly_type", type=str, choices=["Contextual Anomaly", "Structural Anomaly", "Mixed Anomaly"])
    parser.add_argument("--analysis_framework_path", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default="pubmed_fixed_sbert_2_100.pt")
    parser.add_argument("--output_dir", type=str, default="LLM_results/")
    parser.add_argument("--output_file", type=str, default="pubmed_fixed_sbert_2_deepseek-chat.json")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--max_nodes", type=int, default=1000) # -1 means all nodes
    parser.add_argument("--use_human_designed_analysis_framework", action="store_true", default=False)
    parser.add_argument("--use_dummy", action="store_true", default=False)
    parser.add_argument("--test_mode", action="store_true", default=False)
    args = parser.parse_args()

    anomaly_type = args.anomaly_type
    # load the analysis framework
    if os.path.exists(args.analysis_framework_path):
        with open(args.analysis_framework_path, "r") as f:
            analysis_framework = f.read()
    else:
        raise ValueError(f"Analysis framework path: {args.analysis_framework_path} does not exist")
    
    # load the dataset
    dataset_file = args.dataset_file
    data = LLMGNNDataLoader(data_dir=DATA_DIR).load_dataset(dataset_file, is_map_label=False)
    dataset_name = dataset_file.split(".")[0]

    # add the output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # add the processed_text to the structured anomaly data
    if "5" == dataset_name.split("_")[3]:
        print("Structural anomaly")
        data.processed_text = data.raw_texts

    anomaly_labels: List[int] = data.anomaly_labels.tolist()
    model_name = args.model_name
    max_nodes = args.max_nodes
    test_mode = args.test_mode
    use_human_designed_analysis_framework = args.use_human_designed_analysis_framework
    use_dummy = args.use_dummy
    node_num = len(anomaly_labels)

    # decide the user prompt based on the anomaly type and change the analysis framework if needed
    if anomaly_type == "Contextual Anomaly":
        user_prompt = USER_PROMPT_DETECTOR_CONTEXTUAL
        if use_human_designed_analysis_framework:
            if use_dummy:
                analysis_framework = ANALYSIS_FRAMEWORK_DUMMY
            else:
                analysis_framework = ANALYSIS_FRAMEWORK_CONTEXTUAL_HUMAN_DESIGNED
    elif anomaly_type == "Structural Anomaly":
        user_prompt = USER_PROMPT_DETECTOR_STRUCTURAL
        if use_human_designed_analysis_framework:
            if use_dummy:
                analysis_framework = ANALYSIS_FRAMEWORK_DUMMY
            else:
                analysis_framework = ANALYSIS_FRAMEWORK_STRUCTURAL_HUMAN_DESIGNED
    elif anomaly_type == "Mixed Anomaly":
        user_prompt = USER_PROMPT_DETECTOR_MIXED
    else:
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")

    # formulate the testing datasets
    testing_data = []
    if use_human_designed_analysis_framework:
        if use_dummy:
            testing_data_name = dataset_name+"_testing_dataset_human_designed_dummy.json"
        else:
            testing_data_name = dataset_name+"_testing_dataset_human_designed.json"
    else:
        testing_data_name = dataset_name+"_testing_dataset.json"
    testing_data_name = os.path.join(args.output_dir, testing_data_name)
    # check if the testing data exists
    if os.path.exists(testing_data_name):
        print("Loading the testing datasets from the existing file...")
        with open(testing_data_name, "r") as f:
            testing_data = json.load(f)
            print("Length of the testing data: ", len(testing_data))
    else:
        # select the nodes
        selected_nodes = []
        if max_nodes != -1:
            # randomly select max_nodes nodes, ensure the selected nodes has at least 4% and at most 6% of anomaly nodes
            for i in range(MAX_SAMPLE_TRIALS):
                selected_idxs = random.sample(range(node_num), max_nodes)
                if sum([anomaly_labels[idx] for idx in selected_idxs]) >= 0.04 * max_nodes and sum([anomaly_labels[idx] for idx in selected_idxs]) <= 0.06 * max_nodes:
                    break
                if i == MAX_SAMPLE_TRIALS - 1:
                    print(f"The number of anomaly nodes is not in the range of 4% to 6% after {MAX_SAMPLE_TRIALS} trials")
                    sys.exit(1)
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
        # start to formulate the testing datasets
        print("Formulating the testing datasets...")
        for i in range(len(selected_nodes)):
            if i % 100 == 0:
                print(f"Have processed {i} nodes")
            index = selected_nodes[i]["index"]
            label = selected_nodes[i]["anomaly_label"]
            if anomaly_type == "Contextual Anomaly":
                prompt = make_prompt_contextual(data, index, user_prompt, analysis_framework)
            elif anomaly_type == "Structural Anomaly":
                prompt = make_prompt_structural(data, index, user_prompt, analysis_framework)
            elif anomaly_type == "Mixed Anomaly":
                prompt = make_prompt_mixed(data, index, user_prompt, analysis_framework)
            else:
                raise ValueError(f"Invalid anomaly type: {anomaly_type}")
            # count the number of tokens in the prompt
            num_tokens = len(prompt.split())
            if num_tokens > MAX_PROMPT_TOKEN_NUM:
                print(f"The number of tokens in the prompt is too large: {num_tokens}")
                sys.exit(1)
            testing_data.append({"index": index, "prompt": prompt, "ground_truth": label})
        with open(testing_data_name, "w") as f:
            json.dump(testing_data, f, indent=2)
        print(f"Saved the testing datasets to {testing_data_name}")

    # if the test mode is enabled, print a test sample and exit
    if test_mode:
        print(f"Testing mode is enabled, printing a test sample and exiting...")
        print(f"Index: {testing_data[0]['index']}")
        print(f"Prompt:\n{testing_data[0]['prompt']}")
        print(f"Ground truth: {testing_data[0]['ground_truth']}")
        sys.exit(0)

    # check if use the api
    if model_name in OPENAI_MODEL or model_name in DEEPINFRA_MODEL or model_name in DEEPSEEK_MODEL:
        print(f"Using the api to detect the anomaly: {model_name}")
        print(f"Dataset: {dataset_file}")
        results = use_api(testing_data, model_name)
    else:
        # for local inference please use the local_model_inference.py
        raise ValueError(f"Invalid model name: {model_name}")

    # save the results
    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved the results to {output_file}")


def make_prompt_contextual(data: Data, idx: int, user_prompt: str, analysis_framework: str) -> str:
    """
    Make the prompt for the node with the index idx
    Using the first hop neighbors as the context
    """
    # get the first hop neighbors
    first_hop_neighbors = get_k_hop_neighbors(data, idx, 1)
    # reduce the redundancy of the first hop neighbors
    first_hop_neighbors = list(set(first_hop_neighbors))
    # if the number of first hop neighbors is greater than MAX_NEIGHBORS, sample MAX_NEIGHBORS neighbors
    if len(first_hop_neighbors) > MAX_NEIGHBORS:
        print(f"The number of first hop neighbors is greater than {MAX_NEIGHBORS}, sampling {MAX_NEIGHBORS} neighbors")
        rnd = random.Random(42)
        first_hop_neighbors = rnd.sample(first_hop_neighbors, MAX_NEIGHBORS)
    # get the text attributes and their category names
    neighbors_texts = []
    for neighbor_idx in first_hop_neighbors:
        neighbor_text = data.processed_text[neighbor_idx]
        # Discard the neighbor text after MAX_WORD_NUM words
        neighbor_text_split = neighbor_text.split()
        if len(neighbor_text_split) > MAX_WORD_NUM:
            neighbor_text_split = neighbor_text_split[:MAX_WORD_NUM]
        neighbor_text = " ".join(neighbor_text_split)
        neighbors_texts.append(neighbor_text)
    # get the text attribute of the current node, discard the text after MAX_WORD_NUM words
    current_node_text = data.processed_text[idx]
    current_node_text_split = current_node_text.split()
    if len(current_node_text_split) > MAX_WORD_NUM:
        current_node_text_split = current_node_text_split[:MAX_WORD_NUM]
    current_node_text = " ".join(current_node_text_split)
    # construct the prompt
    neighbors_description = ""
    for i in range(len(first_hop_neighbors)):
        neighbors_description += NEIGHBORS_DESCRIPTION.format(neighbor_index=i, neighbor_text=neighbors_texts[i])
    # construct the prompt
    prompt = user_prompt.format(raw_text=current_node_text, num_neighbors=len(first_hop_neighbors), max_word_num=MAX_WORD_NUM, neighbors_description=neighbors_description, analysis_framework=analysis_framework)
    return prompt


def make_prompt_structural(data: Data, idx: int, user_prompt: str, analysis_framework: str) -> str:
    """
    Make the prompt for the node with the index idx
    Using the structure of the subgraph centered at the node as the context
    """
    
    # get the text representation of the subgraph centered at the node
    rnd = random.Random(RANDOM_SEED)
    graph_structure_representation = graph_structure_encoder(data, idx, MAX_NODES, MAX_EDGES, rnd.randint(0, 100000))
    # construct the prompt
    prompt = user_prompt.format(graph_structure_representation=graph_structure_representation, idx=idx, analysis_framework=analysis_framework)
    return prompt


def make_prompt_mixed(data: Data, idx: int, user_prompt: str, analysis_framework: str) -> str:
    max_neighbors = MAX_MIXED_NEIGHBORS
    # get the first hop neighbors
    first_hop_neighbors = get_k_hop_neighbors(data, idx, 1)
    # reduce the redundancy of the first hop neighbors
    first_hop_neighbors = list(set(first_hop_neighbors))
    # if the number of first hop neighbors is greater than max_neighbors, sample max_neighbors neighbors
    if len(first_hop_neighbors) > max_neighbors:
        print(f"The number of first hop neighbors is greater than {max_neighbors}, sampling {max_neighbors} neighbors")
        rnd = random.Random(42)
        first_hop_neighbors = rnd.sample(first_hop_neighbors, max_neighbors)
    # get the text attributes and their category names
    neighbors_texts = []
    neighbors_indices = []
    for neighbor_idx in first_hop_neighbors:
        neighbor_text = data.processed_text[neighbor_idx]
        # Discard the neighbor text after MAX_WORD_NUM words
        neighbor_text_split = neighbor_text.split()
        if len(neighbor_text_split) > MAX_WORD_NUM:
            neighbor_text_split = neighbor_text_split[:MAX_WORD_NUM]
        neighbor_text = " ".join(neighbor_text_split)
        neighbors_texts.append(neighbor_text)
        neighbors_indices.append(neighbor_idx)

    # get the text attribute of the current node, discard the text after MAX_WORD_NUM words
    current_node_text = data.processed_text[idx]
    current_node_text_split = current_node_text.split()
    if len(current_node_text_split) > MAX_WORD_NUM:
        current_node_text_split = current_node_text_split[:MAX_WORD_NUM]
    current_node_text = " ".join(current_node_text_split)
    # construct the prompt
    neighbors_description = ""
    for i in range(len(first_hop_neighbors)):
        neighbors_description += NEIGHBORS_DESCRIPTION.format(neighbor_index=neighbors_indices[i], neighbor_text=neighbors_texts[i])

    # get the graph structure representation
    rnd = random.Random(RANDOM_SEED)
    graph_structure_representation = graph_structure_encoder(data, idx, MAX_MIXED_NODES, MAX_MIXED_EDGES, rnd.randint(0, 100000))
    # construct the prompt
    prompt = user_prompt.format(raw_text=current_node_text,
                                num_neighbors=len(first_hop_neighbors),
                                max_word_num=MAX_WORD_NUM,
                                neighbors_description=neighbors_description,
                                analysis_framework=analysis_framework,
                                graph_structure_representation=graph_structure_representation,
                                idx=idx)
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
        deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        temperature = float(os.getenv("TEMPERATURE"))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # choose the api based on the model name
    if model_name in DEEPINFRA_MODEL:
        results = inference_on_deepinfra(testing_data, SYSTEM_PROMPT, model_name, deepinfra_api_key, temperature)
    elif model_name in DEEPSEEK_MODEL:
        results = inference_on_deepseek(testing_data, SYSTEM_PROMPT, model_name, deepseek_api_key, temperature)
    elif model_name in OPENAI_MODEL:
        results = inference_on_openai(testing_data, SYSTEM_PROMPT, model_name, openai_api_key, temperature)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return results


if __name__ == "__main__":
    main()