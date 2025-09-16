# python make_anomaly.py
from anomaly_generator import llm_generated_contextual_anomaly_generator, heuristic_contextual_anomaly_generator, global_anomaly_generator, dummy_anomaly_generator, structural_anomaly_generator
from data.raw_data_loader import LLMGNNDataLoader
from datasets import load_dataset
from torch_geometric.data import Data
import torch
import os
import argparse
from typing import List

K_NEIGHBORS = 3
K = 50
RANDOM_SEED = 42
M = 10
PUBMED_EACH_NUM = 1000

def save_data(data: Data, data_dir: str, dataset_name: str, anomaly_type: int, anomaly_num: int) -> None:
    file_path = os.path.join(data_dir, f"{dataset_name}_{anomaly_type}_{anomaly_num}.pt")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, file_path)

def make_outliner_texts_pubmed():
    result = []
    outliner_breast_cancer = load_from_huggingface("Gaborandi/breast_cancer_pubmed_abstracts")
    result.extend(outliner_breast_cancer)
    outliner_HIV = load_from_huggingface("Gaborandi/HIV_pubmed_abstracts")
    result.extend(outliner_HIV)
    outliner_brain_tumor = load_from_huggingface("Gaborandi/Brain_Tumor_pubmed_abstracts")
    result.extend(outliner_brain_tumor)
    outliner_Alzheimer = load_from_huggingface("Gaborandi/Alzheimer_pubmed_abstracts")
    result.extend(outliner_Alzheimer)
    return result

def make_outliner_texts_wikics(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("citeseer_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def make_outliner_texts_cora(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("wikics_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def make_outliner_texts_citeseer(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("wikics_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def load_from_huggingface(dataset_name: str) -> List[str]:
    dataset = load_dataset(dataset_name)["train"]
    title = dataset["title"]
    abstract = dataset["abstract"]
    str_list = []
    count = 0
    for title, abstract in zip(title, abstract):
        if title is None or abstract is None:
            continue
        str_list.append("Title: "+ title + "\nAbstract: " + abstract)
        count += 1
        if count >= PUBMED_EACH_NUM:
            break
    print(f"{dataset_name} example:\n{str_list[0]}")
    return str_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikics_fixed_sbert")
    parser.add_argument("--anomaly_type", type=int, default=4)
    parser.add_argument("--anomaly_num", type=int, default=585)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    anomaly_type = args.anomaly_type
    n=args.anomaly_num
    loader = LLMGNNDataLoader(data_dir="data/raw")
    data = loader.load_dataset(dataset_name, is_map_label=True)
    k_neighbors = K_NEIGHBORS
    k = K
    random_seed = RANDOM_SEED

    if anomaly_type == 1:
        data = dummy_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        n=n,
                                        anomaly_type=1,
                                        random_seed=random_seed)
    elif anomaly_type == 2:
        data = llm_generated_contextual_anomaly_generator(data=data,
                                                        dataset_name=dataset_name,
                                                        n=n,
                                                        anomaly_type=2,
                                                        random_seed=random_seed,
                                                        k_neighbors=k_neighbors)
    elif anomaly_type == 3:
        data = heuristic_contextual_anomaly_generator(data=data,
                                                        dataset_name=dataset_name,
                                                        n=n,
                                                        anomaly_type=3,
                                                        k=k,
                                                        random_seed=random_seed)
                                                    
    elif anomaly_type == 4:
        if "pubmed" in dataset_name.lower():
            outliner_texts = make_outliner_texts_pubmed()
        elif "wikics" in dataset_name.lower():
            outliner_texts = make_outliner_texts_wikics(loader)
        elif "cora" in dataset_name.lower():
            outliner_texts = make_outliner_texts_cora(loader)
        elif "citeseer" in dataset_name.lower():
            outliner_texts = make_outliner_texts_citeseer(loader)
        else:
            raise ValueError(f"Dataset name: {dataset_name} is not implemented")
        data = global_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        n=n,
                                        anomaly_type=4,
                                        random_seed=random_seed,
                                        outliner_texts=outliner_texts)
    
    elif anomaly_type == 5:
        n_clique = (n // M) +1
        print(f"Gemerate {n_clique*M} structural anomaly nodes")
        data = structural_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        m=M,
                                        n=n_clique,
                                        anomaly_type=5,
                                        random_seed=random_seed)
        n = n_clique*M
    
    print(data)
    save_data(data, "data/generated", dataset_name, anomaly_type, n)