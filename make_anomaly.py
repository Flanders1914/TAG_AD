# python make_anomaly.py
from anomaly_generator import llm_generated_contextual_anomaly_generator, heuristic_contextual_anomaly_generator, global_anomaly_generator, dummy_anomaly_generator
from data.raw_data_loader import LLMGNNDataLoader
from torch_geometric.data import Data
import torch
import os

def save_data(data: Data, data_dir: str, dataset_name: str, anomaly_type: int, anomaly_num: int) -> None:
    file_path = os.path.join(data_dir, f"{dataset_name}_{anomaly_type}_{anomaly_num}.pt")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, file_path)

if __name__ == "__main__":
    dataset_name="pubmed_fixed_sbert"
    n=986
    loader = LLMGNNDataLoader(data_dir="data/raw")
    data = loader.load_dataset(dataset_name, is_map_label=True)
    anomaly_type = 2
    k_neighbors = 3
    k = 50
    random_seed = 42
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
            data_outliner = loader.load_dataset("citeseer_fixed_sbert", is_map_label=False)
            outliner_texts = data_outliner.raw_texts
        else:
            data_outliner = loader.load_dataset("pubmed_fixed_sbert", is_map_label=False)
            outliner_texts = data_outliner.raw_texts
        data = global_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        n=n,
                                        anomaly_type=4,
                                        random_seed=random_seed,
                                        outliner_texts=outliner_texts)
    
    print(data)
    save_data(data, "data/generated", dataset_name, anomaly_type, n)