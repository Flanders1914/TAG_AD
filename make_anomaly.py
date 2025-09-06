# python make_anomaly.py
from anomaly_generator import llm_generated_contextual_anomaly_generator
from data.raw_data_loader import LLMGNNDataLoader
from torch_geometric.data import Data
import torch
import os

def save_data(data: Data, data_dir: str, dataset_name: str, anomaly_type: int, anomaly_num: int) -> None:
    file_path = os.path.join(data_dir, f"{dataset_name}_{anomaly_type}_{anomaly_num}.pt")
    torch.save(data, file_path)

if __name__ == "__main__":
    loader = LLMGNNDataLoader(data_dir="data/raw")
    data = loader.load_dataset("cora_fixed_sbert")
    data = llm_generated_contextual_anomaly_generator(data=data,
                                                    dataset_name="cora_fixed_sbert",
                                                    n=1,
                                                    anomaly_type=2,
                                                    random_seed=42,
                                                    k_neighbors=2)