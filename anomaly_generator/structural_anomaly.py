# Structural anomaly generation
import random
import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
import pygod

# ---------- main functions ----------

def structural_anomaly_generator(data: Data, dataset_name: str, m: int, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    Generate dummy anomaly for text-attributed graph
    data must have the .edge_index attribute
    """
    if "cora" not in dataset_name.lower() and \
        "citeseer" not in dataset_name.lower() and \
        "pubmed" not in dataset_name.lower() and \
        "wikics" not in dataset_name.lower():
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")
    if ANOMALY_TYPE_LIST[anomaly_type] != "Structural Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    if not hasattr(data, "edge_index"):
        raise ValueError("data must have the .edge_index attribute")

    # use pygod to generate structural anomaly
    data, y_outlier = pygod.generator.gen_structural_outlier(data=data, m=m, n=n, seed=random_seed)
    y_list = y_outlier.tolist()
    assert len([y for y in y_list if y == 1]) == n*m, f"The number of anomaly nodes is not equal to n*m"
    data.anomaly_labels = y_outlier
    data.anomaly_types = [0]*len(y_list)
    for idx in range(len(y_list)):
        if y_list[idx] == 1:
            data.anomaly_types[idx] = anomaly_type
    
    print("Structural anomaly generation completed")
    return data