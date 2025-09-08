# Global anomaly generation
import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import encode_text
import random
from typing import List

# ---------- main functions ----------

def global_anomaly_generator(data: Data, dataset_name: str, n: int, outliner_texts: List[str], anomaly_type: int, random_seed: int) -> Data:
    """
    Generate heuristic contextual anomaly for text-attributed graph
    """
    if ANOMALY_TYPE_LIST[anomaly_type] != "Global Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    if "cora" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "citeseer" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "pubmed" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "arxiv" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")

# ------------ Pipeline ------------------
def global_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, outliner_texts: List[str], random_seed: int) -> Data:
    """
    The pipeline to generate global anomaly
    data must have the following attributes:
    .raw_texts: List[str] the original text attribute of nodes
    .edge_index: torch.Tensor, shape: torch.Size([2, number of edges]), dtype: torch.int64

    Optional:
    .processed_text: List[str], the processed text attribute of nodes
    .anomaly_labels: torch.Tensor, shape: torch.Size([number of nodes]), dtype: torch.int64, the label of anomaly, 0 for normal, 1 for anomaly
    .anomaly_types: List[int], the type of anomaly
    .updated_x: torch.Tensor, shape: torch.Size([number of nodes, embedding_dim]), dtype: torch.float32, the updated text embeddings of nodes
    """
    raw_texts = data.raw_texts
    node_num = len(raw_texts)
    # get the index of normal nodes
    if hasattr(data, "anomaly_labels"):
        normal_idxs = (data.anomaly_labels == 0).nonzero(as_tuple=True)[0]
    else:
        normal_idxs = torch.arange(node_num)
    # if the number of normal nodes is less than n, raise error
    if normal_idxs.shape[0] < n:
        raise ValueError(f"The number of normal nodes is less than n: {normal_idxs.shape[0]} < {n}")
    # randomly select n nodes
    gen = torch.Generator(device=normal_idxs.device).manual_seed(int(random_seed))
    perm = torch.randperm(normal_idxs.numel(), generator=gen, device=normal_idxs.device)
    selected_idxs = normal_idxs[perm[:n]]
    # generate heuristic contextual anomaly
    data = global_anomaly(data, selected_idxs, anomaly_type, outliner_texts, random_seed)
    # encode the text to embeddings
    data = encode_text(data)
    # The updated embeddings should be stored in data.updated_x
    return data

# Dummy anomaly generation
def global_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, outliner_texts: List[str], random_seed: int) -> Data:
    """
    Generate global anomaly
    For each selected node, replace raw texts with random outliner texts
    """
    print("Generating global anomaly...")
    # step 1: initialize the processed text, anomaly labels, and anomaly types
    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    # step 2: replace the original text with the global anomaly
    count = 0
    for idx in selected_idxs.tolist():
        # Use the index of the node as the random seed to ensure the text are different and reproducible
        seed = random_seed + idx
        rng = random.Random(seed)
        # sample a outliner text
        outliner_text = rng.choice(outliner_texts)
        # replace the words with the words from the most distant node in randomly selected k nodes
        processed_text[idx] = outliner_text
        anomaly_labels[idx] = 1
        anomaly_types[idx] = anomaly_type
        count += 1
        if count % 100 == 0:
            print("--------------------------------")
            print(f"Generated {count} nodes")
            print("--------------------------------")

    # step 4: update the data
    data.processed_text = processed_text
    data.anomaly_labels = anomaly_labels
    data.anomaly_types = anomaly_types
    print("Heuristic contextual anomaly generation completed")
    return data
