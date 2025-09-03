# generate text only anomaly for text-attributed graph

import torch
from torch_geometric.data import Data
from text_encoder import encode_text

ANOMALY_TYPE_LIST = {
    0: "Not Anomaly",
    1: "Dummy Text Only Anomaly",
}


def text_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    Generate text only anomaly for text-attributed graph
    """
    if anomaly_type == 0 or anomaly_type not in ANOMALY_TYPE_LIST:
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")

    if "cora" in dataset_name.lower():
        data = cora_anomaly_generator(data, n, anomaly_type, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")
    return data


def cora_anomaly_generator(data: Data, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    Generate text only anomaly for cora dataset
    cora has the following attributes:
    .raw_texts: List[str] the original text of nodes
    .category_names: List[str] the category of nodes
    .label_names: List[str], 7 labels: ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods']
    .x: torch.Tensor, shape: torch.Size([2708, 384]), dtype: torch.float32
    .raw_text: List[str], the original text of nodes
    .val_masks: List[torch.Tensor], length: 10, element type: <class 'torch.Tensor'>
    .edge_index: torch.Tensor, shape: torch.Size([2, 10858]), dtype: torch.int64
    .train_masks: List[torch.Tensor], length: 10, element type: <class 'torch.Tensor'>
    .y: torch.Tensor, shape: torch.Size([2708]), dtype: torch.int64
    .test_masks: List[torch.Tensor], length: 10, element type: <class 'torch.Tensor'>

    Optional:
    .processed_text: List[str], the processed text of nodes
    .anomaly_labels: torch.Tensor, shape: torch.Size([2708]), dtype: torch.int64, the label of anomaly, 0 for normal, 1 for anomaly
    .anomaly_types: List[int], the type of anomaly
    .updated_x: torch.Tensor, shape: torch.Size([2708, 384]), dtype: torch.float32, the updated text embeddings of nodes
    """
    raw_text = data.raw_text
    node_num = len(raw_text)

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

    # generate anomaly according to the anomaly type
    if anomaly_type == 1:
        # generate dummy text only anomaly
        data = dummy_text_only_anomaly(data, selected_idxs)
    else:
        raise ValueError(f"Anomaly type: {anomaly_type} is not implemented")

    # encode the text to embeddings
    data = encode_text(data)
    # The updated embeddings should be stored in data.updated_x
    return data


def dummy_text_only_anomaly(data: Data, selected_idxs: torch.Tensor) -> Data:
    """
    Generate dummy text only anomaly
    replace the original text with a dummy text
    """
    dummy_text = "This is a dummy text"
    
    # get the processed text, anomaly labels, and anomaly types
    processed_text = data.raw_text.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    # replace the original text with the dummy text
    for idx in selected_idxs.tolist():
        processed_text[idx] = dummy_text
        anomaly_labels[idx] = 1
        anomaly_types[idx] = 1

    # update the data
    data.processed_text = processed_text
    data.anomaly_labels = anomaly_labels
    data.anomaly_types = anomaly_types
    return data