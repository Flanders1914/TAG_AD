# Dummy anomaly generation
import random
import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import fit_unigram_and_lengths, sample_length, sample_text, encode_text

# ---------- main functions ----------

def dummy_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    Generate dummy anomaly for text-attributed graph
    """
    if ANOMALY_TYPE_LIST[anomaly_type] != "Dummy Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    if "cora" in dataset_name.lower():
        return dummy_anomaly_generation_pipeline(data, n, anomaly_type, random_seed)
    elif "citeseer" in dataset_name.lower():
        return dummy_anomaly_generation_pipeline(data, n, anomaly_type, random_seed)
    elif "pubmed" in dataset_name.lower():
        return dummy_anomaly_generation_pipeline(data, n, anomaly_type, random_seed)
    elif "arxiv" in dataset_name.lower():
        return dummy_anomaly_generation_pipeline(data, n, anomaly_type, random_seed)
    elif "wikics" in dataset_name.lower():
        return dummy_anomaly_generation_pipeline(data, n, anomaly_type, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")

# ------------ Pipeline ------------------
def dummy_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    The pipeline to generate dummy anomaly
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
    # generate dummy anomaly
    data = dummy_text_only_anomaly(data, selected_idxs, anomaly_type, random_seed)
    # encode the text to embeddings
    data = encode_text(data)
    # The updated embeddings should be stored in data.updated_x
    return data

# Dummy anomaly generation
def dummy_text_only_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, random_seed: int) -> Data:
    """
    Generate dummy text only anomaly
    replace the original text with a dummy text
    """
    print("Generating dummy text only anomaly...")
    # step 1: get the distribution of unigrams and lengths
    unigram_list, unigram_probs, lengths = fit_unigram_and_lengths(data.raw_texts)

    # step 2: initialize the processed text, anomaly labels, and anomaly types
    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    # step 3: replace the original text with the dummy text
    count = 0
    for idx in selected_idxs.tolist():
        print(f"Generating dummy anomaly for node {idx}...")
        # Use the index of the node as the random seed to ensure the text are different and reproducible
        seed = random_seed + idx
        rng = random.Random(seed)
        # sample the length of the text
        length = sample_length(lengths, rng)
        # sample the text
        dummy_text = sample_text(unigram_list, unigram_probs, length, rng)
        processed_text[idx] = dummy_text
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
    print("Dummy text anomaly generation completed")
    return data