# Heuristic contextual anomaly generation
import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import encode_text, TextEncoder
import random

# ---------- main functions ----------

def heuristic_contextual_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, k: int, random_seed: int) -> Data:
    """
    Generate heuristic contextual anomaly for text-attributed graph
    """
    if ANOMALY_TYPE_LIST[anomaly_type] != "Heuristic Contextual Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    if "cora" in dataset_name.lower():
        return heuristic_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "citeseer" in dataset_name.lower():
        return heuristic_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "pubmed" in dataset_name.lower():
        return heuristic_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "arxiv" in dataset_name.lower():
        return heuristic_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "wikics" in dataset_name.lower():
        return heuristic_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")

# ------------ Pipeline ------------------
def heuristic_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, k: int, random_seed: int) -> Data:
    """
    The pipeline to generate heuristic contextual anomaly
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
    data = heuristic_text_only_anomaly(data, selected_idxs, anomaly_type, k, random_seed)
    # encode the text to embeddings
    data = encode_text(data)
    # The updated embeddings should be stored in data.updated_x
    return data

# Dummy anomaly generation
def heuristic_text_only_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, k: int, random_seed: int) -> Data:
    """
    Generate heuristic contextual anomaly
    For each selected node, replace some words with some words from the most distant node in randomly selected k nodes
    """
    print("Generating heuristic contextual anomaly...")
    # step 1: initialize the processed text, anomaly labels, and anomaly types
    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    # step 2: replace the original text with the heuristic contextual anomaly
    count = 0
    text_encoder = TextEncoder()
    for idx in selected_idxs.tolist():
        # Use the index of the node as the random seed to ensure the text are different and reproducible
        seed = random_seed + idx
        gen = torch.Generator(device=selected_idxs.device).manual_seed(int(seed))
        # sample k nodes across all nodes
        num_node = len(data.raw_texts)
        selected_k_idxs = torch.randperm(num_node, generator=gen, device=data.x.device)[:k]
        # get the most distant node in randomly selected k nodes
        most_distant_idx = get_most_distant_node(data, idx, selected_k_idxs, text_encoder)
        # replace the words with the words from the most distant node in randomly selected k nodes
        processed_text[idx] = replace_words(data.raw_texts[idx], data.raw_texts[most_distant_idx], random_seed)
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


def get_most_distant_node(data: Data, idx: int, selected_k_idxs: torch.Tensor, text_encoder: TextEncoder) -> int:
    """
    Get the most distant node in randomly selected k nodes
    distance is measured by cosine similarity
    """
    selected_k_idxs = selected_k_idxs.tolist()
    # get all the text in the selected k nodes
    selected_k_text_list = [data.raw_texts[i] for i in selected_k_idxs]
    # get the current node text
    current_node_text_list = [data.raw_texts[idx]] * len(selected_k_text_list)
    # encode the text to embeddings
    current_node_embeddings = text_encoder.encode_list(current_node_text_list)
    selected_k_embeddings = text_encoder.encode_list(selected_k_text_list)
    # get the similarity between the current node embeddings and the selected k nodes
    similarity = torch.cosine_similarity(current_node_embeddings, selected_k_embeddings)
    # get the most distant node, aka the node with the lowest similarity
    pos = int(similarity.argmin().item())
    most_distant_idx = selected_k_idxs[pos]
    return most_distant_idx


def replace_words(text1: str, text2: str, random_seed: int) -> str:
    """
    Replace some words in text1 with some words in text2
    """
    rng = random.Random(random_seed)
    # split the text into word list
    word_list1 = text1.split()
    len_word_list1 = len(word_list1)
    word_list2 = text2.split()
    len_word_list2 = len(word_list2)
    
    # Handle edge cases
    if len_word_list1 == 0 or len_word_list2 == 0:
        raise ValueError("Text is empty")
    
    # select how many words to replace: randomly select from 1/4*len_word_list1 to 1/2*len_word_list1
    min_replace = max(1, len_word_list1 // 4)  # Ensure at least 1 word
    max_replace = max(min_replace, len_word_list1 // 2)  # Ensure max >= min
    num_words_to_replace = rng.randint(min_replace, max_replace)
    
    # if the number of words to replace is greater than the number of words in text2, set it to the number of words in text2
    if num_words_to_replace > len_word_list2:
        num_words_to_replace = len_word_list2
    
    # Ensure we don't try to replace more words than available in text1
    if num_words_to_replace > len_word_list1:
        num_words_to_replace = len_word_list1
    
    # select num_words_to_replace words from word_list2
    selected_words = rng.sample(word_list2, num_words_to_replace)
    
    # select a start index from word_list1 to replace the selected words
    max_start_idx = len_word_list1 - num_words_to_replace
    start_idx = rng.randint(0, max(0, max_start_idx))
    
    # replace the selected words with the selected words from word_list2
    for i in range(num_words_to_replace):
        word_list1[start_idx + i] = selected_words[i]
    
    # join the word list back into a string
    result_text = " ".join(word_list1)
    return result_text