from collections import Counter
from typing import List, Tuple
import random
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import torch

# ------------- count and sample -------------

# text unigram and length distribution fitting
def fit_unigram_and_lengths(raw_text: List[str]) -> Tuple[List[str], List[float], List[int]]:
    """
    Fit unigram and lengths of texts
    Return:
        - unigram_list: List[str], the list of unigrams
        - unigram_probs: List[float], the probabilities of unigrams
        - lengths: List[int], the lengths of texts
    """
    unigram_counter = Counter()
    lengths = []
    for text in raw_text:
        tokens = text.split()
        unigram_counter.update(tokens)
        lengths.append(len(tokens))
    # calculate the probabilities of unigrams
    items = list(unigram_counter.items())
    total = sum(count for _, count in items)
    if total == 0:
        raise ValueError("Total count of unigrams is 0")
    unigram_probs = [count / total for _, count in items]
    unigram_list = [unigram for unigram, _ in items]
    return unigram_list, unigram_probs, lengths

# sample the length of the text randomly
def sample_length(lengths: List[int], rng: random.Random) -> int:
    """
    Sample the length of the text randomly
    """
    return rng.choices(lengths, k=1)[0]

# sample text based on the unigram probabilities
def sample_text(unigram_list: List[str], unigram_probs: List[float], length: int, rng: random.Random) -> str:
    """
    Sample the text randomly
    """
    text = " ".join(rng.choices(unigram_list, weights=unigram_probs, k=length))
    return text

# ------------- text encoder -------------

def encode_text(data: Data, model_name: str = "all-MiniLM-L6-v2") -> Data:
    """
    This function should update the data.x with new text embeddings.
    Encode the processed text data into embeddings
    Update the data with new text embeddings
    Required attributes:
        data.processed_text: List[str] the new text of node
        data.x: the original node features shape(number of nodes, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    texts = data.processed_text
    if texts is None:
        raise ValueError("data.processed_text is None")
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("data.processed_text must be a non-empty list")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All elements in data.processed_text must be strings")
    
    if not hasattr(data, 'x'):
        raise AttributeError("data.x attribute is missing")
    
    try:
        # encode the texts, precision="float32"
        new_embeddings = model.encode(sentences=texts, precision="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to encode texts: {e}")
    
    if new_embeddings.shape[0] != data.x.shape[0]:
        raise ValueError(f"Mismatch in number of embeddings ({new_embeddings.shape[0]}) and nodes ({data.x.shape[0]})")
    
    # update the data.x with the new embeddings
    data.updated_x = torch.tensor(new_embeddings, dtype=torch.float32, device=data.x.device)
    return data

# ------------- similarity calculation -------------

def calculate_similarity(data: Data) -> Data:
    """
    Calculate the similarity between the original text embeddings and the updated text embeddings
    """
    original_embeddings = data.x
    updated_embeddings = data.updated_x
    similarity = torch.cosine_similarity(original_embeddings, updated_embeddings, dim=1)
    # split the similarity into normal and anomaly
    normal_similarity = similarity[data.anomaly_labels == 0]
    anomaly_similarity = similarity[data.anomaly_labels == 1]
    # calculate the average similarity
    normal_similarity = normal_similarity.mean()
    anomaly_similarity = anomaly_similarity.mean()
    return normal_similarity, anomaly_similarity