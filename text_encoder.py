from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import torch

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
    if not isinstance(texts, list) or len(texts) == 0 or not isinstance(texts[0], str):
        raise ValueError("data.processed_text is empty or not a list of strings")

    # encode the texts, precision="float32"
    new_embeddings = model.encode(sentences=texts, precision="float32")
    if new_embeddings.shape[0] != data.x.shape[0]:
        raise ValueError("the number of embeddings is not equal to the number of nodes")
    # update the data.x with the new embeddings
    data.x = torch.tensor(new_embeddings, dtype=torch.float32)
    return data