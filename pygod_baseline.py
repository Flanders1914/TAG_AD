# use pygod framework to detect anomaly
from data.raw_data_loader import LLMGNNDataLoader
import torch
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from torch_geometric.data import Data
from anomaly_generator import text_anomaly_generator


def clean_data(data: Data) -> Data:
    """
    Clean the data by deleting unncessary attributes
    """
    out = Data()
    N = int(getattr(data, "num_nodes", data.x.size(0)))
    # x
    if hasattr(data, "x"):
        out.x = torch.as_tensor(data.x, dtype=torch.float)
    else:
        raise ValueError("the x attribute is required")
    # y
    if hasattr(data, "y"):
        y = torch.as_tensor(data.y)
        if y.dim() and y.size(0) == N:
            y = y.to(torch.long)
            out.y = y
        else:
            raise ValueError("y has incorrect shape")
    else:
        raise ValueError("the y attribute is required")
    # edge_index
    if hasattr(data, "edge_index"):
        out.edge_index = torch.as_tensor(data.edge_index, dtype=torch.long)
    else:
        raise ValueError("the edge_index attribute is required")
    
    return out


def add_pygod_anomaly(data: Data) -> Data:
    """
    Add pygod anomaly to the data
    """
    # contextual outlier
    # Randomly select n nodes as the attribute perturbation candidates.
    # For each selected node , randomly pick another k nodes from the data
    # and select the node whose attributes deviate the most among k nodes by maximizing the Euclidean distance 
    data, ya = gen_contextual_outlier(data, n=100, k=50)

    # structural outlier
    # Randomly select m nodes from the network and then make those nodes fully connected,
    # and then all the m nodes in the clique are regarded as outliers.
    # Iteratively repeat this process until a number of n cliques are generated
    # The total number of structural outliers is m * n
    data, ys = gen_structural_outlier(data, m=10, n=10)
    data.y = torch.logical_or(ys, ya).long()
    return data


if __name__ == "__main__":
    # load the local 
    loader = LLMGNNDataLoader(data_dir='data/raw')
    data = loader.load_dataset('cora_fixed_sbert')

    # Add anomaly
    print("Adding anomaly...")
    data = text_anomaly_generator(data, 'cora_fixed_sbert', 200, 1, 42)
    data.y = data.anomaly_labels.long()

    # show some example
    # Select one node where y == 1 and print its processed text
    anomaly_indices = (data.y == 1).nonzero(as_tuple=True)[0]
    print(f"Number of nodes with y == 1: {len(anomaly_indices)}")
    if len(anomaly_indices) > 0:
        idx = anomaly_indices[0].item()
        print(f"Example processed_text for node with y==1 (index {idx}):")
        print(data.processed_text[idx])
    else:
        print("No nodes with y == 1 found.")

    # clean the data
    data = clean_data(data)

    # Initialization
    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=100)
    # train the model
    detector.fit(data)

    # predict the anomaly
    pred, score, prob, conf = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
    print('Labels:')
    print(pred)
    print('Raw scores:')
    print(score)
    print('Probability:')
    print(prob)
    print('Confidence:')
    print(conf)
    
    # evaluate the model
    auc_score = eval_roc_auc(data.y, score)
    print('AUC Score:', auc_score)