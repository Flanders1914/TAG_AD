# use pygod framework to detect anomaly
from data.raw_data_loader import LLMGNNDataLoader
import torch
from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GADNR, GAE, GUIDE, OCGNN, ONE, Radar, SCAN
from pygod.metric import eval_roc_auc, eval_average_precision, eval_f1, eval_precision_at_k, eval_recall_at_k
from torch_geometric.data import Data
from anomaly_generator import text_anomaly_generator
from text_encoder import calculate_similarity
import argparse
import random
import numpy as np
import json


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def prepare_data(data_dir: str, dataset_name: str, n: int, anomaly_type: int, random_seed: int) -> Data:
    """
    Prepare the data for the baseline experiment
    """
    # Step 1: load the local data
    print("Loading data...")
    loader = LLMGNNDataLoader(data_dir=data_dir)
    data = loader.load_dataset(dataset_name)

    # Step 2: add anomaly
    print("Adding anomaly...")
    data = text_anomaly_generator(data, dataset_name, n, anomaly_type, random_seed)

    # step 3: validate the anomaly by calculating the similarity
    print("Validating anomaly...")
    normal_similarity, anomaly_similarity = calculate_similarity(data)
    print(f"Normal similarity: {normal_similarity}, Anomaly similarity: {anomaly_similarity}")

    # step 4: update the data with the updated embeddings and anomaly labels
    data.y = data.anomaly_labels.long()
    data.x = data.updated_x
    print("Unique labels in data.y:", torch.unique(data.y))
    assert data.y.min() >= 0 and data.y.max() <= 1, \
        f"Invalid labels found in y: {torch.unique(data.y)}"
    assert data.y.dim() == 1 and data.y.size(0) == data.x.size(0), \
        f"y shape mismatch: {data.y.shape} vs num_nodes={data.x.size(0)}"

    # step 5: show some example
    # Select two nodes where y == 1 and print its processed text
    anomaly_indices = (data.y == 1).nonzero(as_tuple=True)[0]
    print(f"Number of nodes with y == 1: {len(anomaly_indices)}")
    if len(anomaly_indices) > 1:
        idx = anomaly_indices[0].item()
        print(f"Example 1: processed_text for node with y==1 (index {idx}):")
        print(data.processed_text[idx])
        idx = anomaly_indices[1].item()
        print(f"Example 2: processed_text for node with y==1 (index {idx}):")
        print(data.processed_text[idx])
    else:
        print("Less than 2 nodes with y == 1 found.")

    # step 6: clean the data
    print("Cleaning data...")
    data = clean_data(data)
    print("Data preparation completed")
    print()
    return data


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

# ---------- main functions ----------

def baseline_experiment(data: Data, k: int, random_seed: int) -> dict:
    """
    Run the baseline experiment
    """
    result = {
        'random_seed': random_seed
    }

    # random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # AdONE
    print("Running AdONE...")
    detector = AdONE(gpu=0)
    result['AdONE'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # ANOMALOUS
    print("Running ANOMALOUS...")
    detector = ANOMALOUS(gpu=0)
    result['ANOMALOUS'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # AnomalyDAE
    print("Running AnomalyDAE...")
    detector = AnomalyDAE(gpu=0)
    result['AnomalyDAE'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # CoLA
    print("Running CoLA...")
    detector = CoLA(gpu=0)
    result['CoLA'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # CONAD
    print("Running CONAD...")
    detector = CONAD(gpu=0)
    result['CONAD'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # DMGD
    print("Running DMGD...")
    detector = DMGD(gpu=-1) 
    result['DMGD'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # DOMINANT
    print("Running DOMINANT...")
    detector = DOMINANT(gpu=0)
    result['DOMINANT'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # DONE
    print("Running DONE...")
    detector = DONE(gpu=0)
    result['DONE'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # GAAN
    #print("Running GAAN...")
    #detector = GAAN(gpu=0)
    #result['GAAN'] = train_and_evaluate(data.clone(), detector, k)
    #torch.cuda.empty_cache()

    # GADNR
    #print("Running GADNR...")
    #detector = GADNR(gpu=0)
    #result['GADNR'] = train_and_evaluate(data.clone(), detector, k)
    #torch.cuda.empty_cache()

    # GAE
    print("Running GAE...")
    detector = GAE(gpu=0)
    result['GAE'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # GUIDE
    #print("Running GUIDE...")
    #detector = GUIDE(gpu=0)
    #result['GUIDE'] = train_and_evaluate(data.clone(), detector, k)
    #torch.cuda.empty_cache()

    # OCGNN
    print("Running OCGNN...")
    detector = OCGNN(gpu=0)
    result['OCGNN'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # ONE
    print("Running ONE...")
    detector = ONE(gpu=0)
    result['ONE'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()
    
    # Radar
    print("Running Radar...")
    detector = Radar(gpu=0)
    result['Radar'] = train_and_evaluate(data.clone(), detector, k)
    torch.cuda.empty_cache()

    # SCAN
    #print("Running SCAN...")
    #detector = SCAN(gpu=0)
    #result['SCAN'] = train_and_evaluate(data.clone(), detector, k)
    #torch.cuda.empty_cache()
    # return the result
    return result


def train_and_evaluate(data, detector, k: int) -> dict:
    # step 1: train the model
    detector.fit(data)
    # step 2: do prediction
    pred, score = detector.predict(data, return_pred=True, return_score=True)
    # step 3: evaluate the model
    label = data.y
    avg_precision = eval_average_precision(label, score)
    f1_score = eval_f1(label, pred)
    precision_at_k = eval_precision_at_k(label, score, k)
    recall_at_k = eval_recall_at_k(label, score, k)
    auc_score = eval_roc_auc(label, score)

    result = {
        'auc_score': _to_float(auc_score),
        'avg_precision': _to_float(avg_precision),
        'f1_score': _to_float(f1_score),
        'precision_at_k': _to_float(precision_at_k),
        'recall_at_k': _to_float(recall_at_k)
    }
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--dataset_name', type=str, default='cora_fixed_sbert')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--anomaly_type', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--experiment_num', type=int, default=20)
    parser.add_argument('--output_file', type=str, default='results.json')
    args = parser.parse_args()
    
    # prepare the data
    data = prepare_data(args.data_dir, args.dataset_name, args.n, args.anomaly_type, args.random_seed)

    # run the baseline experiment for multiple times
    results = []
    for i in range(args.experiment_num):
        print(f"Running experiment {i+1}...")
        result = baseline_experiment(data, args.n, args.random_seed+i)
        results.append(result)

    # analyze the results
    final_result = {
        'dataset_name': args.dataset_name,
        'anomaly_type': args.anomaly_type,
        'anomaly_num': args.n,
    }
    for key in results[0].keys():
       if key == 'random_seed':
           continue
       auc_mean = np.mean([result[key]['auc_score'] for result in results])
       auc_max = np.max([result[key]['auc_score'] for result in results])
       auc_min = np.min([result[key]['auc_score'] for result in results])
       avg_precision_mean = np.mean([result[key]['avg_precision'] for result in results])
       avg_precision_max = np.max([result[key]['avg_precision'] for result in results])
       avg_precision_min = np.min([result[key]['avg_precision'] for result in results])
       f1_score_mean = np.mean([result[key]['f1_score'] for result in results])
       f1_score_max = np.max([result[key]['f1_score'] for result in results])
       f1_score_min = np.min([result[key]['f1_score'] for result in results])
       precision_at_k_mean = np.mean([result[key]['precision_at_k'] for result in results])
       precision_at_k_max = np.max([result[key]['precision_at_k'] for result in results])
       precision_at_k_min = np.min([result[key]['precision_at_k'] for result in results])
       recall_at_k_mean = np.mean([result[key]['recall_at_k'] for result in results])
       recall_at_k_max = np.max([result[key]['recall_at_k'] for result in results])
       recall_at_k_min = np.min([result[key]['recall_at_k'] for result in results])
       final_result[key] = {
           'auc_mean': auc_mean,
           'auc_max': auc_max,
           'auc_min': auc_min,
           'avg_precision_mean': avg_precision_mean,
           'avg_precision_max': avg_precision_max,
           'avg_precision_min': avg_precision_min,
           'f1_score_mean': f1_score_mean,
           'f1_score_max': f1_score_max,
           'f1_score_min': f1_score_min,
           'precision_at_k_mean': precision_at_k_mean,
           'precision_at_k_max': precision_at_k_max,
           'precision_at_k_min': precision_at_k_min,
           'recall_at_k_mean': recall_at_k_mean,
           'recall_at_k_max': recall_at_k_max,
           'recall_at_k_min': recall_at_k_min
        }

    final_result["experiment_result"] = results
    
    # save the results
    with open(args.output_file, 'w') as f:
        json.dump(final_result, f, indent=4)