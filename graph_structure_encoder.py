from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from typing import List, Set
import random

GRAPH_STRUCTURE_DESCRIPTION = """The {hop}-hop subgraph centered at node {idx}
This subgraph has the following nodes:
"""

def make_incident_list_for_node(direct_neighbors: List[int], origin_node_idx: int, n: int, subnode_set: Set[int], random_seed: int) -> str:
    has_origin_node = False
    incident_list = []
    out_count = 0
    for neighbor_idx in direct_neighbors:
        if neighbor_idx not in subnode_set:
            out_count += 1
            continue
        if neighbor_idx == origin_node_idx:
            has_origin_node = True
            continue
        else:
            incident_list.append(f"{neighbor_idx}")

    len_incident_list = len(incident_list)

    # if the origin node is in the incident list, we need to subtract 1 from n
    if has_origin_node:
        n = max(n-1, 0)

    # sample n edges from the incident list
    if len_incident_list > n:
        rnd = random.Random(random_seed)
        incident_list = rnd.sample(incident_list, n)
    if has_origin_node:
        # prepend the origin node to the incident list to highlight it
        incident_list.insert(0, f"{origin_node_idx}")
    
    # add a message to summarize the missing nodes
    if len_incident_list > n:
        incident_list.append(f"{len_incident_list - n} more nodes inside the subgraph")
    if out_count > 0:
        incident_list.append(f"{out_count} nodes outside the subgraph")
    return ", ".join(incident_list)


def get_k_hop_neighbors(data: Data, idx: int, k: int)-> List[int]:
    """
    Get the k-hop neighbors of the node
    """
    if k == 0:
        return []
    # get the k-hop subgraph
    subset_k, _, _, _ = k_hop_subgraph(idx, k, data.edge_index)
    if k == 1:
        exact_k = subset_k[subset_k != idx].tolist()
        return exact_k
    # get k-1 hop subgraph
    subset_k_1, _, _, _ = k_hop_subgraph(idx, k-1, data.edge_index)
    # get the neighbors of the node
    set_k   = set(subset_k.tolist())
    set_k_1 = set(subset_k_1.tolist())
    exact_k = list(set_k - set_k_1 - {idx})
    return exact_k


def graph_structure_encoder(data: Data, idx: int, m: int, n: int, random_seed: int) -> str:
    """
    Encode the graph structure for the node with index idx
    Using the incident encoding method
    m is the maximum number of nodes included in the result
    n is the maximum number of edges for each node in the result
    """
    # get the first hop neighbors
    first_hop_neighbors = get_k_hop_neighbors(data, idx, 1)
    first_hop_len = len(first_hop_neighbors)
    # get the second hop neighbors
    second_hop_neighbors = get_k_hop_neighbors(data, idx, 2)
    second_hop_len = len(second_hop_neighbors)
    subnode_set = set(first_hop_neighbors + second_hop_neighbors + [idx])

    is_only_first_hop = False
    if first_hop_len > m:
        is_only_first_hop = True
        # sample m nodes from the first hop neighbors
        rnd = random.Random(random_seed)
        first_hop_neighbors = rnd.sample(first_hop_neighbors, m)

    # make the header of the result string
    if is_only_first_hop:
        result_str = GRAPH_STRUCTURE_DESCRIPTION.format(hop=1, idx=idx)
        result_str += f"{idx}"
        for neighbor_idx in first_hop_neighbors:
            result_str += f", {neighbor_idx}"
    else:
        result_str = GRAPH_STRUCTURE_DESCRIPTION.format(hop=2, idx=idx)
        rnd = random.Random(random_seed)
        if second_hop_len > m-len(first_hop_neighbors):
            # sample m-len(first_hop_neighbors) nodes from the second hop neighbors
            second_hop_neighbors = rnd.sample(second_hop_neighbors, m-len(first_hop_neighbors))
        result_str += f"{idx}"
        for neighbor_idx in first_hop_neighbors:
            result_str += f", {neighbor_idx}"
        for neighbor_idx in second_hop_neighbors:
            result_str += f", {neighbor_idx}"
    
    # make the incident list for each node
    result_str += "\n\nIncident list within the subgraph:\n"
    # add the source idx list first
    current_direct_neighbors = get_k_hop_neighbors(data, idx, 1)
    current_list = f"Node {idx} connects to nodes: " + make_incident_list_for_node(current_direct_neighbors, idx, n, subnode_set, random_seed) + "\n"
    result_str += current_list
    # add the first hop neighbors list
    for neighbor_idx in first_hop_neighbors:
        current_direct_neighbors = get_k_hop_neighbors(data, neighbor_idx, 1)
        current_list = f"Node {neighbor_idx} (1-hop) connects to nodes: " + make_incident_list_for_node(current_direct_neighbors, idx, n, subnode_set, random_seed) + "\n"
        result_str += current_list
    if len(first_hop_neighbors) < first_hop_len:
        # add a message to summarize the missing nodes
        result_str += f"{first_hop_len - len(first_hop_neighbors)} 1-hop nodes omitted (limit reached)"
    
    # if only the first hop is included, return the result
    if is_only_first_hop:
        return result_str

    # add the second hop neighbors list
    for neighbor_idx in second_hop_neighbors:
        current_direct_neighbors = get_k_hop_neighbors(data, neighbor_idx, 1)
        current_list = f"Node {neighbor_idx} (2-hop) connects to nodes: " + make_incident_list_for_node(current_direct_neighbors, idx, n, subnode_set, random_seed) + "\n"
        result_str += current_list
    if len(second_hop_neighbors) < second_hop_len:
        # add a message to summarize the missing nodes
        result_str += f"{second_hop_len - len(second_hop_neighbors)} 2-hop nodes omitted (limit reached)"

    return result_str

if __name__ == "__main__":
    # test code
    from data.raw_data_loader import LLMGNNDataLoader
    loader = LLMGNNDataLoader(data_dir="data/generated")
    data = loader.load_dataset("citeseer_fixed_sbert_2_159", is_map_label=False)
    idx = 0
    print(graph_structure_encoder(data, idx, 100, 100, 37))
    # get the 2-hop subgraph
    subset, _, _, _ = k_hop_subgraph(idx, 2, data.edge_index)
    print(subset)
    # get the edge index
    edge_index = data.edge_index
    # transform into a list of tuples
    edge_list = edge_index.T.tolist()
    # filter the edge list
    edge_list = [edge for edge in edge_list if edge[0] == idx or edge[1] == idx]
    print(edge_list)

    print(len(edge_list))
