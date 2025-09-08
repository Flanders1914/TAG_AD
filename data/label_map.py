from torch_geometric.data import Data
from typing import Dict

arxiv_label_map = {
    'arxiv cs na': "Computer Science: Numerical Analysis",
    'arxiv cs mm': "Computer Science: Multimedia",
    'arxiv cs lo': "Computer Science: Logic in Computer Science",
    'arxiv cs cy': "Computer Science: Computers and Society",
    'arxiv cs cr': "Computer Science: Cryptography and Security",
    'arxiv cs dc': "Computer Science: Distributed, Parallel, and Cluster Computing",
    'arxiv cs hc': "Computer Science: Human-Computer Interaction",
    'arxiv cs ce': "Computer Science: Computational Engineering, Finance, and Science",
    'arxiv cs ni': "Computer Science: Networking and Internet Architecture",
    'arxiv cs cc': "Computer Science: Computational Complexity",
    'arxiv cs ai': "Computer Science: Artificial Intelligence",
    'arxiv cs ma': "Computer Science: Multiagent Systems",
    'arxiv cs gl': "Computer Science: General Literature",
    'arxiv cs ne': "Computer Science: Neural and Evolutionary Computing",
    'arxiv cs sc': "Computer Science: Symbolic Computation",
    'arxiv cs ar': "Computer Science: Hardware Architecture",
    'arxiv cs cv': "Computer Science: Computer Vision and Pattern Recognition",
    'arxiv cs gr': "Computer Science: Graphics",
    'arxiv cs et': "Computer Science: Emerging Technologies",
    'arxiv cs sy': "Computer Science: Systems and Control",
    'arxiv cs cg': "Computer Science: Computational Geometry",
    'arxiv cs oh': "Computer Science: Other Computer Science",
    'arxiv cs pl': "Computer Science: Programming Languages",
    'arxiv cs se': "Computer Science: Software Engineering",
    'arxiv cs lg': "Computer Science: Machine Learning",
    'arxiv cs sd': "Computer Science: Sound",
    'arxiv cs si': "Computer Science: Social and Information Networks",
    'arxiv cs ro': "Computer Science: Robotics",
    'arxiv cs it': "Computer Science: Information Theory",
    'arxiv cs pf': "Computer Science: Performance",
    'arxiv cs cl': "Computer Science: Computation and Language",
    'arxiv cs ir': "Computer Science: Information Retrieval",
    'arxiv cs ms': "Computer Science: Mathematical Software",
    'arxiv cs fl': "Computer Science: Formal Languages and Automata Theory",
    'arxiv cs ds': "Computer Science: Data Structures and Algorithms",
    "arxiv cs os": "Computer Science: Operating Systems",
    'arxiv cs gt': "Computer Science: Computer Science and Game Theory",
    'arxiv cs db': "Computer Science: Databases",
    'arxiv cs dl': "Computer Science: Digital Libraries",
    'arxiv cs dm': "Computer Science: Discrete Mathematics",
}

citeseer_label_map = {
    "Agents": "Computer Science: Agents",
    "ML": "Computer Science: Machine Learning",
    "IR": "Computer Science: Information Retrieval",
    "DB": "Computer Science: Database",
    "HCI": "Computer Science: Human Computer Interaction",
    "AI": "Computer Science: Artificial Intelligence",
}

cora_label_map = {
    "Case_Based": "Machine Learning: Case-Based",
    "Genetic_Algorithms": "Machine Learning: Genetic Algorithms",
    "Neural_Networks": "Machine Learning: Neural Networks",
    "Probabilistic_Methods": "Machine Learning: Probabilistic Methods",
    "Reinforcement_Learning": "Machine Learning: Reinforcement Learning",
    "Rule_Learning": "Machine Learning: Rule Learning",
    "Theory": "Machine Learning: Theory",
}

def map_label(data: Data, dataset_name: str) -> Data:
    """
    Map the label of the data to the label of the dataset.
    """
    if "arxiv" in dataset_name.lower():
        print("Mapping labels for arxiv dataset")
        return map_label_with_map(data, arxiv_label_map)
    elif "citeseer" in dataset_name.lower():
        print("Mapping labels for citeseer dataset")
        return map_label_with_map(data, citeseer_label_map)
    elif "cora" in dataset_name.lower():
        print("Mapping labels for cora dataset")
        return map_label_with_map(data, cora_label_map)
    else:
        print(f"No need to map labels for {dataset_name} dataset")
        return data


def map_label_with_map(data: Data, label_map: Dict[str, str]) -> Data:
    """
    Map the label of the data to the label of the arxiv dataset.
    The order of both .label_names and .category_names is preserved.
    Map the following attributes:
    .label_names: List[str]
    .category_names: List[str]
    """
    # map the label_names
    new_label_names = []
    for label in data.label_names:
        if label in label_map:
            new_label_names.append(label_map[label])
        else:
            raise ValueError(f"Label {label} not found in label_map")
    data.label_names = new_label_names
    # map the category_names
    new_category_names = []
    for label in data.category_names:
        if label in label_map:
            new_category_names.append(label_map[label])
        else:
            raise ValueError(f"Label {label} not found in label_map")
    data.category_names = new_category_names
    return data

if __name__ == "__main__":
    labels = citeseer_label_map.values()
    labels_str = ", ".join(labels)
    print(labels_str)