# TAG_AD: Text-Attributed Graph Anomaly Detection

## Overview

TAG_AD provides tools for:
- **Anomaly Generation**: Create synthetic anomalies in text-attributed graphs
- **Text Encoding**: Convert text attributes to embeddings using sentence transformers
- **Anomaly Detection**: Detect anomalies using PyGOD framework algorithms
- **Evaluation**: Comprehensive evaluation metrics and baselines

## Installation

### Prerequisites

```bash
conda create -n ad_env python=3.11
conda activate ad_env
pip install requirements.txt
```

### Clone and Setup

```bash
git clone git@github.com:Flanders1914/TAG_AD.git
cd TAG_AD
```

## Datasets

We use 4 datasets from the [LLMGNN repository](https://github.com/CurryTang/LLMGNN/tree/master):

### Dataset Statistics

| Dataset Name  | #Nodes   | #Edges   | Task Description                                      | Classes                                                                 |
|---------------|----------|----------|-------------------------------------------------------|-------------------------------------------------------------------------|
| CORA          | 2,708    | 5,429    | Given the title and abstract, predict the category of this paper | Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods |
| CITESEER      | 3,186    | 4,277    | Given the title and abstract, predict the category of this paper | Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence |
| PUBMED        | 19,717   | 44,335   | Given the title and abstract, predict the category of this paper | Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 |
| WIKICS        | 11,701   | 215,863  | Given the contents of the Wikipedia article, predict the category of this article | Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocol, Computer file systems, Distributed computing architecture, Web technology, Programming language topics |

### Dataset Setup

1. **Download Dataset**
   ```bash
   # Download small_data.zip from the LLMGNN repository
   # Place it in the ./data folder
   ```

2. **Extract Data**
   ```bash
   unzip ./data/small_data.zip -d ./data/raw
   ```

3. **Verify Installation**
   ```bash
   python data/raw_data_loader.py --dataset cora_fixed_sbert
   ```