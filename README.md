# TAG_AD: Text-Attributed Graph Anomaly Detection

## Overview

TAG_AD provides a comprehensive framework for text-attributed graph anomaly detection with the following capabilities:

- **Anomaly Generation**: Create synthetic anomalies in text-attributed graphs using multiple strategies:
  - Dummy Anomaly: Random text generation based on unigram distributions
  - LLM-Generated Contextual Anomaly: Context-aware anomaly generation using OpenAI GPT models
- **Text Encoding**: Convert text attributes to embeddings using sentence transformers
- **Anomaly Detection**: Detect anomalies using PyGOD framework algorithms with 15+ detection methods
- **Evaluation**: Comprehensive evaluation metrics including AUC, F1-score, precision@k, and recall@k

## Installation

### Prerequisites

```bash
conda create -n ad_env python=3.11
conda activate ad_env
pip install -r requirements.txt
```

### Clone and Setup

```bash
git clone git@github.com:Flanders1914/TAG_AD.git
cd TAG_AD
```

### Configuration

1. **OpenAI API Setup** (required for LLM-based anomaly generation):
   - Copy your OpenAI API key
   - Edit `config.yaml` and replace `"your_openai_api_key_here"` with your actual API key
   - Adjust model and temperature settings as needed

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

## Usage

### 1. Generate Anomalies

#### Dummy Anomaly Generation
```bash
python make_anomaly.py
```

#### LLM-Generated Contextual Anomaly
Modify `make_anomaly.py` to use the LLM-based generator:
```python
from anomaly_generator import llm_generated_contextual_anomaly_generator
from data.raw_data_loader import LLMGNNDataLoader

loader = LLMGNNDataLoader(data_dir="data/raw")
data = loader.load_dataset("cora_fixed_sbert")
data = llm_generated_contextual_anomaly_generator(
    data=data,
    dataset_name="cora_fixed_sbert",
    n=1,
    anomaly_type=2,  # LLM-Generated Contextual Anomaly
    random_seed=42,
    k_neighbors=2
)
```

### 2. Run Baseline Experiments

```bash
python pygod_baseline.py --dataset_name cora_fixed_sbert_2_1 --experiment_num 20 --k 5
```

**Parameters:**
- `--dataset_name`: Name of the dataset to use
- `--experiment_num`: Number of experimental runs (default: 20)
- `--k`: Number of top-k anomalies for precision/recall calculation (default: 5)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--output_file`: Output file for results (default: results.json)

### 3. Supported Detection Methods

The framework includes 15+ anomaly detection algorithms from PyGOD:
- AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD
- DMGD, DOMINANT, DONE, GAE, OCGNN
- ONE, Radar, and more

### 4. Data Loading and Analysis

```bash
# Load and analyze datasets
python data/raw_data_loader.py --dataset cora_fixed_sbert --data_dir data/raw
```

## Project Structure

```
TAG_AD/
├── anomaly_generator/           # Anomaly generation modules
│   ├── __init__.py
│   ├── dummy_anomaly.py        # Random text anomaly generation
│   ├── LLM_contextual_anomaly.py  # LLM-based anomaly generation
│   ├── openai_query.py         # OpenAI API interface
│   ├── prompts.py              # Prompt templates
│   ├── utils.py                # Utility functions
│   └── anomaly_list.py         # Anomaly type definitions
├── data/
│   ├── raw/                    # Dataset files (.pt format)
│   └── raw_data_loader.py      # Dataset loading utilities
├── make_anomaly.py             # Script for anomaly generation
├── pygod_baseline.py           # Baseline evaluation script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Features

### Anomaly Types
- **Type 0**: No Anomaly (normal nodes)
- **Type 1**: Dummy Anomaly (random text based on unigram distributions)
- **Type 2**: LLM-Generated Contextual Anomaly (contextually inconsistent text)

### Evaluation Metrics
- ROC AUC Score
- Average Precision
- F1 Score
- Precision@k
- Recall@k

### Error Handling and Robustness
- Comprehensive error handling for API calls
- Input validation for all major functions
- Device-aware tensor operations
- Graceful handling of missing configuration

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **OpenAI API Errors**: Check your API key in `config.yaml` and ensure you have credits
3. **CUDA Errors**: The framework supports both CPU and GPU operations
4. **Dataset Errors**: Verify dataset files are properly extracted in `data/raw/`

## Security Note

- Never commit API keys to version control
- The `config.yaml` file is gitignored for security
- Use environment variables for production deployments