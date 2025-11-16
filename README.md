# TAG_AD: Text-Attributed Graph Anomaly Detection

## Overview

TAG_AD is an integrated framework for generating, detecting, and analyzing anomalies in text-attributed graphs. It supports the creation of challenging benchmark datasets, baseline evaluations, and advanced LLM-powered detection pipelines for both contextual and structural anomaly detection tasks.

## Installation

### Prerequisites

```bash
conda create -n ad_env python=3.11
conda activate ad_env
pip install -r requirements.txt
```

### API Configuration
```bash
export OPENAI_API_KEY="OPENAI_API_KEY"
export DEEPINFRA_API_KEY="DEEPINFRA_API_KEY"
export DEEPSEEK_API_KEY="DEEPSEEK_API_KEY"
export TEMPERATURE=0
```

## Datasets

We use the following 4 datasets from [LLMGNN repository](https://github.com/CurryTang/LLMGNN/tree/master)

### Dataset Statistics

| Dataset Name  | #Nodes   | #Edges   | Task Description                                      | Classes                                                                 |
|---------------|----------|----------|-------------------------------------------------------|-------------------------------------------------------------------------|
| CORA          | 2,708    | 5,429    | Given the title and abstract, predict the category of this paper | Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods |
| CITESEER      | 3,186    | 4,277    | Given the title and abstract, predict the category of this paper | Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence |
| PUBMED        | 19,717   | 44,335   | Given the title and abstract, predict the category of this paper | Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 |
| wiki-cs       | 11,701   | 43,1726  | Given the title and abstract, predict the category of this paper | Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, Programming language topics |

### Dataset Setup

To set up the datasets:

1. **Download**: Get the archive [`datasets.tar.gz`](https://drive.google.com/file/d/1RqtsxZIut5D0G5gfdjTIDq9i0OTJF4fp/view?usp=sharing) from Google Drive.
2. **Extract**: Unpack the archive:
   ```bash
   tar -xzvf datasets.tar.gz
   ```
3. **Move files**: Place all extracted files and folders into the `data/raw` directory (create it if it doesn't exist).

Your folder structure should look like:
```
TAG_AD/
└── data/
    └── raw/
        ├── <dataset files>
        └── ...
```

## Usage

### 1. Generate Synthetic Anomalies

Use `make_anomaly.py` to create datasets with injected anomalies. Specify the anomaly type, count, and other settings as needed:

```bash
python make_anomaly.py \
  --dataset_name <dataset_name> \
  --anomaly_type <1|2|3|4|5> \
  --anomaly_num <number_of_anomalies> \
  --data_dir <data/raw> \
  --output_dir <data/generated> \
  --is_map_label <True|False>
```

**Parameters:**
- `<dataset_name>`: Name of the dataset (e.g., `pubmed_fixed_sbert_5_290`)
- `<anomaly_type>`:  
    - 1: Dummy anomaly  
    - 2: LLM-generated contextual anomaly  
    - 3: Traditional contextual anomaly  
    - 4: Global anomaly
    - 5: Structural anomaly  
- `<anomaly_num>`: Number of anomalous nodes to generate
- `<data_dir>`: (Optional; default: `data/raw`) Source data directory
- `<output_dir>`: (Optional; default: `data/generated`) Output data directory for generated data
- `<is_map_label>`: Use `True` when generating anomalies for a fresh dataset, otherwise `False` (for additional runs on the same dataset)

**Example:**

Generate 100 LLM-generated contextual anomalies on the PubMed dataset:
```bash
python make_anomaly.py \
  --dataset_name pubmed_fixed_sbert \
  --anomaly_type 2 \
  --anomaly_num 100 \
  --is_map_label True
```

Tips:
- For generating multiple anomaly types on the same dataset, set `--is_map_label` to `True` for the first type, then `False` for subsequent types.
- Output files will be saved in the directory specified by `--output_dir`.

---

### 2. Evaluate Synthetic Anomalies using PyGOD

Use `pygod_baseline.py` to evaluate anomaly detection baselines on your generated datasets.

```bash
python pygod_baseline.py \
  --data_dir <data/generated> \
  --dataset_name <dataset_name> \
  --random_seed <seed> \
  --experiment_num <repeat_times> \
  --output_file <output_json> \
  --k <top_k> \
  --is_structural <False|True>
```

**Parameters:**
- `<data_dir>`: (Optional; default: `data/generated`) Directory where generated datasets are stored
- `<dataset_name>`: Name of the dataset to evaluate (e.g., `pubmed_fixed_sbert_2_100`)
- `<random_seed>`: (Optional; default: 42) Random seed for reproducibility
- `<experiment_num>`: (Optional; default: 1) Number of runs to repeat experiment for statistics
- `<output_file>`: (Optional; default: `results.json`) Where to save evaluation results
- `<k>`: (Optional; default: 20) Number of anomalies used for precision-at-k/recall-at-k
- `<is_structural>`: Set to `True` if evaluating structural anomaly datasets, otherwise `False`

**Example:**

Evaluate a generated PubMed anomaly dataset, running 3 times for statistical robustness:
```bash
python pygod_baseline.py \
  --data_dir data/generated \
  --dataset_name pubmed_fixed_sbert_2_100 \
  --experiment_num 3 \
  --output_file result_pubmed_2_100.json \
  --k 20
```

After running this command, the evaluation results will be saved to the specified output JSON file, including metrics such as AUC, Average Precision, F1 score, Precision@k, and Recall@k for a variety of baselines.

### 3. Create an Analysis Framework for LLM-based Anomaly Detection using RAG

To generate an analysis framework leveraging Retrieval-Augmented Generation (RAG) for text-attributed graph anomaly detection:

1. **Add Papers:**  
   Place relevant text-attributed graph anomaly detection papers (in PDF form) into the `papers/` directory. These papers will serve as the knowledge base for the RAG pipeline.

2. **Run the Framework Generation Script:**  
   You can use the provided shell script (`rag_generate.sh`) or run the Python command directly. This will generate an analysis framework based on your collection of papers using an LLM, with controllable temperature and other parameters.

   **Recommended (using `rag_generate.sh`):**
   ```bash
   bash rag_generate.sh
   ```
   Ensure your OpenAI API key is properly set in your environment. You can edit `rag_generate.sh` to customize arguments or the key.

   **Or direct usage:**
   ```bash
   python analysis_framework_generator.py \
     --type_of_anomaly "Contextual Anomaly" \
     --temperature 0.7 \
     --paper_directory "papers" \
     --output_file "analysis_framework_contextual.txt" \
     --api_key <OPENAI_API_KEY>
   ```

_Notes:_  
- The RAG process will use the contents of all papers found in the `papers/` directory.
- Adjust temperature or anomaly type (e.g., "Structural Anomaly", "Contextual Anomaly", "Mixed Anomaly") as needed using the `--type_of_anomaly` and `--temperature` options.

### 4. Run LLM-Powered Anomaly Detection Pipeline

Once you have prepared your dataset and generated an analysis framework (see sections above), you can apply large language models (LLMs) to the anomaly detection task on text-attributed graph data.

This pipeline supports the following LLM backends:

- **DeepInfra API:**
  - `deepseek-ai/DeepSeek-V3-0324`
  - `Qwen/Qwen3-14B`
  - `google/gemma-3-27b-it`
- **DeepSeek API:**
  - `deepseek-chat`
- **OpenAI API:**
  - `gpt-4o-mini`

Specify the desired model name using the `--model_name` argument when running `LLM_ad_detection.py`. Make sure the corresponding API key is set as an environment variable (`OPENAI_API_KEY`, `DEEPINFRA_API_KEY`, or other as required).

#### Step-by-step Usage

1. **Prepare Required Files and Environment**
   - Ensure your dataset (such as `"pubmed_fixed_sbert_2_100.pt"`) is available.
   - Have an analysis framework file generated (e.g., `"analysis_framework_contextual.txt"`).
   - Confirm your API keys for any required LLM (OpenAI, DeepInfra, DeepSeek, etc.) are set as environment variables, e.g.:
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"
     ```

2. **Run the Detection Script**
   - The main detection pipeline is implemented in `LLM_ad_detection.py`. Run directly:
   ```bash
   python LLM_ad_detection.py \
     --anomaly_type "Contextual Anomaly" \
     --analysis_framework_path "analysis_framework_contextual.txt" \
     --dataset_file "pubmed_fixed_sbert_2_100.pt" \
     --output_dir "LLM_results" \
     --output_file "pubmed_fixed_sbert_2_100_gemma.json" \
     --model_name "google/gemma-3-27b-it" \
     --max_nodes 1000
   ```

   **Key Arguments:**
   - `--anomaly_type`: One of `"Contextual Anomaly"`, `"Structural Anomaly"`, or `"Mixed Anomaly"`.
   - `--analysis_framework_path`: Path to the framework file generated in the previous step.
   - `--dataset_file`: Processed dataset file (.pt file as produced by the data preparation step).
   - `--output_dir`: Directory to save results.
   - `--output_file`: Name for the output results JSON.
   - `--model_name`: The model to use, e.g., `"google/gemma-3-27b-it"`, `"deepseek-ai/DeepSeek-V3-0324"`, etc.
   - `--max_nodes`: Optional, maximum number of nodes (samples) to process.
   - `--use_human_designed_analysis_framework`: Use the provided experted designed framework, rather than LLM-generated.
   - `--use_dummy`: When this flag is set, the script ignores any loaded analysis framework file and instead uses a minimal, placeholder (dummy) analysis framework prompt for each node. You can read the corresponding dummy framework content in `detector_prompts.py` under `ANALYSIS_FRAMEWORK_DUMMY`. This is designed for ablation studies without domain-specific logic.

3. **Results**
   - The script will save detailed per-node anomaly scores and predictions in the specified output file (in JSON format), in the provided output directory.
   - You can use this output for further result analysis or benchmarking.

#### Example: Using a Human-Designed Analysis Framework

This mode uses the expert-designed, interpretable analysis framework provided in `detector_prompts.py` (see `ANALYSIS_FRAMEWORK_CONTEXTUAL_HUMAN_DESIGNED` or `ANALYSIS_FRAMEWORK_STRUCTURAL_HUMAN_DESIGNED` for reference), which is injected at runtime. Be sure to specify the appropriate anomaly type and framework path.

```bash
python LLM_ad_detection.py \
  --anomaly_type "Contextual Anomaly" \
  --analysis_framework_path "analysis_framework_contextual.txt" \
  --dataset_file "pubmed_fixed_sbert_2_100.pt" \
  --output_dir "LLM_results" \
  --output_file "pubmed_fixed_sbert_2_100_gemma_human_designed.json" \
  --model_name "google/gemma-3-27b-it" \
  --max_nodes 1000 \
  --use_human_designed_analysis_framework
```

- When `--use_human_designed_analysis_framework` is specified, the model will use the corresponding human-designed analysis framework for the selected anomaly type. The actual content can be found in `detector_prompts.py`. The `--analysis_framework_path` file is still needed for input, but it will be overridden by the hard-coded human-designed framework (unless `--use_dummy` is also used).
---

#### Example: Dummy Framework Mode (Ablation)

```bash
python LLM_ad_detection.py \
  --anomaly_type "Contextual Anomaly" \
  --analysis_framework_path "analysis_framework_contextual.txt" \
  --dataset_file "pubmed_fixed_sbert_2_100.pt" \
  --output_dir "LLM_results" \
  --output_file "pubmed_fixed_sbert_2_100_gemma_human_designed_dummy.json" \
  --model_name "google/gemma-3-27b-it" \
  --max_nodes 1000 \
  --use_human_designed_analysis_framework \
  --use_dummy
```

For more customizations and an overview of all available arguments, review the `LLM_ad_detection.py` file and refer to comments in `experiment.sh`.

**Note:**  
- Make sure to use the correct API key(s) for your chosen LLM provider.
- The analysis framework file and data file paths must correspond to your setup.

For more advanced multi-run experiments or evaluation of different models, modify or extend `experiment.sh` as needed.



