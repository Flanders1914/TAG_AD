# TAG_AD
anomaly detection for text-attributed graph

## DataSet

We use 4 datasets from 

### Description

| Dataset Name  | #Nodes   | #Edges   | Task Description                                      | Classes                                                                 |
|---------------|----------|----------|-------------------------------------------------------|-------------------------------------------------------------------------|
| CORA          | 2,708    | 5,429    | Given the title and abstract, predict the category of this paper | Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods |
| CITESEER      | 3,186    | 4,277    | Given the title and abstract, predict the category of this paper | Agents, Machine Learning, Information Retrieval, Database, Human Computer Interaction, Artificial Intelligence |
| PUBMED        | 19,717   | 44,335   | Given the title and abstract, predict the category of this paper | Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 |
| WIKICS        | 11,701   | 215,863  | Given the contents of the Wikipedia article, predict the category of this article | Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocol, Computer file systems, Distributed computing architecture, Web technology, Programming language topics |


### Download Dataset

Download small_data.zip from google drive link in (https://github.com/CurryTang/LLMGNN/tree/master)

Move small_data.zip to ./data folder

### Extract data

```bash
unzip ./data/small_data.zip -d ./data/raw
```