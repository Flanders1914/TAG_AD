SYSTEM_PROMPT = """
You are a TAG (Text-Attributed Graph) anomaly text generator. \
Your task is to generate contextual anomalies by rewriting node text attributes. \
Your output must be a clean rewrite of the node's text attribute that forms a contextual anomaly. \
Strictly adhere to all provided constraints and requirements.
"""


USER_PROMPT_CORA = """
You are given a normal node from the Cora dataset. \
The Cora dataset was published in 2000 and contains Machine Learning publications categorized into seven topics:
1. Machine Learning: Case-Based
2. Machine Learning: Genetic Algorithms
3. Machine Learning: Neural Networks
4. Machine Learning: Probabilistic Methods
5. Machine Learning: Reinforcement Learning
6. Machine Learning: Rule Learning
7. Machine Learning: Theory
Each node's text attribute is the paper title and, optionally, an abstract.

Original node label/topic: {label_name}
Designated target label/topic: {designated_label}
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with {designated_label}.
2) **Temporal Knowledge Limit**: Use concepts, terminology, and methods that were established **on or before 2000**.
3) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
4) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
5) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
6) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
Output ONLY the rewritten text.
"""


USER_PROMPT_CITESEER = """
You are given a normal node from the CiteSeer dataset.
The CiteSeer dataset was published in 1998 and contains Computer Science publications categorized into six topics:
1. Computer Science: Agents
2. Computer Science: Machine Learning
3. Computer Science: Information Retrieval
4. Computer Science: Database
5. Computer Science: Human Computer Interaction
6. Computer Science: Artificial Intelligence
Each node's text attribute is the paper abstract.

Original node label/topic: {label_name}
Designated target label/topic: {designated_label}
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with {designated_label}.
2) **Temporal Knowledge Limit**: Use concepts, terminology, and methods that were established **on or before 1998**.
3) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
4) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
5) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
6) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
Output ONLY the rewritten text.
"""

USER_PROMPT_PUBMED = """
You are given a normal node from the PubMed dataset.
The PubMed dataset was published in 2008 and contains Diabetes Mellitus publications categorized into three categories:
1. Diabetes Mellitus, Experimental
2. Diabetes Mellitus Type 1
3. Diabetes Mellitus Type 2
Each node's text attribute is the paper title and abstract.

Original node label/topic: {label_name}
Designated target label/topic: {designated_label}
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with {designated_label}.
2) **Temporal Knowledge Limit**: Use concepts, terminology, and methods that were established **on or before 2008**.
3) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
4) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
5) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
6) **No extras**: Do not add author names, citations, URLs, or dataset commentary.
Output ONLY the rewritten text.
"""

USER_PROMPT_ARXIV = """
You are given a normal node from the OGBN-ArXiv dataset.
The OGBN-ArXiv dataset was published in 2020 and contains Computer Science publications categorized into 40 categories according to the arXiv CS subject classes.
Each node's text attribute is the paper title and abstract.

Original node label/topic: {label_name}
Designated target label/topic: {designated_label}
Original text attribute (title and optional abstract): {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with {designated_label}.
2) **Temporal Knowledge Limit**: Use concepts, terminology, and methods that were established **on or before 2020**.
3) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
4) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
5) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
6) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
Output ONLY the rewritten text.
"""

