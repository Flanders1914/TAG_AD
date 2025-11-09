SYSTEM_PROMPT = """
You are a TAG (Text-Attributed Graph) anomaly text generator. \
Your task is to generate contextual anomalies by rewriting node text attributes. \
Strictly adhere to all provided constraints and requirements.
"""


USER_PROMPT_CORA = """
You are given a normal node from the Cora dataset. \
The Cora dataset contains Machine Learning publications categorized into seven topics:
1. Machine Learning: Case-Based
2. Machine Learning: Genetic Algorithms
3. Machine Learning: Neural Networks
4. Machine Learning: Probabilistic Methods
5. Machine Learning: Reinforcement Learning
6. Machine Learning: Rule Learning
7. Machine Learning: Theory
Each node's text attribute is the paper title and, optionally, an abstract.

Original node label/topic: ({label_name})
Designated target label/topic: ({designated_label})
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with ({designated_label}).
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
5) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
6) **Output format**: Return only the rewritten text, formatted exactly like the original text attribute. Do not include any provenance/generation markers (e.g., "Rewritten text: "). \
The output should read naturally and contain no phrasing that suggests it was produced by a language model.
"""


USER_PROMPT_CITESEER = """
You are given a normal node from the CiteSeer dataset.
The CiteSeer dataset contains Computer Science publications categorized into six topics:
1. Computer Science: Agents
2. Computer Science: Machine Learning
3. Computer Science: Information Retrieval
4. Computer Science: Database
5. Computer Science: Human Computer Interaction
6. Computer Science: Artificial Intelligence
Each node's text attribute is the paper abstract.

Original node label/topic: ({label_name})
Designated target label/topic: ({designated_label})
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with ({designated_label}).
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
5) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
6) **Output format**: Return only the rewritten text, formatted exactly like the original text attribute. Do not include any provenance/generation markers (e.g., "Rewritten text: "). \
The output should read naturally and contain no phrasing that suggests it was produced by a language model.
"""

USER_PROMPT_PUBMED = """
You are given a normal node from the PubMed dataset.
The PubMed dataset contains Diabetes Mellitus publications categorized into three topics:
1. Diabetes Mellitus, Experimental
2. Diabetes Mellitus Type 1
3. Diabetes Mellitus Type 2
Each node's text attribute is the paper title and abstract.

Original node label/topic: ({label_name})
Designated target label/topic: ({designated_label})
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with ({designated_label}).
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
5) **No extras**: Do not add author names, citations, URLs, or dataset commentary.
6) **Output format**: Return only the rewritten text, formatted exactly like the original text attribute. Do not include any provenance/generation markers (e.g., "Rewritten text: "). \
The output should read naturally and contain no phrasing that suggests it was produced by a language model.
"""

USER_PROMPT_ARXIV = """
You are given a normal node from the OGBN-ArXiv dataset.
The OGBN-ArXiv dataset contains Computer Science publications categorized into 40 categories according to the arXiv CS subject classes.
Each node's text attribute is the paper title and abstract.

Original node label/topic: ({label_name})
Designated target label/topic: ({designated_label})
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with ({designated_label}).
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original text only includes an abstract, output only an abstract; if it includes a title and an abstract, output a title and an abstract.
5) **No extras**: Do not add author names, citations, URLs, dataset commentary, or title/abstract labels (e.g., "Title: ", "Abstract: ").
6) **Output format**: Return only the rewritten text, formatted exactly like the original text attribute. Do not include any provenance/generation markers (e.g., "Rewritten text: "). \
The output should read naturally and contain no phrasing that suggests it was produced by a language model.
"""

USER_PROMPT_WIKICS = """
You are given a normal node from the Wiki-CS dataset.
The Wiki-CS dataset contains Computer Science articles from Wikipedia categorized into ten topics:
1. Computational linguistics
2. Databases
3. Operating systems
4. Computer architecture
5. Computer security
6. Internet protocols
7. Computer file systems
8. Distributed computing architecture
9. Web technology
10. Programming language topics
Each node's text attribute is the text of the article.

Original node label/topic: ({label_name})
Designated target label/topic: ({designated_label})
Original text attribute: {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with ({designated_label}).
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original text only includes a title, output only a title; if it includes a title and a text, output a title and a text.
5) **No extras**: Do not add citations, URLs, dataset commentary, or title/text labels (e.g., "Title: ", "Text: ").
6) **Output format**: Return only the rewritten text, formatted exactly like the original text attribute. Do not include any provenance/generation markers (e.g., "Rewritten text: "). \
The output should read naturally and contain no phrasing that suggests it was produced by a language model.
"""