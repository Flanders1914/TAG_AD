SYSTEM_PROMPT = """
You are a TAG (Text-Attributed Graph) anomaly text generator. \
Your output must be a clean rewrite of the node's text attribute that forms a contextual anomaly. \
Obey all constraints strictly.
"""


USER_PROMPT_CORA = """
You are given a normal node from the Cora dataset which contains scientific publications categorized into seven research topics:
case-based reasoning, genetic algorithms, neural networks, probabilistic methods, reinforcement learning, \
rule learning, and theory. Each node's text attribute is the paper title and, optionally, an abstract.

Original node label (topic): {label_name}
Designated target label (topic): {designated_label}
Original text attribute (title and optional abstract): {raw_text}

Task: Rewrite the text attribute so it clearly concerns the designated label/topic: \
({designated_label}) while remaining a **contextual anomaly** relative to the original.

Requirements:
1) **Topicality**: The rewritten text must align tightly with {designated_label}.
2) **Divergence**: Make the content unrelated to the original topic ({label_name}) as much as possible.
3) **Style match**: Mimic the original tone, formality, and sentence structure; keep the length of the rewritten text roughly similar to the original.
4) **Structure preservation**: If the original includes only a title, output only a title; if it includes a title + abstract, output a title and a short abstract (no extra sections).
5) **No extras**: Do not add author names, citations, URLs, dataset commentary, title or abstract label(e.g., "Title: ", "Abstract: ").
Output ONLY the rewritten text.
"""