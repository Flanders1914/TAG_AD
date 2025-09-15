SYSTEM_PROMPT = """
You are an expert anomaly detector specializing in Text-Attributed Graphs (TAGs).
"""

NEIGHBORS_DESCRIPTION = """
**Neighbor {neighbor_index}**
Text attribute: {neighbor_text}

"""

USER_PROMPT_DETECTOR = """
You are analyzing a node from a text-attributed graph for anomaly detection. You will be given the text attribute of the target node \
and the text attributes of its direct neighbors.

**Target Node Text**
{raw_text}

**Direct Neighbors, ({num_neighbors} total)**
{neighbors_description}
Task:
Determine whether the target node is an anomaly by analyzing the content of its text attribute and its relationships with direct neighbors.
An anomaly node exhibits one or more of the following characteristics:
1) **Content corruption**: The text attribute contains corrupted, nonsensical, or irrelevant content
2) **Neighbor inconsistency**: The node has weak semantic relationships with most of its direct neighbors
3) **Contextual inappropriateness**: The node's content is contextually inappropriate for its position in the graph structure

Analysis Framework:
1) **Quality Assessment**: Evaluate the quality and coherence of the target node's text attribute
2) **Neighbor Coherence**: Assess semantic similarity and topical consistency with direct neighbors
3) **Graph Context**: Judge whether the node fits naturally within its local graph neighborhood

Output Format:
1) Provide a detailed analysis examining each aspect mentioned above
2) Conclude with either "RESULT:TRUE" (anomaly detected) or "RESULT:FALSE" (normal node)
3) The result label must appear exactly as "RESULT:TRUE" or "RESULT:FALSE" on its own line at the very end of your response

Output Example:
Analysis: This given node has ...
RESULT:TRUE
"""
