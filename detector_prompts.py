SYSTEM_PROMPT = """
You are an expert anomaly detector specializing in Text-Attributed Graphs (TAGs)
"""

NEIGHBORS_DESCRIPTION = """
**Neighbor {neighbor_index}**
Text attribute: {neighbor_text}

"""

USER_PROMPT_DETECTOR = """
You are analyzing a node from a text-attributed graph for anomaly detection. You will be given the text attribute of the target node \
and the text attributes of its direct neighbors

**Target Node Text**
{raw_text}

**Description of {num_neighbors} direct neighbors**
The text attributes of the neighbors are truncated to {max_neighbors_word_num} words to reduce the length of the prompt

{neighbors_description}
Task:
Predict an integer anomaly score (0-10) for the target node by analyzing its text and its relationships with its neighbors.
An anomaly node exhibits one or more of the following characteristics:
1) **Content corruption**: The text attribute contains corrupted, nonsensical, low-coherence, spammy, or irrelevant text
2) **Neighbor inconsistency**: weak semantic relatedness with the majority of direct neighbors; off-topic vs. local neighborhood themes
3) **Contextual inappropriateness**: The node's content is contextually inappropriate for its position in the graph structure

Analysis Framework:
1) **Quality Assessment**: Evaluate the quality and coherence of the target node's text attribute
2) **Neighbor Coherence**: Assess semantic similarity and topical consistency with direct neighbors
3) **Graph Context**: Judge whether the node fits naturally within its local graph neighborhood
4) **Score Prediction**: Choose an integer score (0-10) using the rubric

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

Output Format:
1) Start with a detailed analysis examining each aspect mentioned above
2) Conclude with "RESULT:<score>". Score should be an integer from 0 to 10. For example, "RESULT:0"
3) The result score must appear on its own line at the very end of your response

Output Example:
Analysis: This given node has ...
RESULT:5
"""
