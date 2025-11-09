SYSTEM_PROMPT = """
You are an expert anomaly detector specializing in Text-Attributed Graphs (TAGs)
"""

NEIGHBORS_DESCRIPTION = """
**Neighbor {neighbor_index}**
Text attribute: {neighbor_text}

"""

USER_PROMPT_DETECTOR_CONTEXTUAL = """
You are analyzing a node from a text-attributed graph for anomaly detection. You will be given the text attribute of the target node \
and the text attributes of its direct neighbors. The text attributes are truncated to {max_word_num} words to reduce the length of the prompt.

**Target Node Text**
{raw_text}

**Description of {num_neighbors} direct neighbors**

{neighbors_description}
Task:
Predict an integer anomaly score (0-10) for the target node by following the analysis framework.

Analysis Framework:
{analysis_framework}

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

Output Format:
1) Start with a detailed analysis following the analysis framework
2) Conclude with "RESULT:<score>". Score should be an integer from 0 to 10. For example, "RESULT:0"
3) The result score must appear on its own line at the very end of your response

Output Example:
Analysis: This given node has ...
RESULT:5
"""



USER_PROMPT_DETECTOR_MIXED = """
You are analyzing a node from a text-attributed graph for anomaly detection. You will be given:
1) The text attribute of the target node, and the text attributes of its direct neighbors. The text attributes are truncated to {max_word_num} words to reduce the length of the prompt.
2) A textual representation describing the structure of a 2-hop subgraph centered at the target node {idx}.

**Target Node Text**
{raw_text}

**Description of {num_neighbors} direct neighbors**

{neighbors_description}

**Subgraph Structure Representation**
{graph_structure_representation}

Task:
Predict an integer anomaly score (0-10) for the target node by following the analysis framework.

Analysis Framework:
{analysis_framework}

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

Output Format:
1) Start with a detailed analysis following the analysis framework
2) Conclude with "RESULT:<score>". Score should be an integer from 0 to 10. For example, "RESULT:0"
3) The result score must appear on its own line at the very end of your response

Output Example:
Analysis: This given node has ...
RESULT:5
"""


USER_PROMPT_DETECTOR_STRUCTURAL = """
You are analyzing a node from a text-attributed graph for structural anomaly detection. You will receive a textual \
representation describing the structure of a subgraph centered around the target node {idx}.

**Subgraph Structure Representation**
{graph_structure_representation}

Task:
Predict an integer anomaly score (0-10) for the target node by following the analysis framework.

Analysis Framework:
{analysis_framework}

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

Output Format:
1) Start with a detailed analysis following the analysis framework
2) Conclude with "RESULT:<score>". Score should be an integer from 0 to 10. For example, "RESULT:0"
3) The result score must appear on its own line at the very end of your response

Output Example:
Analysis: This given node has ...
RESULT:5
"""

ANALYSIS_FRAMEWORK_CONTEXTUAL_HUMAN_DESIGNED = """An contextual anomaly node exhibits one or more of the following characteristics:
1) **Content corruption**: The text attribute contains corrupted, nonsensical, low-coherence, spammy, or irrelevant text.
2) **Neighbor inconsistency**: Weak semantic relatedness with the majority of direct neighbors; off-topic vs. local neighborhood themes.
3) **Contextual inappropriateness**: The node's content is contextually inappropriate for its position in the graph structure.
You should analyze each of the above characteristics from the following aspects:
1) **Quality Assessment**: Evaluate the quality and coherence of the target node's text attribute.
2) **Neighbor Coherence**: Assess semantic similarity and topical consistency with direct neighbors.
3) **Graph Context**: Judge whether the node fits naturally within its local graph neighborhood."""

ANALYSIS_FRAMEWORK_STRUCTURAL_HUMAN_DESIGNED = """A structural anomaly node exhibits one or more of the following characteristics:
1) **Clique-like density spike**: The node sits inside an unusually dense (near-)clique; its immediate neighbors are also heavily interconnected.
2) **Egonet surplus vs. expectation**: The node's egonet (the node and its immediate neighbors, plus all edges among them) contains far more edges or triangles than would be expected for a node of its degree or position in the graph.
3) **Boundary sparsity**: The dense core around the node has relatively few edges that cross to the outside, creating a sharp contrast between internal density and external connectivity.
You should analyze each of the above characteristics from the following aspects:
1) **Local Structural Intensity**: Measure the internal connectivity around the target node (e.g., number of edges, triangles, or density within the egonet). Compare these values to local baselines, such as the egonets of neighbors with similar degree or position.
2) **Boundary & Cut Properties**: Assess the number of edges that connect the egonet (the node and its immediate neighbors) to the rest of the graph. Determine whether there is a sharp drop in connectivity at the boundary, indicating a well-separated dense core.
3) **Community & Positional Consistency**: Evaluate whether the node's structural position and its local community structure are consistent with the broader graph. Consider if the node is embedded in a community in a way that is unusual or inconsistent with typical nodes in the graph."""

ANALYSIS_FRAMEWORK_DUMMY = """You should design an analysis framework by yourself. Then follow your analysis framework to analyze the target node."""