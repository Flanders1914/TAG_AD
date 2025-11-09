# python analysis_framework_generator.py
from paperqa import Settings, ask
import os
import re
import argparse

ANALYSIS_FRAMEWORK_EXAMPLE = """An <type_of_anomaly> anomaly node exhibits one or more of the following characteristics:
1) **<characteristic_1>**: 1 sentences
2) **<characteristic_2>**: 1 sentences
...
You should analyze each of the above characteristics from the following aspects:
1) **<aspect_1>**: 1 sentences
2) **<aspect_2>**: 1 sentences
..."""

CONTEXTUAL_ANOMALY_PROMPT = """
You task is to generate the Analysis Framework Section of a prompt used to detect contextual anomaly.
The Analysis Framework should be concise, practical, and directly usable within an LLM prompt for anomaly detection. Do not include any instructions that go beyond the scope of the inputs. Do not include any instructions that cannot be executed by the large language model(e.g., compare embeddings).
The ONLY inputs available to the model at inference time: the target node's text and the texts of its direct neighbors.
The prompt has already contained the following Rubric Section:

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

You should generate the Analysis Framework Section for the contextual anomaly without any other information such as titles, citations, references, summaries, conclusions, etc.
Here is an example:
{example}
"""

STRUCTURAL_ANOMALY_PROMPT = """
You task is to generate the Analysis Framework Section of a prompt used to detect structural anomaly.
The Analysis Framework should be concise, practical, and directly usable within an LLM prompt for anomaly detection.
Do not include any instructions that go beyond the scope of the inputs. Do not include any instructions that cannot be executed by the large language model(e.g., compare embeddings).
The ONLY inputs available to the model at inference time: the structure of a 2-hop subgraph centered at the target node. The text attribute of the target node is not available.
The prompt has already contained the following Rubric Section:

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

You should generate the Analysis Framework Section for the contextual anomaly without any other information such as titles, citations, references, summaries, conclusions, etc.
Here is an example:
{example}
"""

MIXED_ANOMALY_PROMPT = """
You task is to generate the Analysis Framework Section of a prompt used to detect both contextual and structural anomaly.
The Analysis Framework should be concise, practical, and directly usable within an LLM prompt for anomaly detection.
Do not include any instructions that go beyond the scope of the inputs. Do not include any instructions that cannot be executed by the large language model(e.g., compare embeddings).
The ONLY inputs available to the model at inference time:
1) The text attributes of the target node and its direct neighbors.
2) The structure of a 2-hop subgraph centered at the target node.
The prompt has already contained the following Rubric Section:

Rubric:
The score should be an integer from 0 to 10 (higher = stronger anomaly evidence)
**Score 0**: definitely normal
**Score 1-4**: more likely normal (minor issues)
**Score 5**: equally likely normal and anomalous
**Score 6-9**: more likely anomalous (increasing evidence)
**Score 10**: definitely anomalous

You should generate the Analysis Framework Section for the mixed anomaly without any other information such as titles, citations, references, summaries, conclusions, etc.
Here is an example:
{example}
"""


def remove_citations(text: str) -> str:
    """
    Remove citations from the text
    """
    return re.sub(r'\(.*?\)', '', text)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--type_of_anomaly", type=str, choices=["Contextual Anomaly", "Structural Anomaly", "Mixed Anomaly"])
    args.add_argument("--temperature", type=float, default=0.7)
    args.add_argument("--paper_directory", type=str, default="papers")
    args.add_argument("--output_file", type=str, required=True)
    args.add_argument("--api_key", type=str, required=True)
    args = args.parse_args()

    anomaly_type = args.type_of_anomaly
    temperature = args.temperature
    paper_directory = args.paper_directory
    api_key = args.api_key
    output_file = args.output_file

    os.environ["OPENAI_API_KEY"] = api_key
    if anomaly_type == "Contextual Anomaly":
        prompt = CONTEXTUAL_ANOMALY_PROMPT.format(example=ANALYSIS_FRAMEWORK_EXAMPLE)
    elif anomaly_type == "Structural Anomaly":
        prompt = STRUCTURAL_ANOMALY_PROMPT.format(example=ANALYSIS_FRAMEWORK_EXAMPLE)
    elif anomaly_type == "Mixed Anomaly":
        prompt = MIXED_ANOMALY_PROMPT.format(example=ANALYSIS_FRAMEWORK_EXAMPLE)
    else:
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    answer_response = ask(
        prompt,
        settings=Settings(temperature=temperature, paper_directory=paper_directory),
    )
    
    answer = remove_citations(answer_response.session.answer)
    print(answer)
    # save the answer to a file
    parent_dir = os.path.dirname(output_file)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(output_file, "w") as f:
        f.write(answer)
    print(f"Saved the analysis framework to {output_file}")

if __name__ == "__main__":
    main()