# conda activate LLM_inference
# CUDA_VISIBLE_DEVICES=0 python Local_inference.py
# CUDA_VISIBLE_DEVICES=1 python Local_inference.py
from unsloth import FastLanguageModel
import json
from detector_prompts import SYSTEM_PROMPT

dtype = None
load_in_4bit = True
load_in_8bit = False
max_seq_length = 32768
max_new_tokens = 2048
model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

def local_model_inference(model, tokenizer, input_file, output_file):
    # load the input file
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    dataset_name = input_file.split("_")[0]

    results = []
    count = 0
    # start the inference for each item
    for item in input_data:
        index = item['index']
        user_prompt = item['prompt']
        reference_text = item['ground_truth']
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # inference once
        response = inference_once(model, tokenizer, messages)
        print("Dataset name: ", dataset_name)
        print("Index: ", index)
        print(f"Response: {response}")
        # parse the response to get the prediction
        prediction = None # default value=None
        if response is not None:
            # parse the response to get the prediction
            result_idx = response.find("RESULT:")
            if result_idx != -1:
                res_str = response[result_idx+7]
                if res_str.isdigit() and int(res_str) >= 0 and int(res_str) < 10:
                    score = int(res_str)
                    if score == 1:
                        if result_idx+8 < len(response) and response[result_idx+8] == "0":
                            prediction = 1.0
                        else:
                            prediction = 0.1
                    else:
                        prediction = int(res_str)/10
           
        # save the response
        count += 1
        print()
        print("Dataset name: ", dataset_name)
        print(f"Have processed {count} nodes")
        print("Response:\n", response)
        print()
        print("Ground truth: ", reference_text)
        print("Prediction: ", prediction)
        print("="*100)
        results.append({"index": index, "ground_truth": reference_text, "prediction": prediction})
    # save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved the results to {output_file}")

def inference_once(model, tokenizer, messages) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        enable_thinking = False,
        return_tensors = "pt",
    ).to("cuda")

    num_tokens = inputs.shape[-1]
    if num_tokens >= max_seq_length-max_new_tokens:
        print(f"The number of tokens is >= {max_seq_length-max_new_tokens}, return None")
        return None
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.7)
    # discard inputs, only keep the response
    response_ids = outputs[:, num_tokens:]
    response = tokenizer.batch_decode(response_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)[0]
    return response

if __name__ == "__main__":
    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
    )
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    print("Dataset: cora_fixed_sbert_4_135")
    local_model_inference(model, tokenizer, "cora_fixed_sbert_4_135_testing_dataset.json", "cora_fixed_sbert_4_135_llama.json")
    print("Dataset: citeseer_fixed_sbert_4_159")
    local_model_inference(model, tokenizer, "citeseer_fixed_sbert_4_159_testing_dataset.json", "citeseer_fixed_sbert_4_159_llama.json")
    print("Dataset: pubmed_fixed_sbert_4_986")
    local_model_inference(model, tokenizer, "pubmed_fixed_sbert_4_986_testing_dataset.json", "pubmed_fixed_sbert_4_986_llama.json")
    print("Dataset: wikics_fixed_sbert_4_585")
    local_model_inference(model, tokenizer, "wikics_fixed_sbert_4_585_testing_dataset.json", "wikics_fixed_sbert_4_585_llama.json")