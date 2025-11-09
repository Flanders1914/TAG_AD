import time
from openai import OpenAI
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


DELAY_TIME = 1
REQUEST_INTERVAL = 0.2
MAX_THREADS = 20 # maximum number of threads
MAX_RETRIES = 10 # maximum number of retries

_tls = threading.local()


def get_openai_client(api_key: str) -> OpenAI:
    # get the openai client if the local thread does not have it
    if not hasattr(_tls, "client"):
        _tls.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _tls.client


def call_with_retry(user_prompt, system_prompt, index, model_name, api_key, temperature) -> str:
    """
    Call the deepseek api with exponential backoff
    """
    count = 0
    while count < MAX_RETRIES:
        try:
            time.sleep(REQUEST_INTERVAL)
            return send_query(user_prompt, system_prompt, model_name, api_key, temperature)
        except Exception as e:
            count += 1
            if count >= MAX_RETRIES:
                error_msg = f"Index: {index}\nError: {e}\nMax retries ({MAX_RETRIES}) reached. Giving up."
                print(error_msg)
                raise
            sleep_time = min(DELAY_TIME*(2**count), 60)  # Cap at 60 seconds
            error_msg = f"Index: {index}\nError: {e}\nRetrying the {count}th time after {sleep_time} seconds..."
            print(error_msg)
            time.sleep(sleep_time)


def send_query(user_prompt, system_prompt, model_name, api_key, temperature) -> str:
    """
    Send a query to Deepseek API and get the response
    """
    # get the openai client
    client = get_openai_client(api_key)
    # construct the messages
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    if "r1" in model_name.lower():
        # disable the reasoning
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            reasoning_effort="none"
        )
    else:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

    response = chat_completion.choices[0].message.content
    return response


def parse_response(response: str) -> Optional[float]:
    # parse the response to get the prediction
    prediction = None # default value=None
    result_idx = response.find("RESULT:")
    if result_idx != -1 and result_idx+7 < len(response):
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
    return prediction


def inference_on_deepseek(testing_data: List[Dict], system_prompt: str, model_name: str, api_key: str, temperature: float) -> List[Dict]:
    """
    Inference on the deepseek api, with parallel threads
    testing_data: List[Dict], has the following fields:
        index: int, the index of the data
        prompt: str, the prompt
        ground_truth: int, the ground truth
    system_prompt: str, the system prompt
    model_name: str, the model name
    api_key: str, the api key
    temperature: float, the temperature
    return: List[Dict], has the following fields: index, ground_truth,response, prediction
    """
    results = [None] * len(testing_data)
    # use parallel threads to accelerate the api calling
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(call_with_retry, data["prompt"], system_prompt, data["index"], model_name, api_key, temperature): i for i, data in enumerate(testing_data)}
        count_finished = 0
        for future in as_completed(futures):
            i = futures[future]
            response = future.result()
            prediction = parse_response(response)
            results[i] = {"index": testing_data[i]["index"],
                            "ground_truth": testing_data[i]["ground_truth"],
                            "response": response,
                            "prediction": prediction}
            count_finished += 1
            if count_finished % 50 == 0:
                print(f"Have processed {count_finished} items")
    return results