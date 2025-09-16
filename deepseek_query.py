# python deepseek_query.py
import yaml
import time
from openai import OpenAI

CONFIG_PATH = "./config.yaml"
DELAY_TIME = 5
REQUEST_INTERVAL = 0.2

def send_query_to_deepseek(user_prompt, system_prompt, model_name, api_key, temperature) -> str:
    """
    Send a query to Deepseek API and get the response
    """
    while True:
        try:
            time.sleep(REQUEST_INTERVAL)
            return send_query(user_prompt, system_prompt, model_name, api_key, temperature)
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            time.sleep(DELAY_TIME)


def send_query(user_prompt, system_prompt, model_name, api_key, temperature) -> str:
    """
    Send a query to Deepseek API and get the response
    """
    # send the query to Deepseek API
    openai = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    chat_completion = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=False
    )
    print(f"Making API request with model: {model_name}")
    response = chat_completion.choices[0].message.content
    return response

# test the function
if __name__ == "__main__":
    user_prompt = "Rely only \"hello\" to me"
    system_prompt = "You are a helpful assistant that only replies with \"hello\"."
    response = send_query_to_deepseek(user_prompt, system_prompt)
    print(response)