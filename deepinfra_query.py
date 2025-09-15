# python deepinfra_query.py
import yaml
import time
from openai import OpenAI

CONFIG_PATH = "./config.yaml"
DELAY_TIME = 5
REQUEST_INTERVAL = 0.2

def send_query_to_deepinfra(user_prompt, system_prompt, model_name=None) -> str:
    """
    Send a query to Deepinfra API and get the response
    """
    while True:
        try:
            time.sleep(REQUEST_INTERVAL)
            return send_query(user_prompt, system_prompt, model_name)
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            time.sleep(DELAY_TIME)


def send_query(user_prompt, system_prompt, model_name) -> str:
    """
    Send a query to Deepinfra API and get the response
    """
    try:
        # get the api key, model, and temperature from config.yaml
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        api_key = config.get("DEEPINFRA_API_KEY")
        model = config.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3-0324")
        temperature = config.get("TEMPERATURE", 0.7)
        
        # if model_name is provided, use the model_name
        if model_name:
            model = model_name

        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not found in config.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

    # send the query to Deepinfra API
    openai = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    chat_completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    print(f"Making API request with model: {model}")
    response = chat_completion.choices[0].message.content
    return response

# test the function
if __name__ == "__main__":
    user_prompt = "Rely only \"hello\" to me"
    system_prompt = "You are a helpful assistant that only replies with \"hello\"."
    response = send_query_to_deepinfra(user_prompt, system_prompt)
    print(response)