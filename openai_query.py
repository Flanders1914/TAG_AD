# python openai_query.py
import yaml
import requests
import time

CONFIG_PATH = "./config.yaml"
DELAY_TIME = 1
REQUEST_INTERVAL = 0.2

def send_query_to_openai(user_prompt, system_prompt, model_name=None) -> str:
    """
    Send a query to OpenAI API and get the response
    """
    retry_count = 0
    while True:
        try:
            time.sleep(REQUEST_INTERVAL)
            response = send_query(user_prompt, system_prompt, model_name)
            retry_count = 0
            return response
        except Exception as e:
            print(f"Error: {e}")
            print("User prompt tokens:", len(user_prompt.split()))
            print("User prompt:\n", user_prompt)
            print(f"Retry count: {retry_count}")
            sleep_time = DELAY_TIME*(retry_count+1)
            print(f"Sleeping for {sleep_time} seconds")
            retry_count += 1
            time.sleep(sleep_time)


def send_query(user_prompt, system_prompt, model_name) -> str:
    """
    Send a query to ChatGPT API and get the response
    """
    try:
        # get the api key, model, and temperature from config.yaml
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        api_key = config.get("OPENAI_KEY")
        model = config.get("OPENAI_MODEL", "gpt-3.5-turbo")
        temperature = config.get("TEMPERATURE", 0.7)
        
        # if model_name is provided, use the model_name
        if model_name:
            model = model_name

        if not api_key:
            raise ValueError("OPENAI_KEY not found in config.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

    # send the query to OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }
    print(f"Making API request with model: {model}")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        raise RuntimeError(f"OpenAI API request failed with status {response.status_code}: {response.text}")
    try:
        result = response.json()
        message_content = result["choices"][0]["message"]["content"]
        print(f"API response content: {message_content}")
        print()
        return message_content
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response format: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing API response: {e}")

# test the function
if __name__ == "__main__":
    user_prompt = "Rely only \"hello\" to me"
    system_prompt = "You are a helpful assistant that only replies with \"hello\"."
    response = send_query_to_openai(user_prompt, system_prompt)
    print(response)