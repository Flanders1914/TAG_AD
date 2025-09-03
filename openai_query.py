# python openai_query.py

import yaml
import requests

PROMPT = """
"""
SYSTEM_PROMPT = """
"""

def send_query_to_openai(api_key, model="gpt-3.5-turbo", prompt=PROMPT, system_prompt=SYSTEM_PROMPT, temperature=0.3) -> str:
    """
    Send a query to ChatGPT API and get the response
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
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
        return None
    result = response.json()
    message_content = result["choices"][0]["message"]["content"]
    print(f"API response content: {message_content}")
    return message_content


if __name__ == "__main__":
    # test the function
    # get the api key from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    api_key = config["OPENAI_KEY"]
    model = config["OPENAI_MODEL"]
    prompt = "Rely only \"hello\" to me"
    system_prompt = "You are a helpful assistant that only replies with \"hello\"."
    temperature = 0.3
    response = send_query_to_openai(api_key, model, prompt, system_prompt, temperature)
    print(response)